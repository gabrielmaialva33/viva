defmodule Viva.AI.LLM.TtsClient do
  @moduledoc """
  Text-to-Speech client using NVIDIA Magpie TTS.

  Uses `nvidia/magpie-tts-multilingual` for natural, expressive voices
  in multiple languages including Brazilian Portuguese.

  ## Features

  - Multilingual support (pt-BR, en-US, es-ES, etc.)
  - Voice cloning from audio samples
  - Expressive and natural speech
  - Streaming audio output
  """
  @behaviour Viva.AI.Pipeline.Stage

  require Logger

  alias Viva.AI.LLM
  alias Viva.Avatars.Avatar

  # === Types ===

  @type voice_settings :: %{
          voice_id: String.t(),
          speed: float(),
          pitch: float(),
          language: String.t()
        }

  @doc """
  Synthesize speech from text.

  ## Options

  - `:voice_id` - Voice preset ID or custom voice
  - `:language` - Language code (default: "pt-BR")
  - `:speed` - Speech speed multiplier (default: 1.0)
  - `:pitch` - Pitch adjustment (default: 0.0)
  - `:format` - Audio format: "wav", "mp3", "ogg" (default: "wav")
  """
  @spec synthesize(String.t(), keyword()) :: {:ok, binary()} | {:error, term()}
  def synthesize(text, opts \\ []) do
    model = Keyword.get(opts, :model, LLM.model(:tts))
    language = Keyword.get(opts, :language, "pt-BR")
    voice_id = Keyword.get(opts, :voice_id, default_voice(language))

    body = %{
      model: model,
      input: text,
      voice: voice_id,
      language: language,
      speed: Keyword.get(opts, :speed, 1.0),
      pitch: Keyword.get(opts, :pitch, 0.0),
      response_format: Keyword.get(opts, :format, "wav")
    }

    case LLM.request("/audio/speech", body) do
      {:ok, audio_data} ->
        {:ok, audio_data}

      {:error, reason} ->
        Logger.error("TTS synthesis error: #{inspect(reason)}")
        {:error, reason}
    end
  end

  @doc """
  Synthesize speech with streaming output.
  Calls the callback with audio chunks as they're generated.
  """
  @spec synthesize_stream(String.t(), (binary() -> any()), keyword()) ::
          {:ok, Req.Response.t()} | {:error, term()}
  def synthesize_stream(text, callback, opts \\ []) do
    model = Keyword.get(opts, :model, LLM.model(:tts))
    language = Keyword.get(opts, :language, "pt-BR")
    voice_id = Keyword.get(opts, :voice_id, default_voice(language))

    body = %{
      model: model,
      input: text,
      voice: voice_id,
      language: language,
      speed: Keyword.get(opts, :speed, 1.0),
      response_format: "wav",
      stream: true
    }

    Viva.AI.LLM.stream_request("/audio/speech", body, callback)
  end

  @doc """
  Clone a voice from an audio sample.
  Returns a voice_id that can be used for synthesis.
  """
  @spec clone_voice(binary(), keyword()) :: {:ok, String.t()} | {:error, term()}
  def clone_voice(audio_sample, opts \\ []) do
    model = Keyword.get(opts, :model, "nvidia/magpie-tts-zeroshot")
    name = Keyword.get(opts, :name, "custom_voice")

    body = %{
      model: model,
      audio: Base.encode64(audio_sample),
      name: name
    }

    case LLM.request("/audio/speech_clone", body) do
      {:ok, %{"voice_id" => voice_id}} ->
        {:ok, voice_id}

      {:error, reason} ->
        Logger.error("Voice cloning error: #{inspect(reason)}")
        {:error, reason}
    end
  end

  @doc """
  Get avatar voice settings based on personality.
  """
  @spec voice_for_avatar(Avatar.t()) :: voice_settings()
  def voice_for_avatar(avatar) do
    personality = avatar.personality
    gender = avatar.gender

    %{
      voice_id: select_voice(gender, personality),
      speed: voice_speed(personality),
      pitch: voice_pitch(personality, gender),
      language: personality.native_language
    }
  end

  @doc """
  Synthesize avatar speech with personality-based voice.
  """
  @spec avatar_speak(Avatar.t(), String.t(), keyword()) :: {:ok, binary()} | {:error, term()}
  def avatar_speak(avatar, text, opts \\ []) do
    voice_settings = voice_for_avatar(avatar)

    synthesize(
      text,
      Keyword.merge(
        [
          voice_id: voice_settings.voice_id,
          speed: voice_settings.speed,
          pitch: voice_settings.pitch,
          language: voice_settings.language
        ],
        opts
      )
    )
  end

  defp default_voice("pt-BR"), do: "magpie-pt-br-female-1"
  defp default_voice("en-US"), do: "magpie-en-us-female-1"
  defp default_voice("es-ES"), do: "magpie-es-es-female-1"
  defp default_voice(_), do: "magpie-multilingual-1"

  defp select_voice(:male, personality) do
    # Select male voice based on personality traits
    cond do
      personality.extraversion > 0.7 -> "magpie-pt-br-male-energetic"
      personality.neuroticism > 0.6 -> "magpie-pt-br-male-soft"
      personality.agreeableness > 0.7 -> "magpie-pt-br-male-warm"
      true -> "magpie-pt-br-male-1"
    end
  end

  defp select_voice(:female, personality) do
    cond do
      personality.extraversion > 0.7 -> "magpie-pt-br-female-energetic"
      personality.neuroticism > 0.6 -> "magpie-pt-br-female-soft"
      personality.agreeableness > 0.7 -> "magpie-pt-br-female-warm"
      true -> "magpie-pt-br-female-1"
    end
  end

  defp select_voice(_, _), do: "magpie-pt-br-neutral-1"

  defp voice_speed(personality) do
    # Extraverts speak faster, high conscientiousness = measured pace
    base = 1.0
    extraversion_factor = (personality.extraversion - 0.5) * 0.2
    conscientiousness_factor = (personality.conscientiousness - 0.5) * -0.1

    (base + extraversion_factor + conscientiousness_factor)
    |> max(0.8)
    |> min(1.3)
  end

  defp voice_pitch(personality, gender) do
    # Adjust pitch based on neuroticism and gender
    base = if gender == :male, do: -0.1, else: 0.1
    neuroticism_factor = (personality.neuroticism - 0.5) * 0.1

    (base + neuroticism_factor)
    |> max(-0.3)
    |> min(0.3)
  end
end
