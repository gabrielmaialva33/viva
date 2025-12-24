defmodule Viva.AI.LLM.AudioEnhanceClient do
  @moduledoc """
  Audio enhancement client for premium voice quality.

  Uses:
  - `nvidia/studiovoice` - Studio-quality audio enhancement
  - `nvidia/Background Noise Removal` - Remove background noise

  ## Features

  - Remove background noise from user audio
  - Enhance speech clarity for transcription
  - Studio-quality voice processing
  - Real-time audio enhancement
  """
  require Logger

  # === Types ===

  @type audio_result :: {:ok, binary()} | {:ok, {:url, String.t()}} | {:error, term()}
  @type quality_metrics :: %{
          noise_level: float(),
          clarity: float(),
          loudness_db: float(),
          speech_ratio: float(),
          quality_score: float()
        }
  @type stream_state :: %{
          model: String.t(),
          callback: (term() -> any()),
          buffer: binary(),
          sample_rate: pos_integer()
        }

  @doc """
  Enhance audio to studio quality.
  Improves clarity, removes noise, and balances levels.

  ## Options

  - `:sample_rate` - Output sample rate (default: 48000)
  - `:enhance_vocals` - Focus on voice enhancement (default: true)
  """
  @spec enhance(binary(), keyword()) :: audio_result()
  def enhance(audio_data, opts \\ []) do
    model = Keyword.get(opts, :model, Viva.AI.LLM.model(:audio_enhance))
    sample_rate = Keyword.get(opts, :sample_rate, 48_000)
    enhance_vocals = Keyword.get(opts, :enhance_vocals, true)

    body = %{
      model: model,
      audio: Base.encode64(audio_data),
      sample_rate: sample_rate,
      enhance_vocals: enhance_vocals
    }

    case Viva.AI.LLM.request("/audio/enhance", body, timeout: 60_000) do
      {:ok, %{"audio" => enhanced_audio}} ->
        {:ok, Base.decode64!(enhanced_audio)}

      {:ok, %{"url" => url}} ->
        {:ok, {:url, url}}

      {:error, reason} ->
        Logger.error("Audio enhancement error: #{inspect(reason)}")
        {:error, reason}
    end
  end

  @doc """
  Remove background noise from audio.
  Optimized for speech preservation.

  ## Options

  - `:aggressiveness` - Noise removal level: "low", "medium", "high" (default: "medium")
  - `:preserve_speech` - Prioritize speech preservation (default: true)
  """
  @spec remove_noise(binary(), keyword()) :: {:ok, binary()} | {:error, term()}
  def remove_noise(audio_data, opts \\ []) do
    model = Keyword.get(opts, :model, Viva.AI.LLM.model(:noise_removal))
    aggressiveness = Keyword.get(opts, :aggressiveness, "medium")
    preserve_speech = Keyword.get(opts, :preserve_speech, true)

    body = %{
      model: model,
      audio: Base.encode64(audio_data),
      aggressiveness: aggressiveness,
      preserve_speech: preserve_speech
    }

    case Viva.AI.LLM.request("/audio/denoise", body, timeout: 30_000) do
      {:ok, %{"audio" => denoised_audio}} ->
        {:ok, Base.decode64!(denoised_audio)}

      {:error, reason} ->
        Logger.error("Noise removal error: #{inspect(reason)}")
        {:error, reason}
    end
  end

  @doc """
  Full audio processing pipeline: denoise then enhance.
  """
  @spec process_full(binary(), keyword()) :: {:ok, binary()} | {:error, term()}
  def process_full(audio_data, opts \\ []) do
    with {:ok, denoised} <- remove_noise(audio_data, opts) do
      enhance(denoised, opts)
    end
  end

  @doc """
  Stream audio enhancement in real-time.
  Returns a stream handler for sending audio chunks.
  """
  @spec stream_enhance((term() -> any()), keyword()) :: {:ok, stream_state()}
  def stream_enhance(callback, opts \\ []) do
    model = Keyword.get(opts, :model, Viva.AI.LLM.model(:audio_enhance))

    {:ok,
     %{
       model: model,
       callback: callback,
       buffer: <<>>,
       sample_rate: Keyword.get(opts, :sample_rate, 48_000)
     }}
  end

  @doc """
  Send audio chunk to streaming enhancement.
  """
  @spec stream_chunk(stream_state(), binary()) :: {:ok, stream_state()} | {:error, term()}
  def stream_chunk(stream_state, audio_chunk) do
    body = %{
      model: stream_state.model,
      audio: Base.encode64(audio_chunk),
      sample_rate: stream_state.sample_rate,
      stream: true
    }

    case Viva.AI.LLM.request("/audio/enhance", body) do
      {:ok, %{"audio" => enhanced_chunk, "is_final" => is_final}} ->
        stream_state.callback.({:audio, Base.decode64!(enhanced_chunk), is_final})
        {:ok, stream_state}

      {:error, reason} ->
        {:error, reason}
    end
  end

  @doc """
  Enhance audio before transcription.
  Optimized for ASR accuracy.
  """
  @spec enhance_for_transcription(binary(), keyword()) :: {:ok, binary()} | {:error, term()}
  def enhance_for_transcription(audio_data, opts \\ []) do
    with {:ok, denoised} <-
           remove_noise(audio_data,
             aggressiveness: "high",
             preserve_speech: true
           ) do
      enhance(denoised, Keyword.merge(opts, enhance_vocals: true))
    end
  end

  @doc """
  Normalize audio levels for consistent volume.
  """
  @spec normalize_levels(binary(), keyword()) :: {:ok, binary()} | {:error, term()}
  def normalize_levels(audio_data, opts \\ []) do
    target_db = Keyword.get(opts, :target_db, -14.0)

    body = %{
      model: Viva.AI.LLM.model(:audio_enhance),
      audio: Base.encode64(audio_data),
      normalize: true,
      target_loudness: target_db
    }

    case Viva.AI.LLM.request("/audio/normalize", body) do
      {:ok, %{"audio" => normalized}} ->
        {:ok, Base.decode64!(normalized)}

      {:error, reason} ->
        {:error, reason}
    end
  end

  @doc """
  Analyze audio quality metrics.
  """
  @spec analyze_quality(binary()) :: {:ok, quality_metrics()} | {:error, term()}
  def analyze_quality(audio_data) do
    body = %{
      model: Viva.AI.LLM.model(:audio_enhance),
      audio: Base.encode64(audio_data),
      task: "analyze"
    }

    case Viva.AI.LLM.request("/audio/analyze", body) do
      {:ok, %{"metrics" => metrics}} ->
        {:ok,
         %{
           noise_level: Map.get(metrics, "noise_level", 0.0),
           clarity: Map.get(metrics, "clarity", 0.0),
           loudness_db: Map.get(metrics, "loudness_db", -20.0),
           speech_ratio: Map.get(metrics, "speech_ratio", 0.0),
           quality_score: Map.get(metrics, "quality_score", 0.5)
         }}

      {:error, reason} ->
        {:error, reason}
    end
  end

  @doc """
  Check if audio needs enhancement.
  Returns true if quality is below threshold.
  """
  @spec needs_enhancement?(binary(), float()) :: boolean()
  def needs_enhancement?(audio_data, threshold \\ 0.7) do
    case analyze_quality(audio_data) do
      {:ok, %{quality_score: score}} -> score < threshold
      _ -> true
    end
  end

  @doc """
  Conditionally enhance audio only if needed.
  """
  @spec smart_enhance(binary(), keyword()) :: {:ok, binary()} | {:error, term()}
  def smart_enhance(audio_data, opts \\ []) do
    threshold = Keyword.get(opts, :threshold, 0.7)

    if needs_enhancement?(audio_data, threshold) do
      process_full(audio_data, opts)
    else
      {:ok, audio_data}
    end
  end
end
