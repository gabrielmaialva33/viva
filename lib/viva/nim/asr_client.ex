defmodule Viva.Nim.AsrClient do
  @moduledoc """
  Automatic Speech Recognition client using NVIDIA Parakeet.

  Uses `nvidia/parakeet-1.1b-rnnt-multilingual-asr` for high accuracy
  transcription in 25 languages including Brazilian Portuguese.

  ## Features

  - 25 language support
  - Real-time streaming transcription
  - Punctuation and timestamps
  - Speaker diarization
  """
  require Logger

  alias Viva.Nim

  @doc """
  Transcribe audio to text.

  ## Options

  - `:language` - Target language code (default: "pt-BR")
  - `:punctuation` - Include punctuation (default: true)
  - `:timestamps` - Include word timestamps (default: false)
  - `:diarization` - Enable speaker diarization (default: false)
  """
  def transcribe(audio_data, opts \\ []) do
    model = Keyword.get(opts, :model, Nim.model(:asr))
    language = Keyword.get(opts, :language, "pt-BR")

    body = %{
      model: model,
      audio: encode_audio(audio_data),
      language: language,
      punctuation: Keyword.get(opts, :punctuation, true),
      timestamps: Keyword.get(opts, :timestamps, false),
      diarization: Keyword.get(opts, :diarization, false)
    }

    case Nim.request("/audio/transcriptions", body) do
      {:ok, %{"text" => text} = response} ->
        result = %{
          text: text,
          language: Map.get(response, "language", language),
          duration: Map.get(response, "duration"),
          words: Map.get(response, "words", []),
          segments: Map.get(response, "segments", [])
        }

        {:ok, result}

      {:error, reason} ->
        Logger.error("ASR transcription error: #{inspect(reason)}")
        {:error, reason}
    end
  end

  @doc """
  Stream audio for real-time transcription.
  Calls the callback with partial transcripts as they're generated.
  """
  def transcribe_stream(callback, opts \\ []) do
    model = Keyword.get(opts, :model, Nim.model(:asr))
    language = Keyword.get(opts, :language, "pt-BR")

    # Return a stream handler that can receive audio chunks
    {:ok,
     %{
       model: model,
       language: language,
       callback: callback,
       buffer: <<>>,
       partial_text: ""
     }}
  end

  @doc """
  Send audio chunk to streaming transcription.
  """
  def stream_audio_chunk(stream_state, audio_chunk) do
    body = %{
      model: stream_state.model,
      audio: Base.encode64(audio_chunk),
      language: stream_state.language,
      stream: true
    }

    case Nim.request("/audio/transcriptions", body) do
      {:ok, %{"text" => text, "is_final" => is_final}} ->
        stream_state.callback.({:transcript, text, is_final})
        {:ok, %{stream_state | partial_text: text}}

      {:error, reason} ->
        {:error, reason}
    end
  end

  @doc """
  Transcribe audio file from path.
  """
  def transcribe_file(file_path, opts \\ []) do
    case File.read(file_path) do
      {:ok, audio_data} ->
        transcribe(audio_data, opts)

      {:error, reason} ->
        {:error, {:file_error, reason}}
    end
  end

  @doc """
  Detect the language of spoken audio.
  """
  def detect_language(audio_data) do
    body = %{
      model: Nim.model(:asr),
      audio: encode_audio(audio_data),
      task: "language_detection"
    }

    case Nim.request("/audio/transcriptions", body) do
      {:ok, %{"language" => language, "confidence" => confidence}} ->
        {:ok, %{language: language, confidence: confidence}}

      {:error, reason} ->
        {:error, reason}
    end
  end

  # === Private Functions ===

  defp encode_audio(audio_data) when is_binary(audio_data) do
    Base.encode64(audio_data)
  end
end
