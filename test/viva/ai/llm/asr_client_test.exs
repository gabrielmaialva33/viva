defmodule Viva.AI.LLM.AsrClientTest do
  use ExUnit.Case, async: true

  alias Viva.AI.LLM.AsrClient

  describe "transcribe_stream/2" do
    test "initializes stream state with defaults" do
      callback = fn _ -> :ok end

      {:ok, state} = AsrClient.transcribe_stream(callback)

      assert state.model =~ "nvidia/"
      assert state.language == "pt-BR"
      assert state.callback == callback
      assert state.buffer == <<>>
      assert state.partial_text == ""
    end

    test "accepts custom language option" do
      callback = fn _ -> :ok end

      {:ok, state} = AsrClient.transcribe_stream(callback, language: "en-US")

      assert state.language == "en-US"
    end

    test "accepts custom model option" do
      callback = fn _ -> :ok end
      custom_model = "nvidia/custom-asr-model"

      {:ok, state} = AsrClient.transcribe_stream(callback, model: custom_model)

      assert state.model == custom_model
    end

    test "stores callback function in state" do
      results = []

      callback = fn event ->
        [event | results]
      end

      {:ok, state} = AsrClient.transcribe_stream(callback)

      assert is_function(state.callback, 1)
    end
  end

  describe "transcribe_file/2 with missing file" do
    test "returns error for non-existent file" do
      result = AsrClient.transcribe_file("/nonexistent/path/file.wav")

      assert {:error, {:file_error, :enoent}} = result
    end

    test "returns error for permission denied" do
      # Try to read a directory as a file (will fail)
      result = AsrClient.transcribe_file("/")

      assert {:error, {:file_error, reason}} = result
      assert reason in [:eisdir, :eacces, :enoent]
    end
  end

  describe "module structure" do
    test "exports expected functions" do
      functions = AsrClient.__info__(:functions)

      assert {:transcribe, 1} in functions or {:transcribe, 2} in functions
      assert {:transcribe_stream, 1} in functions or {:transcribe_stream, 2} in functions
      assert {:stream_audio_chunk, 2} in functions
      assert {:transcribe_file, 1} in functions or {:transcribe_file, 2} in functions
      assert {:detect_language, 1} in functions
    end

    test "implements Pipeline.Stage behaviour" do
      behaviours = AsrClient.__info__(:attributes)[:behaviour] || []
      assert Viva.AI.Pipeline.Stage in behaviours
    end
  end
end
