defmodule Viva.AI.LLM.AudioEnhanceClientTest do
  use ExUnit.Case, async: true

  alias Viva.AI.LLM.AudioEnhanceClient

  describe "stream_enhance/2" do
    test "initializes stream state with defaults" do
      callback = fn _ -> :ok end

      {:ok, state} = AudioEnhanceClient.stream_enhance(callback)

      assert is_binary(state.model) or is_nil(state.model)
      assert state.callback == callback
      assert state.buffer == <<>>
      assert state.sample_rate == 48_000
    end

    test "accepts custom sample rate" do
      callback = fn _ -> :ok end

      {:ok, state} = AudioEnhanceClient.stream_enhance(callback, sample_rate: 44_100)

      assert state.sample_rate == 44_100
    end

    test "accepts custom model" do
      callback = fn _ -> :ok end
      custom_model = "nvidia/custom-audio-model"

      {:ok, state} = AudioEnhanceClient.stream_enhance(callback, model: custom_model)

      assert state.model == custom_model
    end

    test "stores callback function in state" do
      callback = fn event -> event end

      {:ok, state} = AudioEnhanceClient.stream_enhance(callback)

      assert is_function(state.callback, 1)
    end
  end

  describe "module structure" do
    test "exports expected functions" do
      functions = AudioEnhanceClient.__info__(:functions)

      assert {:enhance, 1} in functions or {:enhance, 2} in functions
      assert {:remove_noise, 1} in functions or {:remove_noise, 2} in functions
      assert {:process_full, 1} in functions or {:process_full, 2} in functions
      assert {:stream_enhance, 1} in functions or {:stream_enhance, 2} in functions
      assert {:stream_chunk, 2} in functions

      assert {:enhance_for_transcription, 1} in functions or
               {:enhance_for_transcription, 2} in functions

      assert {:normalize_levels, 1} in functions or {:normalize_levels, 2} in functions
      assert {:analyze_quality, 1} in functions
      assert {:needs_enhancement?, 1} in functions or {:needs_enhancement?, 2} in functions
      assert {:smart_enhance, 1} in functions or {:smart_enhance, 2} in functions
    end
  end
end
