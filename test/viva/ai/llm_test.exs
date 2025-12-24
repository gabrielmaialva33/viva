defmodule Viva.AI.LLMTest do
  use ExUnit.Case, async: true

  alias Viva.AI.LLM

  setup_all do
    # Ensure the module is fully loaded before testing
    {:module, _} = Code.ensure_loaded(LLM)
    :ok
  end

  describe "config/0" do
    test "returns keyword list from application config" do
      config = LLM.config()
      assert is_list(config)
    end
  end

  describe "base_url/0" do
    test "returns default NVIDIA API URL when not configured" do
      assert LLM.base_url() =~ "integrate.api.nvidia.com"
    end
  end

  describe "image_base_url/0" do
    test "returns default image generation API URL" do
      assert LLM.image_base_url() =~ "ai.api.nvidia.com"
    end
  end

  describe "timeout/0" do
    test "returns timeout value" do
      timeout = LLM.timeout()
      assert is_integer(timeout)
      assert timeout > 0
    end

    test "timeout is at least 30 seconds" do
      # Timeout should be reasonable (at least 30s for LLM calls)
      assert LLM.timeout() >= 30_000
    end
  end

  describe "model/1" do
    test "returns default LLM model" do
      model = LLM.model(:llm)
      assert model =~ "nvidia/"
      assert model =~ "nemotron"
    end

    test "returns default embedding model" do
      model = LLM.model(:embedding)
      assert model =~ "nvidia/"
      assert model =~ "embed"
    end

    test "returns default TTS model" do
      model = LLM.model(:tts)
      assert model =~ "nvidia/"
    end

    test "returns default ASR model" do
      model = LLM.model(:asr)
      assert model =~ "nvidia/"
    end

    test "returns default safety model" do
      model = LLM.model(:safety)
      assert model =~ "nvidia/"
    end

    test "returns default VLM model" do
      model = LLM.model(:vlm)
      assert model =~ "nvidia/"
    end

    test "returns nil for unknown model type" do
      assert LLM.model(:unknown_type) == nil
    end
  end

  describe "auth_headers/0" do
    test "returns authorization headers list" do
      headers = LLM.auth_headers()
      assert is_list(headers)
      assert length(headers) == 2

      {auth_header, auth_value} = Enum.find(headers, fn {k, _} -> k == "Authorization" end)
      assert auth_header == "Authorization"
      assert auth_value =~ "Bearer"

      {ct_header, ct_value} = Enum.find(headers, fn {k, _} -> k == "Content-Type" end)
      assert ct_header == "Content-Type"
      assert ct_value == "application/json"
    end
  end

  describe "health/0" do
    test "returns health status map" do
      health = LLM.health()

      assert is_map(health)
      assert Map.has_key?(health, :circuit_breaker)
      assert Map.has_key?(health, :rate_limiter)
      assert Map.has_key?(health, :api_key_configured)
    end
  end
end
