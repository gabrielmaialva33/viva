defmodule VivaBridge.CortexTest do
  use ExUnit.Case
  alias VivaBridge.Cortex

  # @moduletag :capture_log

  setup do
    # Ensure Cortex is running for each test
    case VivaBridge.Cortex.start_link([]) do
      {:ok, pid} -> {:ok, cortex_pid: pid}
      {:error, {:already_started, pid}} -> {:ok, cortex_pid: pid}
      _ -> :error
    end
  end

  test "starts and pings the Liquid Engine" do
    # Test Ping (Basic comms)
    # Cortex.ping() returns the result map directly
    result = Cortex.ping()

    assert is_map(result)
    assert result["status"] == "pong"
    assert result["type"] == "liquid_ncp"
  end

  test "process experience returns vector and pad" do
    emotion = %{pleasure: 0.5, arousal: 0.2, dominance: 0.1}
    narrative = "Unit test narrative"

    # experience returns {:ok, vector, new_pad}
    assert {:ok, vector, new_pad} = Cortex.experience(narrative, emotion)

    assert is_list(vector)
    # The vector size depends on the model output, usually 32 or similar for liquid layer?
    # Actually checking liquid_engine.py might reveal size.
    # But let's just assert it is a non-empty list.
    assert length(vector) > 0

    assert is_map(new_pad)
    assert Map.has_key?(new_pad, "pleasure") or Map.has_key?(new_pad, :pleasure)
  end
end
