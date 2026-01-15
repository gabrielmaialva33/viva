defmodule VivaBridgeTest do
  use ExUnit.Case

  @moduletag :bridge

  describe "Body NIF" do
    test "alive/0 returns confirmation string" do
      assert VivaBridge.Body.alive() == "VIVA body is alive"
    end

    test "feel_hardware/0 returns hardware metrics" do
      result = VivaBridge.Body.feel_hardware()

      assert is_map(result)
      assert Map.has_key?(result, :cpu_usage)
      assert Map.has_key?(result, :memory_used_percent)
      assert Map.has_key?(result, :memory_available_gb)
      assert Map.has_key?(result, :uptime_seconds)

      # CPU usage should be between 0-100
      assert result.cpu_usage >= 0.0
      assert result.cpu_usage <= 100.0

      # Memory percent should be between 0-100
      assert result.memory_used_percent >= 0.0
      assert result.memory_used_percent <= 100.0
    end

    test "hardware_to_qualia/0 returns PAD deltas" do
      {p, a, d} = VivaBridge.Body.hardware_to_qualia()

      # All should be floats
      assert is_float(p)
      assert is_float(a)
      assert is_float(d)

      # Deltas should be small (within expected ranges)
      assert p >= -0.1 and p <= 0.1
      assert a >= -0.1 and a <= 0.2
      assert d >= -0.1 and d <= 0.1
    end
  end

  describe "VivaBridge integration" do
    test "alive?/0 returns true when NIF is loaded" do
      assert VivaBridge.alive?() == true
    end

    test "feel_hardware/0 delegates to Body" do
      result = VivaBridge.feel_hardware()
      assert is_map(result)
      assert Map.has_key?(result, :cpu_usage)
    end
  end
end
