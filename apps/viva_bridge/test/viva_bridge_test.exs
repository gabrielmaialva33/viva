defmodule VivaBridgeTest do
  use ExUnit.Case

  @moduletag :bridge

  describe "Body NIF - Basic" do
    test "alive/0 returns confirmation" do
      assert VivaBridge.Body.alive() == "VIVA body is alive"
    end
  end

  describe "Body NIF - Hardware Sensing (Interoception)" do
    test "feel_hardware/0 returns CPU metrics" do
      result = VivaBridge.Body.feel_hardware()

      assert is_map(result)
      assert Map.has_key?(result, :cpu_usage)
      assert Map.has_key?(result, :cpu_count)

      # CPU usage should be between 0-100
      assert result.cpu_usage >= 0.0
      assert result.cpu_usage <= 100.0

      # Should have at least 1 CPU
      assert result.cpu_count >= 1
    end

    test "feel_hardware/0 returns memory metrics" do
      result = VivaBridge.Body.feel_hardware()

      assert Map.has_key?(result, :memory_used_percent)
      assert Map.has_key?(result, :memory_available_gb)
      assert Map.has_key?(result, :memory_total_gb)
      assert Map.has_key?(result, :swap_used_percent)

      # Memory percent should be between 0-100
      assert result.memory_used_percent >= 0.0
      assert result.memory_used_percent <= 100.0

      # Should have available memory
      assert result.memory_total_gb > 0.0
    end

    test "feel_hardware/0 returns temperature (optional)" do
      result = VivaBridge.Body.feel_hardware()

      # cpu_temp may be nil if not available
      assert Map.has_key?(result, :cpu_temp)

      case result.cpu_temp do
        # Temperature not available - OK
        nil ->
          :ok

        temp ->
          # If available, should be in reasonable range (0-150C)
          assert is_float(temp)
          assert temp >= 0.0
          assert temp < 150.0
      end
    end

    test "feel_hardware/0 returns GPU metrics (optional)" do
      result = VivaBridge.Body.feel_hardware()

      # All GPU metrics may be nil
      assert Map.has_key?(result, :gpu_usage)
      assert Map.has_key?(result, :gpu_vram_used_percent)
      assert Map.has_key?(result, :gpu_temp)
      assert Map.has_key?(result, :gpu_name)

      # If GPU available, values should be valid
      if result.gpu_usage != nil do
        assert result.gpu_usage >= 0.0
        assert result.gpu_usage <= 100.0
      end

      if result.gpu_vram_used_percent != nil do
        assert result.gpu_vram_used_percent >= 0.0
        assert result.gpu_vram_used_percent <= 100.0
      end
    end

    test "feel_hardware/0 returns disk metrics" do
      result = VivaBridge.Body.feel_hardware()

      assert Map.has_key?(result, :disk_usage_percent)
      assert Map.has_key?(result, :disk_read_bytes)
      assert Map.has_key?(result, :disk_write_bytes)

      # Disk usage should be between 0-100
      assert result.disk_usage_percent >= 0.0
      assert result.disk_usage_percent <= 100.0
    end

    test "feel_hardware/0 returns network metrics" do
      result = VivaBridge.Body.feel_hardware()

      assert Map.has_key?(result, :net_rx_bytes)
      assert Map.has_key?(result, :net_tx_bytes)

      # Bytes must be non-negative
      assert result.net_rx_bytes >= 0
      assert result.net_tx_bytes >= 0
    end

    test "feel_hardware/0 returns system metrics" do
      result = VivaBridge.Body.feel_hardware()

      assert Map.has_key?(result, :uptime_seconds)
      assert Map.has_key?(result, :process_count)
      assert Map.has_key?(result, :load_avg_1m)
      assert Map.has_key?(result, :load_avg_5m)
      assert Map.has_key?(result, :load_avg_15m)

      # Uptime should be positive
      assert result.uptime_seconds > 0

      # Should have at least some processes
      assert result.process_count > 0

      # Load average non-negative
      assert result.load_avg_1m >= 0.0
    end
  end

  describe "Body NIF - Qualia (Hardware -> PAD)" do
    test "hardware_to_qualia/0 returns tuple of PAD deltas" do
      {p, a, d} = VivaBridge.Body.hardware_to_qualia()

      # All should be floats
      assert is_float(p)
      assert is_float(a)
      assert is_float(d)

      # Pleasure delta: negative or zero (stress never increases pleasure)
      assert p <= 0.0
      # Max stress does not exceed -0.08
      assert p >= -0.1

      # Arousal delta: typically positive
      assert a >= 0.0
      # Max ~0.12
      assert a <= 0.15

      # Dominance delta: negative or zero
      assert d <= 0.0
      assert d >= -0.1
    end

    test "hardware_to_qualia/0 is deterministic in short term" do
      # Two close calls should give similar results
      {p1, a1, d1} = VivaBridge.Body.hardware_to_qualia()
      Process.sleep(10)
      {p2, a2, d2} = VivaBridge.Body.hardware_to_qualia()

      # Difference should be small (< 0.05)
      assert abs(p1 - p2) < 0.05
      assert abs(a1 - a2) < 0.05
      assert abs(d1 - d2) < 0.05
    end
  end

  describe "VivaBridge integration" do
    test "alive?/0 returns true when NIF loaded" do
      assert VivaBridge.alive?() == true
    end

    test "feel_hardware/0 delegates to Body" do
      result = VivaBridge.feel_hardware()
      assert is_map(result)
      assert Map.has_key?(result, :cpu_usage)
      assert Map.has_key?(result, :cpu_temp)
      assert Map.has_key?(result, :gpu_usage)
    end

    test "sync_body_to_soul/0 applies qualia to Emotional" do
      # Start an isolated Emotional for test
      {:ok, _pid} = VivaCore.Emotional.start_link(name: :test_emotional_sync)

      # Initial state is neutral
      initial = VivaCore.Emotional.get_state(:test_emotional_sync)
      assert initial.pleasure == 0.0
      assert initial.arousal == 0.0
      assert initial.dominance == 0.0

      # Apply qualia directly (not via sync_body_to_soul which uses the global)
      {p, a, d} = VivaBridge.hardware_to_qualia()
      VivaCore.Emotional.apply_hardware_qualia(p, a, d, :test_emotional_sync)

      # Small delay for the cast to process
      Process.sleep(10)

      # State should have changed
      new_state = VivaCore.Emotional.get_state(:test_emotional_sync)

      # If there was any stress, pleasure should have decreased
      if p < 0, do: assert(new_state.pleasure < 0)

      # Arousal should have increased (stress increases arousal)
      if a > 0, do: assert(new_state.arousal > 0)

      GenServer.stop(:test_emotional_sync)
    end
  end
end
