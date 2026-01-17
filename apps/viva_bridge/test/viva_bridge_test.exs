defmodule VivaBridgeTest do
  use ExUnit.Case
  @moduletag :bridge

  describe "Body NIF - ECS Architecture" do
    @tag :nif
    test "alive/0 returns confirmation" do
      assert VivaBridge.Body.alive() =~ "VIVA body is alive"
    end

    test "body_tick/0 returns complete BodyState" do
      state = VivaBridge.Body.body_tick()

      assert is_map(state)
      assert Map.has_key?(state, :tick)
      assert Map.has_key?(state, :stress_level)
      assert Map.has_key?(state, :pleasure)
      assert Map.has_key?(state, :hardware)

      # Check Hardware State
      hw = state.hardware
      assert is_map(hw)
      assert Map.has_key?(hw, :cpu_usage)

      # CPU Count might be 0 in some envs, but should exist
      assert is_integer(hw.cpu_count)
      assert hw.uptime_seconds >= 0
    end
  end

  describe "Soul Component Integration" do
    test "body_tick/0 returns dynamics" do
      state = VivaBridge.Body.body_tick()
      # Dynamics run purely in Rust now
      assert is_float(state.pleasure)
      assert is_float(state.arousal)
      assert is_float(state.dominance)

      # Bounds check
      assert state.pleasure >= -1.0 and state.pleasure <= 1.0
    end
  end
end
