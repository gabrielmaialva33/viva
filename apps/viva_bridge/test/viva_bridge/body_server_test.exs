defmodule VivaBridge.BodyServerTest do
  use ExUnit.Case
  alias VivaBridge.BodyServer

  @moduletag :capture_log
  # All tests require the Rust NIF to be compiled
  @moduletag :nif

  # DeepSeek Analysis:
  # The BodyServer is a singleton wrapping a Rust Singleton (ECS World).
  # We cannot spawn new isolated instances. We must test the running instance.

  setup do
    # Ensure server is running (it should be part of app supervision)
    pid = Process.whereis(BodyServer)

    if pid && Process.alive?(pid) do
      {:ok, pid: pid}
    else
      # If not running (e.g. app not started in test env?), try to start it
      start_supervised!(BodyServer)
      {:ok, pid: Process.whereis(BodyServer)}
    end
  end

  test "get_state/0 returns valid BodyState immediately" do
    state = BodyServer.get_state()

    assert is_map(state), "State should never be nil"

    # Structural Integrity Check
    assert Map.has_key?(state, :stress_level)
    assert Map.has_key?(state, :pleasure)
    assert Map.has_key?(state, :hardware)
    assert Map.has_key?(state, :tick)

    # Hardware Metrics
    hw = state.hardware
    assert is_map(hw)
    assert is_number(hw.cpu_usage)
  end

  test "get_pad/0 returns numeric tuple" do
    {p, a, d} = BodyServer.get_pad()
    assert is_float(p)
    assert is_float(a)
    assert is_float(d)

    # Check bounds [-1, 1]
    assert p >= -1.0 and p <= 1.0
  end

  @tag :external
  test "apply_stimulus/4 modifies emotional state" do
    # 1. Get baseline
    {p1, _a1, _d1} = BodyServer.get_pad()

    # 2. Apply strong positive stimulus
    BodyServer.apply_stimulus(0.5, 0.1, 0.1)

    # 3. Force tick to process updates from Rust
    BodyServer.force_tick()

    # 4. Check effect
    {p2, _a2, _d2} = BodyServer.get_pad()

    # We expect pleasure to increase (or stay maxed) and arousal to change
    # Note: O-U dynamics constantly pull back to 0, so small delta might be lost if we wait too long.
    # But since we just cast it, it should be visible.

    # Assert change occurred (unless already maxed)
    if p1 < 0.9 do
      assert p2 > p1, "Positive stimulus should increase pleasure"
    end
  end
end
