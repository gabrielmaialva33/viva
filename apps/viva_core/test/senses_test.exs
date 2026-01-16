defmodule VivaCore.SensesTest do
  use ExUnit.Case

  @moduletag :senses

  describe "Senses - Initialization" do
    test "start_link/1 starts with default state" do
      {:ok, pid} = VivaCore.Senses.start_link(name: :test_senses_init, enabled: false)

      state = VivaCore.Senses.get_state(:test_senses_init)

      assert state.interval_ms == 1000
      assert state.enabled == false
      assert state.heartbeat_count == 0
      assert state.last_reading == nil
      assert state.last_qualia == nil

      GenServer.stop(pid)
    end

    test "start_link/1 accepts custom options" do
      {:ok, pid} =
        VivaCore.Senses.start_link(
          name: :test_senses_custom,
          interval_ms: 500,
          enabled: false
        )

      state = VivaCore.Senses.get_state(:test_senses_custom)
      assert state.interval_ms == 500

      GenServer.stop(pid)
    end
  end

  describe "Senses - Heartbeat" do
    test "pulse/1 forces immediate reading" do
      # Start isolated Emotional for the test
      {:ok, emotional_pid} = VivaCore.Emotional.start_link(name: :test_emotional_pulse)

      {:ok, senses_pid} =
        VivaCore.Senses.start_link(
          name: :test_senses_pulse,
          emotional_server: :test_emotional_pulse,
          enabled: false
        )

      # Initial Emotional state is neutral
      initial = VivaCore.Emotional.get_state(:test_emotional_pulse)
      assert initial.pleasure == 0.0
      assert initial.arousal == 0.0
      assert initial.dominance == 0.0

      # Force a pulse
      {:ok, {p, a, d}} = VivaCore.Senses.pulse(:test_senses_pulse)

      # Qualia should have been calculated
      assert is_float(p)
      assert is_float(a)
      assert is_float(d)

      # Senses state should have been updated
      state = VivaCore.Senses.get_state(:test_senses_pulse)
      assert state.heartbeat_count == 1
      assert state.last_qualia == {p, a, d}
      assert state.last_reading != nil

      # Small delay for the cast to process
      Process.sleep(10)

      # Emotional should have received the qualia
      new_state = VivaCore.Emotional.get_state(:test_emotional_pulse)

      # If there was hardware stress, pleasure should have changed
      if p != 0.0, do: assert(new_state.pleasure != 0.0)

      GenServer.stop(senses_pid)
      GenServer.stop(emotional_pid)
    end

    test "automatic heartbeat increments counter" do
      {:ok, emotional_pid} = VivaCore.Emotional.start_link(name: :test_emotional_auto)

      {:ok, senses_pid} =
        VivaCore.Senses.start_link(
          name: :test_senses_auto,
          emotional_server: :test_emotional_auto,
          # 100ms for fast test
          interval_ms: 100,
          enabled: true
        )

      # Wait for 3 heartbeats (~300ms + margin)
      Process.sleep(350)

      state = VivaCore.Senses.get_state(:test_senses_auto)
      assert state.heartbeat_count >= 3

      GenServer.stop(senses_pid)
      GenServer.stop(emotional_pid)
    end
  end

  describe "Senses - Control" do
    test "pause/1 stops automatic sensing" do
      {:ok, emotional_pid} = VivaCore.Emotional.start_link(name: :test_emotional_pause)

      {:ok, senses_pid} =
        VivaCore.Senses.start_link(
          name: :test_senses_pause,
          emotional_server: :test_emotional_pause,
          interval_ms: 100,
          enabled: true
        )

      # Wait for some heartbeats
      Process.sleep(250)

      state1 = VivaCore.Senses.get_state(:test_senses_pause)
      count1 = state1.heartbeat_count

      # Pause
      VivaCore.Senses.pause(:test_senses_pause)

      # Wait more time
      Process.sleep(250)

      state2 = VivaCore.Senses.get_state(:test_senses_pause)
      count2 = state2.heartbeat_count

      # Counter should not have increased (or at most +1 if it was in the middle)
      assert count2 <= count1 + 1
      assert state2.enabled == false

      GenServer.stop(senses_pid)
      GenServer.stop(emotional_pid)
    end

    test "resume/1 resumes sensing" do
      {:ok, emotional_pid} = VivaCore.Emotional.start_link(name: :test_emotional_resume)

      {:ok, senses_pid} =
        VivaCore.Senses.start_link(
          name: :test_senses_resume,
          emotional_server: :test_emotional_resume,
          interval_ms: 100,
          # Start paused
          enabled: false
        )

      state1 = VivaCore.Senses.get_state(:test_senses_resume)
      assert state1.enabled == false

      # Resume
      VivaCore.Senses.resume(:test_senses_resume)
      Process.sleep(250)

      state2 = VivaCore.Senses.get_state(:test_senses_resume)
      assert state2.enabled == true
      assert state2.heartbeat_count >= 1

      GenServer.stop(senses_pid)
      GenServer.stop(emotional_pid)
    end

    test "set_interval/2 changes frequency at runtime" do
      {:ok, senses_pid} =
        VivaCore.Senses.start_link(
          name: :test_senses_interval,
          enabled: false
        )

      state1 = VivaCore.Senses.get_state(:test_senses_interval)
      assert state1.interval_ms == 1000

      VivaCore.Senses.set_interval(500, :test_senses_interval)

      state2 = VivaCore.Senses.get_state(:test_senses_interval)
      assert state2.interval_ms == 500

      GenServer.stop(senses_pid)
    end
  end

  describe "Senses - Hardware Integration" do
    test "last_reading contains hardware metrics" do
      {:ok, emotional_pid} = VivaCore.Emotional.start_link(name: :test_emotional_hw)

      {:ok, senses_pid} =
        VivaCore.Senses.start_link(
          name: :test_senses_hw,
          emotional_server: :test_emotional_hw,
          enabled: false
        )

      # Force a pulse to get a reading
      VivaCore.Senses.pulse(:test_senses_hw)

      state = VivaCore.Senses.get_state(:test_senses_hw)

      # last_reading should contain hardware metrics
      assert is_map(state.last_reading)
      assert Map.has_key?(state.last_reading, :cpu_usage)
      assert Map.has_key?(state.last_reading, :memory_used_percent)
      assert Map.has_key?(state.last_reading, :gpu_usage)

      GenServer.stop(senses_pid)
      GenServer.stop(emotional_pid)
    end
  end
end
