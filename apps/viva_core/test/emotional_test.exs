defmodule VivaCore.EmotionalTest do
  use ExUnit.Case, async: true
  doctest VivaCore.Emotional

  alias VivaCore.Emotional

  @moduletag :emotional

  describe "start_link/1" do
    test "starts with neutral state by default" do
      {:ok, pid} = Emotional.start_link(name: :test_emotional_1)

      state = Emotional.get_state(pid)

      assert state.pleasure == 0.0
      assert state.arousal == 0.0
      assert state.dominance == 0.0

      GenServer.stop(pid)
    end

    test "accepts custom initial state" do
      initial = %{pleasure: 0.5, arousal: -0.3, dominance: 0.2}
      {:ok, pid} = Emotional.start_link(name: :test_emotional_2, initial_state: initial)

      state = Emotional.get_state(pid)

      assert state.pleasure == 0.5
      assert state.arousal == -0.3
      assert state.dominance == 0.2

      GenServer.stop(pid)
    end
  end

  describe "feel/4" do
    test "rejection decreases pleasure and dominance, increases arousal" do
      {:ok, pid} = Emotional.start_link(name: :test_emotional_3)

      before = Emotional.get_state(pid)
      Emotional.feel(:rejection, "human_test", 1.0, pid)

      # Give time for the cast to be processed
      :timer.sleep(50)

      after_state = Emotional.get_state(pid)

      assert after_state.pleasure < before.pleasure
      assert after_state.arousal > before.arousal
      assert after_state.dominance < before.dominance

      GenServer.stop(pid)
    end

    test "acceptance increases pleasure, arousal and dominance" do
      {:ok, pid} = Emotional.start_link(name: :test_emotional_4)

      before = Emotional.get_state(pid)
      Emotional.feel(:acceptance, "human_test", 1.0, pid)
      :timer.sleep(50)

      after_state = Emotional.get_state(pid)

      assert after_state.pleasure > before.pleasure
      assert after_state.arousal > before.arousal
      assert after_state.dominance > before.dominance

      GenServer.stop(pid)
    end

    test "intensity modulates the impact" do
      {:ok, pid1} = Emotional.start_link(name: :test_emotional_5a)
      {:ok, pid2} = Emotional.start_link(name: :test_emotional_5b)

      # Low intensity
      Emotional.feel(:rejection, "test", 0.2, pid1)
      # High intensity
      Emotional.feel(:rejection, "test", 1.0, pid2)
      :timer.sleep(50)

      state_low = Emotional.get_state(pid1)
      state_high = Emotional.get_state(pid2)

      # High intensity should cause greater negative impact
      assert abs(state_high.pleasure) > abs(state_low.pleasure)

      GenServer.stop(pid1)
      GenServer.stop(pid2)
    end

    test "unknown stimulus does not alter state" do
      {:ok, pid} = Emotional.start_link(name: :test_emotional_6)

      before = Emotional.get_state(pid)
      Emotional.feel(:unknown_stimulus, "test", 1.0, pid)
      :timer.sleep(50)

      after_state = Emotional.get_state(pid)

      assert after_state == before

      GenServer.stop(pid)
    end

    test "hardware_stress simulates stress qualia" do
      {:ok, pid} = Emotional.start_link(name: :test_emotional_7)

      before = Emotional.get_state(pid)
      Emotional.feel(:hardware_stress, "cpu_monitor", 1.0, pid)
      :timer.sleep(50)

      after_state = Emotional.get_state(pid)

      # Stress increases arousal and decreases pleasure/dominance
      assert after_state.arousal > before.arousal
      assert after_state.pleasure < before.pleasure

      GenServer.stop(pid)
    end
  end

  describe "introspect/1" do
    test "returns semantic interpretation of state" do
      {:ok, pid} = Emotional.start_link(name: :test_emotional_8)

      introspection = Emotional.introspect(pid)

      assert Map.has_key?(introspection, :pad)
      assert Map.has_key?(introspection, :mood)
      assert Map.has_key?(introspection, :energy)
      assert Map.has_key?(introspection, :agency)
      assert Map.has_key?(introspection, :self_assessment)
      assert is_binary(introspection.self_assessment)

      GenServer.stop(pid)
    end

    test "mood reflects pleasure correctly" do
      # Test with happy state
      {:ok, pid_happy} = Emotional.start_link(
        name: :test_emotional_9a,
        initial_state: %{pleasure: 0.7, arousal: 0.0, dominance: 0.0}
      )

      # Test with sad state
      {:ok, pid_sad} = Emotional.start_link(
        name: :test_emotional_9b,
        initial_state: %{pleasure: -0.7, arousal: 0.0, dominance: 0.0}
      )

      happy_intro = Emotional.introspect(pid_happy)
      sad_intro = Emotional.introspect(pid_sad)

      assert happy_intro.mood == :joyful
      assert sad_intro.mood == :depressed

      GenServer.stop(pid_happy)
      GenServer.stop(pid_sad)
    end
  end

  describe "get_happiness/1" do
    test "returns normalized value 0-1" do
      {:ok, pid} = Emotional.start_link(name: :test_emotional_10)

      happiness = Emotional.get_happiness(pid)

      assert happiness >= 0.0
      assert happiness <= 1.0
      # Neutral state should be 0.5
      assert happiness == 0.5

      GenServer.stop(pid)
    end

    test "maximum happiness returns ~1.0" do
      {:ok, pid} = Emotional.start_link(
        name: :test_emotional_11,
        initial_state: %{pleasure: 1.0, arousal: 0.0, dominance: 0.0}
      )

      happiness = Emotional.get_happiness(pid)
      assert happiness == 1.0

      GenServer.stop(pid)
    end
  end

  describe "reset/1" do
    test "returns to neutral state" do
      {:ok, pid} = Emotional.start_link(
        name: :test_emotional_12,
        initial_state: %{pleasure: 0.8, arousal: -0.5, dominance: 0.3}
      )

      # Verify non-neutral state
      before = Emotional.get_state(pid)
      assert before.pleasure == 0.8

      # Reset
      Emotional.reset(pid)
      :timer.sleep(50)

      # Verify neutral state
      after_state = Emotional.get_state(pid)
      assert after_state.pleasure == 0.0
      assert after_state.arousal == 0.0
      assert after_state.dominance == 0.0

      GenServer.stop(pid)
    end
  end

  describe "decay" do
    test "values decay toward neutral" do
      {:ok, pid} = Emotional.start_link(
        name: :test_emotional_13,
        initial_state: %{pleasure: 0.5, arousal: 0.5, dominance: 0.5}
      )

      before = Emotional.get_state(pid)

      # Apply decay manually
      Emotional.decay(pid)
      :timer.sleep(50)

      after_state = Emotional.get_state(pid)

      # Positive values should decrease
      assert after_state.pleasure < before.pleasure
      assert after_state.arousal < before.arousal
      assert after_state.dominance < before.dominance

      GenServer.stop(pid)
    end

    test "negative values increase toward neutral" do
      {:ok, pid} = Emotional.start_link(
        name: :test_emotional_14,
        initial_state: %{pleasure: -0.5, arousal: -0.5, dominance: -0.5}
      )

      before = Emotional.get_state(pid)

      Emotional.decay(pid)
      :timer.sleep(50)

      after_state = Emotional.get_state(pid)

      # Negative values should increase (toward 0)
      assert after_state.pleasure > before.pleasure
      assert after_state.arousal > before.arousal
      assert after_state.dominance > before.dominance

      GenServer.stop(pid)
    end
  end

  describe "value limits" do
    test "values are limited to [-1.0, 1.0]" do
      {:ok, pid} = Emotional.start_link(name: :test_emotional_15)

      # Apply many positive stimuli
      for _ <- 1..20 do
        Emotional.feel(:success, "test", 1.0, pid)
      end
      :timer.sleep(100)

      state = Emotional.get_state(pid)

      assert state.pleasure <= 1.0
      assert state.arousal <= 1.0
      assert state.dominance <= 1.0

      GenServer.stop(pid)
    end
  end
end
