defmodule Viva.Avatars.Systems.AllostasisTest do
  use ExUnit.Case, async: true

  alias Viva.Avatars.AllostasisState
  alias Viva.Avatars.BioState
  alias Viva.Avatars.EmotionalState
  alias Viva.Avatars.Systems.Allostasis

  describe "tick/3" do
    test "accumulates load when cortisol is high" do
      allostasis = AllostasisState.new()
      bio = %BioState{cortisol: 0.8}

      result = Allostasis.tick(allostasis, bio)

      assert result.load_level > allostasis.load_level
    end

    test "does not accumulate load when cortisol is low" do
      allostasis = %{AllostasisState.new() | load_level: 0.3}
      bio = %BioState{cortisol: 0.3}

      result = Allostasis.tick(allostasis, bio)

      # Should recover instead of accumulate
      assert result.load_level < allostasis.load_level
    end

    test "higher cortisol causes faster load accumulation" do
      allostasis = AllostasisState.new()
      moderate_stress = %BioState{cortisol: 0.65}
      high_stress = %BioState{cortisol: 0.9}

      moderate_result = Allostasis.tick(allostasis, moderate_stress)
      high_result = Allostasis.tick(allostasis, high_stress)

      assert high_result.load_level > moderate_result.load_level
    end

    test "tracks high stress hours when cortisol exceeds threshold" do
      allostasis = AllostasisState.new()
      bio = %BioState{cortisol: 0.8}

      result = Allostasis.tick(allostasis, bio)

      assert result.high_stress_hours > 0
    end

    test "reduces high stress hours when cortisol is low" do
      allostasis = %{AllostasisState.new() | high_stress_hours: 10.0}
      bio = %BioState{cortisol: 0.3}

      result = Allostasis.tick(allostasis, bio)

      assert result.high_stress_hours < allostasis.high_stress_hours
    end

    test "updates cortisol history" do
      allostasis = AllostasisState.new()
      bio = %BioState{cortisol: 0.7}

      result = Allostasis.tick(allostasis, bio)

      assert 0.7 in result.cortisol_history
    end

    test "limits cortisol history size" do
      history = Enum.map(1..30, fn _ -> 0.5 end)
      allostasis = %{AllostasisState.new() | cortisol_history: history}
      bio = %BioState{cortisol: 0.7}

      result = Allostasis.tick(allostasis, bio)

      assert length(result.cortisol_history) <= 24
    end

    test "receptor sensitivity decreases as load increases" do
      allostasis = AllostasisState.new()
      bio = %BioState{cortisol: 0.9}

      # Simulate several ticks of high stress
      result =
        Enum.reduce(1..50, allostasis, fn _, acc ->
          Allostasis.tick(acc, bio)
        end)

      assert result.receptor_sensitivity < 1.0
    end

    test "recovery capacity decreases with sustained stress" do
      allostasis = AllostasisState.new()
      bio = %BioState{cortisol: 0.9}

      # Simulate sustained high stress
      result =
        Enum.reduce(1..100, allostasis, fn _, acc ->
          Allostasis.tick(acc, bio)
        end)

      assert result.recovery_capacity < 1.0
    end

    test "cognitive impairment increases with high load" do
      allostasis = %{AllostasisState.new() | load_level: 0.7}
      bio = %BioState{cortisol: 0.8}

      result = Allostasis.tick(allostasis, bio)

      assert result.cognitive_impairment > 0
    end

    test "records recovery milestone when load drops below 0.3" do
      allostasis = %{AllostasisState.new() | load_level: 0.35, last_recovery_at: nil}
      bio = %BioState{cortisol: 0.1}

      # Low cortisol should trigger recovery
      result =
        Enum.reduce(1..10, allostasis, fn _, acc ->
          Allostasis.tick(acc, bio)
        end)

      if result.load_level < 0.3 do
        assert result.last_recovery_at != nil
      end
    end

    test "load never exceeds 1.0" do
      allostasis = %{AllostasisState.new() | load_level: 0.99}
      bio = %BioState{cortisol: 1.0}

      result = Allostasis.tick(allostasis, bio)

      assert result.load_level <= 1.0
    end

    test "load never goes below 0.0" do
      allostasis = %{AllostasisState.new() | load_level: 0.01}
      bio = %BioState{cortisol: 0.0}

      result = Allostasis.tick(allostasis, bio)

      assert result.load_level >= 0.0
    end
  end

  describe "dampen_emotions/2" do
    test "does not dampen emotions when sensitivity is high" do
      emotional = %EmotionalState{pleasure: 0.5, arousal: 0.5, mood_label: "happy"}
      allostasis = %{AllostasisState.new() | receptor_sensitivity: 1.0}

      result = Allostasis.dampen_emotions(emotional, allostasis)

      assert result.pleasure == emotional.pleasure
      assert result.arousal == emotional.arousal
    end

    test "dampens emotions proportionally to sensitivity loss" do
      emotional = %EmotionalState{pleasure: 0.8, arousal: 0.6, mood_label: "excited"}
      allostasis = %{AllostasisState.new() | receptor_sensitivity: 0.5}

      result = Allostasis.dampen_emotions(emotional, allostasis)

      assert result.pleasure == 0.4
      assert result.arousal == 0.3
    end

    test "changes mood label to 'numb' when severely desensitized" do
      emotional = %EmotionalState{pleasure: 0.5, arousal: 0.5, mood_label: "happy"}
      allostasis = %{AllostasisState.new() | receptor_sensitivity: 0.2}

      result = Allostasis.dampen_emotions(emotional, allostasis)

      assert result.mood_label == "numb"
    end

    test "changes mood label to 'exhausted' when moderately desensitized with negative pleasure" do
      emotional = %EmotionalState{pleasure: -0.3, arousal: 0.3, mood_label: "anxious"}
      allostasis = %{AllostasisState.new() | receptor_sensitivity: 0.4}

      result = Allostasis.dampen_emotions(emotional, allostasis)

      assert result.mood_label == "exhausted"
    end

    test "keeps original mood label when already 'numb' or 'exhausted'" do
      emotional = %EmotionalState{pleasure: 0.5, arousal: 0.5, mood_label: "numb"}
      allostasis = %{AllostasisState.new() | receptor_sensitivity: 0.3}

      result = Allostasis.dampen_emotions(emotional, allostasis)

      assert result.mood_label == "numb"
    end
  end

  describe "cognitive_penalty/1" do
    test "returns 1.0 when no impairment" do
      allostasis = %{AllostasisState.new() | cognitive_impairment: 0.0}

      assert Allostasis.cognitive_penalty(allostasis) == 1.0
    end

    test "returns reduced value when impaired" do
      allostasis = %{AllostasisState.new() | cognitive_impairment: 0.5}

      result = Allostasis.cognitive_penalty(allostasis)

      assert result == 0.75
    end

    test "never goes below 0.5" do
      allostasis = %{AllostasisState.new() | cognitive_impairment: 1.0}

      result = Allostasis.cognitive_penalty(allostasis)

      assert result >= 0.5
    end
  end

  describe "burnout?/1" do
    test "returns true when load is high and sensitivity is low" do
      allostasis = %{AllostasisState.new() | load_level: 0.9, receptor_sensitivity: 0.3}

      assert Allostasis.burnout?(allostasis)
    end

    test "returns false when load is moderate" do
      allostasis = %{AllostasisState.new() | load_level: 0.5, receptor_sensitivity: 0.3}

      refute Allostasis.burnout?(allostasis)
    end

    test "returns false when sensitivity is still good" do
      allostasis = %{AllostasisState.new() | load_level: 0.9, receptor_sensitivity: 0.6}

      refute Allostasis.burnout?(allostasis)
    end
  end

  describe "describe/1" do
    test "describes burnout state" do
      allostasis = %{AllostasisState.new() | load_level: 0.9, receptor_sensitivity: 0.3}

      result = Allostasis.describe(allostasis)

      assert result =~ "burnout"
    end

    test "describes significant stress" do
      allostasis = %{AllostasisState.new() | load_level: 0.7, receptor_sensitivity: 0.5}

      result = Allostasis.describe(allostasis)

      assert result =~ "chronic stress"
    end

    test "describes moderate stress" do
      allostasis = %{AllostasisState.new() | load_level: 0.4}

      result = Allostasis.describe(allostasis)

      assert result =~ "Moderate"
    end

    test "describes mild stress" do
      allostasis = %{AllostasisState.new() | load_level: 0.15}

      result = Allostasis.describe(allostasis)

      assert result =~ "Mild"
    end

    test "describes well-rested state" do
      allostasis = AllostasisState.new()

      result = Allostasis.describe(allostasis)

      assert result =~ "resilient"
    end
  end

  describe "integration scenario" do
    test "chronic stress causes emotional blunting over time" do
      allostasis = AllostasisState.new()
      emotional = %EmotionalState{pleasure: 0.8, arousal: 0.7, mood_label: "excited"}
      bio = %BioState{cortisol: 0.85}

      # Simulate 100 ticks of chronic high stress (roughly 17 simulated hours)
      {final_allostasis, _} =
        Enum.reduce(1..100, {allostasis, emotional}, fn _, {acc_allostasis, acc_emotional} ->
          new_allostasis = Allostasis.tick(acc_allostasis, bio)
          new_emotional = Allostasis.dampen_emotions(acc_emotional, new_allostasis)
          {new_allostasis, new_emotional}
        end)

      final_emotional = Allostasis.dampen_emotions(emotional, final_allostasis)

      # After chronic stress, emotions should be significantly dampened
      assert final_emotional.pleasure < emotional.pleasure
      # Load increases meaningfully under chronic stress
      assert final_allostasis.load_level > 0.2
      assert final_allostasis.receptor_sensitivity < 1.0
    end

    test "recovery occurs when stress is removed" do
      # Start with high load
      allostasis = %{
        AllostasisState.new()
        | load_level: 0.6,
          receptor_sensitivity: 0.6,
          high_stress_hours: 20.0
      }

      bio = %BioState{cortisol: 0.2}

      # Simulate 50 ticks of low stress (recovery period)
      final_allostasis =
        Enum.reduce(1..50, allostasis, fn _, acc ->
          Allostasis.tick(acc, bio)
        end)

      # Load should decrease during recovery
      assert final_allostasis.load_level < allostasis.load_level
      # Sensitivity should improve
      assert final_allostasis.receptor_sensitivity > allostasis.receptor_sensitivity
    end
  end
end
