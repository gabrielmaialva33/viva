defmodule Viva.Avatars.Systems.AllostasisTest do
  use Viva.DataCase, async: true

  alias Viva.Avatars.AllostasisState
  alias Viva.Avatars.BioState
  alias Viva.Avatars.EmotionalState
  alias Viva.Avatars.Personality
  alias Viva.Avatars.Systems.Allostasis

  setup do
    allostasis_data = AllostasisState.new()
    # Defaults to cortisol 0.2
    bio_data = %BioState{}

    {:ok, allostasis: allostasis_data, bio: bio_data}
  end

  describe "tick/3" do
    test "accumulates load when cortisol is high", %{allostasis: allostasis} do
      # High stress (> 0.6)
      bio = %BioState{cortisol: 0.8}

      updated = Allostasis.tick(allostasis, bio)

      assert updated.load_level > allostasis.load_level
      assert updated.high_stress_hours > allostasis.high_stress_hours
      assert length(updated.cortisol_history) == 1
      assert updated.receptor_sensitivity < 1.0
    end

    test "recovers load when cortisol is low", %{allostasis: allostasis} do
      # Start with some load
      allostasis = %{allostasis | load_level: 0.5, high_stress_hours: 5.0}
      # Low stress
      bio = %BioState{cortisol: 0.2}

      updated = Allostasis.tick(allostasis, bio)

      assert updated.load_level < 0.5
      assert updated.high_stress_hours < 5.0
      assert updated.recovery_capacity > 0.1
    end

    test "updates recovery timestamp when milestone reached", %{allostasis: allostasis} do
      allostasis = %{allostasis | load_level: 0.31, last_recovery_at: nil}
      bio = %BioState{cortisol: 0.1}

      # Should recover enough to cross 0.3 threshold downwards?
      # Assuming 1 hour elapsed to force significant drop
      updated = Allostasis.tick(allostasis, bio, 5.0)

      assert updated.load_level < 0.3
      assert updated.last_recovery_at != nil
    end
  end

  describe "dampen_emotions/2" do
    test "dampens pleasure and arousal based on sensitivity" do
      # Sensitivity 0.5 means emotions are halved
      allostasis = %AllostasisState{receptor_sensitivity: 0.5}
      emotional = %EmotionalState{pleasure: 0.8, arousal: 0.6, mood_label: "happy"}

      dampened = Allostasis.dampen_emotions(emotional, allostasis)

      assert_in_delta dampened.pleasure, 0.4, 0.01
      assert_in_delta dampened.arousal, 0.3, 0.01
      # Not low enough to change label yet
      assert dampened.mood_label == "happy"
    end

    test "changes mood label when sensitivity is very low" do
      # Very low
      allostasis = %AllostasisState{receptor_sensitivity: 0.2}
      emotional = %EmotionalState{pleasure: 0.5, arousal: 0.5, mood_label: "happy"}

      dampened = Allostasis.dampen_emotions(emotional, allostasis)

      assert dampened.mood_label == "numb"
    end

    test "changes mood label to exhausted when sensitivity low and pleasure negative" do
      allostasis = %AllostasisState{receptor_sensitivity: 0.4}
      emotional = %EmotionalState{pleasure: -0.5, arousal: 0.5, mood_label: "sad"}

      dampened = Allostasis.dampen_emotions(emotional, allostasis)

      assert dampened.mood_label == "exhausted"
    end

    test "changes mood label to depleted when sensitivity low and pleasure positive" do
      allostasis = %AllostasisState{receptor_sensitivity: 0.4}
      emotional = %EmotionalState{pleasure: 0.5, arousal: 0.5, mood_label: "happy"}

      dampened = Allostasis.dampen_emotions(emotional, allostasis)

      assert dampened.mood_label == "depleted"
    end
  end

  describe "cognitive_penalty/1" do
    test "returns penalty based on impairment" do
      allostasis = %AllostasisState{cognitive_impairment: 0.5}
      # 1.0 - 0.5 * 0.5 = 0.75
      assert Allostasis.cognitive_penalty(allostasis) == 0.75
    end
  end

  describe "burnout?/1" do
    test "returns true when load high and sensitivity low" do
      allostasis = %AllostasisState{load_level: 0.9, receptor_sensitivity: 0.3}
      assert Allostasis.burnout?(allostasis)
    end

    test "returns false otherwise" do
      allostasis = %AllostasisState{load_level: 0.5, receptor_sensitivity: 0.8}
      refute Allostasis.burnout?(allostasis)
    end
  end

  describe "describe/1" do
    test "returns correct description for each level" do
      allo_state = %AllostasisState{}

      assert Allostasis.describe(%{allo_state | load_level: 0.9, receptor_sensitivity: 0.3}) =~
               "burnout"

      assert Allostasis.describe(%{allo_state | load_level: 0.7}) =~ "significant chronic stress"
      assert Allostasis.describe(%{allo_state | load_level: 0.4}) =~ "Moderate stress"
      assert Allostasis.describe(%{allo_state | load_level: 0.2}) =~ "Mild stress"
      assert Allostasis.describe(%{allo_state | load_level: 0.0}) =~ "Well-rested"
    end
  end

  describe "phenomenology/1" do
    test "returns subjective experience for levels" do
      allo_state = %AllostasisState{}

      assert Allostasis.phenomenology(%{allo_state | load_level: 0.1}).quality == :peaceful
      assert Allostasis.phenomenology(%{allo_state | load_level: 0.4}).quality == :pressured
      assert Allostasis.phenomenology(%{allo_state | load_level: 0.7}).quality == :overwhelmed
      assert Allostasis.phenomenology(%{allo_state | load_level: 0.9}).quality == :burnout
    end
  end

  describe "decision_impairment/1" do
    test "returns impairment metrics" do
      # Simulate a state where impairment has been calculated
      allostasis = %AllostasisState{load_level: 0.8, cognitive_impairment: 0.7}
      imp = Allostasis.decision_impairment(allostasis)

      assert imp.cognitive_penalty < 1.0
      assert imp.impulsivity_bonus > 0.0
      assert imp.risk_aversion_shift != 0.0
    end

    test "returns different metrics for low load" do
      allostasis = %AllostasisState{load_level: 0.1}
      imp = Allostasis.decision_impairment(allostasis)

      assert imp.impulsivity_bonus == 0.0
      assert imp.risk_aversion_shift == 0.0
    end

    test "returns metrics for medium load" do
      allostasis = %AllostasisState{load_level: 0.4}
      imp = Allostasis.decision_impairment(allostasis)

      assert imp.impulsivity_bonus > 0.0
      assert imp.risk_aversion_shift == -0.2
    end

    test "returns metrics for high load" do
      allostasis = %AllostasisState{load_level: 0.65}
      imp = Allostasis.decision_impairment(allostasis)

      assert imp.impulsivity_bonus > 0.1
      assert imp.risk_aversion_shift == -0.3
    end
  end

  describe "generate_recovery_fantasy/2" do
    test "generates fantasy when load > 0.6 based on personality" do
      allostasis = %AllostasisState{load_level: 0.7}

      # Extravert
      p_ext = %Personality{extraversion: 0.8}
      {:fantasy_activated, f_ext} = Allostasis.generate_recovery_fantasy(allostasis, p_ext)
      assert f_ext =~ "surrounded by friends"

      # Introvert
      p_int = %Personality{extraversion: 0.2}
      {:fantasy_activated, f_int} = Allostasis.generate_recovery_fantasy(allostasis, p_int)
      assert f_int =~ "crave solitude"

      # Openness
      p_open = %Personality{extraversion: 0.5, openness: 0.8}
      {:fantasy_activated, f_open} = Allostasis.generate_recovery_fantasy(allostasis, p_open)
      assert f_open =~ "escaping somewhere new"

      # Neuroticism
      p_neur = %Personality{extraversion: 0.5, openness: 0.5, neuroticism: 0.8}
      {:fantasy_activated, f_neur} = Allostasis.generate_recovery_fantasy(allostasis, p_neur)
      assert f_neur =~ "feel safe"

      # Default
      p_def = %Personality{extraversion: 0.5, openness: 0.5, neuroticism: 0.5}
      {:fantasy_activated, f_def} = Allostasis.generate_recovery_fantasy(allostasis, p_def)
      assert f_def =~ "break"
    end

    test "returns :no_fantasy when load is low" do
      allostasis = %AllostasisState{load_level: 0.5}
      p_traits = %Personality{}
      assert Allostasis.generate_recovery_fantasy(allostasis, p_traits) == :no_fantasy
    end
  end

  describe "attention_constraints/1" do
    test "maps phenomenology attention capacity to constraints" do
      allo_state = %AllostasisState{}

      # Broad
      c_broad = Allostasis.attention_constraints(%{allo_state | load_level: 0.1})
      assert c_broad.can_plan_ahead == true

      # Moderate
      c_mod = Allostasis.attention_constraints(%{allo_state | load_level: 0.4})
      assert c_mod.can_consider_abstract == :limited

      # Narrow
      c_nar = Allostasis.attention_constraints(%{allo_state | load_level: 0.7})
      assert c_nar.can_plan_ahead == false

      # Fragmented
      c_frag = Allostasis.attention_constraints(%{allo_state | load_level: 0.9})
      assert c_frag.memory_access == :impaired
    end
  end
end
