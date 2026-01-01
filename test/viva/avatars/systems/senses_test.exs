defmodule Viva.Avatars.Systems.SensesTest do
  use Viva.DataCase, async: true
  import Mox

  alias Viva.Avatars.EmotionalState

  alias Viva.Avatars.Personality

  alias Viva.Avatars.SensoryState

  alias Viva.Avatars.Systems.Senses

  setup :verify_on_exit!

  setup do
    sensory = SensoryState.new()

    personality = %Personality{openness: 0.5, neuroticism: 0.5, extraversion: 0.5}

    emotional = %EmotionalState{pleasure: 0.0, arousal: 0.0}

    stimulus = %{
      type: :social,
      source: "friend",
      intensity: 0.5,
      valence: 0.5,
      novelty: 0.5,
      threat_level: 0.0
    }

    {:ok, sensory: sensory, personality: personality, emotional: emotional, stimulus: stimulus}
  end

  describe "perceive/4" do
    test "processes stimulus and updates focus", context do
      %{sensory: s, personality: p, emotional: e, stimulus: stim} = context

      # Mock LLM for qualia (may or may not be called due to caching/probability)
      stub(Viva.AI.LLM.MockClient, :generate, fn _, _ ->
        {:ok, "I feel a warm connection."}
      end)

      {updated, effects} = Senses.perceive(s, stim, p, e)

      assert updated.attention_focus == "social"

      # Narrative should be a non-empty string (could be from LLM, cache, or fallback)
      assert is_binary(updated.current_qualia.narrative)
      assert String.length(updated.current_qualia.narrative) > 0

      assert is_float(updated.surprise_level)

      assert is_list(effects)
    end
  end

  describe "calculate_salience/3" do
    test "boosts salience for threat", %{emotional: e} do
      low_threat = %{intensity: 0.5, perceived_threat: 0.1}
      high_threat = %{intensity: 0.5, perceived_threat: 0.9}

      s_low = Senses.calculate_salience(low_threat, e, nil)
      s_high = Senses.calculate_salience(high_threat, e, nil)

      assert s_high > s_low
    end

    test "boosts salience for mood congruence", %{stimulus: stim} do
      e_happy = %EmotionalState{pleasure: 0.8}
      e_sad = %EmotionalState{pleasure: -0.8}

      # stim has valence 0.5 (positive)
      s_happy = Senses.calculate_salience(stim, e_happy, nil)
      s_sad = Senses.calculate_salience(stim, e_sad, nil)

      assert s_happy > s_sad
    end
  end

  describe "calculate_sensory_pleasure/3" do
    test "influenced by mood and personality", %{stimulus: stim} do
      # More positive baseline
      p = %Personality{neuroticism: 0.1}
      e = %EmotionalState{pleasure: 0.8}

      pleasure = Senses.calculate_sensory_pleasure(stim, p, e)
      # Base 0.5 + mood bonus + personality bonus
      assert pleasure > stim.valence
    end
  end

  describe "calculate_sensory_pain/3" do
    test "amplified by neuroticism", %{stimulus: stim, emotional: e} do
      stim_pain = %{stim | valence: -0.8, threat_level: 0.5}

      p_calm = %Personality{neuroticism: 0.0}
      p_sensitive = %Personality{neuroticism: 1.0}

      pain_low = Senses.calculate_sensory_pain(stim_pain, p_calm, e)
      pain_high = Senses.calculate_sensory_pain(stim_pain, p_sensitive, e)

      assert pain_high > pain_low
    end
  end

  describe "tick/2" do
    test "decays attention and surprise", %{personality: p} do
      s = %SensoryState{attention_intensity: 0.8, surprise_level: 0.8}
      updated = Senses.tick(s, p)

      assert updated.attention_intensity < 0.8
      assert updated.surprise_level < 0.8
    end
  end

  describe "surprise_to_neurochemistry/1" do
    test "returns effects for levels of surprise" do
      assert Senses.surprise_to_neurochemistry(0.9) == [:surprise_high]
      assert Senses.surprise_to_neurochemistry(0.5) == [:surprise_moderate]
      assert Senses.surprise_to_neurochemistry(0.1) == []
    end
  end
end
