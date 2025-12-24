defmodule Viva.Avatars.Systems.PsychologyTest do
  use ExUnit.Case, async: true

  alias Viva.Avatars.BioState
  alias Viva.Avatars.Personality
  alias Viva.Avatars.Systems.Psychology

  @base_personality %Personality{
    openness: 0.5,
    conscientiousness: 0.5,
    extraversion: 0.5,
    agreeableness: 0.5,
    neuroticism: 0.5
  }

  describe "calculate_emotional_state/2" do
    test "returns EmotionalState struct" do
      bio = %BioState{dopamine: 0.5, oxytocin: 0.5, cortisol: 0.5, adenosine: 0.5, libido: 0.5}
      result = Psychology.calculate_emotional_state(bio, @base_personality)

      assert is_struct(result, Viva.Avatars.EmotionalState)
    end

    test "calculates pleasure from dopamine + oxytocin - cortisol" do
      bio = %BioState{dopamine: 0.8, oxytocin: 0.6, cortisol: 0.2, adenosine: 0.5, libido: 0.5}
      result = Psychology.calculate_emotional_state(bio, @base_personality)

      # Pleasure = 0.8 + 0.6 - 0.2 = 1.2, clamped to 1.0
      assert result.pleasure == 1.0
    end

    test "calculates arousal from dopamine + libido + cortisol - adenosine" do
      bio = %BioState{dopamine: 0.6, oxytocin: 0.5, cortisol: 0.4, adenosine: 0.2, libido: 0.3}
      result = Psychology.calculate_emotional_state(bio, @base_personality)

      # Arousal = 0.6 + 0.3 + 0.4 - 0.2 = 1.1, clamped to 1.0
      assert result.arousal == 1.0
    end

    test "calculates dominance from low cortisol and extraversion" do
      bio = %BioState{dopamine: 0.5, oxytocin: 0.5, cortisol: 0.2, adenosine: 0.5, libido: 0.5}
      high_extraversion = %Personality{@base_personality | extraversion: 1.0}
      result = Psychology.calculate_emotional_state(bio, high_extraversion)

      # Dominance = 1.0 - 0.2 + 1.0 * 0.5 = 1.3, clamped to 1.0
      assert result.dominance == 1.0
    end

    test "clamps values to -1.0 to 1.0 range" do
      # Very negative state
      sad_bio = %BioState{dopamine: 0.0, oxytocin: 0.0, cortisol: 1.0, adenosine: 1.0, libido: 0.0}
      low_extraversion = %Personality{@base_personality | extraversion: 0.0}
      result = Psychology.calculate_emotional_state(sad_bio, low_extraversion)

      assert result.pleasure >= -1.0 and result.pleasure <= 1.0
      assert result.arousal >= -1.0 and result.arousal <= 1.0
      assert result.dominance >= -1.0 and result.dominance <= 1.0
    end
  end

  describe "mood classification" do
    test "excited when high pleasure and high arousal" do
      bio = %BioState{dopamine: 0.9, oxytocin: 0.7, cortisol: 0.0, adenosine: 0.0, libido: 0.8}
      result = Psychology.calculate_emotional_state(bio, @base_personality)

      # Pleasure > 0.5 and arousal > 0.5
      assert result.mood_label == "excited"
    end

    test "relaxed when high pleasure and low arousal" do
      bio = %BioState{dopamine: 0.4, oxytocin: 0.7, cortisol: 0.0, adenosine: 0.8, libido: 0.0}
      result = Psychology.calculate_emotional_state(bio, @base_personality)

      # Pleasure > 0.5, arousal < 0.5
      assert result.mood_label == "relaxed"
    end

    test "happy when moderate positive pleasure" do
      bio = %BioState{dopamine: 0.4, oxytocin: 0.4, cortisol: 0.4, adenosine: 0.5, libido: 0.3}
      result = Psychology.calculate_emotional_state(bio, @base_personality)

      # Pleasure between 0.0 and 0.5
      assert result.mood_label == "happy"
    end

    test "angry when very negative pleasure and high arousal" do
      bio = %BioState{dopamine: 0.2, oxytocin: 0.0, cortisol: 0.9, adenosine: 0.0, libido: 0.5}
      result = Psychology.calculate_emotional_state(bio, @base_personality)

      # Pleasure < -0.5, arousal > 0.5
      assert result.mood_label == "angry"
    end

    test "depressed when very negative pleasure and low arousal" do
      bio = %BioState{dopamine: 0.0, oxytocin: 0.0, cortisol: 0.8, adenosine: 0.9, libido: 0.0}
      result = Psychology.calculate_emotional_state(bio, @base_personality)

      # Pleasure < -0.5, arousal < 0.5
      assert result.mood_label == "depressed"
    end

    test "anxious when negative pleasure and positive arousal" do
      bio = %BioState{dopamine: 0.3, oxytocin: 0.1, cortisol: 0.6, adenosine: 0.2, libido: 0.2}
      result = Psychology.calculate_emotional_state(bio, @base_personality)

      # Pleasure between -0.5 and 0.0, arousal > 0.0
      assert result.mood_label == "anxious"
    end

    test "sad when negative pleasure and negative arousal" do
      bio = %BioState{dopamine: 0.1, oxytocin: 0.1, cortisol: 0.4, adenosine: 0.7, libido: 0.1}
      result = Psychology.calculate_emotional_state(bio, @base_personality)

      # Pleasure between -0.5 and 0.0, arousal < 0.0
      assert result.mood_label == "sad"
    end
  end

  describe "personality influence on emotional state" do
    test "high extraversion increases dominance" do
      bio = %BioState{dopamine: 0.5, oxytocin: 0.5, cortisol: 0.5, adenosine: 0.5, libido: 0.5}
      low_extraversion = %Personality{@base_personality | extraversion: 0.0}
      high_extraversion = %Personality{@base_personality | extraversion: 1.0}

      low_result = Psychology.calculate_emotional_state(bio, low_extraversion)
      high_result = Psychology.calculate_emotional_state(bio, high_extraversion)

      assert high_result.dominance > low_result.dominance
    end
  end
end
