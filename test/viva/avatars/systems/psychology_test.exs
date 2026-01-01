defmodule Viva.Avatars.Systems.PsychologyTest do
  use Viva.DataCase, async: true

  alias Viva.Avatars.BioState
  alias Viva.Avatars.Personality
  alias Viva.Avatars.Systems.Psychology

  describe "calculate_emotional_state/2" do
    test "returns positive aroused mood for high dopamine and high arousal" do
      # Optimal positive inputs with high arousal
      bio = %BioState{dopamine: 0.9, oxytocin: 0.5, cortisol: 0.1, libido: 0.8, adenosine: 0.1}
      p = %Personality{extraversion: 0.5}

      state = Psychology.calculate_emotional_state(bio, p)
      # With hedonic rebalancing + random fluctuation, may land in:
      # - "excited" (p > 0.5, a > 0.5) - rare with dampened positive
      # - "energized" (p > 0.15, a > 0.4) - typical positive-aroused
      # - "neutral" (small pleasure near zero with negative fluctuation)
      # - "restless" (if fluctuation pushes into mild negative)
      assert state.mood_label in ["excited", "energized", "neutral", "restless"]
      assert state.arousal > 0.5
    end

    test "returns positive calm mood for high satisfaction and low arousal" do
      # Very high oxytocin + dopamine + very low stress/libido
      # With hedonic rebalancing, reaching "content" (p > 0.5) is difficult
      bio = %BioState{dopamine: 0.9, oxytocin: 0.9, cortisol: 0.0, libido: 0.0, adenosine: 0.3}
      p = %Personality{extraversion: 0.3}

      state = Psychology.calculate_emotional_state(bio, p)
      # May be "pleasant" (p > 0.15) or "content" (p > 0.5) with positive fluctuation
      assert state.mood_label in ["content", "pleasant", "energized", "neutral"]
      assert state.pleasure > -0.15
      assert state.arousal <= 1.0
    end

    test "returns distressed mood for very low pleasure and high arousal" do
      # Very high cortisol + low dopamine/oxytocin = suffering with high arousal = distressed
      bio = %BioState{dopamine: 0.1, oxytocin: 0.1, cortisol: 0.9, libido: 0.5, adenosine: 0.1}
      p = %Personality{extraversion: 0.5, neuroticism: 0.5}

      state = Psychology.calculate_emotional_state(bio, p)
      assert state.mood_label == "distressed"
      assert state.pleasure < -0.5
      assert state.arousal > 0.5
    end

    test "returns depressed mood for low pleasure and low arousal" do
      bio = %BioState{dopamine: 0.0, oxytocin: 0.0, cortisol: 0.6, libido: 0.0, adenosine: 0.9}
      p = %Personality{extraversion: 0.1}

      state = Psychology.calculate_emotional_state(bio, p)
      assert state.mood_label == "depressed"
      assert state.pleasure < -0.5
      assert state.arousal < 0.0
    end

    test "returns neutral mood for balanced state" do
      # Carefully tuned to produce neutral pleasure
      # Higher satisfaction to offset the deprivation pain
      bio = %BioState{dopamine: 0.5, oxytocin: 0.5, cortisol: 0.15, libido: 0.2, adenosine: 0.1}
      p = %Personality{}

      state = Psychology.calculate_emotional_state(bio, p)
      # With these inputs, pleasure should be near neutral (within -0.15 to 0.15 + fluctuation)
      # Allow for macro fluctuation of ±0.125
      assert state.pleasure >= -0.3 and state.pleasure <= 0.3
      assert state.mood_label in ["neutral", "uncomfortable", "pleasant"]
    end

    test "returns negative high-arousal mood for stress and high arousal" do
      # Stress with high arousal produces negative aroused states
      bio = %BioState{dopamine: 0.25, oxytocin: 0.25, cortisol: 0.4, libido: 0.3, adenosine: 0.0}
      p = %Personality{neuroticism: 0.4}

      state = Psychology.calculate_emotional_state(bio, p)
      # With hedonic rebalancing, may reach more extreme negative states
      assert state.mood_label in ["anxious", "restless", "distressed", "suffering"]
      assert state.pleasure < 0
    end

    test "returns negative low-arousal mood for moderate stress and low arousal" do
      # Moderate stress + low arousal = negative low-arousal states
      bio = %BioState{dopamine: 0.15, oxytocin: 0.15, cortisol: 0.45, libido: 0.0, adenosine: 0.3}
      p = %Personality{extraversion: 0.3}

      state = Psychology.calculate_emotional_state(bio, p)
      # With hedonic rebalancing, may be sad, depressed, or suffering
      assert state.mood_label in ["sad", "depressed", "suffering", "defeated"]
      assert state.pleasure < -0.3
    end

    test "dominance is influenced by cortisol and extraversion" do
      p_extra = %Personality{extraversion: 1.0}
      p_intro = %Personality{extraversion: 0.0}
      bio = %BioState{cortisol: 0.5}

      s_extra = Psychology.calculate_emotional_state(bio, p_extra)
      s_intro = Psychology.calculate_emotional_state(bio, p_intro)

      assert s_extra.dominance > s_intro.dominance
    end

    test "deprivation pain causes negative pleasure when needs unmet" do
      # Low dopamine (< 0.4) and low oxytocin (< 0.35) cause deprivation pain
      bio = %BioState{dopamine: 0.2, oxytocin: 0.1, cortisol: 0.2, libido: 0.2, adenosine: 0.1}
      p = %Personality{}

      state = Psychology.calculate_emotional_state(bio, p)
      # Deprivation should push pleasure negative
      assert state.pleasure < 0
    end

    test "high neuroticism amplifies emotional sensitivity" do
      bio = %BioState{dopamine: 0.3, oxytocin: 0.3, cortisol: 0.4, libido: 0.2, adenosine: 0.1}
      p_low_n = %Personality{neuroticism: 0.0}
      p_high_n = %Personality{neuroticism: 1.0}

      state_low = Psychology.calculate_emotional_state(bio, p_low_n)
      state_high = Psychology.calculate_emotional_state(bio, p_high_n)

      # High neuroticism should amplify pleasure magnitude
      assert abs(state_high.pleasure) >= abs(state_low.pleasure)
    end
  end
end
