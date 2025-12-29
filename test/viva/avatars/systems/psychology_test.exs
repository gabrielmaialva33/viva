defmodule Viva.Avatars.Systems.PsychologyTest do
  use Viva.DataCase, async: true

  alias Viva.Avatars.BioState
  alias Viva.Avatars.Personality
  alias Viva.Avatars.Systems.Psychology

  describe "calculate_emotional_state/2" do
    test "returns excited mood for high dopamine and high arousal" do
      bio = %BioState{dopamine: 0.9, oxytocin: 0.5, cortisol: 0.1, libido: 0.8, adenosine: 0.1}
      p = %Personality{extraversion: 0.5}

      state = Psychology.calculate_emotional_state(bio, p)
      assert state.mood_label == "excited"
      assert state.pleasure > 0.5
      assert state.arousal > 0.5
    end

    test "returns relaxed mood for high pleasure but low arousal" do
      bio = %BioState{dopamine: 0.4, oxytocin: 0.8, cortisol: 0.1, libido: 0.1, adenosine: 0.5}
      p = %Personality{extraversion: 0.5}

      state = Psychology.calculate_emotional_state(bio, p)
      assert state.mood_label == "relaxed"
      assert state.pleasure > 0.5
      assert state.arousal <= 0.5
    end

    test "returns angry mood for low pleasure and high arousal" do
      bio = %BioState{dopamine: 0.1, oxytocin: 0.1, cortisol: 0.9, libido: 0.1, adenosine: 0.1}
      p = %Personality{extraversion: 0.5}

      state = Psychology.calculate_emotional_state(bio, p)
      assert state.mood_label == "angry"
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

    test "returns happy mood for moderate pleasure" do
      bio = %BioState{dopamine: 0.4, oxytocin: 0.2, cortisol: 0.2, libido: 0.2, adenosine: 0.2}
      p = %Personality{}

      state = Psychology.calculate_emotional_state(bio, p)
      assert state.mood_label == "happy"
    end

    test "returns anxious mood for low pleasure and moderate arousal" do
      bio = %BioState{dopamine: 0.2, oxytocin: 0.1, cortisol: 0.4, libido: 0.2, adenosine: 0.1}
      p = %Personality{}

      state = Psychology.calculate_emotional_state(bio, p)
      assert state.mood_label == "anxious"
    end

    test "returns sad mood for low pleasure and low but not depressed arousal" do
      # pleasure = 0.0(d) + 0.0(o) - 0.4(c) = -0.4 (Negative)
      # arousal = 0.0(d) + 0.0(l) + 0.4(c) - 0.5(a) = -0.1 (<= 0.0)
      bio = %BioState{dopamine: 0.0, oxytocin: 0.0, cortisol: 0.4, libido: 0.0, adenosine: 0.5}
      p = %Personality{}

      state = Psychology.calculate_emotional_state(bio, p)
      assert state.mood_label == "sad"
    end

    test "dominance is influenced by cortisol and extraversion" do
      p_extra = %Personality{extraversion: 1.0}
      p_intro = %Personality{extraversion: 0.0}
      bio = %BioState{cortisol: 0.5}

      s_extra = Psychology.calculate_emotional_state(bio, p_extra)
      s_intro = Psychology.calculate_emotional_state(bio, p_intro)

      assert s_extra.dominance > s_intro.dominance
    end
  end
end
