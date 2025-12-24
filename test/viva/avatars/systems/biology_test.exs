defmodule Viva.Avatars.Systems.BiologyTest do
  use ExUnit.Case, async: true

  alias Viva.Avatars.BioState
  alias Viva.Avatars.Personality
  alias Viva.Avatars.Systems.Biology

  describe "tick/2" do
    test "decays dopamine based on personality extraversion" do
      bio = %BioState{dopamine: 0.5, oxytocin: 0.5, cortisol: 0.5, adenosine: 0.5, libido: 0.5}
      personality = %Personality{extraversion: 0.0, neuroticism: 0.0}

      result = Biology.tick(bio, personality)

      # Base decay is 0.05, with extraversion 0.0: decay = 0.05 * 1.0 = 0.05
      # Dopamine also affected by fatigue factor
      assert result.dopamine < bio.dopamine
    end

    test "extraverts burn dopamine faster" do
      bio = %BioState{dopamine: 0.5, oxytocin: 0.5, cortisol: 0.5, adenosine: 0.0, libido: 0.5}
      low_extraversion = %Personality{extraversion: 0.0, neuroticism: 0.0}
      high_extraversion = %Personality{extraversion: 1.0, neuroticism: 0.0}

      low_result = Biology.tick(bio, low_extraversion)
      high_result = Biology.tick(bio, high_extraversion)

      # High extraversion should cause more dopamine decay
      assert high_result.dopamine < low_result.dopamine
    end

    test "neurotics hold onto cortisol longer" do
      bio = %BioState{dopamine: 0.5, oxytocin: 0.5, cortisol: 0.5, adenosine: 0.0, libido: 0.5}
      low_neuroticism = %Personality{extraversion: 0.5, neuroticism: 0.0}
      high_neuroticism = %Personality{extraversion: 0.5, neuroticism: 1.0}

      low_result = Biology.tick(bio, low_neuroticism)
      high_result = Biology.tick(bio, high_neuroticism)

      # High neuroticism means slower cortisol decay (higher remaining cortisol)
      assert high_result.cortisol > low_result.cortisol
    end

    test "decays oxytocin at fixed rate" do
      bio = %BioState{dopamine: 0.5, oxytocin: 0.5, cortisol: 0.5, adenosine: 0.0, libido: 0.5}
      personality = %Personality{extraversion: 0.5, neuroticism: 0.5}

      result = Biology.tick(bio, personality)

      # Oxytocin decay is fixed at 0.02
      assert_in_delta result.oxytocin, 0.48, 0.001
    end

    test "accumulates adenosine (fatigue)" do
      bio = %BioState{dopamine: 0.5, oxytocin: 0.5, cortisol: 0.5, adenosine: 0.5, libido: 0.5}
      personality = %Personality{extraversion: 0.5, neuroticism: 0.5}

      result = Biology.tick(bio, personality)

      # Adenosine builds at 0.005 per tick
      assert result.adenosine > bio.adenosine
    end

    test "adenosine caps at 1.0" do
      bio = %BioState{dopamine: 0.5, oxytocin: 0.5, cortisol: 0.5, adenosine: 0.999, libido: 0.5}
      personality = %Personality{extraversion: 0.5, neuroticism: 0.5}

      result = Biology.tick(bio, personality)

      assert result.adenosine <= 1.0
    end

    test "high cortisol kills libido" do
      bio = %BioState{dopamine: 0.5, oxytocin: 0.5, cortisol: 0.7, adenosine: 0.0, libido: 0.8}
      personality = %Personality{extraversion: 0.5, neuroticism: 0.5}

      result = Biology.tick(bio, personality)

      # Cortisol > 0.6 should set libido to 0
      assert result.libido == 0.0
    end

    test "low cortisol preserves libido" do
      bio = %BioState{dopamine: 0.5, oxytocin: 0.5, cortisol: 0.3, adenosine: 0.0, libido: 0.8}
      personality = %Personality{extraversion: 0.5, neuroticism: 0.5}

      result = Biology.tick(bio, personality)

      # Cortisol <= 0.6 should preserve libido
      assert result.libido == 0.8
    end

    test "high adenosine suppresses dopamine" do
      bio = %BioState{dopamine: 0.5, oxytocin: 0.5, cortisol: 0.0, adenosine: 0.9, libido: 0.5}
      personality = %Personality{extraversion: 0.0, neuroticism: 0.0}

      result = Biology.tick(bio, personality)

      # Fatigue factor = 1.0 - 0.9 * 0.3 = 0.73
      # Dopamine should be significantly reduced
      assert result.dopamine < bio.dopamine * 0.8
    end

    test "hormones never go below 0" do
      bio = %BioState{dopamine: 0.01, oxytocin: 0.01, cortisol: 0.01, adenosine: 0.0, libido: 0.5}
      personality = %Personality{extraversion: 1.0, neuroticism: 0.0}

      result = Biology.tick(bio, personality)

      assert result.dopamine >= 0.0
      assert result.oxytocin >= 0.0
      assert result.cortisol >= 0.0
    end
  end
end
