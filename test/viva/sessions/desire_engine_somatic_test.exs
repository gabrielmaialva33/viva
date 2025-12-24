defmodule Viva.Sessions.DesireEngineSomaticTest do
  use ExUnit.Case, async: true

  alias Viva.Avatars.BioState
  alias Viva.Avatars.EmotionalState
  alias Viva.Avatars.Personality
  alias Viva.Sessions.DesireEngine

  describe "determine/4 with somatic_bias" do
    test "returns desire without somatic_bias (backward compatible)" do
      bio = %BioState{adenosine: 0.2, oxytocin: 0.2, dopamine: 0.5, libido: 0.3}
      emotional = %EmotionalState{pleasure: 0.0, arousal: 0.3}
      personality = %Personality{extraversion: 0.8, openness: 0.5}

      desire = DesireEngine.determine(bio, emotional, personality)

      assert desire == :wants_attention
    end

    test "body warning suppresses social desire" do
      bio = %BioState{adenosine: 0.2, oxytocin: 0.2, dopamine: 0.5, libido: 0.3}
      emotional = %EmotionalState{pleasure: 0.0, arousal: 0.3}
      personality = %Personality{extraversion: 0.8, openness: 0.5}
      somatic_bias = %{bias: -0.5, signal: "Warning", markers_activated: 1}

      desire = DesireEngine.determine(bio, emotional, personality, somatic_bias)

      assert desire == :wants_something_new
    end

    test "body attraction amplifies crush desire" do
      bio = %BioState{adenosine: 0.2, oxytocin: 0.4, dopamine: 0.5, libido: 0.5}
      emotional = %EmotionalState{pleasure: 0.0, arousal: 0.3}
      personality = %Personality{extraversion: 0.4, openness: 0.4}
      somatic_bias = %{bias: 0.5, signal: "Attraction", markers_activated: 1}

      desire = DesireEngine.determine(bio, emotional, personality, somatic_bias)

      assert desire == :wants_to_see_crush
    end

    test "positive bias increases social desire likelihood" do
      # With positive bias > 0.2, oxytocin < 0.4 condition kicks in
      bio = %BioState{adenosine: 0.2, oxytocin: 0.25, dopamine: 0.5, libido: 0.3}
      emotional = %EmotionalState{pleasure: 0.0, arousal: 0.3}
      personality = %Personality{extraversion: 0.7, openness: 0.4}
      somatic_bias = %{bias: 0.4, signal: "Warm feeling", markers_activated: 1}

      desire = DesireEngine.determine(bio, emotional, personality, somatic_bias)

      assert desire == :wants_attention
    end

    test "rest still takes priority over somatic influence" do
      bio = %BioState{adenosine: 0.9, oxytocin: 0.2, dopamine: 0.5, libido: 0.3}
      emotional = %EmotionalState{pleasure: 0.0, arousal: 0.3}
      personality = %Personality{extraversion: 0.8, openness: 0.5}
      somatic_bias = %{bias: 0.8, signal: "Strong attraction", markers_activated: 2}

      desire = DesireEngine.determine(bio, emotional, personality, somatic_bias)

      assert desire == :wants_rest
    end

    test "nil somatic_bias works like no bias" do
      bio = %BioState{adenosine: 0.2, oxytocin: 0.2, dopamine: 0.5, libido: 0.3}
      emotional = %EmotionalState{pleasure: 0.0, arousal: 0.3}
      personality = %Personality{extraversion: 0.8, openness: 0.5}

      desire_with_nil = DesireEngine.determine(bio, emotional, personality, nil)
      desire_without = DesireEngine.determine(bio, emotional, personality)

      assert desire_with_nil == desire_without
    end

    test "weak body warning does not suppress social desire" do
      bio = %BioState{adenosine: 0.2, oxytocin: 0.2, dopamine: 0.5, libido: 0.3}
      emotional = %EmotionalState{pleasure: 0.0, arousal: 0.3}
      personality = %Personality{extraversion: 0.8, openness: 0.5}
      somatic_bias = %{bias: -0.2, signal: "Slight unease", markers_activated: 1}

      desire = DesireEngine.determine(bio, emotional, personality, somatic_bias)

      assert desire == :wants_attention
    end

    test "body warning only affects social desires" do
      bio = %BioState{adenosine: 0.2, oxytocin: 0.6, dopamine: 0.1, libido: 0.3}
      emotional = %EmotionalState{pleasure: 0.0, arousal: 0.3}
      personality = %Personality{extraversion: 0.3, openness: 0.8}
      somatic_bias = %{bias: -0.6, signal: "Strong warning", markers_activated: 2}

      desire = DesireEngine.determine(bio, emotional, personality, somatic_bias)

      assert desire == :wants_something_new
    end
  end
end
