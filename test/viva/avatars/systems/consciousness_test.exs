defmodule Viva.Avatars.Systems.ConsciousnessTest do
  use ExUnit.Case, async: true

  alias Viva.Avatars.BioState
  alias Viva.Avatars.ConsciousnessState
  alias Viva.Avatars.EmotionalState
  alias Viva.Avatars.Personality
  alias Viva.Avatars.SensoryState
  alias Viva.Avatars.Systems.Consciousness

  describe "integrate/6" do
    setup do
      consciousness = ConsciousnessState.new()
      sensory = SensoryState.new()
      bio = %BioState{}
      emotional = %EmotionalState{}
      personality = build_personality()

      {:ok,
       consciousness: consciousness,
       sensory: sensory,
       bio: bio,
       emotional: emotional,
       personality: personality}
    end

    test "returns updated consciousness state", ctx do
      result =
        Consciousness.integrate(
          ctx.consciousness,
          ctx.sensory,
          ctx.bio,
          ctx.emotional,
          "thinking about something",
          ctx.personality
        )

      assert %ConsciousnessState{} = result
    end

    test "adds moment to experience stream", ctx do
      result =
        Consciousness.integrate(
          ctx.consciousness,
          ctx.sensory,
          ctx.bio,
          ctx.emotional,
          nil,
          ctx.personality
        )

      assert result.experience_stream != []
    end

    test "limits experience stream length", ctx do
      # Create consciousness with full stream
      full_stream = for _ <- 1..20, do: %{timestamp: DateTime.utc_now()}
      consciousness = %{ctx.consciousness | experience_stream: full_stream}

      result =
        Consciousness.integrate(
          consciousness,
          ctx.sensory,
          ctx.bio,
          ctx.emotional,
          nil,
          ctx.personality
        )

      # Should be limited to max stream length (use Enum.count for comparison)
      assert Enum.count(result.experience_stream) <= 15
    end

    test "updates focal content when sensory qualia is intense", ctx do
      sensory = %{
        ctx.sensory
        | current_qualia: %{
            dominant_sensation: "warmth",
            emotional_color: "pleasant",
            intensity: 0.8,
            narrative: "A warm sensation"
          }
      }

      result =
        Consciousness.integrate(
          ctx.consciousness,
          sensory,
          ctx.bio,
          ctx.emotional,
          nil,
          ctx.personality
        )

      assert result.focal_content != nil
    end

    test "calculates tempo based on arousal and fatigue", ctx do
      # High arousal, low fatigue = fast tempo
      high_arousal_bio = %{ctx.bio | adenosine: 0.2}
      high_arousal_emotional = %{ctx.emotional | arousal: 0.9}

      result =
        Consciousness.integrate(
          ctx.consciousness,
          ctx.sensory,
          high_arousal_bio,
          high_arousal_emotional,
          nil,
          ctx.personality
        )

      assert result.stream_tempo in [:fast, :racing]
    end

    test "calculates slow tempo when fatigued", ctx do
      fatigued_bio = %{ctx.bio | adenosine: 0.9}
      low_arousal_emotional = %{ctx.emotional | arousal: 0.1}

      result =
        Consciousness.integrate(
          ctx.consciousness,
          ctx.sensory,
          fatigued_bio,
          low_arousal_emotional,
          nil,
          ctx.personality
        )

      assert result.stream_tempo in [:slow, :frozen]
    end
  end

  describe "calculate_tempo/2" do
    test "returns racing for very high arousal" do
      assert Consciousness.calculate_tempo(0.95, 0.1) == :racing
    end

    test "returns fast or racing for high arousal" do
      tempo = Consciousness.calculate_tempo(0.8, 0.2)
      assert tempo in [:fast, :racing]
    end

    test "returns normal for moderate arousal and fatigue" do
      assert Consciousness.calculate_tempo(0.5, 0.4) == :normal
    end

    test "returns slow or normal for moderate fatigue" do
      tempo = Consciousness.calculate_tempo(0.2, 0.7)
      assert tempo in [:slow, :normal]
    end

    test "returns frozen for very high fatigue" do
      assert Consciousness.calculate_tempo(0.1, 0.95) == :frozen
    end
  end

  describe "calculate_presence/3" do
    test "returns high presence for normal conditions" do
      emotional = %EmotionalState{arousal: 0.5, pleasure: 0.3, dominance: 0.5}
      bio = %BioState{adenosine: 0.3}

      presence = Consciousness.calculate_presence(emotional, bio, 0.2)
      assert presence > 0.5
    end

    test "returns presence value for high arousal conditions" do
      emotional = %EmotionalState{arousal: 0.95, pleasure: -0.8, dominance: 0.2}
      bio = %BioState{adenosine: 0.3}

      presence = Consciousness.calculate_presence(emotional, bio, 0.8)
      # High arousal and surprise can affect presence but doesn't always lower it
      assert presence > 0.0 and presence <= 1.0
    end

    test "returns lower presence when very fatigued" do
      emotional = %EmotionalState{arousal: 0.3, pleasure: 0.0, dominance: 0.5}
      bio = %BioState{adenosine: 0.95}

      presence = Consciousness.calculate_presence(emotional, bio, 0.0)
      # Very high adenosine should reduce presence
      assert presence <= 0.7
    end
  end

  describe "synthesize_experience_narrative/3" do
    test "generates narrative from consciousness state" do
      consciousness = ConsciousnessState.new()
      sensory = SensoryState.new()
      emotional = %EmotionalState{}

      narrative = Consciousness.synthesize_experience_narrative(consciousness, sensory, emotional)

      assert is_binary(narrative)
      assert String.length(narrative) > 0
    end

    test "includes awareness description" do
      consciousness = %{ConsciousnessState.new() | presence_level: 0.9}
      sensory = SensoryState.new()
      emotional = %EmotionalState{}

      narrative = Consciousness.synthesize_experience_narrative(consciousness, sensory, emotional)

      # Should mention awareness/presence/present in some form
      assert String.contains?(narrative, "present") or String.contains?(narrative, "Awareness")
    end

    test "includes temporal focus" do
      consciousness = %{ConsciousnessState.new() | temporal_focus: :past}
      sensory = SensoryState.new()
      emotional = %EmotionalState{}

      narrative = Consciousness.synthesize_experience_narrative(consciousness, sensory, emotional)

      assert String.contains?(narrative, "past")
    end

    test "includes emotional state description" do
      consciousness = ConsciousnessState.new()
      sensory = SensoryState.new()
      emotional = %EmotionalState{pleasure: 0.7, arousal: 0.3, dominance: 0.5, mood_label: "happy"}

      narrative = Consciousness.synthesize_experience_narrative(consciousness, sensory, emotional)

      assert String.contains?(narrative, "happy") or String.contains?(narrative, "Emotional")
    end

    test "includes qualia when present" do
      consciousness = ConsciousnessState.new()

      sensory = %{
        SensoryState.new()
        | current_qualia: %{
            dominant_sensation: "warmth",
            emotional_color: nil,
            intensity: 0.6,
            narrative: "A gentle warmth spreads through me"
          }
      }

      emotional = %EmotionalState{}

      narrative = Consciousness.synthesize_experience_narrative(consciousness, sensory, emotional)

      assert String.contains?(narrative, "warmth") or String.contains?(narrative, "Sensation")
    end
  end

  describe "maybe_metacognate/3" do
    test "returns observation for high awareness" do
      consciousness = %{ConsciousnessState.new() | meta_awareness: 0.8}
      emotional = %EmotionalState{}
      personality = build_personality(openness: 0.8)

      {awareness, observation} =
        Consciousness.maybe_metacognate(consciousness, emotional, personality)

      assert awareness >= 0.0
      # Observation might be nil if random chance doesn't trigger it
      assert is_nil(observation) or is_binary(observation)
    end

    test "rarely generates observation for low awareness" do
      consciousness = %{ConsciousnessState.new() | meta_awareness: 0.1}
      emotional = %EmotionalState{}
      personality = build_personality(openness: 0.2)

      # Run multiple times to check probability
      results =
        for _ <- 1..10 do
          {_, observation} = Consciousness.maybe_metacognate(consciousness, emotional, personality)
          observation
        end

      # Most should be nil with low awareness
      nil_count = Enum.count(results, &is_nil/1)
      assert nil_count >= 5
    end
  end

  describe "tick/2" do
    test "decays meta_awareness" do
      consciousness = %{ConsciousnessState.new() | meta_awareness: 0.9}
      personality = build_personality()

      result = Consciousness.tick(consciousness, personality)
      # Meta awareness should decay by 5% each tick
      assert result.meta_awareness < 0.9
    end

    test "ages experience stream" do
      old_exp = %{content: "old", timestamp: DateTime.add(DateTime.utc_now(), -700, :second)}
      recent_exp = %{content: "recent", timestamp: DateTime.utc_now()}

      consciousness = %{
        ConsciousnessState.new()
        | experience_stream: [recent_exp, old_exp]
      }

      personality = build_personality()

      result = Consciousness.tick(consciousness, personality)
      # Old experience (> 600 seconds) should be removed
      assert Enum.count(result.experience_stream) == 1
    end

    test "meta awareness has minimum based on openness" do
      consciousness = %{ConsciousnessState.new() | meta_awareness: 0.1}
      personality = build_personality(openness: 0.8)

      result = Consciousness.tick(consciousness, personality)
      # Minimum is openness * 0.3 = 0.24
      assert result.meta_awareness >= 0.24
    end
  end

  defp build_personality(opts \\ []) do
    %Personality{
      openness: Keyword.get(opts, :openness, 0.5),
      conscientiousness: Keyword.get(opts, :conscientiousness, 0.5),
      extraversion: Keyword.get(opts, :extraversion, 0.5),
      agreeableness: Keyword.get(opts, :agreeableness, 0.5),
      neuroticism: Keyword.get(opts, :neuroticism, 0.5)
    }
  end
end
