defmodule Viva.Avatars.Systems.MetacognitionTest do
  use ExUnit.Case, async: true

  alias Viva.Avatars.ConsciousnessState
  alias Viva.Avatars.EmotionalState
  alias Viva.Avatars.Personality
  alias Viva.Avatars.SelfModel
  alias Viva.Avatars.Systems.Metacognition

  describe "process/4" do
    test "skips processing when tick count is not at interval" do
      consciousness = ConsciousnessState.new()
      emotional = %EmotionalState{}
      personality = %Personality{}

      # Tick 1 should skip (interval is 10)
      {updated, result} = Metacognition.process(consciousness, emotional, personality, 1)

      assert result.patterns_detected == []
      assert result.insight == nil
      assert updated == consciousness
    end

    test "processes at interval tick" do
      consciousness = %ConsciousnessState{
        ConsciousnessState.new()
        | meta_awareness: 0.8,
          experience_stream: build_experience_stream(7)
      }

      emotional = %EmotionalState{pleasure: 0.5, arousal: 0.3, dominance: 0.6}
      personality = %Personality{openness: 0.7, neuroticism: 0.4}

      # Tick 10 should process (interval is 10)
      {_, result} = Metacognition.process(consciousness, emotional, personality, 10)

      # Should have run and potentially detected patterns
      assert is_list(result.patterns_detected)
      assert is_map(result.alignment)
    end

    test "runs more frequently with high meta_awareness" do
      consciousness = %ConsciousnessState{
        ConsciousnessState.new()
        | meta_awareness: 0.9,
          experience_stream: build_experience_stream(5)
      }

      emotional = %EmotionalState{pleasure: 0.3, arousal: 0.2, dominance: 0.5}
      personality = %Personality{openness: 0.6}

      # With high meta_awareness (0.9 > 0.6), interval is reduced to 7
      # So tick 7 should process
      {_, result} = Metacognition.process(consciousness, emotional, personality, 7)

      # Should have processed (not skipped)
      assert is_map(result.alignment)
    end
  end

  describe "detect_emotional_patterns/1" do
    test "returns empty list when insufficient experiences" do
      stream = build_experience_stream(3)
      patterns = Metacognition.detect_emotional_patterns(stream)
      assert patterns == []
    end

    test "detects recurring emotional patterns" do
      # Build stream with same mood appearing frequently
      stream =
        Enum.map(1..10, fn i ->
          %{
            timestamp: DateTime.utc_now(),
            emotion: %{mood: if(i <= 6, do: "sad", else: "neutral")},
            thought: nil
          }
        end)

      patterns = Metacognition.detect_emotional_patterns(stream)

      assert patterns != []
      assert Enum.any?(patterns, fn p -> p.response == "sad" end)
    end

    test "ignores moods below frequency threshold" do
      # Build stream where no mood appears in > 40% of experiences
      stream =
        Enum.map(1..10, fn i ->
          mood =
            case rem(i, 5) do
              0 -> "happy"
              1 -> "sad"
              2 -> "angry"
              3 -> "anxious"
              4 -> "neutral"
            end

          %{
            timestamp: DateTime.utc_now(),
            emotion: %{mood: mood},
            thought: nil
          }
        end)

      patterns = Metacognition.detect_emotional_patterns(stream)

      # Each mood appears only 2 times (20%), below 40% threshold
      assert patterns == []
    end
  end

  describe "check_alignment/2" do
    test "returns neutral when no self_model" do
      emotional = %EmotionalState{pleasure: 0.5, arousal: 0.3, dominance: 0.5}
      alignment = Metacognition.check_alignment(emotional, nil)

      assert alignment.ideal_alignment == 0.0
      assert alignment.feared_alignment == 0.0
      assert alignment.dominant_direction == :neutral
    end

    test "detects toward_ideal when positive emotional state" do
      emotional = %EmotionalState{pleasure: 0.8, arousal: 0.3, dominance: 0.7}

      self_model = %SelfModel{
        SelfModel.new()
        | ideal_self: "Happy and confident",
          feared_self: "Anxious and withdrawn"
      }

      alignment = Metacognition.check_alignment(emotional, self_model)

      assert alignment.ideal_alignment > alignment.feared_alignment
      assert alignment.dominant_direction == :toward_ideal
    end

    test "detects toward_feared when negative emotional state" do
      emotional = %EmotionalState{pleasure: -0.7, arousal: 0.8, dominance: -0.5}

      self_model = %SelfModel{
        SelfModel.new()
        | ideal_self: "Calm and collected",
          feared_self: "Anxious mess"
      }

      alignment = Metacognition.check_alignment(emotional, self_model)

      assert alignment.feared_alignment > alignment.ideal_alignment
      assert alignment.dominant_direction == :toward_feared
    end

    test "returns neutral when alignments are close" do
      emotional = %EmotionalState{pleasure: 0.0, arousal: 0.0, dominance: 0.0}

      self_model = %SelfModel{
        SelfModel.new()
        | ideal_self: "Balanced",
          feared_self: "Extreme"
      }

      alignment = Metacognition.check_alignment(emotional, self_model)

      # When difference is < 0.2, should be neutral
      assert alignment.dominant_direction == :neutral
    end
  end

  describe "maybe_generate_insight/4" do
    test "returns nil when meta_awareness is low" do
      consciousness = %ConsciousnessState{ConsciousnessState.new() | meta_awareness: 0.3}
      patterns = [%{trigger: "stress", response: "sad", frequency: 5}]
      alignment = %{ideal_alignment: 0.5, feared_alignment: 0.3, dominant_direction: :toward_ideal}
      personality = %Personality{openness: 0.6}

      insight =
        Metacognition.maybe_generate_insight(consciousness, patterns, alignment, personality)

      assert insight == nil
    end

    test "generates insight when meta_awareness is high and patterns exist" do
      consciousness = %ConsciousnessState{ConsciousnessState.new() | meta_awareness: 0.8}
      patterns = [%{trigger: "various", response: "anxious", frequency: 7}]
      alignment = %{ideal_alignment: 0.5, feared_alignment: 0.3, dominant_direction: :toward_ideal}
      personality = %Personality{openness: 0.7}

      insight =
        Metacognition.maybe_generate_insight(consciousness, patterns, alignment, personality)

      assert insight != nil
      assert is_binary(insight)
      assert insight =~ "anxious"
    end

    test "generates alignment-based insight when no patterns but significant direction" do
      consciousness = %ConsciousnessState{
        ConsciousnessState.new()
        | meta_awareness: 0.8,
          self_model: %SelfModel{SelfModel.new() | ideal_self: "Confident leader"}
      }

      patterns = []
      alignment = %{ideal_alignment: 0.8, feared_alignment: 0.2, dominant_direction: :toward_ideal}
      personality = %Personality{openness: 0.5}

      insight =
        Metacognition.maybe_generate_insight(consciousness, patterns, alignment, personality)

      assert insight != nil
      assert insight =~ "becoming" or insight =~ "good"
    end

    test "generates negative insight when toward feared self" do
      consciousness = %ConsciousnessState{
        ConsciousnessState.new()
        | meta_awareness: 0.8,
          self_model: %SelfModel{SelfModel.new() | feared_self: "Failure"}
      }

      patterns = []
      alignment = %{ideal_alignment: 0.2, feared_alignment: 0.8, dominant_direction: :toward_feared}
      personality = %Personality{neuroticism: 0.7}

      insight =
        Metacognition.maybe_generate_insight(consciousness, patterns, alignment, personality)

      assert insight != nil
      assert insight =~ "worried" or insight =~ "fear" or insight =~ "change" or insight =~ "right"
    end
  end

  describe "update_self_model/3" do
    test "merges new patterns with existing" do
      self_model = %SelfModel{
        SelfModel.new()
        | emotional_patterns: [%{situation: "work", typical_emotion: "stressed"}]
      }

      new_patterns = [%{trigger: "social", response: "anxious", frequency: 5}]
      alignment = %{ideal_alignment: 0.5, feared_alignment: 0.3, dominant_direction: :neutral}

      updated = Metacognition.update_self_model(self_model, new_patterns, alignment)

      assert length(updated.emotional_patterns) == 2
    end

    test "increases self_esteem when toward_ideal" do
      self_model = %SelfModel{SelfModel.new() | self_esteem: 0.5}
      patterns = []
      alignment = %{ideal_alignment: 0.8, feared_alignment: 0.2, dominant_direction: :toward_ideal}

      updated = Metacognition.update_self_model(self_model, patterns, alignment)

      assert updated.self_esteem > 0.5
    end

    test "decreases self_esteem when toward_feared" do
      self_model = %SelfModel{SelfModel.new() | self_esteem: 0.5}
      patterns = []
      alignment = %{ideal_alignment: 0.2, feared_alignment: 0.8, dominant_direction: :toward_feared}

      updated = Metacognition.update_self_model(self_model, patterns, alignment)

      assert updated.self_esteem < 0.5
    end

    test "respects self_esteem bounds" do
      high_esteem = %SelfModel{SelfModel.new() | self_esteem: 0.89}
      low_esteem = %SelfModel{SelfModel.new() | self_esteem: 0.11}

      toward_ideal = %{
        ideal_alignment: 0.9,
        feared_alignment: 0.1,
        dominant_direction: :toward_ideal
      }

      toward_feared = %{
        ideal_alignment: 0.1,
        feared_alignment: 0.9,
        dominant_direction: :toward_feared
      }

      updated_high = Metacognition.update_self_model(high_esteem, [], toward_ideal)
      updated_low = Metacognition.update_self_model(low_esteem, [], toward_feared)

      assert updated_high.self_esteem <= 0.9
      assert updated_low.self_esteem >= 0.1
    end
  end

  describe "calculate_congruence_delta/1" do
    test "returns positive delta for toward_ideal" do
      alignment = %{dominant_direction: :toward_ideal}
      delta = Metacognition.calculate_congruence_delta(alignment)
      assert delta > 0
    end

    test "returns negative delta for toward_feared" do
      alignment = %{dominant_direction: :toward_feared}
      delta = Metacognition.calculate_congruence_delta(alignment)
      assert delta < 0
    end

    test "returns small positive delta for neutral" do
      alignment = %{dominant_direction: :neutral}
      delta = Metacognition.calculate_congruence_delta(alignment)
      assert delta > 0
      assert delta < 0.05
    end
  end

  describe "describe/1" do
    test "describes high meta_awareness state" do
      consciousness = %ConsciousnessState{
        ConsciousnessState.new()
        | meta_awareness: 0.8,
          meta_observation: "I notice I'm overthinking"
      }

      description = Metacognition.describe(consciousness)
      assert description =~ "overthinking"
    end

    test "describes moderate meta_awareness" do
      consciousness = %ConsciousnessState{
        ConsciousnessState.new()
        | meta_awareness: 0.5
      }

      description = Metacognition.describe(consciousness)
      assert description =~ "Some self-awareness"
    end

    test "describes low meta_awareness" do
      consciousness = %ConsciousnessState{
        ConsciousnessState.new()
        | meta_awareness: 0.2
      }

      description = Metacognition.describe(consciousness)
      assert description =~ "autopilot"
    end
  end

  describe "integration scenarios" do
    test "detect_emotional_patterns finds recurring mood" do
      # Direct test of pattern detection
      experience_stream =
        Enum.map(1..10, fn _ ->
          %{
            timestamp: DateTime.utc_now(),
            emotion: %{mood: "anxious"},
            thought: "worrying"
          }
        end)

      patterns = Metacognition.detect_emotional_patterns(experience_stream)

      # Should detect the anxious pattern (100% of experiences)
      assert patterns != []
      assert Enum.any?(patterns, fn p -> p.response == "anxious" end)
    end

    test "check_alignment returns correct direction with self_model" do
      # Direct test of alignment calculation
      self_model = %{
        SelfModel.new()
        | ideal_self: "Happy and confident",
          feared_self: "Depressed and anxious"
      }

      positive_emotional = %EmotionalState{pleasure: 0.9, arousal: 0.2, dominance: 0.8}
      negative_emotional = %EmotionalState{pleasure: -0.7, arousal: 0.8, dominance: -0.5}

      positive_alignment = Metacognition.check_alignment(positive_emotional, self_model)
      negative_alignment = Metacognition.check_alignment(negative_emotional, self_model)

      # Positive state should have higher ideal alignment
      assert positive_alignment.ideal_alignment > 0
      # Negative state should have higher feared alignment
      assert negative_alignment.feared_alignment > 0
    end

    test "process runs and returns valid result structure" do
      consciousness = %{
        ConsciousnessState.new()
        | meta_awareness: 0.8,
          experience_stream: build_experience_stream(5)
      }

      emotional = %EmotionalState{pleasure: 0.3, arousal: 0.2, dominance: 0.5}
      personality = %Personality{openness: 0.6}

      {updated, result} = Metacognition.process(consciousness, emotional, personality, 10)

      # Should have proper result structure
      assert is_list(result.patterns_detected)
      assert is_map(result.alignment)
      assert is_float(result.self_congruence_delta)

      # Should return consciousness struct
      assert Map.has_key?(updated, :meta_awareness)
    end
  end

  # Helper to build experience stream
  defp build_experience_stream(count) do
    Enum.map(1..count, fn _ ->
      %{
        timestamp: DateTime.utc_now(),
        emotion: %{mood: Enum.random(["happy", "sad", "neutral"])},
        thought: nil
      }
    end)
  end
end
