defmodule Viva.Avatars.Systems.MetacognitionTest do
  use Viva.DataCase, async: true

  alias Viva.Avatars.ConsciousnessState
  alias Viva.Avatars.EmotionalState
  alias Viva.Avatars.Personality
  alias Viva.Avatars.SelfModel
  alias Viva.Avatars.Systems.Metacognition

  setup do
    personality = %Personality{openness: 0.5, neuroticism: 0.5}
    self_model = %SelfModel{ideal_self: "Strong", feared_self: "Weak", self_esteem: 0.5}

    consciousness = %ConsciousnessState{
      meta_awareness: 0.5,
      self_model: self_model,
      self_congruence: 0.5
    }

    emotional = %EmotionalState{pleasure: 0.0, arousal: 0.0, dominance: 0.0}

    {:ok, c: consciousness, e: emotional, p: personality, s: self_model}
  end

  describe "process/4" do
    test "runs only on interval", %{c: c, e: e, p: p} do
      # Meta awareness 0.5 -> interval 10
      {_, res1} = Metacognition.process(c, e, p, 1)
      assert res1.patterns_detected == []

      {_, res10} = Metacognition.process(c, e, p, 10)
      assert match?(%{ideal_alignment: _}, res10.alignment)
    end

    test "runs more often with high meta awareness", %{c: c, e: e, p: p} do
      c_high = %{c | meta_awareness: 0.8}
      # High awareness -> interval 10-3 = 7
      {_, res7} = Metacognition.process(c_high, e, p, 7)
      assert match?(%{ideal_alignment: _}, res7.alignment)
    end
  end

  describe "detect_emotional_patterns/1" do
    test "returns empty for short stream" do
      assert Metacognition.detect_emotional_patterns([%{}, %{}]) == []
    end

    test "detects recurring emotions" do
      stream = [
        %{emotion: %{mood: "happy"}},
        %{emotion: %{mood: "happy"}},
        %{emotion: %{mood: "happy"}},
        %{emotion: %{mood: "sad"}},
        %{emotion: %{mood: "sad"}}
      ]

      patterns = Metacognition.detect_emotional_patterns(stream)
      assert length(patterns) == 2
      assert Enum.any?(patterns, fn p -> p.response == "happy" end)
    end
  end

  describe "check_alignment/2" do
    test "detects direction toward ideal", %{e: e, s: s} do
      # High pleasure/dominance -> Toward Ideal
      e_ideal = %{e | pleasure: 0.9, dominance: 0.9}
      res = Metacognition.check_alignment(e_ideal, s)
      assert res.dominant_direction == :toward_ideal
    end

    test "detects direction toward feared", %{e: e, s: s} do
      # Low pleasure/dominance + High arousal -> Toward Feared
      e_feared = %{e | pleasure: -0.9, dominance: -0.9, arousal: 0.8}
      res = Metacognition.check_alignment(e_feared, s)
      assert res.dominant_direction == :toward_feared
    end
  end

  describe "maybe_generate_insight/4" do
    test "generates insight when awareness high and pattern found", %{c: c, p: p} do
      c_insight = %{c | meta_awareness: 0.8}
      patterns = [%{response: "happy", frequency: 5}]
      alignment = %{dominant_direction: :neutral}

      insight = Metacognition.maybe_generate_insight(c_insight, patterns, alignment, p)
      assert insight =~ "pattern"
      assert insight =~ "happy"
    end

    test "returns nil when awareness low", %{c: c, p: p} do
      c_low = %{c | meta_awareness: 0.3}
      patterns = [%{response: "happy", frequency: 5}]
      alignment = %{dominant_direction: :neutral}

      assert Metacognition.maybe_generate_insight(c_low, patterns, alignment, p) == nil
    end
  end

  describe "update_self_model/3" do
    test "adjusts self-esteem based on alignment", %{s: s} do
      align_ideal = %{dominant_direction: :toward_ideal}
      updated = Metacognition.update_self_model(s, [], align_ideal)
      assert updated.self_esteem > s.self_esteem

      align_feared = %{dominant_direction: :toward_feared}
      updated_f = Metacognition.update_self_model(s, [], align_feared)
      assert updated_f.self_esteem < s.self_esteem
    end
  end

  describe "describe/1" do
    test "returns correct strings" do
      assert Metacognition.describe(%ConsciousnessState{
               meta_awareness: 0.8,
               meta_observation: "I see"
             }) == "I see"

      assert Metacognition.describe(%ConsciousnessState{meta_awareness: 0.5}) =~
               "Some self-awareness"

      assert Metacognition.describe(%ConsciousnessState{meta_awareness: 0.1}) =~ "autopilot"
    end
  end
end
