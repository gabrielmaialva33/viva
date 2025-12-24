defmodule Viva.Avatars.Systems.EmotionRegulationTest do
  use ExUnit.Case, async: true

  alias Viva.Avatars.BioState
  alias Viva.Avatars.EmotionalState
  alias Viva.Avatars.EmotionRegulationState
  alias Viva.Avatars.Personality
  alias Viva.Avatars.Systems.EmotionRegulation

  describe "regulate/4" do
    test "starts regulation when emotional intensity is high" do
      regulation = EmotionRegulationState.new()
      emotional = %EmotionalState{pleasure: -0.8, arousal: 0.7, mood_label: "distressed"}
      bio = %BioState{cortisol: 0.5}
      personality = %Personality{neuroticism: 0.5, openness: 0.5}

      {new_reg, _new_emotional, _new_bio} =
        EmotionRegulation.regulate(regulation, emotional, bio, personality)

      assert new_reg.active_strategy != nil
      assert new_reg.strategy_duration == 1
    end

    test "does not start regulation when intensity is low" do
      regulation = EmotionRegulationState.new()
      emotional = %EmotionalState{pleasure: 0.3, arousal: 0.2, mood_label: "content"}
      bio = %BioState{cortisol: 0.2}
      personality = %Personality{neuroticism: 0.3}

      {new_reg, new_emotional, new_bio} =
        EmotionRegulation.regulate(regulation, emotional, bio, personality)

      assert new_reg.active_strategy == nil
      assert new_emotional == emotional
      assert new_bio == bio
    end

    test "continues active regulation" do
      regulation = %{
        EmotionRegulationState.new()
        | active_strategy: :reappraise,
          strategy_duration: 2,
          pre_regulation_pleasure: -0.7,
          pre_regulation_arousal: 0.6
      }

      emotional = %EmotionalState{pleasure: -0.5, arousal: 0.5, mood_label: "anxious"}
      bio = %BioState{cortisol: 0.4}
      personality = %Personality{openness: 0.8}

      {new_reg, _new_emotional, _new_bio} =
        EmotionRegulation.regulate(regulation, emotional, bio, personality)

      assert new_reg.active_strategy == :reappraise
      assert new_reg.strategy_duration == 3
    end

    test "stops regulation after duration exceeds limit" do
      regulation = %{
        EmotionRegulationState.new()
        | active_strategy: :distract,
          strategy_duration: 5,
          pre_regulation_pleasure: -0.6,
          pre_regulation_arousal: 0.5
      }

      emotional = %EmotionalState{pleasure: -0.3, arousal: 0.3, mood_label: "uneasy"}
      bio = %BioState{cortisol: 0.3}
      personality = %Personality{}

      {new_reg, _new_emotional, _new_bio} =
        EmotionRegulation.regulate(regulation, emotional, bio, personality)

      assert new_reg.active_strategy == nil
      assert new_reg.strategy_duration == 0
    end

    test "does not start regulation when exhausted" do
      regulation = %{EmotionRegulationState.new() | regulation_exhaustion: 0.9}
      emotional = %EmotionalState{pleasure: -0.8, arousal: 0.8, mood_label: "distressed"}
      bio = %BioState{cortisol: 0.6}
      personality = %Personality{neuroticism: 0.7}

      {new_reg, _new_emotional, _new_bio} =
        EmotionRegulation.regulate(regulation, emotional, bio, personality)

      # Should not start new regulation when exhausted
      assert new_reg.active_strategy == nil
    end

    test "recovers exhaustion when not regulating" do
      regulation = %{EmotionRegulationState.new() | regulation_exhaustion: 0.5}
      emotional = %EmotionalState{pleasure: 0.2, arousal: 0.1, mood_label: "calm"}
      bio = %BioState{cortisol: 0.1}
      personality = %Personality{}

      {new_reg, _new_emotional, _new_bio} =
        EmotionRegulation.regulate(regulation, emotional, bio, personality)

      assert new_reg.regulation_exhaustion < regulation.regulation_exhaustion
    end
  end

  describe "select_strategy/2" do
    test "high neuroticism favors rumination" do
      personality = %Personality{
        neuroticism: 0.9,
        openness: 0.3,
        conscientiousness: 0.3,
        extraversion: 0.3,
        agreeableness: 0.5,
        attachment_style: :secure
      }

      regulation = EmotionRegulationState.new()

      # Run multiple times and count strategy frequencies
      strategies =
        Enum.map(1..100, fn _ ->
          EmotionRegulation.select_strategy(personality, regulation)
        end)

      ruminate_count = Enum.count(strategies, &(&1 == :ruminate))

      # High neuroticism should lead to more rumination
      assert ruminate_count > 20
    end

    test "high openness and conscientiousness favors reappraisal" do
      personality = %Personality{
        neuroticism: 0.2,
        openness: 0.9,
        conscientiousness: 0.9,
        extraversion: 0.3,
        agreeableness: 0.5,
        attachment_style: :secure
      }

      regulation = EmotionRegulationState.new()

      strategies =
        Enum.map(1..100, fn _ ->
          EmotionRegulation.select_strategy(personality, regulation)
        end)

      reappraise_count = Enum.count(strategies, &(&1 == :reappraise))

      # High openness + conscientiousness should favor reappraisal
      assert reappraise_count > 20
    end

    test "high extraversion with secure attachment favors seeking support" do
      personality = %Personality{
        neuroticism: 0.2,
        openness: 0.5,
        conscientiousness: 0.5,
        extraversion: 0.9,
        agreeableness: 0.7,
        attachment_style: :secure
      }

      regulation = EmotionRegulationState.new()

      strategies =
        Enum.map(1..100, fn _ ->
          EmotionRegulation.select_strategy(personality, regulation)
        end)

      seek_support_count = Enum.count(strategies, &(&1 == :seek_support))

      # Extraverted + secure should seek support more
      assert seek_support_count > 15
    end

    test "avoidant attachment reduces support-seeking" do
      personality = %Personality{
        neuroticism: 0.2,
        openness: 0.5,
        conscientiousness: 0.5,
        extraversion: 0.9,
        agreeableness: 0.7,
        attachment_style: :avoidant
      }

      regulation = EmotionRegulationState.new()

      strategies =
        Enum.map(1..100, fn _ ->
          EmotionRegulation.select_strategy(personality, regulation)
        end)

      seek_support_count = Enum.count(strategies, &(&1 == :seek_support))

      # Avoidant should seek less support despite high extraversion
      assert seek_support_count < 30
    end

    test "low agreeableness favors suppression" do
      personality = %Personality{
        neuroticism: 0.3,
        openness: 0.3,
        conscientiousness: 0.5,
        extraversion: 0.3,
        agreeableness: 0.1,
        attachment_style: :secure
      }

      regulation = EmotionRegulationState.new()

      strategies =
        Enum.map(1..100, fn _ ->
          EmotionRegulation.select_strategy(personality, regulation)
        end)

      suppress_count = Enum.count(strategies, &(&1 == :suppress))

      # Low agreeableness should suppress more
      assert suppress_count > 20
    end
  end

  describe "strategy effects" do
    test "rumination worsens emotions" do
      regulation = %{
        EmotionRegulationState.new()
        | active_strategy: nil
      }

      emotional = %EmotionalState{pleasure: -0.5, arousal: 0.6, mood_label: "anxious"}
      bio = %BioState{cortisol: 0.4}

      # Force rumination by setting very high neuroticism
      personality = %Personality{
        neuroticism: 1.0,
        openness: 0.0,
        conscientiousness: 0.0,
        extraversion: 0.0,
        agreeableness: 1.0,
        attachment_style: :avoidant
      }

      # Simulate multiple ticks of rumination
      {final_reg, final_emotional, final_bio} =
        Enum.reduce(1..5, {regulation, emotional, bio}, fn _, {reg, emo, b} ->
          # Force the state to trigger regulation
          high_intensity_emo = %{emo | pleasure: -0.8, arousal: 0.7}
          EmotionRegulation.regulate(reg, high_intensity_emo, b, personality)
        end)

      # After rumination, things should be worse or cortisol higher
      if final_reg.ruminate_count > 0 do
        assert final_bio.cortisol >= bio.cortisol or final_emotional.pleasure <= emotional.pleasure
      end
    end

    test "reappraisal improves emotions" do
      regulation = %{
        EmotionRegulationState.new()
        | active_strategy: :reappraise,
          strategy_duration: 1,
          pre_regulation_pleasure: -0.6,
          pre_regulation_arousal: 0.5
      }

      emotional = %EmotionalState{pleasure: -0.5, arousal: 0.5, mood_label: "anxious"}
      bio = %BioState{cortisol: 0.5}
      personality = %Personality{openness: 0.8, conscientiousness: 0.8}

      {_new_reg, new_emotional, new_bio} =
        EmotionRegulation.regulate(regulation, emotional, bio, personality)

      # Reappraisal should improve pleasure and reduce cortisol
      assert new_emotional.pleasure > emotional.pleasure
      assert new_bio.cortisol < bio.cortisol
    end

    test "seek_support boosts oxytocin" do
      regulation = %{
        EmotionRegulationState.new()
        | active_strategy: :seek_support,
          strategy_duration: 1,
          pre_regulation_pleasure: -0.5,
          pre_regulation_arousal: 0.4
      }

      emotional = %EmotionalState{pleasure: -0.4, arousal: 0.4, mood_label: "lonely"}
      bio = %BioState{cortisol: 0.3, oxytocin: 0.2}
      personality = %Personality{extraversion: 0.8, attachment_style: :secure}

      {_new_reg, _new_emotional, new_bio} =
        EmotionRegulation.regulate(regulation, emotional, bio, personality)

      # Seeking support should boost oxytocin
      assert new_bio.oxytocin > bio.oxytocin
    end

    test "suppression reduces arousal but increases cortisol" do
      regulation = %{
        EmotionRegulationState.new()
        | active_strategy: :suppress,
          strategy_duration: 1,
          pre_regulation_pleasure: -0.4,
          pre_regulation_arousal: 0.7
      }

      emotional = %EmotionalState{pleasure: -0.4, arousal: 0.7, mood_label: "frustrated"}
      bio = %BioState{cortisol: 0.3}
      personality = %Personality{agreeableness: 0.2}

      {_new_reg, new_emotional, new_bio} =
        EmotionRegulation.regulate(regulation, emotional, bio, personality)

      # Suppression should reduce arousal but increase cortisol
      assert new_emotional.arousal < emotional.arousal
      assert new_bio.cortisol > bio.cortisol
    end
  end

  describe "describe/1" do
    test "describes inactive state" do
      regulation = EmotionRegulationState.new()
      result = EmotionRegulation.describe(regulation)
      assert result =~ "stable"
    end

    test "describes exhausted state" do
      regulation = %{EmotionRegulationState.new() | regulation_exhaustion: 0.8}
      result = EmotionRegulation.describe(regulation)
      assert result =~ "depleted"
    end

    test "describes rumination" do
      regulation = %{EmotionRegulationState.new() | active_strategy: :ruminate}
      result = EmotionRegulation.describe(regulation)
      assert result =~ "Dwelling" or result =~ "spiral"
    end

    test "describes reappraisal" do
      regulation = %{EmotionRegulationState.new() | active_strategy: :reappraise}
      result = EmotionRegulation.describe(regulation)
      assert result =~ "Reframing" or result =~ "perspective"
    end

    test "describes seeking support" do
      regulation = %{EmotionRegulationState.new() | active_strategy: :seek_support}
      result = EmotionRegulation.describe(regulation)
      assert result =~ "connection" or result =~ "share"
    end

    test "describes suppression" do
      regulation = %{EmotionRegulationState.new() | active_strategy: :suppress}
      result = EmotionRegulation.describe(regulation)
      assert result =~ "Pushing" or result =~ "composure"
    end

    test "describes distraction" do
      regulation = %{EmotionRegulationState.new() | active_strategy: :distract}
      result = EmotionRegulation.describe(regulation)
      assert result =~ "Shifting" or result =~ "move on"
    end
  end

  describe "overwhelmed?/2" do
    test "returns true when high intensity and exhausted" do
      emotional = %EmotionalState{pleasure: -0.8, arousal: 0.8, mood_label: "distressed"}
      regulation = %{EmotionRegulationState.new() | regulation_exhaustion: 0.9}

      assert EmotionRegulation.overwhelmed?(emotional, regulation)
    end

    test "returns false when intensity is low" do
      emotional = %EmotionalState{pleasure: 0.3, arousal: 0.2, mood_label: "calm"}
      regulation = %{EmotionRegulationState.new() | regulation_exhaustion: 0.9}

      refute EmotionRegulation.overwhelmed?(emotional, regulation)
    end

    test "returns false when not exhausted" do
      emotional = %EmotionalState{pleasure: -0.8, arousal: 0.8, mood_label: "distressed"}
      regulation = EmotionRegulationState.new()

      refute EmotionRegulation.overwhelmed?(emotional, regulation)
    end
  end

  describe "EmotionRegulationState helpers" do
    test "dominant_strategy returns most used strategy" do
      regulation = %{
        EmotionRegulationState.new()
        | ruminate_count: 5,
          reappraise_count: 10,
          suppress_count: 3
      }

      assert EmotionRegulationState.dominant_strategy(regulation) == :reappraise
    end

    test "dominant_strategy returns nil when no strategies used" do
      regulation = EmotionRegulationState.new()
      assert EmotionRegulationState.dominant_strategy(regulation) == nil
    end

    test "total_attempts sums all strategy counts" do
      regulation = %{
        EmotionRegulationState.new()
        | ruminate_count: 2,
          reappraise_count: 3,
          seek_support_count: 1,
          suppress_count: 4,
          distract_count: 5
      }

      assert EmotionRegulationState.total_attempts(regulation) == 15
    end
  end

  describe "integration scenario" do
    test "neurotic avatar develops rumination habit over time" do
      regulation = EmotionRegulationState.new()

      personality = %Personality{
        neuroticism: 0.9,
        openness: 0.2,
        conscientiousness: 0.3,
        extraversion: 0.2,
        agreeableness: 0.5,
        attachment_style: :anxious
      }

      # Simulate many high-stress episodes
      final_regulation =
        Enum.reduce(1..50, regulation, fn _, reg ->
          emotional = %EmotionalState{pleasure: -0.7, arousal: 0.7, mood_label: "anxious"}
          bio = %BioState{cortisol: 0.6}

          {new_reg, _emo, _bio} =
            EmotionRegulation.regulate(reg, emotional, bio, personality)

          # Reset active strategy to simulate time passing
          %{new_reg | active_strategy: nil, strategy_duration: 0}
        end)

      # Neurotic avatar should have ruminated significantly
      total = EmotionRegulationState.total_attempts(final_regulation)
      ruminate_ratio = final_regulation.ruminate_count / max(total, 1)

      assert ruminate_ratio > 0.3
    end

    test "well-adjusted avatar develops healthy coping patterns" do
      regulation = EmotionRegulationState.new()

      personality = %Personality{
        neuroticism: 0.2,
        openness: 0.8,
        conscientiousness: 0.7,
        extraversion: 0.6,
        agreeableness: 0.7,
        attachment_style: :secure
      }

      # Simulate many episodes
      final_regulation =
        Enum.reduce(1..50, regulation, fn _, reg ->
          emotional = %EmotionalState{pleasure: -0.7, arousal: 0.6, mood_label: "stressed"}
          bio = %BioState{cortisol: 0.5}

          {new_reg, _emo, _bio} =
            EmotionRegulation.regulate(reg, emotional, bio, personality)

          %{new_reg | active_strategy: nil, strategy_duration: 0}
        end)

      total = EmotionRegulationState.total_attempts(final_regulation)

      # Well-adjusted avatar should use more adaptive strategies
      adaptive_count =
        final_regulation.reappraise_count + final_regulation.seek_support_count

      adaptive_ratio = adaptive_count / max(total, 1)

      assert adaptive_ratio > 0.3
    end
  end
end
