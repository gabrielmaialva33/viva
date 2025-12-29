defmodule Viva.Avatars.Systems.EmotionRegulationTest do
  use Viva.DataCase, async: true

  alias Viva.Avatars.BioState
  alias Viva.Avatars.EmotionalState
  alias Viva.Avatars.EmotionRegulationState
  alias Viva.Avatars.Personality
  alias Viva.Avatars.Systems.EmotionRegulation

  setup do
    regulation = EmotionRegulationState.new()
    emotional = %EmotionalState{pleasure: 0.0, arousal: 0.0}
    bio = %BioState{}
    personality = %Personality{}

    {:ok, regulation: regulation, emotional: emotional, bio: bio, personality: personality}
  end

  describe "regulate/4" do
    test "stays idle when intensity is low", %{regulation: r, emotional: e, bio: b, personality: p} do
      {new_r, new_e, new_b} = EmotionRegulation.regulate(r, e, b, p)

      assert new_r.active_strategy == nil
      assert new_e == e
      assert new_b == b
    end

    test "starts regulation when intensity is high", %{regulation: r, bio: b, personality: p} do
      # Intensity = max(-pleasure, abs(arousal)). Threshold = 0.6
      emotional = %EmotionalState{pleasure: -0.8, arousal: 0.5}

      {new_r, new_e, _} = EmotionRegulation.regulate(r, emotional, b, p)

      assert new_r.active_strategy != nil
      assert new_r.strategy_duration == 1
      assert new_r.pre_regulation_pleasure == -0.8
      # Should have improved slightly
      assert new_e.pleasure > -0.8 or new_e.arousal < 0.5
    end

    test "continues regulation and eventually stops", %{bio: b, personality: p} do
      r = %EmotionRegulationState{
        active_strategy: :reappraise,
        strategy_duration: 1,
        pre_regulation_pleasure: -0.8,
        pre_regulation_arousal: 0.5
      }

      emotional = %EmotionalState{pleasure: -0.7, arousal: 0.4}

      {new_r, _, _} = EmotionRegulation.regulate(r, emotional, b, p)

      assert new_r.strategy_duration == 2

      # Simulate stop after duration
      r_long = %{r | strategy_duration: 5}
      {stopped_r, _, _} = EmotionRegulation.regulate(r_long, emotional, b, p)
      assert stopped_r.active_strategy == nil
      assert stopped_r.reappraise_effectiveness != r.reappraise_effectiveness
    end

    test "recovers exhaustion when not regulating", %{emotional: e, bio: b, personality: p} do
      r = %EmotionRegulationState{regulation_exhaustion: 0.5}
      {new_r, _, _} = EmotionRegulation.regulate(r, e, b, p)

      assert new_r.regulation_exhaustion < 0.5
    end
  end

  describe "select_strategy/2" do
    test "favors ruminate for high neuroticism" do
      p = %Personality{neuroticism: 1.0}
      r = EmotionRegulationState.new()

      # Sample multiple times to account for randomness
      strategies = for _ <- 1..50, do: EmotionRegulation.select_strategy(p, r)
      assert :ruminate in strategies
    end

    test "favors seek_support for high extraversion and secure attachment" do
      p = %Personality{extraversion: 1.0, attachment_style: :secure}
      r = EmotionRegulationState.new()

      strategies = for _ <- 1..50, do: EmotionRegulation.select_strategy(p, r)
      assert :seek_support in strategies
    end
  end

  describe "overwhelmed?/2" do
    test "returns true if intensity high and exhausted" do
      e = %EmotionalState{pleasure: -0.9}
      r = %EmotionRegulationState{regulation_exhaustion: 0.9}
      assert EmotionRegulation.overwhelmed?(e, r)
    end

    test "returns false otherwise" do
      e = %EmotionalState{pleasure: 0.0}
      r = %EmotionRegulationState{regulation_exhaustion: 0.9}
      refute EmotionRegulation.overwhelmed?(e, r)
    end
  end

  describe "describe/1" do
    test "returns correct strings" do
      s = %EmotionRegulationState{}
      assert EmotionRegulation.describe(%{s | active_strategy: nil}) =~ "stable"

      assert EmotionRegulation.describe(%{s | active_strategy: nil, regulation_exhaustion: 0.7}) =~
               "depleted"

      assert EmotionRegulation.describe(%{s | active_strategy: :ruminate}) =~ "Dwelling"
      assert EmotionRegulation.describe(%{s | active_strategy: :reappraise}) =~ "Reframing"

      assert EmotionRegulation.describe(%{s | active_strategy: :seek_support}) =~
               "Seeking connection"

      assert EmotionRegulation.describe(%{s | active_strategy: :suppress}) =~
               "Pushing emotions down"

      assert EmotionRegulation.describe(%{s | active_strategy: :distract}) =~ "Shifting attention"
    end
  end
end
