defmodule Viva.Sessions.DesireEngineTest do
  use Viva.DataCase, async: true
  alias Viva.Avatars.BioState
  alias Viva.Avatars.EmotionalState
  alias Viva.Avatars.Personality
  alias Viva.Sessions.DesireEngine

  setup do
    bio = %BioState{dopamine: 0.5, oxytocin: 0.5, cortisol: 0.2, adenosine: 0.0, libido: 0.4}
    emotional = %EmotionalState{pleasure: 0.0, arousal: 0.0}
    personality = %Personality{extraversion: 0.5, openness: 0.5}

    {:ok, bio: bio, emotional: emotional, personality: personality}
  end

  describe "determine/5" do
    test "rest takes priority when exhausted", %{bio: bio, emotional: e, personality: p} do
      bio_tired = %{bio | adenosine: 0.9}
      assert DesireEngine.determine(bio_tired, e, p) == :wants_rest
    end

    test "drive-based desires take priority", %{bio: bio, emotional: e, personality: p} do
      # Status drive -> wants to express
      assert DesireEngine.determine(bio, e, p, nil, :status) == :wants_to_express

      # Belonging drive + low oxytocin -> wants attention
      bio_lonely = %{bio | oxytocin: 0.2}
      assert DesireEngine.determine(bio_lonely, e, p, nil, :belonging) == :wants_attention
    end

    test "negative somatic bias suppresses social desires", %{
      bio: bio,
      emotional: e,
      personality: p
    } do
      # Setup social desire conditions
      bio_lonely = %{bio | oxytocin: 0.2}
      p_extra = %{p | extraversion: 0.8}

      # Negative bias (-0.5)
      somatic = %{bias: -0.5, signal: "Warning", markers_activated: 1}

      # Should fall back to something else, like wants_something_new (novelty)
      # if conditions met, or :none
      res = DesireEngine.determine(bio_lonely, e, p_extra, somatic)
      refute res == :wants_attention
    end

    test "novelty desire when dopamine is low", %{bio: bio, emotional: e, personality: p} do
      bio_bored = %{bio | dopamine: 0.1}
      p_open = %{p | openness: 0.8}

      assert DesireEngine.determine(bio_bored, e, p_open) == :wants_something_new
    end

    test "expression desire when emotions are intense", %{bio: bio, personality: p} do
      e_intense = %EmotionalState{arousal: 0.8, pleasure: 0.0}
      assert DesireEngine.determine(bio, e_intense, p) == :wants_to_express
    end

    test "crush desire when libido is high and bias positive", %{
      bio: bio,
      emotional: e,
      personality: p
    } do
      bio_excited = %{bio | libido: 0.7, oxytocin: 0.4}
      somatic = %{bias: 0.4, signal: "Attraction", markers_activated: 1}

      assert DesireEngine.determine(bio_excited, e, p, somatic) == :wants_to_see_crush
    end
  end

  describe "desire_from_drive/3" do
    # Since it's private, we test via determine with appropriate setups
    test "survival drive", %{bio: bio, emotional: e, personality: p} do
      bio_sleepy = %{bio | adenosine: 0.7}
      assert DesireEngine.determine(bio_sleepy, e, p, nil, :survival) == :wants_rest
    end

    test "safety drive", %{bio: bio, emotional: e, personality: p} do
      bio_stressed = %{bio | cortisol: 0.6}
      assert DesireEngine.determine(bio_stressed, e, p, nil, :safety) == :wants_rest
    end
  end
end
