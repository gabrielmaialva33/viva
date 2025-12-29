defmodule Viva.Avatars.Systems.BiologyTest do
  use Viva.DataCase, async: true

  alias Viva.Avatars.BioState
  alias Viva.Avatars.Personality
  alias Viva.Avatars.Systems.Biology

  setup do
    bio = %BioState{
      dopamine: 1.0,
      cortisol: 1.0,
      oxytocin: 1.0,
      adenosine: 0.0,
      libido: 0.5
    }

    personality = %Personality{
      extraversion: 0.5,
      neuroticism: 0.5
    }

    {:ok, bio: bio, personality: personality}
  end

  describe "tick/2" do
    test "decays hormones", %{bio: bio, personality: p} do
      updated = Biology.tick(bio, p)

      assert updated.dopamine < bio.dopamine
      assert updated.oxytocin < bio.oxytocin
      assert updated.cortisol < bio.cortisol
    end

    test "accumulates adenosine", %{bio: bio, personality: p} do
      updated = Biology.tick(bio, p)
      assert updated.adenosine > bio.adenosine
    end

    test "extraversion accelerates dopamine decay", %{bio: bio} do
      p_low = %Personality{extraversion: 0.0}
      p_high = %Personality{extraversion: 1.0}

      updated_low = Biology.tick(bio, p_low)
      updated_high = Biology.tick(bio, p_high)

      # Higher extraversion = faster decay = lower remaining value
      assert updated_high.dopamine < updated_low.dopamine
    end

    test "neuroticism slows cortisol decay", %{bio: bio} do
      p_low = %Personality{neuroticism: 0.0}
      p_high = %Personality{neuroticism: 1.0}

      updated_low = Biology.tick(bio, p_low)
      updated_high = Biology.tick(bio, p_high)

      # Higher neuroticism = slower decay = higher remaining value
      assert updated_high.cortisol > updated_low.cortisol
    end

    test "high cortisol kills libido", %{bio: bio, personality: p} do
      # Bio already has 1.0 cortisol in setup
      updated = Biology.tick(bio, p)
      assert updated.libido == 0.0
    end

    test "low cortisol preserves libido", %{bio: bio, personality: p} do
      bio = %{bio | cortisol: 0.2}
      updated = Biology.tick(bio, p)
      assert updated.libido == 0.5
    end

    test "adenosine suppresses dopamine", %{bio: bio, personality: p} do
      # Compare decay with and without fatigue
      bio_tired = %{bio | adenosine: 1.0}
      bio_fresh = %{bio | adenosine: 0.0}

      updated_tired = Biology.tick(bio_tired, p)
      updated_fresh = Biology.tick(bio_fresh, p)

      # Fatigue factor is applied at end: dopamine * (1 - adenosine*0.3)
      # So tired one should be significantly lower
      assert updated_tired.dopamine < updated_fresh.dopamine
    end
  end
end
