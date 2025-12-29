defmodule Viva.Avatars.Systems.NeurochemistryTest do
  use Viva.DataCase, async: true
  alias Viva.Avatars.BioState
  alias Viva.Avatars.Systems.Neurochemistry

  setup do
    {:ok, bio: %BioState{dopamine: 0.5, oxytocin: 0.5, cortisol: 0.5, adenosine: 0.5, libido: 0.5}}
  end

  describe "apply_effect/2" do
    test "interaction_start boosts dopamine and oxytocin", %{bio: bio} do
      updated = Neurochemistry.apply_effect(bio, :interaction_start)
      assert updated.dopamine > bio.dopamine
      assert updated.oxytocin > bio.oxytocin
      assert updated.libido > bio.libido
    end

    test "interaction_ongoing boosts oxytocin", %{bio: bio} do
      updated = Neurochemistry.apply_effect(bio, :interaction_ongoing)
      assert updated.oxytocin > bio.oxytocin
    end

    test "interaction_end dampens dopamine", %{bio: bio} do
      updated = Neurochemistry.apply_effect(bio, :interaction_end)
      assert updated.dopamine < bio.dopamine
    end

    test "thought_generated boosts dopamine slightly", %{bio: bio} do
      updated = Neurochemistry.apply_effect(bio, :thought_generated)
      assert updated.dopamine > bio.dopamine
    end

    test "deep_sleep_tick dampens adenosine and cortisol", %{bio: bio} do
      updated = Neurochemistry.apply_effect(bio, :deep_sleep_tick)
      assert updated.adenosine < bio.adenosine
      assert updated.cortisol < bio.cortisol
    end

    test "stress_event boosts cortisol and dampens dopamine", %{bio: bio} do
      updated = Neurochemistry.apply_effect(bio, :stress_event)
      assert updated.cortisol > bio.cortisol
      assert updated.dopamine < bio.dopamine
    end

    test "surprise_high boosts dopamine and cortisol", %{bio: bio} do
      updated = Neurochemistry.apply_effect(bio, :surprise_high)
      assert updated.dopamine > bio.dopamine
      assert updated.cortisol > bio.cortisol
    end

    test "surprise_moderate boosts dopamine slightly", %{bio: bio} do
      updated = Neurochemistry.apply_effect(bio, :surprise_moderate)
      assert updated.dopamine > bio.dopamine
      assert updated.cortisol == bio.cortisol
    end

    test "sensory_pain boosts cortisol", %{bio: bio} do
      updated = Neurochemistry.apply_effect(bio, :sensory_pain)
      assert updated.cortisol > bio.cortisol
      assert updated.dopamine < bio.dopamine
    end

    test "sensory_pleasure boosts dopamine", %{bio: bio} do
      updated = Neurochemistry.apply_effect(bio, :sensory_pleasure)
      assert updated.dopamine > bio.dopamine
      assert updated.oxytocin > bio.oxytocin
    end

    test "unknown event returns original state", %{bio: bio} do
      assert Neurochemistry.apply_effect(bio, :unknown) == bio
    end

    test "clamps values at 0.0 and 1.0" do
      low_bio = %BioState{dopamine: 0.01}
      high_bio = %BioState{dopamine: 0.99}

      assert Neurochemistry.apply_effect(low_bio, :interaction_end).dopamine == 0.0
      assert Neurochemistry.apply_effect(high_bio, :interaction_start).dopamine == 1.0
    end
  end
end
