defmodule Viva.Avatars.Systems.NeurochemistryTest do
  use ExUnit.Case, async: true

  alias Viva.Avatars.BioState
  alias Viva.Avatars.Systems.Neurochemistry

  @base_bio %BioState{dopamine: 0.5, oxytocin: 0.5, cortisol: 0.5, adenosine: 0.5, libido: 0.5}

  describe "apply_effect/2 with :interaction_start" do
    test "boosts dopamine" do
      result = Neurochemistry.apply_effect(@base_bio, :interaction_start)
      assert result.dopamine > @base_bio.dopamine
    end

    test "boosts oxytocin" do
      result = Neurochemistry.apply_effect(@base_bio, :interaction_start)
      assert result.oxytocin > @base_bio.oxytocin
    end

    test "increases adenosine (energy cost)" do
      result = Neurochemistry.apply_effect(@base_bio, :interaction_start)
      assert result.adenosine > @base_bio.adenosine
    end

    test "boosts libido" do
      result = Neurochemistry.apply_effect(@base_bio, :interaction_start)
      assert result.libido > @base_bio.libido
    end
  end

  describe "apply_effect/2 with :interaction_ongoing" do
    test "boosts oxytocin moderately" do
      result = Neurochemistry.apply_effect(@base_bio, :interaction_ongoing)
      assert result.oxytocin == @base_bio.oxytocin + 0.05
    end

    test "slightly increases adenosine" do
      result = Neurochemistry.apply_effect(@base_bio, :interaction_ongoing)
      assert result.adenosine == @base_bio.adenosine + 0.02
    end

    test "doesn't affect dopamine" do
      result = Neurochemistry.apply_effect(@base_bio, :interaction_ongoing)
      assert result.dopamine == @base_bio.dopamine
    end
  end

  describe "apply_effect/2 with :interaction_end" do
    test "dampens dopamine (crash)" do
      result = Neurochemistry.apply_effect(@base_bio, :interaction_end)
      assert result.dopamine < @base_bio.dopamine
    end

    test "doesn't affect other hormones" do
      result = Neurochemistry.apply_effect(@base_bio, :interaction_end)
      assert result.oxytocin == @base_bio.oxytocin
      assert result.cortisol == @base_bio.cortisol
      assert result.adenosine == @base_bio.adenosine
    end
  end

  describe "apply_effect/2 with :thought_generated" do
    test "slightly boosts dopamine" do
      result = Neurochemistry.apply_effect(@base_bio, :thought_generated)
      assert result.dopamine == @base_bio.dopamine + 0.02
    end

    test "slightly increases adenosine" do
      result = Neurochemistry.apply_effect(@base_bio, :thought_generated)
      assert result.adenosine == @base_bio.adenosine + 0.01
    end
  end

  describe "apply_effect/2 with :deep_sleep_tick" do
    test "reduces adenosine" do
      result = Neurochemistry.apply_effect(@base_bio, :deep_sleep_tick)
      assert result.adenosine < @base_bio.adenosine
    end

    test "reduces cortisol" do
      result = Neurochemistry.apply_effect(@base_bio, :deep_sleep_tick)
      assert result.cortisol < @base_bio.cortisol
    end

    test "reduces dopamine slightly" do
      result = Neurochemistry.apply_effect(@base_bio, :deep_sleep_tick)
      assert result.dopamine < @base_bio.dopamine
    end
  end

  describe "apply_effect/2 with :stress_event" do
    test "spikes cortisol significantly" do
      result = Neurochemistry.apply_effect(@base_bio, :stress_event)
      assert result.cortisol == @base_bio.cortisol + 0.3
    end

    test "drops dopamine significantly" do
      result = Neurochemistry.apply_effect(@base_bio, :stress_event)
      assert result.dopamine == @base_bio.dopamine - 0.2
    end

    test "drops oxytocin" do
      result = Neurochemistry.apply_effect(@base_bio, :stress_event)
      assert result.oxytocin == @base_bio.oxytocin - 0.1
    end
  end

  describe "apply_effect/2 with unknown event" do
    test "returns bio unchanged" do
      result = Neurochemistry.apply_effect(@base_bio, :unknown_event)
      assert result == @base_bio
    end
  end

  describe "hormone bounds" do
    test "hormones cap at 1.0" do
      high_bio = %BioState{
        dopamine: 0.95,
        oxytocin: 0.95,
        cortisol: 0.95,
        adenosine: 0.95,
        libido: 0.95
      }

      result = Neurochemistry.apply_effect(high_bio, :interaction_start)

      assert result.dopamine <= 1.0
      assert result.oxytocin <= 1.0
      assert result.adenosine <= 1.0
      assert result.libido <= 1.0
    end

    test "hormones don't go below 0.0" do
      low_bio = %BioState{
        dopamine: 0.05,
        oxytocin: 0.05,
        cortisol: 0.05,
        adenosine: 0.05,
        libido: 0.05
      }

      result = Neurochemistry.apply_effect(low_bio, :stress_event)

      assert result.dopamine >= 0.0
      assert result.oxytocin >= 0.0
      assert result.cortisol >= 0.0
    end
  end
end
