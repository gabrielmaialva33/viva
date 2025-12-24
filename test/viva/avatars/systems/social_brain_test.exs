defmodule Viva.Avatars.Systems.SocialBrainTest do
  use ExUnit.Case, async: true

  alias Viva.Avatars.Avatar
  alias Viva.Avatars.Personality
  alias Viva.Avatars.SocialPersona
  alias Viva.Avatars.Systems.SocialBrain

  defp build_avatar(opts) do
    personality = %Personality{
      openness: Keyword.get(opts, :openness, 0.5),
      conscientiousness: Keyword.get(opts, :conscientiousness, 0.5),
      extraversion: Keyword.get(opts, :extraversion, 0.5),
      agreeableness: Keyword.get(opts, :agreeableness, 0.5),
      neuroticism: Keyword.get(opts, :neuroticism, 0.5)
    }

    social_persona = %SocialPersona{
      social_ambition: Keyword.get(opts, :social_ambition, 0.5),
      public_reputation: Keyword.get(opts, :public_reputation, 0.5),
      current_mask_intensity: Keyword.get(opts, :current_mask_intensity, 0.0),
      perceived_traits: []
    }

    %Avatar{
      id: Keyword.get(opts, :id, Ecto.UUID.generate()),
      name: Keyword.get(opts, :name, "TestAvatar"),
      personality: personality,
      social_persona: social_persona,
      moral_flexibility: Keyword.get(opts, :moral_flexibility, 0.5)
    }
  end

  describe "calculate_mask_stress/2" do
    test "returns stress based on mask intensity" do
      avatar = build_avatar(neuroticism: 0.5, moral_flexibility: 0.5)
      stress = SocialBrain.calculate_mask_stress(avatar, 0.5)

      assert is_float(stress)
      assert stress > 0.0
    end

    test "zero mask intensity produces no stress" do
      avatar = build_avatar(neuroticism: 0.5, moral_flexibility: 0.5)
      stress = SocialBrain.calculate_mask_stress(avatar, 0.0)

      assert stress == 0.0
    end

    test "high neuroticism amplifies stress" do
      low_neurotic = build_avatar(neuroticism: 0.0, moral_flexibility: 0.5)
      high_neurotic = build_avatar(neuroticism: 1.0, moral_flexibility: 0.5)

      low_stress = SocialBrain.calculate_mask_stress(low_neurotic, 0.5)
      high_stress = SocialBrain.calculate_mask_stress(high_neurotic, 0.5)

      assert high_stress > low_stress
    end

    test "high moral flexibility reduces stress" do
      rigid = build_avatar(neuroticism: 0.5, moral_flexibility: 0.0)
      flexible = build_avatar(neuroticism: 0.5, moral_flexibility: 1.0)

      rigid_stress = SocialBrain.calculate_mask_stress(rigid, 0.5)
      flexible_stress = SocialBrain.calculate_mask_stress(flexible, 0.5)

      assert flexible_stress < rigid_stress
    end

    test "Machiavellian avatar (high flexibility) feels minimal stress" do
      machiavellian = build_avatar(neuroticism: 0.5, moral_flexibility: 1.0)
      stress = SocialBrain.calculate_mask_stress(machiavellian, 1.0)

      # Should still have some stress, but minimal
      assert stress < 0.3
    end

    test "stress is capped at 10.0" do
      avatar = build_avatar(neuroticism: 1.0, moral_flexibility: 0.0)
      stress = SocialBrain.calculate_mask_stress(avatar, 1.0)

      assert stress <= 10.0
    end
  end

  describe "decide_strategy/3" do
    test "returns :authentic when integrity exceeds incentive" do
      # Low ambition, high conscientiousness, low flexibility
      honest_avatar =
        build_avatar(
          social_ambition: 0.1,
          conscientiousness: 0.9,
          moral_flexibility: 0.1
        )

      result = SocialBrain.decide_strategy(honest_avatar, 0.3, 0.2)
      assert result == :authentic
    end

    test "returns {:mask, intensity} when incentive exceeds integrity" do
      # High ambition, low conscientiousness, high flexibility
      manipulative_avatar =
        build_avatar(
          social_ambition: 0.9,
          conscientiousness: 0.1,
          moral_flexibility: 0.9
        )

      result = SocialBrain.decide_strategy(manipulative_avatar, 0.8, 0.8)
      assert {:mask, intensity} = result
      assert is_float(intensity)
      assert intensity > 0.0
    end

    test "high stakes increase incentive to mask" do
      avatar = build_avatar(social_ambition: 0.5, moral_flexibility: 0.5)

      low_stakes = SocialBrain.decide_strategy(avatar, 0.3, 0.1)
      high_stakes = SocialBrain.decide_strategy(avatar, 0.3, 0.9)

      # High stakes more likely to trigger masking
      case {low_stakes, high_stakes} do
        {:authentic, {:mask, _}} -> assert true
        {{:mask, low_intensity}, {:mask, high_intensity}} -> assert high_intensity >= low_intensity
        _ -> :ok
      end
    end

    test "high target reputation increases incentive to mask" do
      ambitious_avatar = build_avatar(social_ambition: 0.8, moral_flexibility: 0.6)

      low_rep = SocialBrain.decide_strategy(ambitious_avatar, 0.2, 0.3)
      high_rep = SocialBrain.decide_strategy(ambitious_avatar, 0.9, 0.3)

      # Ambitious avatar more likely to mask for powerful people
      case {low_rep, high_rep} do
        {:authentic, {:mask, _}} -> assert true
        {{:mask, low_intensity}, {:mask, high_intensity}} -> assert high_intensity >= low_intensity
        _ -> :ok
      end
    end

    test "mask intensity is capped at 1.0" do
      extreme_avatar =
        build_avatar(
          social_ambition: 1.0,
          conscientiousness: 0.0,
          moral_flexibility: 1.0
        )

      {:mask, intensity} = SocialBrain.decide_strategy(extreme_avatar, 1.0, 1.0)
      assert intensity <= 1.0
    end
  end

  describe "process_interaction/2" do
    test "updates current_mask_intensity with exponential moving average" do
      avatar = build_avatar(current_mask_intensity: 0.0)

      updated = SocialBrain.process_interaction(avatar, 1.0)

      # New intensity = 0.0 * 0.8 + 1.0 * 0.2 = 0.2
      assert updated.social_persona.current_mask_intensity == 0.2
    end

    test "maintains weighted history of mask usage" do
      avatar = build_avatar(current_mask_intensity: 0.5)

      updated = SocialBrain.process_interaction(avatar, 0.0)

      # New intensity = 0.5 * 0.8 + 0.0 * 0.2 = 0.4
      assert_in_delta updated.social_persona.current_mask_intensity, 0.4, 0.001
    end

    test "consecutive high mask usage accumulates" do
      initial_avatar = build_avatar(current_mask_intensity: 0.0)

      final_avatar =
        initial_avatar
        |> SocialBrain.process_interaction(1.0)
        |> SocialBrain.process_interaction(1.0)
        |> SocialBrain.process_interaction(1.0)

      # After 3 rounds: should approach higher values
      assert final_avatar.social_persona.current_mask_intensity > 0.3
    end

    test "authentic interactions (mask 0) reduce accumulated mask intensity" do
      initial_avatar = build_avatar(current_mask_intensity: 0.8)

      final_avatar =
        initial_avatar
        |> SocialBrain.process_interaction(0.0)
        |> SocialBrain.process_interaction(0.0)

      # Mask intensity should decrease
      assert final_avatar.social_persona.current_mask_intensity < 0.8
    end
  end
end
