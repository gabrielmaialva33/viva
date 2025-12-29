defmodule Viva.Avatars.Systems.AttachmentBiasTest do
  use Viva.DataCase, async: true
  alias Viva.Avatars.Personality
  alias Viva.Avatars.Systems.AttachmentBias

  describe "interpret/2" do
    test "applies secure bias (positive valence shift, lower threat)" do
      p = %Personality{attachment_style: :secure}
      stim = %{type: :social, valence: 0.0, intensity: 0.5, threat_level: 0.1}

      {updated, interpretation} = AttachmentBias.interpret(stim, p)

      assert updated.valence > 0.0
      assert updated.threat_level < 0.1
      assert interpretation.bias_applied == true
    end

    test "applies anxious bias (negative valence shift, higher intensity and threat)" do
      p = %Personality{attachment_style: :anxious}
      stim = %{type: :social, valence: 0.0, intensity: 0.5, threat_level: 0.1}

      {updated, _} = AttachmentBias.interpret(stim, p)

      assert updated.valence < 0.0
      assert updated.intensity > 0.5
      assert updated.threat_level > 0.1
    end

    test "applies avoidant bias (dampened intensity)" do
      p = %Personality{attachment_style: :avoidant}
      stim = %{type: :social, intensity: 0.5}

      {updated, _} = AttachmentBias.interpret(stim, p)
      assert updated.intensity < 0.5
    end

    test "does not apply bias to non-social stimuli" do
      p = %Personality{attachment_style: :anxious}
      stim = %{type: :physical, valence: 0.0}

      {updated, interpretation} = AttachmentBias.interpret(stim, p)
      assert updated == stim
      assert interpretation.bias_applied == false
    end
  end

  describe "interpret_situation/2" do
    test "covers all situations and styles" do
      styles = [:secure, :anxious, :avoidant, :fearful]
      situations = [:no_response, :criticism, :more_time_together, :distance, :affection, :conflict]

      for style <- styles, situation <- situations do
        res = AttachmentBias.interpret_situation(situation, style)
        assert is_binary(res)
      end
    end

    test "handles unknown situations with default" do
      assert AttachmentBias.interpret_situation(:unknown, :secure) =~ "fine"
      assert AttachmentBias.interpret_situation(:unknown, :anxious) =~ "okay"
    end
  end

  describe "helpers" do
    test "describe_style/1 returns strings" do
      assert is_binary(AttachmentBias.describe_style(:secure))
    end

    test "social_initiative/1 returns floats" do
      assert AttachmentBias.social_initiative(:secure) == 0.7
      assert AttachmentBias.social_initiative(:avoidant) == 0.3
    end

    test "uncertainty_response/1 returns atoms" do
      assert AttachmentBias.uncertainty_response(:secure) == :calm_inquiry
      assert AttachmentBias.uncertainty_response(:anxious) == :protest_behavior
    end
  end
end
