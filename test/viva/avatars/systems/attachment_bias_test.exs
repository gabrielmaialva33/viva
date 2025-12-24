defmodule Viva.Avatars.Systems.AttachmentBiasTest do
  use ExUnit.Case, async: true

  alias Viva.Avatars.Personality
  alias Viva.Avatars.Systems.AttachmentBias

  describe "interpret/2" do
    test "applies bias to social stimuli" do
      stimulus = %{
        type: :social,
        source: "conversation_partner",
        intensity: 0.5,
        valence: 0.0,
        novelty: 0.3,
        threat_level: 0.0,
        social_context: :conversation
      }

      personality = %Personality{attachment_style: :anxious}

      {biased_stimulus, interpretation} = AttachmentBias.interpret(stimulus, personality)

      assert interpretation.bias_applied == true
      assert interpretation.style == :anxious
      # Anxious style should make valence more negative
      assert biased_stimulus.valence < stimulus.valence
      # Anxious style should increase threat perception
      assert biased_stimulus.threat_level > stimulus.threat_level
    end

    test "does not apply bias to non-social stimuli" do
      stimulus = %{
        type: :ambient,
        source: "environment",
        intensity: 0.3,
        valence: 0.0,
        novelty: 0.2,
        threat_level: 0.0
      }

      personality = %Personality{attachment_style: :anxious}

      {biased_stimulus, interpretation} = AttachmentBias.interpret(stimulus, personality)

      assert interpretation.bias_applied == false
      # Stimulus should be unchanged
      assert biased_stimulus.valence == stimulus.valence
      assert biased_stimulus.threat_level == stimulus.threat_level
    end

    test "secure attachment adds positive bias" do
      stimulus = %{
        type: :social,
        source: "owner_presence",
        intensity: 0.5,
        valence: 0.0,
        novelty: 0.3,
        threat_level: 0.1
      }

      personality = %Personality{attachment_style: :secure}

      {biased_stimulus, interpretation} = AttachmentBias.interpret(stimulus, personality)

      assert interpretation.style == :secure
      # Secure style should make valence more positive
      assert biased_stimulus.valence > stimulus.valence
      # Secure style should reduce threat perception
      assert biased_stimulus.threat_level < stimulus.threat_level
    end

    test "anxious attachment amplifies intensity" do
      stimulus = %{
        type: :social_ambient,
        source: "owner_presence",
        intensity: 0.5,
        valence: 0.0,
        novelty: 0.3,
        threat_level: 0.0
      }

      personality = %Personality{attachment_style: :anxious}

      {biased_stimulus, _interpretation} = AttachmentBias.interpret(stimulus, personality)

      # Anxious style should amplify intensity
      assert biased_stimulus.intensity > stimulus.intensity
    end

    test "avoidant attachment dampens intensity" do
      stimulus = %{
        type: :social,
        source: "conversation_partner",
        intensity: 0.6,
        valence: 0.2,
        novelty: 0.3,
        threat_level: 0.0,
        social_context: :conversation
      }

      personality = %Personality{attachment_style: :avoidant}

      {biased_stimulus, _interpretation} = AttachmentBias.interpret(stimulus, personality)

      # Avoidant style should dampen intensity
      assert biased_stimulus.intensity < stimulus.intensity
    end

    test "fearful attachment increases threat and decreases valence" do
      stimulus = %{
        type: :social,
        source: "conversation_partner",
        intensity: 0.5,
        valence: 0.1,
        novelty: 0.3,
        threat_level: 0.0,
        social_context: :conversation
      }

      personality = %Personality{attachment_style: :fearful}

      {biased_stimulus, _interpretation} = AttachmentBias.interpret(stimulus, personality)

      # Fearful style should increase threat and decrease valence
      assert biased_stimulus.threat_level > stimulus.threat_level
      assert biased_stimulus.valence < stimulus.valence
    end

    test "adds attachment interpretation to stimulus" do
      stimulus = %{
        type: :social_ambient,
        source: "owner_presence",
        intensity: 0.3,
        valence: 0.0,
        novelty: 0.2,
        threat_level: 0.0
      }

      personality = %Personality{attachment_style: :anxious}

      {biased_stimulus, _interpretation} = AttachmentBias.interpret(stimulus, personality)

      assert biased_stimulus.attachment_interpretation != nil
      assert biased_stimulus.attachment_style == :anxious
    end
  end

  describe "interpret_situation/2 for :no_response" do
    test "secure interprets as busy" do
      result = AttachmentBias.interpret_situation(:no_response, :secure)
      assert result =~ "busy"
    end

    test "anxious interprets as rejection" do
      result = AttachmentBias.interpret_situation(:no_response, :anxious)
      assert result =~ "ignoring" or result =~ "wrong"
    end

    test "avoidant dismisses the need" do
      result = AttachmentBias.interpret_situation(:no_response, :avoidant)
      assert result =~ "don't need"
    end

    test "fearful expects rejection" do
      result = AttachmentBias.interpret_situation(:no_response, :fearful)
      assert result =~ "reject"
    end
  end

  describe "interpret_situation/2 for :criticism" do
    test "secure sees as feedback" do
      result = AttachmentBias.interpret_situation(:criticism, :secure)
      assert result =~ "feedback"
    end

    test "anxious feels hated" do
      result = AttachmentBias.interpret_situation(:criticism, :anxious)
      assert result =~ "hate" or result =~ "not good enough"
    end

    test "avoidant sees as controlling" do
      result = AttachmentBias.interpret_situation(:criticism, :avoidant)
      assert result =~ "controlling"
    end

    test "fearful feels disappointment" do
      result = AttachmentBias.interpret_situation(:criticism, :fearful)
      assert result =~ "disappoint"
    end
  end

  describe "interpret_situation/2 for :more_time_together" do
    test "secure is positive" do
      result = AttachmentBias.interpret_situation(:more_time_together, :secure)
      assert result =~ "nice"
    end

    test "anxious is eager" do
      result = AttachmentBias.interpret_situation(:more_time_together, :anxious)
      assert result =~ "Finally" or result =~ "waiting"
    end

    test "avoidant feels overwhelmed" do
      result = AttachmentBias.interpret_situation(:more_time_together, :avoidant)
      assert result =~ "space" or result =~ "much"
    end

    test "fearful is conflicted" do
      result = AttachmentBias.interpret_situation(:more_time_together, :fearful)
      assert result =~ "want" and result =~ "wrong"
    end
  end

  describe "interpret_situation/2 for :distance" do
    test "secure accepts it calmly" do
      result = AttachmentBias.interpret_situation(:distance, :secure)
      assert result =~ "space"
    end

    test "anxious worries about abandonment" do
      result = AttachmentBias.interpret_situation(:distance, :anxious)
      assert result =~ "pulling away" or result =~ "what did I do"
    end

    test "avoidant welcomes independence" do
      result = AttachmentBias.interpret_situation(:distance, :avoidant)
      assert result =~ "independence" or result =~ "Good"
    end

    test "fearful confirms expectations" do
      result = AttachmentBias.interpret_situation(:distance, :fearful)
      assert result =~ "expected" or result =~ "leave"
    end
  end

  describe "interpret_situation/2 for :affection" do
    test "secure appreciates it" do
      result = AttachmentBias.interpret_situation(:affection, :secure)
      assert result =~ "sweet" or result =~ "appreciate"
    end

    test "anxious doubts it" do
      result = AttachmentBias.interpret_situation(:affection, :anxious)
      assert result =~ "really mean" or result =~ "hope"
    end

    test "avoidant feels pressured" do
      result = AttachmentBias.interpret_situation(:affection, :avoidant)
      assert result =~ "intense" or result =~ "slow"
    end

    test "fearful wants to believe but is scared" do
      result = AttachmentBias.interpret_situation(:affection, :fearful)
      assert result =~ "believe" and result =~ "scared"
    end
  end

  describe "interpret_situation/2 for :conflict" do
    test "secure wants to work through it" do
      result = AttachmentBias.interpret_situation(:conflict, :secure)
      assert result =~ "work through"
    end

    test "anxious fears abandonment" do
      result = AttachmentBias.interpret_situation(:conflict, :anxious)
      assert result =~ "leave" or result =~ "terrible"
    end

    test "avoidant wants to escape" do
      result = AttachmentBias.interpret_situation(:conflict, :avoidant)
      assert result =~ "done" or result =~ "out"
    end

    test "fearful confirms negative expectations" do
      result = AttachmentBias.interpret_situation(:conflict, :fearful)
      assert result =~ "knew" and result =~ "wrong"
    end
  end

  describe "describe_style/1" do
    test "describes secure style" do
      result = AttachmentBias.describe_style(:secure)
      assert result =~ "intimacy" and result =~ "trusts"
    end

    test "describes anxious style" do
      result = AttachmentBias.describe_style(:anxious)
      assert result =~ "closeness" and result =~ "abandonment"
    end

    test "describes avoidant style" do
      result = AttachmentBias.describe_style(:avoidant)
      assert result =~ "independence"
    end

    test "describes fearful style" do
      result = AttachmentBias.describe_style(:fearful)
      assert result =~ "connection" and result =~ "rejection"
    end
  end

  describe "social_initiative/1" do
    test "anxious has highest initiative" do
      assert AttachmentBias.social_initiative(:anxious) > AttachmentBias.social_initiative(:secure)
    end

    test "avoidant has lowest initiative" do
      assert AttachmentBias.social_initiative(:avoidant) <
               AttachmentBias.social_initiative(:fearful)
    end

    test "secure has moderate initiative" do
      secure = AttachmentBias.social_initiative(:secure)
      assert secure > 0.5 and secure < 0.9
    end
  end

  describe "uncertainty_response/1" do
    test "secure responds with calm inquiry" do
      assert AttachmentBias.uncertainty_response(:secure) == :calm_inquiry
    end

    test "anxious responds with protest behavior" do
      assert AttachmentBias.uncertainty_response(:anxious) == :protest_behavior
    end

    test "avoidant responds with deactivation" do
      assert AttachmentBias.uncertainty_response(:avoidant) == :deactivation
    end

    test "fearful responds with freeze" do
      assert AttachmentBias.uncertainty_response(:fearful) == :freeze_response
    end
  end

  describe "integration scenarios" do
    test "anxious attachment creates more negative experience from neutral social input" do
      neutral_stimulus = %{
        type: :social_ambient,
        source: "owner_presence",
        intensity: 0.4,
        valence: 0.0,
        novelty: 0.3,
        threat_level: 0.0
      }

      anxious_personality = %Personality{attachment_style: :anxious}
      secure_personality = %Personality{attachment_style: :secure}

      {anxious_result, _} = AttachmentBias.interpret(neutral_stimulus, anxious_personality)
      {secure_result, _} = AttachmentBias.interpret(neutral_stimulus, secure_personality)

      # Anxious should have more negative valence and higher threat
      assert anxious_result.valence < secure_result.valence
      assert anxious_result.threat_level > secure_result.threat_level
    end

    test "avoidant and anxious interpret same stimulus very differently" do
      social_stimulus = %{
        type: :social,
        source: "conversation_partner",
        intensity: 0.6,
        valence: 0.3,
        novelty: 0.4,
        threat_level: 0.0,
        social_context: :conversation
      }

      avoidant = %Personality{attachment_style: :avoidant}
      anxious = %Personality{attachment_style: :anxious}

      {avoidant_result, avoidant_interp} = AttachmentBias.interpret(social_stimulus, avoidant)
      {anxious_result, anxious_interp} = AttachmentBias.interpret(social_stimulus, anxious)

      # Avoidant dampens, anxious amplifies
      assert avoidant_result.intensity < anxious_result.intensity

      # Different interpretations
      assert avoidant_interp.style == :avoidant
      assert anxious_interp.style == :anxious
    end
  end
end
