defmodule Viva.Avatars.SelfModelTest do
  use ExUnit.Case, async: true

  alias Viva.Avatars.Personality
  alias Viva.Avatars.SelfModel

  describe "new/0" do
    test "creates a new self-model with defaults" do
      model = SelfModel.new()

      assert model.identity_narrative == "I am still discovering who I am."
      assert model.core_values == []
      assert model.core_beliefs == []
      assert model.perceived_strengths == []
      assert model.perceived_weaknesses == []
      assert model.self_esteem == 0.5
      assert model.self_efficacy == 0.5
      assert model.behavioral_patterns == []
      assert model.emotional_patterns == []
      assert model.current_goals == []
      assert model.ideal_self == nil
      assert model.feared_self == nil
      assert model.attachment_narrative == nil
      assert model.social_identities == []
    end
  end

  describe "changeset/2" do
    test "validates self_esteem range" do
      model = SelfModel.new()

      invalid_high = SelfModel.changeset(model, %{self_esteem: 1.5})
      refute invalid_high.valid?
      assert "must be less than or equal to 1.0" in errors_on(invalid_high).self_esteem

      invalid_low = SelfModel.changeset(model, %{self_esteem: -0.5})
      refute invalid_low.valid?
      assert "must be greater than or equal to 0.0" in errors_on(invalid_low).self_esteem

      valid_changeset = SelfModel.changeset(model, %{self_esteem: 0.7})
      assert valid_changeset.valid?
    end

    test "validates self_efficacy range" do
      model = SelfModel.new()

      invalid_changeset = SelfModel.changeset(model, %{self_efficacy: 2.0})
      refute invalid_changeset.valid?

      valid_changeset = SelfModel.changeset(model, %{self_efficacy: 0.8})
      assert valid_changeset.valid?
    end

    test "accepts valid attributes" do
      model = SelfModel.new()

      changeset =
        SelfModel.changeset(model, %{
          identity_narrative: "I am a creative problem solver.",
          core_values: ["honesty", "creativity"],
          perceived_strengths: ["empathy"],
          self_esteem: 0.7
        })

      assert changeset.valid?
    end
  end

  describe "from_personality/1" do
    test "initializes from a high openness personality" do
      personality = %Personality{
        openness: 0.8,
        conscientiousness: 0.5,
        extraversion: 0.5,
        agreeableness: 0.5,
        neuroticism: 0.5,
        attachment_style: :secure,
        values: ["creativity", "freedom"]
      }

      model = SelfModel.from_personality(personality)

      assert "creative thinking" in model.perceived_strengths
      assert "curiosity and imagination" in model.perceived_strengths
      assert model.core_values == ["creativity", "freedom"]

      assert model.attachment_narrative ==
               "I generally trust others and feel comfortable with intimacy."
    end

    test "initializes from a high conscientiousness personality" do
      personality = %Personality{
        openness: 0.5,
        conscientiousness: 0.8,
        extraversion: 0.5,
        agreeableness: 0.5,
        neuroticism: 0.5,
        attachment_style: :anxious,
        values: []
      }

      model = SelfModel.from_personality(personality)

      assert "reliability and organization" in model.perceived_strengths
      assert "discipline and focus" in model.perceived_strengths
      assert model.attachment_narrative == "I worry about abandonment and crave closeness."
    end

    test "derives weaknesses from low traits" do
      personality = %Personality{
        openness: 0.2,
        conscientiousness: 0.2,
        extraversion: 0.2,
        agreeableness: 0.2,
        neuroticism: 0.5,
        attachment_style: :avoidant,
        values: []
      }

      model = SelfModel.from_personality(personality)

      assert "resistance to change" in model.perceived_weaknesses
      assert "disorganization" in model.perceived_weaknesses
      assert "social withdrawal" in model.perceived_weaknesses
      assert "difficulty trusting others" in model.perceived_weaknesses

      assert model.attachment_narrative ==
               "I value independence and sometimes struggle with closeness."
    end

    test "derives weaknesses from high neuroticism" do
      personality = %Personality{
        openness: 0.5,
        conscientiousness: 0.5,
        extraversion: 0.5,
        agreeableness: 0.5,
        neuroticism: 0.85,
        attachment_style: :fearful,
        values: []
      }

      model = SelfModel.from_personality(personality)

      assert "emotional sensitivity" in model.perceived_weaknesses
      assert "anxiety proneness" in model.perceived_weaknesses
      assert model.attachment_narrative == "I desire closeness but fear getting hurt."
    end

    test "calculates base esteem from personality" do
      # High extraversion, low neuroticism = higher self-esteem
      personality = %Personality{
        openness: 0.5,
        conscientiousness: 0.5,
        extraversion: 0.8,
        agreeableness: 0.5,
        neuroticism: 0.2,
        attachment_style: :secure,
        values: []
      }

      model = SelfModel.from_personality(personality)
      # Expected: 0.5 + (0.5 - 0.2) * 0.3 + (0.8 - 0.5) * 0.2 = 0.5 + 0.09 + 0.06 = 0.65
      assert model.self_esteem >= 0.6
      assert model.self_esteem <= 0.7
    end

    test "calculates base efficacy from personality" do
      # High conscientiousness, high openness = higher self-efficacy
      personality = %Personality{
        openness: 0.8,
        conscientiousness: 0.8,
        extraversion: 0.5,
        agreeableness: 0.5,
        neuroticism: 0.5,
        attachment_style: :secure,
        values: []
      }

      model = SelfModel.from_personality(personality)
      # Expected: 0.5 + (0.8 - 0.5) * 0.3 + (0.8 - 0.5) * 0.2 = 0.5 + 0.09 + 0.06 = 0.65
      assert model.self_efficacy >= 0.6
      assert model.self_efficacy <= 0.7
    end
  end

  describe "query functions" do
    test "low_self_esteem?/1 returns true when esteem is low" do
      low_esteem = %SelfModel{self_esteem: 0.3}
      assert SelfModel.low_self_esteem?(low_esteem)

      normal_esteem = %SelfModel{self_esteem: 0.5}
      refute SelfModel.low_self_esteem?(normal_esteem)
    end

    test "confident?/1 returns true when efficacy is high" do
      confident = %SelfModel{self_efficacy: 0.7}
      assert SelfModel.confident?(confident)

      not_confident = %SelfModel{self_efficacy: 0.5}
      refute SelfModel.confident?(not_confident)
    end

    test "get_beliefs_by_domain/2 filters beliefs" do
      model = %SelfModel{
        core_beliefs: [
          %{domain: "self", belief: "I am capable", strength: 0.8},
          %{domain: "others", belief: "People are generally kind", strength: 0.6},
          %{domain: "self", belief: "I deserve respect", strength: 0.7}
        ]
      }

      self_beliefs = SelfModel.get_beliefs_by_domain(model, "self")
      assert length(self_beliefs) == 2
      assert Enum.all?(self_beliefs, &(&1.domain == "self"))

      other_beliefs = SelfModel.get_beliefs_by_domain(model, "others")
      assert length(other_beliefs) == 1
    end
  end

  defp errors_on(changeset) do
    Ecto.Changeset.traverse_errors(changeset, fn {msg, opts} ->
      Regex.replace(~r"%{(\w+)}", msg, fn _, key ->
        opts
        |> Keyword.get(String.to_existing_atom(key), key)
        |> to_string()
      end)
    end)
  end
end
