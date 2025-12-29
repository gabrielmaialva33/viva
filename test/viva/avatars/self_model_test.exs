defmodule Viva.Avatars.SelfModelTest do
  use Viva.DataCase, async: true

  alias Viva.Avatars.Personality
  alias Viva.Avatars.SelfModel

  describe "new/0" do
    test "returns default state" do
      model = SelfModel.new()
      assert model.self_esteem == 0.5
      assert model.coherence_level == 0.8
    end
  end

  describe "from_personality/1" do
    test "derives strengths and weaknesses from Big Five" do
      p = %Personality{
        # Creative
        openness: 0.9,
        # Disorganized
        conscientiousness: 0.1,
        # Social energy
        extraversion: 0.8,
        # Difficulty trusting
        agreeableness: 0.2,
        # Anxiety
        neuroticism: 0.9
      }

      model = SelfModel.from_personality(p)

      assert "creative thinking" in model.perceived_strengths
      assert "disorganization" in model.perceived_weaknesses
      assert "anxiety proneness" in model.perceived_weaknesses
    end

    test "derives beliefs from Enneagram" do
      p = %Personality{enneagram_type: :type_5}
      model = SelfModel.from_personality(p)

      assert model.identity_narrative =~ "understand the world"
      assert Enum.any?(model.core_beliefs, fn b -> b.belief =~ "desire To be capable" end)
    end

    test "calculates esteem and efficacy" do
      # High N lowers esteem, High C increases efficacy
      p = %Personality{neuroticism: 0.9, conscientiousness: 0.9}
      model = SelfModel.from_personality(p)

      # Esteem: 0.5 + (0.5 - 0.9)*0.3 = 0.5 - 0.12 = 0.38
      assert model.self_esteem < 0.5

      # Efficacy: 0.5 + (0.9 - 0.5)*0.3 = 0.5 + 0.12 = 0.62
      assert model.self_efficacy > 0.5
    end
  end

  describe "integrate_experience/2" do
    test "reinforces identity on success" do
      model = SelfModel.new()
      exp = %{type: :success, intensity: 0.8}

      updated = SelfModel.integrate_experience(model, exp)

      assert updated.self_efficacy > model.self_efficacy
      assert updated.coherence_level > model.coherence_level
    end

    test "reinforces esteem on kindness received" do
      model = SelfModel.new()
      exp = %{type: :kindness_received, intensity: 0.7}

      updated = SelfModel.integrate_experience(model, exp)

      assert updated.self_esteem > model.self_esteem
    end

    test "reduces coherence on contradiction" do
      # Setup a belief "I am competent"
      model = %SelfModel{
        core_beliefs: [%{domain: "self", belief: "I am competent"}],
        coherence_level: 0.8,
        contradictions: []
      }

      # Failure contradicts competence
      exp = %{type: :failure, intensity: 0.8}

      updated = SelfModel.integrate_experience(model, exp)

      assert updated.coherence_level < 0.8
      assert length(updated.contradictions) == 1
      assert length(updated.identity_negotiations) == 1
    end

    test "triggers crisis narrative when coherence drops low" do
      model = %SelfModel{
        core_beliefs: [%{domain: "self", belief: "I am brave"}],
        # Already low
        coherence_level: 0.35,
        # Already has some history
        contradictions: [%{}, %{}],
        identity_narrative: "I am brave and strong"
      }

      # Cowardice contradicts bravery
      exp = %{type: :cowardice, intensity: 0.9}

      updated = SelfModel.integrate_experience(model, exp)

      assert updated.coherence_level < 0.3
      assert updated.identity_narrative =~ "not sure anymore"
    end
  end

  describe "recover_coherence/2" do
    test "increases coherence" do
      model = %SelfModel{coherence_level: 0.5}
      updated = SelfModel.recover_coherence(model)
      assert updated.coherence_level > 0.5
    end
  end

  describe "query helpers" do
    test "low_self_esteem?/1" do
      assert SelfModel.low_self_esteem?(%SelfModel{self_esteem: 0.3})
      refute SelfModel.low_self_esteem?(%SelfModel{self_esteem: 0.5})
    end

    test "confident?/1" do
      assert SelfModel.confident?(%SelfModel{self_efficacy: 0.7})
      refute SelfModel.confident?(%SelfModel{self_efficacy: 0.5})
    end

    test "identity_crisis?/1" do
      assert SelfModel.identity_crisis?(%SelfModel{coherence_level: 0.3})
      refute SelfModel.identity_crisis?(%SelfModel{coherence_level: 0.5})
    end

    test "describe_coherence/1" do
      model_crisis = %SelfModel{coherence_level: 0.2, contradictions: []}
      assert SelfModel.describe_coherence(model_crisis) =~ "don't recognize myself"

      model_clear = %SelfModel{coherence_level: 0.9}
      assert SelfModel.describe_coherence(model_clear) =~ "clear sense"
    end
  end

  describe "get_beliefs_by_domain/2" do
    test "filters beliefs" do
      model = %SelfModel{
        core_beliefs: [
          %{domain: "self", belief: "A"},
          %{domain: "world", belief: "B"}
        ]
      }

      results = SelfModel.get_beliefs_by_domain(model, "self")
      assert length(results) == 1
      assert hd(results).belief == "A"
    end
  end
end
