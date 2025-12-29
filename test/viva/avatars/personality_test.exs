defmodule Viva.Avatars.PersonalityTest do
  use Viva.DataCase, async: true

  alias Viva.Avatars.Personality

  describe "changeset/2" do
    test "validates numeric ranges 0.0-1.0" do
      params = %{
        openness: 1.1,
        conscientiousness: -0.1,
        extraversion: 0.5,
        agreeableness: 0.5,
        neuroticism: 0.5
      }

      changeset = Personality.changeset(%Personality{}, params)
      refute changeset.valid?
      assert %{openness: ["must be less than or equal to 1.0"]} = errors_on(changeset)
      assert %{conscientiousness: ["must be greater than or equal to 0.0"]} = errors_on(changeset)
    end

    test "accepts valid attributes" do
      params = %{
        openness: 0.8,
        conscientiousness: 0.2,
        extraversion: 0.5,
        agreeableness: 0.9,
        neuroticism: 0.1,
        enneagram_type: :type_5,
        humor_style: :sarcastic,
        native_language: "en-US"
      }

      changeset = Personality.changeset(%Personality{}, params)
      assert changeset.valid?
    end
  end

  describe "temperament/1" do
    test "identifies sanguine (high E, low N)" do
      p = %Personality{extraversion: 0.6, neuroticism: 0.4}
      assert Personality.temperament(p) == :sanguine
    end

    test "identifies choleric (high E, high N)" do
      p = %Personality{extraversion: 0.6, neuroticism: 0.5}
      assert Personality.temperament(p) == :choleric

      p2 = %Personality{extraversion: 0.6, neuroticism: 0.8}
      assert Personality.temperament(p2) == :choleric
    end

    test "identifies phlegmatic (low E, low N)" do
      p = %Personality{extraversion: 0.5, neuroticism: 0.4}
      assert Personality.temperament(p) == :phlegmatic

      p2 = %Personality{extraversion: 0.2, neuroticism: 0.4}
      assert Personality.temperament(p2) == :phlegmatic
    end

    test "identifies melancholic (low E, high N)" do
      p = %Personality{extraversion: 0.5, neuroticism: 0.5}
      assert Personality.temperament(p) == :melancholic

      p2 = %Personality{extraversion: 0.2, neuroticism: 0.8}
      assert Personality.temperament(p2) == :melancholic
    end
  end

  describe "describe_temperament/1" do
    test "returns string descriptions" do
      assert Personality.describe_temperament(:sanguine) =~ "optimistic"
      assert Personality.describe_temperament(:choleric) =~ "intense"
      assert Personality.describe_temperament(:phlegmatic) =~ "calm"
      assert Personality.describe_temperament(:melancholic) =~ "introspective"
    end
  end

  describe "language_name/1" do
    test "returns human readable names" do
      assert Personality.language_name("pt-BR") == "Portuguese (Brazilian)"
      assert Personality.language_name("en-US") == "English (American)"
    end

    test "returns code for unknown language" do
      assert Personality.language_name("xx-YY") == "xx-YY"
    end
  end

  describe "random/0" do
    test "generates valid personality struct" do
      p = Personality.random()
      assert %Personality{} = p

      assert p.openness >= 0.0 and p.openness <= 1.0
      # Logic is * 0.7 but still <= 1.0
      assert p.neuroticism >= 0.0 and p.neuroticism <= 1.0

      assert p.enneagram_type in [
               :type_1,
               :type_2,
               :type_3,
               :type_4,
               :type_5,
               :type_6,
               :type_7,
               :type_8,
               :type_9
             ]

      assert p.humor_style in [:witty, :sarcastic, :wholesome, :dark, :absurd]
      assert p.attachment_style in [:secure, :anxious, :avoidant, :fearful]
    end
  end

  describe "enneagram_data/1" do
    test "returns enneagram details" do
      p = %Personality{enneagram_type: :type_5}
      # Assuming Enneagram module works, we just check if it returns a struct/map with data
      data = Personality.enneagram_data(p)
      assert data.number == 5
      assert data.name == "The Investigator"
    end
  end
end
