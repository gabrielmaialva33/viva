defmodule Viva.AI.LLM.TtsClientTest do
  use ExUnit.Case, async: true

  alias Viva.AI.LLM.TtsClient
  alias Viva.Avatars.Avatar
  alias Viva.Avatars.Personality

  defp build_personality(opts \\ []) do
    %Personality{
      openness: Keyword.get(opts, :openness, 0.5),
      conscientiousness: Keyword.get(opts, :conscientiousness, 0.5),
      extraversion: Keyword.get(opts, :extraversion, 0.5),
      agreeableness: Keyword.get(opts, :agreeableness, 0.5),
      neuroticism: Keyword.get(opts, :neuroticism, 0.5),
      native_language: Keyword.get(opts, :native_language, "pt-BR"),
      humor_style: "witty",
      attachment_style: "secure",
      enneagram_type: 5
    }
  end

  defp build_avatar(opts \\ []) do
    %Avatar{
      id: Ecto.UUID.generate(),
      name: Keyword.get(opts, :name, "Test Avatar"),
      gender: Keyword.get(opts, :gender, :female),
      age: Keyword.get(opts, :age, 25),
      personality: Keyword.get(opts, :personality, build_personality(opts))
    }
  end

  describe "voice_for_avatar/1" do
    test "returns voice settings map with required keys" do
      avatar = build_avatar()
      settings = TtsClient.voice_for_avatar(avatar)

      assert is_map(settings)
      assert Map.has_key?(settings, :voice_id)
      assert Map.has_key?(settings, :speed)
      assert Map.has_key?(settings, :pitch)
      assert Map.has_key?(settings, :language)
    end

    test "uses avatar's native language" do
      avatar = build_avatar(native_language: "en-US")
      settings = TtsClient.voice_for_avatar(avatar)

      assert settings.language == "en-US"
    end

    test "returns energetic voice for high extraversion female" do
      avatar = build_avatar(gender: :female, extraversion: 0.9)
      settings = TtsClient.voice_for_avatar(avatar)

      assert settings.voice_id =~ "energetic"
      assert settings.voice_id =~ "female"
    end

    test "returns energetic voice for high extraversion male" do
      avatar = build_avatar(gender: :male, extraversion: 0.9)
      settings = TtsClient.voice_for_avatar(avatar)

      assert settings.voice_id =~ "energetic"
      assert settings.voice_id =~ "male"
    end

    test "returns soft voice for high neuroticism female" do
      avatar = build_avatar(gender: :female, extraversion: 0.3, neuroticism: 0.8)
      settings = TtsClient.voice_for_avatar(avatar)

      assert settings.voice_id =~ "soft"
      assert settings.voice_id =~ "female"
    end

    test "returns soft voice for high neuroticism male" do
      avatar = build_avatar(gender: :male, extraversion: 0.3, neuroticism: 0.8)
      settings = TtsClient.voice_for_avatar(avatar)

      assert settings.voice_id =~ "soft"
      assert settings.voice_id =~ "male"
    end

    test "returns warm voice for high agreeableness female" do
      avatar =
        build_avatar(gender: :female, extraversion: 0.3, neuroticism: 0.3, agreeableness: 0.9)

      settings = TtsClient.voice_for_avatar(avatar)

      assert settings.voice_id =~ "warm"
      assert settings.voice_id =~ "female"
    end

    test "returns warm voice for high agreeableness male" do
      avatar = build_avatar(gender: :male, extraversion: 0.3, neuroticism: 0.3, agreeableness: 0.9)
      settings = TtsClient.voice_for_avatar(avatar)

      assert settings.voice_id =~ "warm"
      assert settings.voice_id =~ "male"
    end

    test "returns neutral voice for non-binary gender" do
      avatar = build_avatar(gender: :non_binary)
      settings = TtsClient.voice_for_avatar(avatar)

      assert settings.voice_id =~ "neutral"
    end

    test "returns default female voice when no strong traits" do
      avatar =
        build_avatar(
          gender: :female,
          extraversion: 0.5,
          neuroticism: 0.3,
          agreeableness: 0.5
        )

      settings = TtsClient.voice_for_avatar(avatar)

      assert settings.voice_id == "magpie-pt-br-female-1"
    end

    test "returns default male voice when no strong traits" do
      avatar =
        build_avatar(
          gender: :male,
          extraversion: 0.5,
          neuroticism: 0.3,
          agreeableness: 0.5
        )

      settings = TtsClient.voice_for_avatar(avatar)

      assert settings.voice_id == "magpie-pt-br-male-1"
    end

    test "calculates speed based on extraversion and conscientiousness" do
      # High extraversion = faster
      avatar_extraverted = build_avatar(extraversion: 1.0, conscientiousness: 0.5)
      settings_extraverted = TtsClient.voice_for_avatar(avatar_extraverted)

      # Low extraversion = slower
      avatar_introverted = build_avatar(extraversion: 0.0, conscientiousness: 0.5)
      settings_introverted = TtsClient.voice_for_avatar(avatar_introverted)

      assert settings_extraverted.speed > settings_introverted.speed
    end

    test "high conscientiousness reduces speed" do
      avatar_high_c = build_avatar(extraversion: 0.5, conscientiousness: 1.0)
      settings_high_c = TtsClient.voice_for_avatar(avatar_high_c)

      avatar_low_c = build_avatar(extraversion: 0.5, conscientiousness: 0.0)
      settings_low_c = TtsClient.voice_for_avatar(avatar_low_c)

      assert settings_high_c.speed < settings_low_c.speed
    end

    test "speed is clamped between 0.8 and 1.3" do
      # Extreme high
      avatar_fast = build_avatar(extraversion: 1.0, conscientiousness: 0.0)
      settings_fast = TtsClient.voice_for_avatar(avatar_fast)
      assert settings_fast.speed <= 1.3

      # Extreme low
      avatar_slow = build_avatar(extraversion: 0.0, conscientiousness: 1.0)
      settings_slow = TtsClient.voice_for_avatar(avatar_slow)
      assert settings_slow.speed >= 0.8
    end

    test "pitch is adjusted by neuroticism and gender" do
      # Female has higher base pitch
      avatar_female = build_avatar(gender: :female, neuroticism: 0.5)
      settings_female = TtsClient.voice_for_avatar(avatar_female)

      # Male has lower base pitch
      avatar_male = build_avatar(gender: :male, neuroticism: 0.5)
      settings_male = TtsClient.voice_for_avatar(avatar_male)

      assert settings_female.pitch > settings_male.pitch
    end

    test "high neuroticism increases pitch" do
      avatar_high_n = build_avatar(gender: :female, neuroticism: 1.0)
      settings_high_n = TtsClient.voice_for_avatar(avatar_high_n)

      avatar_low_n = build_avatar(gender: :female, neuroticism: 0.0)
      settings_low_n = TtsClient.voice_for_avatar(avatar_low_n)

      assert settings_high_n.pitch > settings_low_n.pitch
    end

    test "pitch is clamped between -0.3 and 0.3" do
      # Extreme high neuroticism
      avatar_high = build_avatar(gender: :female, neuroticism: 1.0)
      settings_high = TtsClient.voice_for_avatar(avatar_high)
      assert settings_high.pitch <= 0.3

      # Extreme low neuroticism
      avatar_low = build_avatar(gender: :male, neuroticism: 0.0)
      settings_low = TtsClient.voice_for_avatar(avatar_low)
      assert settings_low.pitch >= -0.3
    end
  end
end
