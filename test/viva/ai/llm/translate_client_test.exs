defmodule Viva.AI.LLM.TranslateClientTest do
  use ExUnit.Case, async: true

  alias Viva.AI.LLM.TranslateClient
  alias Viva.Avatars.Avatar
  alias Viva.Avatars.Personality

  describe "supported?/1" do
    test "returns true for supported language codes" do
      assert TranslateClient.supported?("en") == true
      assert TranslateClient.supported?("pt") == true
      assert TranslateClient.supported?("es") == true
      assert TranslateClient.supported?("de") == true
      assert TranslateClient.supported?("fr") == true
      assert TranslateClient.supported?("ja") == true
      assert TranslateClient.supported?("zh") == true
    end

    test "returns false for unsupported language codes" do
      assert TranslateClient.supported?("xyz") == false
      assert TranslateClient.supported?("qq") == false
      assert TranslateClient.supported?("zzz") == false
    end

    test "normalizes language codes with regions" do
      assert TranslateClient.supported?("pt-BR") == true
      assert TranslateClient.supported?("en-US") == true
      assert TranslateClient.supported?("es-ES") == true
    end

    test "handles atom language codes" do
      assert TranslateClient.supported?(:en) == true
      assert TranslateClient.supported?(:pt) == true
    end
  end

  describe "supported_languages/0" do
    test "returns a list of language codes" do
      languages = TranslateClient.supported_languages()

      assert is_list(languages)
      assert length(languages) > 30
      assert "en" in languages
      assert "pt" in languages
      assert "es" in languages
    end

    test "all codes are strings" do
      languages = TranslateClient.supported_languages()

      Enum.each(languages, fn lang ->
        assert is_binary(lang)
      end)
    end
  end

  describe "language_name/1" do
    test "returns full name for known language codes" do
      assert TranslateClient.language_name("en") == "English"
      assert TranslateClient.language_name("pt") == "Portuguese"
      assert TranslateClient.language_name("es") == "Spanish"
      assert TranslateClient.language_name("de") == "German"
      assert TranslateClient.language_name("fr") == "French"
      assert TranslateClient.language_name("ja") == "Japanese"
      assert TranslateClient.language_name("zh") == "Chinese (Simplified)"
      # zh-TW is normalized to "zh" by normalize_lang, so it returns Simplified
      # The full "zh-TW" code in supported_languages would return Traditional directly
    end

    test "returns code for unknown language" do
      assert TranslateClient.language_name("xyz") == "xyz"
    end

    test "handles atom language codes" do
      assert TranslateClient.language_name(:en) == "English"
      assert TranslateClient.language_name(:pt) == "Portuguese"
    end

    test "normalizes regional codes" do
      assert TranslateClient.language_name("pt-BR") == "Portuguese"
      assert TranslateClient.language_name("en-US") == "English"
    end
  end

  describe "translate/4 same language" do
    test "returns original text when languages are the same" do
      text = "Hello world"

      assert {:ok, ^text} = TranslateClient.translate(text, "en", "en")
    end

    test "normalizes and compares regional codes" do
      text = "Hello world"

      assert {:ok, ^text} = TranslateClient.translate(text, "en-US", "en-GB")
    end

    test "handles atom language codes" do
      text = "Olá mundo"

      assert {:ok, ^text} = TranslateClient.translate(text, :pt, :pt)
    end
  end

  describe "translate_batch/4 same language" do
    test "returns original texts when languages are the same" do
      texts = ["Hello", "World", "Test"]

      assert {:ok, ^texts} = TranslateClient.translate_batch(texts, "en", "en")
    end
  end

  describe "common_language/2" do
    defp build_avatar(native_language, other_languages \\ []) do
      personality = %Personality{
        openness: 0.5,
        conscientiousness: 0.5,
        extraversion: 0.5,
        agreeableness: 0.5,
        neuroticism: 0.5,
        native_language: native_language,
        other_languages: other_languages,
        humor_style: "witty",
        attachment_style: "secure",
        enneagram_type: 5
      }

      %Avatar{
        id: Ecto.UUID.generate(),
        name: "Test",
        gender: :female,
        age: 25,
        personality: personality
      }
    end

    test "returns common language when both speak the same native" do
      avatar_a = build_avatar("pt-BR")
      avatar_b = build_avatar("pt-BR")

      assert {:ok, "pt"} = TranslateClient.common_language(avatar_a, avatar_b)
    end

    test "returns native of first when second speaks it" do
      avatar_a = build_avatar("en-US")
      avatar_b = build_avatar("pt-BR", ["en-US"])

      assert {:ok, "en"} = TranslateClient.common_language(avatar_a, avatar_b)
    end

    test "returns native of second when first speaks it" do
      avatar_a = build_avatar("pt-BR", ["en-US"])
      avatar_b = build_avatar("en-US")

      assert {:ok, "en"} = TranslateClient.common_language(avatar_a, avatar_b)
    end

    test "returns shared non-native language" do
      avatar_a = build_avatar("pt-BR", ["es-ES"])
      avatar_b = build_avatar("fr-FR", ["es-ES"])

      assert {:ok, "es"} = TranslateClient.common_language(avatar_a, avatar_b)
    end

    test "returns none when no common language" do
      avatar_a = build_avatar("pt-BR")
      avatar_b = build_avatar("ja-JP")

      assert {:none, {"pt-BR", "ja-JP"}} = TranslateClient.common_language(avatar_a, avatar_b)
    end
  end

  describe "module structure" do
    test "exports expected functions" do
      functions = TranslateClient.__info__(:functions)

      assert {:translate, 3} in functions or {:translate, 4} in functions
      assert {:translate_batch, 3} in functions or {:translate_batch, 4} in functions
      assert {:detect_language, 1} in functions

      assert {:translate_avatar_message, 3} in functions or
               {:translate_avatar_message, 4} in functions

      assert {:translate_conversation, 2} in functions or {:translate_conversation, 3} in functions
      assert {:supported?, 1} in functions
      assert {:supported_languages, 0} in functions
      assert {:language_name, 1} in functions
      assert {:common_language, 2} in functions
    end

    test "implements Pipeline.Stage behaviour" do
      behaviours = TranslateClient.__info__(:attributes)[:behaviour] || []
      assert Viva.AI.Pipeline.Stage in behaviours
    end
  end
end
