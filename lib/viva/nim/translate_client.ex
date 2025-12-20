defmodule Viva.Nim.TranslateClient do
  @moduledoc """
  Translation client for global avatar communication.

  Uses `nvidia/riva-translate-1.6b` for smooth translation
  across 36 languages.

  ## Features

  - Translate messages between avatars
  - Detect message language
  - Batch translation for memories
  - Preserve context and tone
  """
  require Logger

  alias Viva.Nim

  @supported_languages [
    "ar",
    "bg",
    "cs",
    "da",
    "de",
    "el",
    "en",
    "es",
    "et",
    "fi",
    "fr",
    "hi",
    "hr",
    "hu",
    "id",
    "it",
    "ja",
    "ko",
    "lt",
    "lv",
    "nl",
    "no",
    "pl",
    "pt",
    "ro",
    "ru",
    "sk",
    "sl",
    "sv",
    "th",
    "tr",
    "uk",
    "vi",
    "zh",
    "zh-TW"
  ]

  @language_names %{
    "ar" => "Arabic",
    "bg" => "Bulgarian",
    "cs" => "Czech",
    "da" => "Danish",
    "de" => "German",
    "el" => "Greek",
    "en" => "English",
    "es" => "Spanish",
    "et" => "Estonian",
    "fi" => "Finnish",
    "fr" => "French",
    "hi" => "Hindi",
    "hr" => "Croatian",
    "hu" => "Hungarian",
    "id" => "Indonesian",
    "it" => "Italian",
    "ja" => "Japanese",
    "ko" => "Korean",
    "lt" => "Lithuanian",
    "lv" => "Latvian",
    "nl" => "Dutch",
    "no" => "Norwegian",
    "pl" => "Polish",
    "pt" => "Portuguese",
    "ro" => "Romanian",
    "ru" => "Russian",
    "sk" => "Slovak",
    "sl" => "Slovenian",
    "sv" => "Swedish",
    "th" => "Thai",
    "tr" => "Turkish",
    "uk" => "Ukrainian",
    "vi" => "Vietnamese",
    "zh" => "Chinese (Simplified)",
    "zh-TW" => "Chinese (Traditional)"
  }

  @doc """
  Translate text from one language to another.

  ## Options

  - `:preserve_tone` - Try to preserve emotional tone (default: true)
  - `:formality` - "formal", "informal", "auto" (default: "auto")
  """
  def translate(text, from_lang, to_lang, opts \\ []) do
    model = Keyword.get(opts, :model, Nim.model(:translate))
    preserve_tone = Keyword.get(opts, :preserve_tone, true)
    formality = Keyword.get(opts, :formality, "auto")

    from_lang = normalize_lang(from_lang)
    to_lang = normalize_lang(to_lang)

    if from_lang == to_lang do
      {:ok, text}
    else
      body = %{
        model: model,
        text: text,
        source_language: from_lang,
        target_language: to_lang,
        preserve_tone: preserve_tone,
        formality: formality
      }

      case Nim.request("/translate", body) do
        {:ok, %{"translation" => translation}} ->
          {:ok, translation}

        {:ok, %{"translations" => [%{"text" => translation} | _]}} ->
          {:ok, translation}

        {:error, reason} ->
          Logger.error("Translation error: #{inspect(reason)}")
          {:error, reason}
      end
    end
  end

  @doc """
  Translate multiple texts in batch.
  More efficient than individual calls.
  """
  def translate_batch(texts, from_lang, to_lang, opts \\ []) when is_list(texts) do
    model = Keyword.get(opts, :model, Nim.model(:translate))

    from_lang = normalize_lang(from_lang)
    to_lang = normalize_lang(to_lang)

    if from_lang == to_lang do
      {:ok, texts}
    else
      body = %{
        model: model,
        texts: texts,
        source_language: from_lang,
        target_language: to_lang
      }

      case Nim.request("/translate/batch", body) do
        {:ok, %{"translations" => translations}} ->
          {:ok, Enum.map(translations, & &1["text"])}

        {:error, reason} ->
          {:error, reason}
      end
    end
  end

  @doc """
  Detect the language of a text.
  """
  def detect_language(text) do
    body = %{
      model: Nim.model(:translate),
      text: text,
      task: "detect"
    }

    case Nim.request("/translate/detect", body) do
      {:ok, %{"language" => lang, "confidence" => conf}} ->
        {:ok, %{language: lang, confidence: conf, name: @language_names[lang] || lang}}

      {:error, reason} ->
        {:error, reason}
    end
  end

  @doc """
  Translate a message between two avatars.
  Automatically detects languages from avatar settings.
  """
  def translate_avatar_message(message, from_avatar, to_avatar, opts \\ []) do
    from_lang = from_avatar.personality.native_language
    to_lang = to_avatar.personality.native_language

    translate(message, from_lang, to_lang, opts)
  end

  @doc """
  Translate conversation history for an avatar.
  Preserves speaker attribution.
  """
  def translate_conversation(messages, to_lang, opts \\ []) do
    # Group messages by language to batch translate
    grouped =
      messages
      |> Enum.group_by(fn msg ->
        msg.language || detect_message_language(msg.content)
      end)

    results =
      Enum.flat_map(grouped, fn {from_lang, msgs} ->
        texts = Enum.map(msgs, & &1.content)

        case translate_batch(texts, from_lang, to_lang, opts) do
          {:ok, translations} ->
            Enum.zip(msgs, translations)
            |> Enum.map(fn {msg, translation} ->
              Map.put(msg, :translated_content, translation)
            end)

          {:error, _} ->
            # Keep original on error
            Enum.map(msgs, &Map.put(&1, :translated_content, &1.content))
        end
      end)

    {:ok, Enum.sort_by(results, & &1.inserted_at)}
  end

  @doc """
  Check if a language is supported.
  """
  def supported?(lang) do
    normalize_lang(lang) in @supported_languages
  end

  @doc "List all supported languages"
  def supported_languages, do: @supported_languages

  @doc "Get language name from code"
  def language_name(code) do
    @language_names[normalize_lang(code)] || code
  end

  @doc """
  Get common language between two avatars.
  Returns the language both speak, or nil if none.
  """
  def common_language(avatar_a, avatar_b) do
    a_languages = [avatar_a.personality.native_language | avatar_a.personality.other_languages]
    b_languages = [avatar_b.personality.native_language | avatar_b.personality.other_languages]

    a_set = MapSet.new(Enum.map(a_languages, &normalize_lang/1))
    b_set = MapSet.new(Enum.map(b_languages, &normalize_lang/1))

    common = MapSet.intersection(a_set, b_set)

    if MapSet.size(common) > 0 do
      # Prefer native language of either
      native_a = normalize_lang(avatar_a.personality.native_language)
      native_b = normalize_lang(avatar_b.personality.native_language)

      cond do
        native_a in common -> {:ok, native_a}
        native_b in common -> {:ok, native_b}
        true -> {:ok, Enum.at(MapSet.to_list(common), 0)}
      end
    else
      {:none, {avatar_a.personality.native_language, avatar_b.personality.native_language}}
    end
  end


  defp normalize_lang(lang) when is_binary(lang) do
    # Convert "pt-BR" to "pt", "en-US" to "en", etc.
    lang
    |> String.split("-")
    |> List.first()
    |> String.downcase()
  end

  defp normalize_lang(lang) when is_atom(lang) do
    normalize_lang(Atom.to_string(lang))
  end

  defp detect_message_language(content) do
    case detect_language(content) do
      {:ok, %{language: lang}} -> lang
      _ -> "en"
    end
  end
end
