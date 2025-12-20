defmodule Viva.Avatars.Personality do
  @moduledoc """
  Embedded schema for avatar personality traits.
  Based on Big Five personality model + Enneagram + derived Temperament.
  """
  use Ecto.Schema
  import Ecto.Changeset

  alias Viva.Avatars.Enneagram

  @primary_key false
  embedded_schema do
    # Big Five personality traits (0.0 to 1.0)
    field :openness, :float, default: 0.5
    field :conscientiousness, :float, default: 0.5
    field :extraversion, :float, default: 0.5
    field :agreeableness, :float, default: 0.5
    field :neuroticism, :float, default: 0.3

    # Enneagram type (1-9) - deep motivations and growth path
    field :enneagram_type, Ecto.Enum,
      values: [:type_1, :type_2, :type_3, :type_4, :type_5, :type_6, :type_7, :type_8, :type_9],
      default: :type_9

    # Communication style
    field :humor_style, Ecto.Enum,
      values: [:witty, :sarcastic, :wholesome, :dark, :absurd],
      default: :witty

    # Love language preference
    field :love_language, Ecto.Enum,
      values: [:words, :time, :gifts, :touch, :service],
      default: :words

    # Attachment style (affects relationships)
    field :attachment_style, Ecto.Enum,
      values: [:secure, :anxious, :avoidant, :fearful],
      default: :secure

    # Interests (affects compatibility)
    field :interests, {:array, :string}, default: []

    # Core values
    field :values, {:array, :string}, default: []

    # Voice characteristics for TTS
    field :voice_pitch, :float, default: 0.5
    field :voice_speed, :float, default: 0.5
    field :voice_warmth, :float, default: 0.5

    # Language settings
    field :native_language, :string, default: "pt-BR"
    field :other_languages, {:array, :string}, default: []
  end

  @language_names %{
    "pt-BR" => "Portuguese (Brazilian)",
    "pt-PT" => "Portuguese (European)",
    "en-US" => "English (American)",
    "en-GB" => "English (British)",
    "es-ES" => "Spanish (Spain)",
    "es-MX" => "Spanish (Mexican)",
    "es-AR" => "Spanish (Argentine)",
    "fr-FR" => "French",
    "de-DE" => "German",
    "it-IT" => "Italian",
    "ja-JP" => "Japanese",
    "ko-KR" => "Korean",
    "zh-CN" => "Chinese (Simplified)",
    "ru-RU" => "Russian"
  }

  @doc "Get human-readable language name"
  def language_name(code) do
    Map.get(@language_names, code, code)
  end

  @doc """
  Derives the classical temperament from Big Five traits.
  Based on Extraversion and Neuroticism dimensions.

  Returns one of: :sanguine, :choleric, :phlegmatic, :melancholic
  """
  def temperament(%__MODULE__{} = p) do
    cond do
      p.extraversion > 0.5 and p.neuroticism < 0.5 -> :sanguine
      p.extraversion > 0.5 and p.neuroticism >= 0.5 -> :choleric
      p.extraversion <= 0.5 and p.neuroticism < 0.5 -> :phlegmatic
      true -> :melancholic
    end
  end

  @doc "Get the Enneagram type data for this personality"
  def enneagram_data(%__MODULE__{enneagram_type: type}) do
    Enneagram.get_type(type)
  end

  @doc "Describe the temperament in human-readable terms"
  def describe_temperament(:sanguine), do: "optimistic, sociable, enthusiastic, and expressive"
  def describe_temperament(:choleric), do: "intense, passionate, driven, and assertive"
  def describe_temperament(:phlegmatic), do: "calm, patient, reliable, and thoughtful"
  def describe_temperament(:melancholic), do: "introspective, analytical, sensitive, and deep"

  def changeset(personality, attrs) do
    personality
    |> cast(attrs, [
      :openness,
      :conscientiousness,
      :extraversion,
      :agreeableness,
      :neuroticism,
      :enneagram_type,
      :humor_style,
      :love_language,
      :attachment_style,
      :interests,
      :values,
      :voice_pitch,
      :voice_speed,
      :voice_warmth,
      :native_language,
      :other_languages
    ])
    |> validate_number(:openness, greater_than_or_equal_to: 0.0, less_than_or_equal_to: 1.0)
    |> validate_number(:conscientiousness,
      greater_than_or_equal_to: 0.0,
      less_than_or_equal_to: 1.0
    )
    |> validate_number(:extraversion, greater_than_or_equal_to: 0.0, less_than_or_equal_to: 1.0)
    |> validate_number(:agreeableness, greater_than_or_equal_to: 0.0, less_than_or_equal_to: 1.0)
    |> validate_number(:neuroticism, greater_than_or_equal_to: 0.0, less_than_or_equal_to: 1.0)
  end

  @doc "Generate a random personality"
  def random do
    %__MODULE__{
      openness: :rand.uniform(),
      conscientiousness: :rand.uniform(),
      extraversion: :rand.uniform(),
      agreeableness: :rand.uniform(),
      neuroticism: :rand.uniform() * 0.7,
      enneagram_type:
        Enum.random([
          :type_1,
          :type_2,
          :type_3,
          :type_4,
          :type_5,
          :type_6,
          :type_7,
          :type_8,
          :type_9
        ]),
      humor_style: Enum.random([:witty, :sarcastic, :wholesome, :dark, :absurd]),
      love_language: Enum.random([:words, :time, :gifts, :touch, :service]),
      attachment_style:
        weighted_random([
          {:secure, 0.5},
          {:anxious, 0.2},
          {:avoidant, 0.2},
          {:fearful, 0.1}
        ]),
      native_language: "pt-BR",
      other_languages: []
    }
  end

  defp weighted_random(options) do
    total = Enum.reduce(options, 0, fn {_, weight}, acc -> acc + weight end)
    random = :rand.uniform() * total

    Enum.reduce_while(options, 0, fn {value, weight}, acc ->
      new_acc = acc + weight
      if random <= new_acc, do: {:halt, value}, else: {:cont, new_acc}
    end)
  end
end
