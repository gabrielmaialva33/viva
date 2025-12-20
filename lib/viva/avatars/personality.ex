defmodule Viva.Avatars.Personality do
  @moduledoc """
  Embedded schema for avatar personality traits.
  Based on Big Five personality model + additional traits.
  """
  use Ecto.Schema
  import Ecto.Changeset

  @primary_key false
  embedded_schema do
    # Big Five personality traits (0.0 to 1.0)
    field :openness, :float, default: 0.5
    field :conscientiousness, :float, default: 0.5
    field :extraversion, :float, default: 0.5
    field :agreeableness, :float, default: 0.5
    field :neuroticism, :float, default: 0.3

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
  end

  def changeset(personality, attrs) do
    personality
    |> cast(attrs, [
      :openness,
      :conscientiousness,
      :extraversion,
      :agreeableness,
      :neuroticism,
      :humor_style,
      :love_language,
      :attachment_style,
      :interests,
      :values,
      :voice_pitch,
      :voice_speed,
      :voice_warmth
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
      humor_style: Enum.random([:witty, :sarcastic, :wholesome, :dark, :absurd]),
      love_language: Enum.random([:words, :time, :gifts, :touch, :service]),
      attachment_style:
        weighted_random([
          {:secure, 0.5},
          {:anxious, 0.2},
          {:avoidant, 0.2},
          {:fearful, 0.1}
        ])
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
