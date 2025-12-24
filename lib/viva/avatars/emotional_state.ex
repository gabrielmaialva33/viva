defmodule Viva.Avatars.EmotionalState do
  @moduledoc """
  Embedded schema for the avatar's emotional state based on the PAD model.
  Pleasure (Valence), Arousal (Activation), Dominance (Power).
  Values range from -1.0 to 1.0.
  """
  use Ecto.Schema
  import Ecto.Changeset

  @primary_key false
  embedded_schema do
    # -1.0 (Unhappy) to 1.0 (Happy)
    field :pleasure, :float, default: 0.2
    # -1.0 (Calm) to 1.0 (Excited)
    field :arousal, :float, default: 0.0
    # -1.0 (Submissive) to 1.0 (Influential)
    field :dominance, :float, default: 0.1

    # High-level label calculated from PAD
    field :mood_label, :string, default: "neutral"
  end

  def changeset(state, attrs) do
    state
    |> cast(attrs, [:pleasure, :arousal, :dominance, :mood_label])
    |> validate_number(:pleasure, greater_than_or_equal_to: -1.0, less_than_or_equal_to: 1.0)
    |> validate_number(:arousal, greater_than_or_equal_to: -1.0, less_than_or_equal_to: 1.0)
    |> validate_number(:dominance, greater_than_or_equal_to: -1.0, less_than_or_equal_to: 1.0)
  end
end
