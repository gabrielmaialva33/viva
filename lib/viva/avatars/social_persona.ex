defmodule Viva.Avatars.SocialPersona do
  @moduledoc """
  Schema to track an avatar's public face vs private reality.
  Part of the Social Complexity Engine.
  """
  use Ecto.Schema
  import Ecto.Changeset

  @primary_key false
  embedded_schema do
    # Public Reputation (Calculated by Graph Theory Engine)
    # 0.0 (Pariah) -> 1.0 (Celebrity)
    field :public_reputation, :float, default: 0.1

    # Current Mask Intensity
    # 0.0 (Authentic) -> 1.0 (Total Imposter)
    # High mask = high stress (cortisol)
    field :current_mask_intensity, :float, default: 0.0

    # How society perceives this avatar (may be wrong!)
    # E.g., a "Introvert" might be perceived as "Arrogant" if mask is bad.
    field :perceived_traits, {:array, :string}, default: []

    # Social Ambition
    # How much they care about potential status games
    field :social_ambition, :float, default: 0.5
  end

  @type t :: %__MODULE__{}

  @spec changeset(t(), map()) :: Ecto.Changeset.t()
  def changeset(persona, attrs) do
    persona
    |> cast(attrs, [
      :public_reputation,
      :current_mask_intensity,
      :perceived_traits,
      :social_ambition
    ])
    |> validate_number(:public_reputation,
      greater_than_or_equal_to: 0.0,
      less_than_or_equal_to: 1.0
    )
    |> validate_number(:current_mask_intensity,
      greater_than_or_equal_to: 0.0,
      less_than_or_equal_to: 1.0
    )
    |> validate_number(:social_ambition, greater_than_or_equal_to: 0.0, less_than_or_equal_to: 1.0)
  end

  @spec new() :: t()
  def new do
    %__MODULE__{}
  end
end
