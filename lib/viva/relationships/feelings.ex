defmodule Viva.Relationships.Feelings do
  @moduledoc "Individual avatar's feelings about the relationship"
  use Ecto.Schema

  @type t :: %__MODULE__{}

  @primary_key false
  embedded_schema do
    field :thinks_about_often, :boolean, default: false
    field :feels_understood, :float, default: 0.5
    field :wants_more_time, :boolean, default: false
    field :romantic_interest, :float, default: 0.0
    field :jealousy, :float, default: 0.0
    field :admiration, :float, default: 0.0
    field :comfort_level, :float, default: 0.5
    field :excitement_to_see, :float, default: 0.3

    # Social Complexity Fields - NEW
    # How much I trust them (subjective)
    field :perceived_trust, :float, default: 0.5
    # How much I am faking in this relationship (0.0 to 1.0)
    field :active_mask_intensity, :float, default: 0.0
    # How I see their status relative to mine (-1.0 lower, 1.0 higher)
    field :perceived_status_diff, :float, default: 0.0
  end

  @spec changeset(t(), map()) :: Ecto.Changeset.t()
  def changeset(feelings, attrs) do
    Ecto.Changeset.cast(feelings, attrs, [
      :thinks_about_often,
      :feels_understood,
      :wants_more_time,
      :romantic_interest,
      :jealousy,
      :admiration,
      :comfort_level,
      :excitement_to_see,
      :perceived_trust,
      :active_mask_intensity,
      :perceived_status_diff
    ])
  end
end
