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
      :excitement_to_see
    ])
  end
end
