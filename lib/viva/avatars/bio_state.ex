defmodule Viva.Avatars.BioState do
  @moduledoc """
  Embedded schema for the avatar's biological/hormonal state.
  Values typically range from 0.0 to 1.0.
  """
  use Ecto.Schema
  import Ecto.Changeset

  @primary_key false
  embedded_schema do
    # Reward/Focus
    field :dopamine, :float, default: 0.5
    # Stress/Survival
    field :cortisol, :float, default: 0.2
    # Bonding/Trust
    field :oxytocin, :float, default: 0.3
    # Sleep pressure
    field :adenosine, :float, default: 0.0
    # Drive/Attraction
    field :libido, :float, default: 0.4

    # Circadian config
    field :chronotype, Ecto.Enum, values: [:lark, :owl, :hummingbird], default: :hummingbird
    field :sleep_start_hour, :integer, default: 23
    field :wake_start_hour, :integer, default: 7
  end

  @type t :: %__MODULE__{}

  @spec changeset(t(), map()) :: Ecto.Changeset.t()
  def changeset(state, attrs) do
    state
    |> cast(attrs, [
      :dopamine,
      :cortisol,
      :oxytocin,
      :adenosine,
      :libido,
      :chronotype,
      :sleep_start_hour,
      :wake_start_hour
    ])
    |> validate_number(:dopamine, greater_than_or_equal_to: 0.0, less_than_or_equal_to: 1.0)
    |> validate_number(:cortisol, greater_than_or_equal_to: 0.0, less_than_or_equal_to: 1.0)

    # etc...
  end
end
