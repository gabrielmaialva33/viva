defmodule Viva.Avatars.BioState do
  @moduledoc """
  Embedded schema for the avatar's biological/hormonal state.
  Values typically range from 0.0 to 1.0.
  """
  use Ecto.Schema
  import Ecto.Changeset

  @primary_key false
  embedded_schema do
    # Reward/Focus - lowered to create mild boredom (need for stimulation)
    field :dopamine, :float, default: 0.35
    # Stress/Survival - slightly elevated for baseline alertness
    field :cortisol, :float, default: 0.25
    # Bonding/Trust - lowered to create mild loneliness (need for connection)
    field :oxytocin, :float, default: 0.25
    # Sleep pressure - slight fatigue to motivate rest-seeking
    field :adenosine, :float, default: 0.1
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
