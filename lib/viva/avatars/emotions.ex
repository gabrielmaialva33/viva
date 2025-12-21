defmodule Viva.Avatars.Emotions do
  @moduledoc "Embedded schema for emotion values."
  use Ecto.Schema

  @type t :: %__MODULE__{}

  @primary_key false
  embedded_schema do
    field :joy, :float, default: 0.5
    field :sadness, :float, default: 0.0
    field :anger, :float, default: 0.0
    field :fear, :float, default: 0.0
    field :surprise, :float, default: 0.0
    field :disgust, :float, default: 0.0
    field :love, :float, default: 0.0
    field :loneliness, :float, default: 0.3
    field :curiosity, :float, default: 0.4
    field :excitement, :float, default: 0.2
  end

  @spec changeset(t(), map()) :: Ecto.Changeset.t()
  def changeset(emotions, attrs) do
    Ecto.Changeset.cast(emotions, attrs, [
      :joy,
      :sadness,
      :anger,
      :fear,
      :surprise,
      :disgust,
      :love,
      :loneliness,
      :curiosity,
      :excitement
    ])
  end
end
