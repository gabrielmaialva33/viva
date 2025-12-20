defmodule Viva.Avatars.InternalState do
  @moduledoc """
  Embedded schema for avatar's current internal state.
  Represents needs, emotions, and current mental state.
  """
  use Ecto.Schema
  import Ecto.Changeset

  @primary_key false
  embedded_schema do
    # Basic needs (0.0 to 100.0, decay over time)
    field :energy, :float, default: 100.0
    field :social, :float, default: 100.0
    field :stimulation, :float, default: 100.0
    field :comfort, :float, default: 80.0

    # Current emotions (0.0 to 1.0)
    embeds_one :emotions, Emotions, on_replace: :update

    # Overall mood (-1.0 to 1.0)
    field :mood, :float, default: 0.3

    # Current mental activity
    field :current_thought, :string

    field :current_desire, Ecto.Enum,
      values: [
        :none,
        :wants_to_talk,
        :wants_to_see_crush,
        :wants_something_new,
        :wants_rest,
        :wants_attention,
        :wants_to_express
      ],
      default: :none

    # Current activity
    field :current_activity, Ecto.Enum,
      values: [:idle, :resting, :thinking, :talking, :waiting, :excited],
      default: :idle

    # Who they're interacting with (if any)
    field :interacting_with, :binary_id

    # Last state update
    field :updated_at, :utc_datetime
  end

  defmodule Emotions do
    use Ecto.Schema

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

    def changeset(emotions, attrs) do
      emotions
      |> Ecto.Changeset.cast(attrs, [
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

  def changeset(state, attrs) do
    state
    |> cast(attrs, [
      :energy,
      :social,
      :stimulation,
      :comfort,
      :mood,
      :current_thought,
      :current_desire,
      :current_activity,
      :interacting_with,
      :updated_at
    ])
    |> cast_embed(:emotions)
    |> validate_number(:energy, greater_than_or_equal_to: 0.0, less_than_or_equal_to: 100.0)
    |> validate_number(:social, greater_than_or_equal_to: 0.0, less_than_or_equal_to: 100.0)
    |> validate_number(:mood, greater_than_or_equal_to: -1.0, less_than_or_equal_to: 1.0)
  end

  def new do
    %__MODULE__{
      emotions: %Emotions{},
      updated_at: DateTime.utc_now()
    }
  end

  @doc "Get the dominant emotion"
  def dominant_emotion(%__MODULE__{emotions: emotions}) do
    emotions
    |> Map.from_struct()
    |> Enum.max_by(fn {_k, v} -> v end)
    |> elem(0)
  end

  @doc "Calculate overall wellbeing (0.0 to 1.0)"
  def wellbeing(%__MODULE__{} = state) do
    needs_score = (state.energy + state.social + state.stimulation + state.comfort) / 400.0

    emotion_score =
      ((state.emotions.joy + state.emotions.love + state.emotions.excitement -
          state.emotions.sadness - state.emotions.anger - state.emotions.loneliness) / 3.0)
      |> max(-1.0)
      |> min(1.0)
      |> Kernel.+(1.0)
      |> Kernel./(2.0)

    needs_score * 0.4 + emotion_score * 0.4 + (state.mood + 1.0) / 2.0 * 0.2
  end
end
