defmodule Viva.Avatars.InternalState do
  @moduledoc """
  The aggregate state of an avatar, combining biological, emotional and cognitive aspects.
  """
  use Ecto.Schema
  import Ecto.Changeset

  alias Viva.Avatars.BioState
  alias Viva.Avatars.EmotionalState

  @primary_key false
  embedded_schema do
    # Layers of the Synthetic Soul
    embeds_one :bio, BioState, on_replace: :update
    embeds_one :emotional, EmotionalState, on_replace: :update

    # Cognitive / Activity state
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

    field :current_activity, Ecto.Enum,
      values: [:idle, :resting, :thinking, :talking, :waiting, :excited, :sleeping],
      default: :idle

    field :interacting_with, :binary_id

    field :updated_at, :utc_datetime
  end

  @type t :: %__MODULE__{}

  @spec changeset(t(), map()) :: Ecto.Changeset.t()
  def changeset(state, attrs) do
    state
    |> cast(attrs, [
      :current_thought,
      :current_desire,
      :current_activity,
      :interacting_with,
      :updated_at
    ])
    |> cast_embed(:bio)
    |> cast_embed(:emotional)
  end

  @doc """
  Returns a label for the current dominant emotion.
  """
  @spec dominant_emotion(t()) :: String.t()
  def dominant_emotion(%__MODULE__{emotional: emotional}) do
    emotional.mood_label
  end

  @doc """
  Calculates overall wellbeing (0.0 to 1.0).
  A mix of high Pleasure, low Cortisol, and low Adenosine.
  """
  @spec wellbeing(t()) :: float()
  def wellbeing(%__MODULE__{bio: bio, emotional: emotional}) do
    # Pleasure is -1 to 1, we normalize to 0 to 1
    pleasure_score = (emotional.pleasure + 1.0) / 2.0

    # Stress and Fatigue are 0 to 1, we invert them
    stress_penalty = bio.cortisol
    fatigue_penalty = bio.adenosine

    (pleasure_score * 0.6 + (1.0 - stress_penalty) * 0.2 + (1.0 - fatigue_penalty) * 0.2)
    |> max(0.0)
    |> min(1.0)
  end

  @spec new() :: t()
  def new do
    %__MODULE__{
      bio: %BioState{},
      emotional: %EmotionalState{},
      updated_at: DateTime.utc_now(:second)
    }
  end
end
