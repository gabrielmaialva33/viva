defmodule Viva.Avatars.InternalState do
  @moduledoc """
  The aggregate state of an avatar, combining biological, emotional,
  sensory, and conscious aspects.

  This is the complete internal state of the avatar's "synthetic soul":
  - Bio: Hormonal/physiological state
  - Emotional: PAD model emotions
  - Sensory: Perception and qualia (subjective experience)
  - Consciousness: Stream of experience and self-model
  """
  use Ecto.Schema
  import Ecto.Changeset

  alias Viva.Avatars.BioState
  alias Viva.Avatars.ConsciousnessState
  alias Viva.Avatars.EmotionalState
  alias Viva.Avatars.SensoryState

  @primary_key false
  embedded_schema do
    # === Layers of the Synthetic Soul ===

    # Layer 1: Biological (hormones, physiology)
    embeds_one :bio, BioState, on_replace: :update

    # Layer 2: Emotional (PAD model)
    embeds_one :emotional, EmotionalState, on_replace: :update

    # Layer 3: Sensory (perception, qualia, attention)
    embeds_one :sensory, SensoryState, on_replace: :update

    # Layer 4: Consciousness (experience stream, self-model, metacognition)
    embeds_one :consciousness, ConsciousnessState, on_replace: :update

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
    |> cast_embed(:sensory)
    |> cast_embed(:consciousness)
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
      sensory: SensoryState.new(),
      consciousness: ConsciousnessState.new(),
      updated_at: DateTime.utc_now(:second)
    }
  end

  @doc """
  Creates a new internal state with consciousness initialized from personality.
  """
  @spec from_personality(Viva.Avatars.Personality.t()) :: t()
  def from_personality(personality) do
    %__MODULE__{
      bio: %BioState{},
      emotional: %EmotionalState{},
      sensory: SensoryState.new(),
      consciousness: ConsciousnessState.from_personality(personality),
      updated_at: DateTime.utc_now(:second)
    }
  end

  @doc """
  Returns the current qualia narrative (subjective experience description).
  """
  @spec qualia_narrative(t()) :: String.t() | nil
  def qualia_narrative(%__MODULE__{sensory: sensory}) do
    SensoryState.experience_narrative(sensory)
  end

  @doc """
  Returns true if the avatar is currently surprised.
  """
  @spec surprised?(t()) :: boolean()
  def surprised?(%__MODULE__{sensory: sensory}) do
    SensoryState.surprised?(sensory)
  end

  @doc """
  Returns true if the avatar is in a dissociative state.
  """
  @spec dissociated?(t()) :: boolean()
  def dissociated?(%__MODULE__{consciousness: consciousness}) do
    ConsciousnessState.dissociated?(consciousness)
  end

  @doc """
  Returns the current metacognitive observation.
  """
  @spec meta_observation(t()) :: String.t() | nil
  def meta_observation(%__MODULE__{consciousness: consciousness}) do
    consciousness.meta_observation
  end
end
