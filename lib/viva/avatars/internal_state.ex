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

  alias Viva.Avatars.AllostasisState
  alias Viva.Avatars.BioState
  alias Viva.Avatars.ConsciousnessState
  alias Viva.Avatars.EmotionalState
  alias Viva.Avatars.EmotionRegulationState
  alias Viva.Avatars.MotivationState
  alias Viva.Avatars.SensoryState
  alias Viva.Avatars.SomaticMarkersState

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

    # Layer 5: Allostasis (chronic stress tracking)
    embeds_one :allostasis, AllostasisState, on_replace: :update

    # Layer 6: Emotion Regulation (coping strategies)
    embeds_one :regulation, EmotionRegulationState, on_replace: :update

    # Layer 7: Somatic Markers (body memory influencing decisions)
    embeds_one :somatic, SomaticMarkersState, on_replace: :update

    # Layer 8: Motivation (hierarchical drives system)
    embeds_one :motivation, MotivationState, on_replace: :update

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
    |> cast_embed(:allostasis)
    |> cast_embed(:regulation)
    |> cast_embed(:somatic)
    |> cast_embed(:motivation)
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
      allostasis: AllostasisState.new(),
      regulation: EmotionRegulationState.new(),
      somatic: SomaticMarkersState.new(),
      motivation: MotivationState.new(),
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
      allostasis: AllostasisState.new(),
      regulation: EmotionRegulationState.new(),
      somatic: SomaticMarkersState.new(),
      motivation: MotivationState.from_enneagram(personality.enneagram_type),
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

  @doc """
  Ensures all internal state components are properly initialized.
  Handles nil values that may exist in persisted states.
  """
  @spec ensure_integrity(t() | nil, Viva.Avatars.Personality.t()) :: t()
  def ensure_integrity(nil, personality), do: from_personality(personality)

  def ensure_integrity(state, personality) do
    state
    |> ensure_bio()
    |> ensure_emotional()
    |> ensure_sensory()
    |> ensure_consciousness(personality)
    |> ensure_allostasis()
    |> ensure_regulation()
    |> ensure_somatic()
    |> ensure_motivation(personality)
  end

  defp ensure_bio(%{bio: nil} = state), do: %{state | bio: %BioState{}}
  defp ensure_bio(state), do: state

  defp ensure_emotional(%{emotional: nil} = state), do: %{state | emotional: %EmotionalState{}}
  defp ensure_emotional(state), do: state

  defp ensure_sensory(%{sensory: nil} = state), do: %{state | sensory: SensoryState.new()}
  defp ensure_sensory(state), do: state

  defp ensure_consciousness(%{consciousness: nil} = state, personality) do
    %{state | consciousness: ConsciousnessState.from_personality(personality)}
  end

  defp ensure_consciousness(state, _), do: state

  defp ensure_allostasis(%{allostasis: nil} = state) do
    %{state | allostasis: AllostasisState.new()}
  end

  defp ensure_allostasis(state), do: state

  defp ensure_regulation(%{regulation: nil} = state) do
    %{state | regulation: EmotionRegulationState.new()}
  end

  defp ensure_regulation(state), do: state

  defp ensure_somatic(%{somatic: nil} = state) do
    %{state | somatic: SomaticMarkersState.new()}
  end

  defp ensure_somatic(state), do: state

  defp ensure_motivation(%{motivation: nil} = state, personality) do
    %{state | motivation: MotivationState.from_enneagram(personality.enneagram_type)}
  end

  defp ensure_motivation(state, _), do: state
end
