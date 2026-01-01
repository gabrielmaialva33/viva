defmodule Viva.Avatars.SensoryState do
  @moduledoc """
  Embedded schema for the avatar's sensory/perceptual state.
  Represents what the avatar is currently perceiving and attending to.

  This is the "Senses" layer that creates subjective experience (qualia)
  from raw environmental stimuli.
  """
  use Ecto.Schema
  import Ecto.Changeset

  @primary_key false
  embedded_schema do
    # === Attention System ===
    # What the avatar is focused on
    field :attention_focus, :string
    # Intensity of focus (0.0 = distracted, 1.0 = laser focused)
    field :attention_intensity, :float, default: 0.5
    # Attention bandwidth used (0.0 = bored, 1.0 = overwhelmed)
    field :cognitive_load, :float, default: 0.3

    # === Working Perceptual Memory ===
    # Last N perceived stimuli with subjective coloring
    # Each: %{stimulus: x, qualia: y, timestamp: z, salience: 0.0-1.0}
    field :active_percepts, {:array, :map}, default: []

    # === Prediction Engine ===
    # Current expectations about what will happen
    # Each: %{context: x, prediction: y, confidence: 0.0-1.0}
    field :expectations, {:array, :map}, default: []

    # Surprise/prediction error (0.0 = expected, 1.0 = completely unexpected)
    field :surprise_level, :float, default: 0.0
    field :last_prediction_error, :string

    # === Qualia (Subjective Experience) ===
    # Current sensory experience colored by personality/emotion
    field :current_qualia, :map,
      default: %{
        dominant_sensation: nil,
        emotional_color: nil,
        intensity: 0.0,
        narrative: nil
      }

    # === Hedonic Signals ===
    # Immediate pleasure/pain response (-1.0 to 1.0)
    field :sensory_pleasure, :float, default: 0.0
    # Pain/discomfort level (0.0 to 1.0)
    field :sensory_pain, :float, default: 0.0

    # === Recurrent Processing ===
    # Sensitivity to novel stimuli (modulated by consciousness reentry)
    field :novelty_sensitivity, :float, default: 0.5
  end

  @type t :: %__MODULE__{
          attention_focus: String.t() | nil,
          attention_intensity: float(),
          cognitive_load: float(),
          active_percepts: list(map()),
          expectations: list(map()),
          surprise_level: float(),
          last_prediction_error: String.t() | nil,
          current_qualia: map(),
          sensory_pleasure: float(),
          sensory_pain: float(),
          novelty_sensitivity: float()
        }

  @spec changeset(t(), map()) :: Ecto.Changeset.t()
  def changeset(state, attrs) do
    state
    |> cast(attrs, [
      :attention_focus,
      :attention_intensity,
      :cognitive_load,
      :active_percepts,
      :expectations,
      :surprise_level,
      :last_prediction_error,
      :current_qualia,
      :sensory_pleasure,
      :sensory_pain,
      :novelty_sensitivity
    ])
    |> validate_number(:attention_intensity,
      greater_than_or_equal_to: 0.0,
      less_than_or_equal_to: 1.0
    )
    |> validate_number(:cognitive_load,
      greater_than_or_equal_to: 0.0,
      less_than_or_equal_to: 1.0
    )
    |> validate_number(:surprise_level,
      greater_than_or_equal_to: 0.0,
      less_than_or_equal_to: 1.0
    )
    |> validate_number(:sensory_pleasure,
      greater_than_or_equal_to: -1.0,
      less_than_or_equal_to: 1.0
    )
    |> validate_number(:sensory_pain,
      greater_than_or_equal_to: 0.0,
      less_than_or_equal_to: 1.0
    )
    |> validate_number(:novelty_sensitivity,
      greater_than_or_equal_to: 0.0,
      less_than_or_equal_to: 1.0
    )
  end

  @spec new() :: t()
  def new do
    %__MODULE__{
      attention_focus: nil,
      attention_intensity: 0.5,
      cognitive_load: 0.3,
      active_percepts: [],
      expectations: [],
      surprise_level: 0.0,
      last_prediction_error: nil,
      current_qualia: %{
        dominant_sensation: nil,
        emotional_color: nil,
        intensity: 0.0,
        narrative: nil
      },
      sensory_pleasure: 0.0,
      sensory_pain: 0.0,
      novelty_sensitivity: 0.5
    }
  end

  @doc """
  Returns the dominant sensation being experienced.
  """
  @spec dominant_sensation(t()) :: String.t() | nil
  def dominant_sensation(%__MODULE__{current_qualia: qualia}) do
    Map.get(qualia, :dominant_sensation)
  end

  @doc """
  Returns the narrative description of current experience.
  """
  @spec experience_narrative(t()) :: String.t() | nil
  def experience_narrative(%__MODULE__{current_qualia: qualia}) do
    Map.get(qualia, :narrative)
  end

  @doc """
  Checks if the avatar is currently surprised (prediction error > 0.5).
  """
  @spec surprised?(t()) :: boolean()
  def surprised?(%__MODULE__{surprise_level: level}), do: level > 0.5

  @doc """
  Checks if the avatar is experiencing pleasure.
  """
  @spec experiencing_pleasure?(t()) :: boolean()
  def experiencing_pleasure?(%__MODULE__{sensory_pleasure: pleasure}), do: pleasure > 0.3

  @doc """
  Checks if the avatar is experiencing pain/discomfort.
  """
  @spec experiencing_pain?(t()) :: boolean()
  def experiencing_pain?(%__MODULE__{sensory_pain: pain}), do: pain > 0.3

  @doc """
  Checks if attention is highly focused.
  """
  @spec focused?(t()) :: boolean()
  def focused?(%__MODULE__{attention_intensity: intensity}), do: intensity > 0.7

  @doc """
  Checks if cognitive load is high (overwhelmed).
  """
  @spec overwhelmed?(t()) :: boolean()
  def overwhelmed?(%__MODULE__{cognitive_load: load}), do: load > 0.8
end
