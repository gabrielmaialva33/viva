defmodule Viva.Avatars.SomaticMarkersState do
  @moduledoc """
  Tracks somatic markers - body-based memories that influence decisions.

  Based on Damasio's somatic marker hypothesis: emotionally intense experiences
  create lasting associations in the body. When similar situations arise,
  the body "warns" (negative marker) or "attracts" (positive marker) before
  conscious reasoning occurs.

  Markers are learned from intense experiences and influence:
  - Approach/avoidance tendencies
  - Gut feelings about situations
  - Decision-making biases

  Each marker stores:
  - Stimulus features that triggered the experience
  - The body state at the time (valence, intensity)
  - Strength of the association (decays over time)
  """
  use Ecto.Schema
  import Ecto.Changeset

  @primary_key false
  embedded_schema do
    # Markers for social interactions (by partner type or source)
    # Map of source -> %{valence: float, strength: float, last_activated: datetime}
    field :social_markers, :map, default: %{}

    # Markers for activity types
    # Map of activity_type -> %{valence: float, strength: float, last_activated: datetime}
    field :activity_markers, :map, default: %{}

    # Markers for emotional contexts
    # Map of emotional_context -> %{valence: float, strength: float, last_activated: datetime}
    field :context_markers, :map, default: %{}

    # Current somatic bias (computed from active markers)
    # Positive = body attraction, Negative = body warning
    field :current_bias, :float, default: 0.0

    # Current body signal description
    field :body_signal, :string

    # Learning threshold (minimum emotional intensity to form a marker)
    field :learning_threshold, :float, default: 0.7

    # How many markers have been formed (for tracking development)
    field :markers_formed, :integer, default: 0

    # Last time a marker was activated
    field :last_marker_activation, :utc_datetime
  end

  @type marker :: %{
          valence: float(),
          strength: float(),
          last_activated: DateTime.t() | nil,
          context: String.t() | nil
        }

  @type t :: %__MODULE__{
          social_markers: %{String.t() => marker()},
          activity_markers: %{atom() => marker()},
          context_markers: %{String.t() => marker()},
          current_bias: float(),
          body_signal: String.t() | nil,
          learning_threshold: float(),
          markers_formed: non_neg_integer(),
          last_marker_activation: DateTime.t() | nil
        }

  @spec changeset(t(), map()) :: Ecto.Changeset.t()
  def changeset(state, attrs) do
    state
    |> cast(attrs, [
      :social_markers,
      :activity_markers,
      :context_markers,
      :current_bias,
      :body_signal,
      :learning_threshold,
      :markers_formed,
      :last_marker_activation
    ])
    |> validate_number(:current_bias, greater_than_or_equal_to: -1.0, less_than_or_equal_to: 1.0)
    |> validate_number(:learning_threshold,
      greater_than_or_equal_to: 0.0,
      less_than_or_equal_to: 1.0
    )
  end

  @spec new() :: t()
  def new do
    %__MODULE__{
      social_markers: %{},
      activity_markers: %{},
      context_markers: %{},
      current_bias: 0.0,
      body_signal: nil,
      learning_threshold: 0.7,
      markers_formed: 0,
      last_marker_activation: nil
    }
  end

  @doc """
  Returns the total number of markers across all categories.
  """
  @spec total_markers(t()) :: non_neg_integer()
  def total_markers(state) do
    map_size(state.social_markers) +
      map_size(state.activity_markers) +
      map_size(state.context_markers)
  end

  @doc """
  Returns true if the avatar has developed significant somatic memory.
  """
  @spec has_body_memory?(t()) :: boolean()
  def has_body_memory?(state) do
    state.markers_formed >= 3
  end

  @doc """
  Returns the strongest marker (positive or negative).
  """
  @spec strongest_marker(t()) :: {atom(), String.t(), marker()} | nil
  def strongest_marker(state) do
    all_markers =
      Enum.concat([
        Enum.map(state.social_markers, fn {k, v} -> {:social, k, v} end),
        Enum.map(state.activity_markers, fn {k, v} -> {:activity, to_string(k), v} end),
        Enum.map(state.context_markers, fn {k, v} -> {:context, k, v} end)
      ])

    case Enum.max_by(all_markers, fn {_, _, m} -> abs(m.valence) * m.strength end, fn -> nil end) do
      nil -> nil
      marker -> marker
    end
  end
end
