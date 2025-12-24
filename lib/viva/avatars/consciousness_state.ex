defmodule Viva.Avatars.ConsciousnessState do
  @moduledoc """
  Embedded schema for the avatar's consciousness/awareness state.
  Represents the unified subjective experience and self-awareness.

  This is the "Consciousness" layer that integrates perception, emotion,
  thought, and memory into a coherent stream of experience.
  """
  use Ecto.Schema
  import Ecto.Changeset

  alias Viva.Avatars.SelfModel

  @primary_key false
  embedded_schema do
    # === Stream of Consciousness ===
    # Recent moments of experience (limited working memory)
    # Each: %{content: x, type: y, timestamp: z, intensity: 0.0-1.0}
    field :experience_stream, {:array, :map}, default: []

    # Current stream tempo (thought speed)
    field :stream_tempo, Ecto.Enum,
      values: [:frozen, :slow, :normal, :fast, :racing],
      default: :normal

    # === Self Model ===
    embeds_one :self_model, SelfModel, on_replace: :update

    # How aligned current behavior is with self-image (0.0 to 1.0)
    field :self_congruence, :float, default: 0.8

    # === Metacognition ===
    # Level of self-awareness (0.0 = autopilot, 1.0 = highly reflective)
    field :meta_awareness, :float, default: 0.3

    # Current metacognitive observation ("I notice that I'm...")
    field :meta_observation, :string

    # === Global Workspace ===
    # What is currently in focal awareness
    field :focal_content, :map, default: %{type: nil, content: nil, source: nil}

    # Background awareness (things noticed but not focused on)
    field :peripheral_content, {:array, :map}, default: []

    # === Temporal Orientation ===
    # Where attention is in time
    field :temporal_focus, Ecto.Enum,
      values: [:past, :present, :future],
      default: :present

    # How far ahead/behind thinking extends (in simulated minutes)
    field :time_horizon, :integer, default: 0

    # === Experience Quality ===
    # How present/grounded vs dissociated (0.0 to 1.0)
    field :presence_level, :float, default: 0.7

    # Vividness of experience (0.0 = numb, 1.0 = intensely alive)
    field :experience_intensity, :float, default: 0.5

    # Flow state indicator (0.0 = fragmented, 1.0 = fully absorbed)
    field :flow_state, :float, default: 0.3
  end

  @type t :: %__MODULE__{
          experience_stream: list(map()),
          stream_tempo: atom(),
          self_model: SelfModel.t() | nil,
          self_congruence: float(),
          meta_awareness: float(),
          meta_observation: String.t() | nil,
          focal_content: map(),
          peripheral_content: list(map()),
          temporal_focus: atom(),
          time_horizon: integer(),
          presence_level: float(),
          experience_intensity: float(),
          flow_state: float()
        }

  @spec changeset(t(), map()) :: Ecto.Changeset.t()
  def changeset(state, attrs) do
    state
    |> cast(attrs, [
      :experience_stream,
      :stream_tempo,
      :self_congruence,
      :meta_awareness,
      :meta_observation,
      :focal_content,
      :peripheral_content,
      :temporal_focus,
      :time_horizon,
      :presence_level,
      :experience_intensity,
      :flow_state
    ])
    |> cast_embed(:self_model)
    |> validate_number(:self_congruence,
      greater_than_or_equal_to: 0.0,
      less_than_or_equal_to: 1.0
    )
    |> validate_number(:meta_awareness,
      greater_than_or_equal_to: 0.0,
      less_than_or_equal_to: 1.0
    )
    |> validate_number(:presence_level,
      greater_than_or_equal_to: 0.0,
      less_than_or_equal_to: 1.0
    )
    |> validate_number(:experience_intensity,
      greater_than_or_equal_to: 0.0,
      less_than_or_equal_to: 1.0
    )
    |> validate_number(:flow_state,
      greater_than_or_equal_to: 0.0,
      less_than_or_equal_to: 1.0
    )
  end

  @spec new() :: t()
  def new do
    %__MODULE__{
      experience_stream: [],
      stream_tempo: :normal,
      self_model: SelfModel.new(),
      self_congruence: 0.8,
      meta_awareness: 0.3,
      meta_observation: nil,
      focal_content: %{type: nil, content: nil, source: nil},
      peripheral_content: [],
      temporal_focus: :present,
      time_horizon: 0,
      presence_level: 0.7,
      experience_intensity: 0.5,
      flow_state: 0.3
    }
  end

  @doc """
  Initialize consciousness state with a self-model derived from personality.
  """
  @spec from_personality(Viva.Avatars.Personality.t()) :: t()
  def from_personality(personality) do
    %__MODULE__{
      experience_stream: [],
      stream_tempo: :normal,
      self_model: SelfModel.from_personality(personality),
      self_congruence: 0.8,
      meta_awareness: personality.openness * 0.4 + 0.2,
      meta_observation: nil,
      focal_content: %{type: nil, content: nil, source: nil},
      peripheral_content: [],
      temporal_focus: :present,
      time_horizon: 0,
      presence_level: 0.7,
      experience_intensity: 0.5,
      flow_state: 0.3
    }
  end

  # === Query Functions ===

  @doc """
  Returns true if the avatar is in a dissociative state.
  """
  @spec dissociated?(t()) :: boolean()
  def dissociated?(%__MODULE__{presence_level: presence}), do: presence < 0.4

  @doc """
  Returns true if the avatar is in a flow state.
  """
  @spec in_flow?(t()) :: boolean()
  def in_flow?(%__MODULE__{flow_state: flow}), do: flow > 0.7

  @doc """
  Returns true if the avatar is highly self-aware.
  """
  @spec self_aware?(t()) :: boolean()
  def self_aware?(%__MODULE__{meta_awareness: awareness}), do: awareness > 0.6

  @doc """
  Returns true if thoughts are racing.
  """
  @spec racing_thoughts?(t()) :: boolean()
  def racing_thoughts?(%__MODULE__{stream_tempo: tempo}), do: tempo == :racing

  @doc """
  Returns true if experience is frozen (e.g., shock, extreme fatigue).
  """
  @spec frozen?(t()) :: boolean()
  def frozen?(%__MODULE__{stream_tempo: tempo}), do: tempo == :frozen

  @doc """
  Returns true if focused on the past (ruminating).
  """
  @spec ruminating?(t()) :: boolean()
  def ruminating?(%__MODULE__{temporal_focus: focus}), do: focus == :past

  @doc """
  Returns true if focused on the future (anticipating/worrying).
  """
  @spec anticipating?(t()) :: boolean()
  def anticipating?(%__MODULE__{temporal_focus: focus}), do: focus == :future

  @doc """
  Returns the current focal content.
  """
  @spec current_focus(t()) :: map()
  def current_focus(%__MODULE__{focal_content: content}), do: content

  @doc """
  Gets the most recent experience from the stream.
  """
  @spec latest_experience(t()) :: map() | nil
  def latest_experience(%__MODULE__{experience_stream: []}), do: nil
  def latest_experience(%__MODULE__{experience_stream: [latest | _]}), do: latest

  @doc """
  Describes the current state of consciousness in words.
  """
  @spec describe_state(t()) :: String.t()
  def describe_state(%__MODULE__{} = state) do
    presence_desc = presence_description(state.presence_level)
    tempo_desc = tempo_description(state.stream_tempo)

    "Awareness is #{presence_desc}, thoughts are #{tempo_desc}."
  end

  defp presence_description(p) when p > 0.8, do: "fully present and grounded"
  defp presence_description(p) when p > 0.6, do: "mostly present"
  defp presence_description(p) when p > 0.4, do: "somewhat distracted"
  defp presence_description(p) when p > 0.2, do: "feeling distant"
  defp presence_description(_), do: "dissociated and foggy"

  defp tempo_description(:frozen), do: "completely stopped"
  defp tempo_description(:slow), do: "moving slowly and heavily"
  defp tempo_description(:normal), do: "flowing naturally"
  defp tempo_description(:fast), do: "moving quickly"
  defp tempo_description(:racing), do: "racing uncontrollably"
end
