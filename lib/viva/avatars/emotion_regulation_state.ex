defmodule Viva.Avatars.EmotionRegulationState do
  @moduledoc """
  Tracks emotion regulation strategies and their usage over time.

  Different personalities favor different coping strategies:
  - Rumination: Dwelling on negative emotions (high neuroticism)
  - Reappraisal: Reframing situations positively (high openness + conscientiousness)
  - Seek Support: Reaching out to others (high extraversion, secure attachment)
  - Suppression: Hiding/suppressing emotions (low agreeableness)
  - Distraction: Shifting attention away (neutral/default)

  The system tracks which strategies are used and their effectiveness,
  allowing avatars to develop habitual coping patterns over time.
  """
  use Ecto.Schema
  import Ecto.Changeset

  @primary_key false
  embedded_schema do
    # Current active strategy (nil if not actively regulating)
    field :active_strategy, Ecto.Enum,
      values: [:ruminate, :reappraise, :seek_support, :suppress, :distract],
      default: nil

    # How long the current strategy has been active (in ticks)
    field :strategy_duration, :integer, default: 0

    # Usage counts for each strategy (tracks habitual patterns)
    field :ruminate_count, :integer, default: 0
    field :reappraise_count, :integer, default: 0
    field :seek_support_count, :integer, default: 0
    field :suppress_count, :integer, default: 0
    field :distract_count, :integer, default: 0

    # Effectiveness tracking (running average of how well each strategy worked)
    field :ruminate_effectiveness, :float, default: 0.0
    field :reappraise_effectiveness, :float, default: 0.5
    field :seek_support_effectiveness, :float, default: 0.5
    field :suppress_effectiveness, :float, default: 0.3
    field :distract_effectiveness, :float, default: 0.4

    # Emotional state before regulation started (to measure effectiveness)
    field :pre_regulation_pleasure, :float
    field :pre_regulation_arousal, :float

    # Regulation exhaustion (0.0 = fresh, 1.0 = depleted)
    # High exhaustion makes regulation less effective
    field :regulation_exhaustion, :float, default: 0.0

    # Last time regulation was attempted
    field :last_regulation_at, :utc_datetime
  end

  @type strategy :: :ruminate | :reappraise | :seek_support | :suppress | :distract
  @type t :: %__MODULE__{
          active_strategy: strategy() | nil,
          strategy_duration: non_neg_integer(),
          ruminate_count: non_neg_integer(),
          reappraise_count: non_neg_integer(),
          seek_support_count: non_neg_integer(),
          suppress_count: non_neg_integer(),
          distract_count: non_neg_integer(),
          ruminate_effectiveness: float(),
          reappraise_effectiveness: float(),
          seek_support_effectiveness: float(),
          suppress_effectiveness: float(),
          distract_effectiveness: float(),
          pre_regulation_pleasure: float() | nil,
          pre_regulation_arousal: float() | nil,
          regulation_exhaustion: float(),
          last_regulation_at: DateTime.t() | nil
        }

  @spec changeset(t(), map()) :: Ecto.Changeset.t()
  def changeset(state, attrs) do
    state
    |> cast(attrs, [
      :active_strategy,
      :strategy_duration,
      :ruminate_count,
      :reappraise_count,
      :seek_support_count,
      :suppress_count,
      :distract_count,
      :ruminate_effectiveness,
      :reappraise_effectiveness,
      :seek_support_effectiveness,
      :suppress_effectiveness,
      :distract_effectiveness,
      :pre_regulation_pleasure,
      :pre_regulation_arousal,
      :regulation_exhaustion,
      :last_regulation_at
    ])
    |> validate_number(:regulation_exhaustion,
      greater_than_or_equal_to: 0.0,
      less_than_or_equal_to: 1.0
    )
  end

  @spec new() :: t()
  def new do
    %__MODULE__{
      active_strategy: nil,
      strategy_duration: 0,
      ruminate_count: 0,
      reappraise_count: 0,
      seek_support_count: 0,
      suppress_count: 0,
      distract_count: 0,
      ruminate_effectiveness: 0.0,
      reappraise_effectiveness: 0.5,
      seek_support_effectiveness: 0.5,
      suppress_effectiveness: 0.3,
      distract_effectiveness: 0.4,
      pre_regulation_pleasure: nil,
      pre_regulation_arousal: nil,
      regulation_exhaustion: 0.0,
      last_regulation_at: nil
    }
  end

  @doc """
  Returns the most used strategy (habitual coping pattern).
  """
  @spec dominant_strategy(t()) :: strategy() | nil
  def dominant_strategy(state) do
    counts = [
      {:ruminate, state.ruminate_count},
      {:reappraise, state.reappraise_count},
      {:seek_support, state.seek_support_count},
      {:suppress, state.suppress_count},
      {:distract, state.distract_count}
    ]

    case Enum.max_by(counts, fn {_, count} -> count end) do
      {_, 0} -> nil
      {strategy, _} -> strategy
    end
  end

  @doc """
  Returns the total number of regulation attempts.
  """
  @spec total_attempts(t()) :: non_neg_integer()
  def total_attempts(state) do
    state.ruminate_count +
      state.reappraise_count +
      state.seek_support_count +
      state.suppress_count +
      state.distract_count
  end
end
