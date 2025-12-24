defmodule Viva.Avatars.Systems.Metacognition do
  @moduledoc """
  Implements metacognitive processes for avatars.

  Metacognition is "thinking about thinking" - the avatar's ability to:
  - Detect patterns in their own behavior and emotions
  - Compare current behavior with ideal_self/feared_self
  - Generate self-insights through reflection
  - Update their self-model based on observations

  ## Pattern Detection
  Analyzes the experience_stream for recurring emotional states and
  behavioral patterns. When patterns are detected, they're stored in
  the self_model for future reference.

  ## Self-Alignment
  Compares current behavior/emotions with ideal_self and feared_self.
  High alignment with ideal increases self_congruence.
  High alignment with feared decreases it and may trigger anxiety.

  ## Insight Generation
  Under certain conditions (high meta_awareness, detected patterns),
  the avatar may generate insights via LLM about their own psychology.
  Insights are rare but meaningful observations like:
  "I notice I always withdraw when I feel criticized."

  ## Integration
  This system runs after Consciousness.integrate() and periodically
  (not every tick) to avoid excessive processing.
  """

  require Logger

  alias Viva.Avatars.ConsciousnessState
  alias Viva.Avatars.EmotionalState
  alias Viva.Avatars.Personality
  alias Viva.Avatars.SelfModel

  # Minimum experiences needed to detect patterns
  @min_experiences_for_pattern 5

  # How often to run full metacognition (every N ticks)
  @metacognition_interval 10

  # Threshold for generating LLM insight
  @insight_threshold 0.7

  # Maximum patterns to store per category
  @max_patterns 10

  @type pattern :: %{
          trigger: String.t(),
          response: String.t(),
          frequency: non_neg_integer(),
          last_detected: DateTime.t()
        }

  @type alignment :: %{
          ideal_alignment: float(),
          feared_alignment: float(),
          dominant_direction: :toward_ideal | :toward_feared | :neutral
        }

  @type metacognition_result :: %{
          patterns_detected: list(pattern()),
          alignment: alignment(),
          insight: String.t() | nil,
          self_congruence_delta: float()
        }

  @doc """
  Main metacognition processing function.
  Called periodically (not every tick) to analyze patterns and self-alignment.
  """
  @spec process(ConsciousnessState.t(), EmotionalState.t(), Personality.t(), non_neg_integer()) ::
          {ConsciousnessState.t(), metacognition_result()}
  def process(consciousness, emotional, personality, tick_count) do
    if should_run_metacognition?(tick_count, consciousness.meta_awareness) do
      do_process(consciousness, emotional, personality)
    else
      {consciousness,
       %{
         patterns_detected: [],
         alignment: %{ideal_alignment: 0.0, feared_alignment: 0.0, dominant_direction: :neutral},
         insight: nil,
         self_congruence_delta: 0.0
       }}
    end
  end

  @doc """
  Detect emotional patterns in the experience stream.
  Returns a list of detected patterns.
  """
  @spec detect_emotional_patterns(list(map())) :: list(pattern())
  def detect_emotional_patterns(experience_stream)
      when length(experience_stream) < @min_experiences_for_pattern do
    []
  end

  def detect_emotional_patterns(experience_stream) do
    # Group experiences by dominant emotion
    emotion_counts =
      experience_stream
      |> Enum.map(fn exp -> exp.emotion[:mood] end)
      |> Enum.filter(& &1)
      |> Enum.frequencies()

    # Find recurring emotions (appear in > 40% of experiences)
    threshold = length(experience_stream) * 0.4

    emotion_counts
    |> Enum.filter(fn {_, count} -> count >= threshold end)
    |> Enum.map(fn {mood, count} ->
      %{
        trigger: "various situations",
        response: mood,
        frequency: count,
        last_detected: DateTime.utc_now(:second)
      }
    end)
  end

  @doc """
  Check alignment between current state and ideal/feared self.
  Returns alignment scores and dominant direction.
  """
  @spec check_alignment(EmotionalState.t(), SelfModel.t() | nil) :: alignment()
  def check_alignment(_emotional, nil) do
    %{ideal_alignment: 0.0, feared_alignment: 0.0, dominant_direction: :neutral}
  end

  def check_alignment(emotional, self_model) do
    ideal_alignment = calculate_ideal_alignment(emotional, self_model)
    feared_alignment = calculate_feared_alignment(emotional, self_model)

    dominant_direction =
      cond do
        ideal_alignment > feared_alignment + 0.2 -> :toward_ideal
        feared_alignment > ideal_alignment + 0.2 -> :toward_feared
        true -> :neutral
      end

    %{
      ideal_alignment: ideal_alignment,
      feared_alignment: feared_alignment,
      dominant_direction: dominant_direction
    }
  end

  @doc """
  Generate a self-insight based on patterns and alignment.
  Only generates insight when conditions are right (high meta_awareness, patterns detected).
  """
  @spec maybe_generate_insight(
          ConsciousnessState.t(),
          list(pattern()),
          alignment(),
          Personality.t()
        ) :: String.t() | nil
  def maybe_generate_insight(consciousness, patterns, alignment, personality) do
    # Only generate insight if:
    # 1. Meta-awareness is high enough
    # 2. There are patterns to reflect on OR alignment is significant
    should_generate =
      consciousness.meta_awareness >= @insight_threshold and
        (length(patterns) > 0 or alignment.dominant_direction != :neutral)

    if should_generate do
      generate_insight(consciousness, patterns, alignment, personality)
    else
      nil
    end
  end

  @doc """
  Update self-model with newly detected patterns.
  Merges new patterns with existing ones, updating frequencies.
  """
  @spec update_self_model(SelfModel.t(), list(pattern()), alignment()) :: SelfModel.t()
  def update_self_model(self_model, new_patterns, alignment) do
    updated_emotional_patterns = merge_patterns(self_model.emotional_patterns, new_patterns)

    # Adjust self-esteem based on alignment
    esteem_delta =
      case alignment.dominant_direction do
        :toward_ideal -> 0.02
        :toward_feared -> -0.03
        :neutral -> 0.0
      end

    new_esteem =
      (self_model.self_esteem + esteem_delta)
      |> max(0.1)
      |> min(0.9)

    %{
      self_model
      | emotional_patterns: Enum.take(updated_emotional_patterns, @max_patterns),
        self_esteem: new_esteem
    }
  end

  @doc """
  Calculate self-congruence delta based on alignment.
  """
  @spec calculate_congruence_delta(alignment()) :: float()
  def calculate_congruence_delta(alignment) do
    case alignment.dominant_direction do
      :toward_ideal -> 0.05
      :toward_feared -> -0.08
      :neutral -> 0.01
    end
  end

  @doc """
  Describe current metacognitive state.
  """
  @spec describe(ConsciousnessState.t()) :: String.t()
  def describe(%ConsciousnessState{meta_awareness: awareness, meta_observation: obs})
      when awareness > 0.7 do
    obs || "Highly self-aware, observing my own thoughts and feelings"
  end

  def describe(%ConsciousnessState{meta_awareness: awareness})
      when awareness > 0.4 do
    "Some self-awareness, occasionally noticing patterns"
  end

  def describe(%ConsciousnessState{}) do
    "Running on autopilot, not much self-reflection"
  end

  # === Private Functions ===

  defp should_run_metacognition?(tick_count, meta_awareness) do
    # Run more often if meta_awareness is high
    interval =
      if meta_awareness > 0.6, do: @metacognition_interval - 3, else: @metacognition_interval

    rem(tick_count, max(interval, 1)) == 0
  end

  defp do_process(consciousness, emotional, personality) do
    self_model = consciousness.self_model

    # 1. Detect patterns in experience stream
    patterns = detect_emotional_patterns(consciousness.experience_stream)

    # 2. Check alignment with ideal/feared self
    alignment = check_alignment(emotional, self_model)

    # 3. Maybe generate insight
    insight = maybe_generate_insight(consciousness, patterns, alignment, personality)

    # 4. Update self-model with new patterns
    updated_self_model =
      if self_model do
        update_self_model(self_model, patterns, alignment)
      else
        self_model
      end

    # 5. Calculate congruence delta
    congruence_delta = calculate_congruence_delta(alignment)

    new_congruence =
      (consciousness.self_congruence + congruence_delta)
      |> max(0.2)
      |> min(1.0)

    # 6. Update consciousness with new observation
    updated_consciousness =
      %{
        consciousness
        | self_model: updated_self_model,
          self_congruence: new_congruence,
          meta_observation: insight || consciousness.meta_observation
      }

    result = %{
      patterns_detected: patterns,
      alignment: alignment,
      insight: insight,
      self_congruence_delta: congruence_delta
    }

    {updated_consciousness, result}
  end

  defp calculate_ideal_alignment(_emotional, %SelfModel{ideal_self: nil}), do: 0.0

  defp calculate_ideal_alignment(emotional, %SelfModel{ideal_self: _ideal_self}) do
    # Higher pleasure + higher self-esteem indicators = closer to ideal
    # This is a simplified heuristic - ideal_self is a string description
    positive_emotional_state = (emotional.pleasure + 1) / 2
    dominance_factor = (emotional.dominance + 1) / 2

    (positive_emotional_state * 0.6 + dominance_factor * 0.4)
    |> min(1.0)
  end

  defp calculate_feared_alignment(_emotional, %SelfModel{feared_self: nil}), do: 0.0

  defp calculate_feared_alignment(emotional, %SelfModel{feared_self: _feared_self}) do
    # High negative emotion + low dominance = closer to feared self
    negative_emotional_state = (1 - emotional.pleasure) / 2
    low_dominance_factor = (1 - emotional.dominance) / 2
    high_arousal_factor = (emotional.arousal + 1) / 2

    # Stressed, anxious state is often feared
    (negative_emotional_state * 0.4 + low_dominance_factor * 0.3 + high_arousal_factor * 0.3)
    |> min(1.0)
  end

  defp generate_insight(consciousness, patterns, alignment, personality) do
    # Generate insight based on detected patterns and alignment
    cond do
      # Pattern-based insight
      length(patterns) > 0 ->
        pattern = hd(patterns)
        generate_pattern_insight(pattern, personality)

      # Alignment-based insight
      alignment.dominant_direction == :toward_ideal ->
        generate_positive_alignment_insight(consciousness.self_model)

      alignment.dominant_direction == :toward_feared ->
        generate_negative_alignment_insight(consciousness.self_model, personality)

      true ->
        nil
    end
  end

  defp generate_pattern_insight(pattern, personality) do
    mood = pattern.response
    frequency = pattern.frequency

    base_insight =
      cond do
        frequency >= 7 ->
          "I notice I've been feeling #{mood} a lot lately."

        frequency >= 5 ->
          "There's a pattern - #{mood} keeps coming up for me."

        true ->
          "I'm aware that #{mood} is becoming a theme."
      end

    # Add personality-influenced reflection
    if personality.openness > 0.6 do
      base_insight <> " I wonder what this means about what I need."
    else
      base_insight
    end
  end

  defp generate_positive_alignment_insight(%SelfModel{ideal_self: ideal}) when is_binary(ideal) do
    "I feel like I'm becoming more of who I want to be."
  end

  defp generate_positive_alignment_insight(_), do: "I feel good about how I'm being right now."

  defp generate_negative_alignment_insight(%SelfModel{feared_self: feared}, personality)
       when is_binary(feared) do
    if personality.neuroticism > 0.6 do
      "I'm worried I'm becoming what I fear... I need to change something."
    else
      "I notice I'm not quite myself lately. Something feels off."
    end
  end

  defp generate_negative_alignment_insight(_, _) do
    "Something doesn't feel right about how I'm being."
  end

  defp merge_patterns(existing, new) do
    # Merge new patterns with existing, updating frequencies
    Enum.reduce(new, existing, fn new_pattern, acc ->
      new_emotion = new_pattern.response

      # Find existing pattern with same typical_emotion
      case Enum.find_index(acc, fn p -> p[:typical_emotion] == new_emotion end) do
        nil ->
          # New pattern, add it
          [convert_to_stored_pattern(new_pattern) | acc]

        _index ->
          # Existing pattern already exists, skip duplicate
          acc
      end
    end)
  end

  defp convert_to_stored_pattern(pattern) do
    %{
      situation: pattern.trigger,
      typical_emotion: pattern.response
    }
  end
end
