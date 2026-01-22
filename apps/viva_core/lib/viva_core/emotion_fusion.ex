defmodule VivaCore.EmotionFusion do
  @moduledoc """
  Emotion Fusion - Dual-Source Emotional Processing

  Based on "Emotions in Artificial Intelligence" (Borotschnig, 2025).

  Combines three sources of emotional information:
  1. **Need-based emotions** (from Interoception) - Current physiological state
  2. **Past emotions** (from Memory retrieval) - Emotional tags from similar episodes
  3. **Personality bias** (trait baseline) - Long-term emotional tendencies

  The fusion uses adaptive weights that vary based on context:
  - High arousal → trust immediate needs more
  - High confidence → trust past experiences more
  - High novelty → rely on personality defaults

  Output: Unified PreActionAffect state for action selection.

  ## Paper Reference
  ```
  FusedEmotions := FuseEmotions(CurrentEmotions, PastEmotions)
  PreActionAffect := [FusedEmotions, Mood, Personality]
  ```
  """

  require VivaLog

  alias VivaCore.Personality

  # Default fusion weights
  @default_need_weight 0.4
  @default_past_weight 0.4
  @default_personality_weight 0.2

  # Mood smoothing factor (α = 0.95 → ~20-step half-life)
  @mood_alpha 0.95

  @type pad :: %{pleasure: float(), arousal: float(), dominance: float()}
  @type context :: %{
    arousal: float(),
    confidence: float(),
    novelty: float()
  }
  @type fusion_result :: %{
    fused_pad: pad(),
    mood: pad(),
    pre_action_affect: map(),
    weights: {float(), float(), float()}
  }

  # ============================================================
  # Main API
  # ============================================================

  @doc """
  Main fusion function - combines multiple emotion sources.

  ## Parameters
  - need_pad: PAD from Interoception (current needs/physiological state)
  - past_pad: PAD from Memory retrieval (emotional tags from similar episodes)
  - personality: %Personality{} struct with baseline, reactivity, volatility
  - mood: Current mood state (EMA of recent emotions)
  - context: %{arousal: float, confidence: float, novelty: float}
    - arousal: Current arousal level (high → trust needs more)
    - confidence: Similarity score of retrieved memories (high → trust past more)
    - novelty: 1 - max_similarity (high → new situation, rely on personality)

  ## Returns
  %{
    fused_pad: PAD,           # Final fused emotion
    mood: PAD,                # Updated mood (EMA)
    pre_action_affect: map(), # Full affect state for action selection
    weights: {w_need, w_past, w_pers}
  }
  """
  @spec fuse(pad(), pad(), Personality.t(), pad(), context()) :: fusion_result()
  def fuse(need_pad, past_pad, personality, mood, context) do
    # 1. Calculate adaptive weights based on context
    {w_need, w_past, w_pers} = calculate_weights(context)

    # 2. Weighted fusion of three sources
    fused = weighted_fusion(
      need_pad, past_pad, personality.baseline,
      {w_need, w_past, w_pers}
    )

    # 3. Apply personality reactivity (amplify/dampen deviations)
    fused = apply_reactivity(fused, personality)

    # 4. Clamp to valid PAD range
    fused = clamp_pad(fused)

    # 5. Update mood using EMA
    new_mood = update_mood(mood, fused)

    # 6. Build PreActionAffect (full affect state for action selection)
    pre_action_affect = %{
      emotion: fused,
      mood: new_mood,
      personality_baseline: personality.baseline,
      personality_traits: personality.traits,
      confidence: Map.get(context, :confidence, 0.5),
      novelty: Map.get(context, :novelty, 0.5),
      fusion_weights: %{
        need: w_need,
        past: w_past,
        personality: w_pers
      }
    }

    VivaLog.debug(:emotion_fusion, :fused,
      pleasure: Float.round(fused.pleasure, 3),
      arousal: Float.round(fused.arousal, 3),
      dominance: Float.round(fused.dominance, 3),
      w_need: Float.round(w_need, 2),
      w_past: Float.round(w_past, 2)
    )

    %{
      fused_pad: fused,
      mood: new_mood,
      pre_action_affect: pre_action_affect,
      weights: {w_need, w_past, w_pers}
    }
  end

  @doc """
  Simple fusion without full context (for quick updates).

  Uses default weights.
  """
  @spec simple_fuse(pad(), pad()) :: pad()
  def simple_fuse(need_pad, past_pad) do
    weighted_fusion(
      need_pad, past_pad, Personality.neutral_pad(),
      {@default_need_weight, @default_past_weight, @default_personality_weight}
    )
    |> clamp_pad()
  end

  @doc """
  Calculate adaptive weights based on context.

  Weight adaptation rules:
  - High arousal → trust immediate needs more (emergency response)
  - High confidence → trust past experiences more (familiar situation)
  - High novelty → rely on personality defaults (new situation)

  ## Parameters
  - context: %{arousal: float, confidence: float, novelty: float}

  ## Returns
  {w_need, w_past, w_personality} - Normalized weights summing to 1.0
  """
  @spec calculate_weights(context()) :: {float(), float(), float()}
  def calculate_weights(context) do
    arousal = Map.get(context, :arousal, 0.0) |> abs()
    confidence = Map.get(context, :confidence, 0.5)
    novelty = Map.get(context, :novelty, 0.5)

    # Base weights modified by context
    # High arousal → increase need weight (emergency mode)
    need_mod = @default_need_weight * (1.0 + arousal * 0.5)

    # High confidence (from memory retrieval) → increase past weight
    past_mod = @default_past_weight * (0.5 + confidence)

    # High novelty (new situation) → increase personality weight
    pers_mod = @default_personality_weight * (0.5 + novelty)

    # Normalize to sum to 1.0
    total = need_mod + past_mod + pers_mod

    {
      need_mod / total,
      past_mod / total,
      pers_mod / total
    }
  end

  @doc """
  Update mood using Exponential Moving Average (EMA).

  Mood changes slowly (α = 0.95), providing emotional stability.

  Mood[t] = α × Mood[t-1] + (1-α) × Emotion[t]

  ## Parameters
  - current_mood: Previous mood PAD
  - new_emotion: Current fused emotion PAD

  ## Returns
  Updated mood PAD
  """
  @spec update_mood(pad(), pad()) :: pad()
  def update_mood(current_mood, new_emotion) do
    alpha = @mood_alpha

    %{
      pleasure: alpha * get_value(current_mood, :pleasure) +
                (1 - alpha) * get_value(new_emotion, :pleasure),
      arousal: alpha * get_value(current_mood, :arousal) +
               (1 - alpha) * get_value(new_emotion, :arousal),
      dominance: alpha * get_value(current_mood, :dominance) +
                 (1 - alpha) * get_value(new_emotion, :dominance)
    }
  end

  @doc """
  Get the mood smoothing factor (alpha).
  """
  @spec mood_alpha() :: float()
  def mood_alpha, do: @mood_alpha

  @doc """
  Create a neutral context for default fusion.
  """
  @spec neutral_context() :: context()
  def neutral_context do
    %{
      arousal: 0.0,
      confidence: 0.5,
      novelty: 0.5
    }
  end

  @doc """
  Get the neutral PAD state.
  """
  @spec neutral_pad() :: pad()
  def neutral_pad, do: %{pleasure: 0.0, arousal: 0.0, dominance: 0.0}

  @doc """
  Compute emotional distance between two PAD vectors.

  Uses Euclidean distance in PAD space.
  """
  @spec emotional_distance(pad(), pad()) :: float()
  def emotional_distance(pad1, pad2) do
    dp = get_value(pad1, :pleasure) - get_value(pad2, :pleasure)
    da = get_value(pad1, :arousal) - get_value(pad2, :arousal)
    dd = get_value(pad1, :dominance) - get_value(pad2, :dominance)

    :math.sqrt(dp * dp + da * da + dd * dd)
  end

  @doc """
  Classify emotion based on PAD octant.

  Returns a basic emotion label based on which octant of PAD space
  the emotion falls into.
  """
  @spec classify_emotion(pad()) :: atom()
  def classify_emotion(pad) do
    p = get_value(pad, :pleasure)
    a = get_value(pad, :arousal)
    d = get_value(pad, :dominance)

    cond do
      p > 0.2 and a > 0.2 and d > 0.2 -> :exuberant
      p > 0.2 and a > 0.2 and d <= 0.2 -> :dependent_joy
      p > 0.2 and a <= 0.2 and d > 0.2 -> :relaxed
      p > 0.2 and a <= 0.2 and d <= 0.2 -> :docile
      p <= 0.2 and a > 0.2 and d > 0.2 -> :hostile
      p <= 0.2 and a > 0.2 and d <= 0.2 -> :anxious
      p <= 0.2 and a <= 0.2 and d > 0.2 -> :disdainful
      p <= 0.2 and a <= 0.2 and d <= 0.2 -> :bored
      true -> :neutral
    end
  end

  # ============================================================
  # Private Functions
  # ============================================================

  defp weighted_fusion(need_pad, past_pad, personality_baseline, {w_need, w_past, w_pers}) do
    %{
      pleasure: w_need * get_value(need_pad, :pleasure) +
                w_past * get_value(past_pad, :pleasure) +
                w_pers * get_value(personality_baseline, :pleasure),
      arousal: w_need * get_value(need_pad, :arousal) +
               w_past * get_value(past_pad, :arousal) +
               w_pers * get_value(personality_baseline, :arousal),
      dominance: w_need * get_value(need_pad, :dominance) +
                 w_past * get_value(past_pad, :dominance) +
                 w_pers * get_value(personality_baseline, :dominance)
    }
  end

  defp apply_reactivity(fused, %Personality{} = personality) do
    # Reactivity amplifies deviation from personality baseline
    baseline = personality.baseline
    reactivity = personality.reactivity

    %{
      pleasure: baseline.pleasure +
                (fused.pleasure - baseline.pleasure) * reactivity,
      arousal: baseline.arousal +
               (fused.arousal - baseline.arousal) * reactivity,
      dominance: baseline.dominance +
                 (fused.dominance - baseline.dominance) * reactivity
    }
  end

  defp clamp_pad(pad) do
    %{
      pleasure: clamp(get_value(pad, :pleasure), -1.0, 1.0),
      arousal: clamp(get_value(pad, :arousal), -1.0, 1.0),
      dominance: clamp(get_value(pad, :dominance), -1.0, 1.0)
    }
  end

  defp clamp(value, min_val, max_val) do
    value |> max(min_val) |> min(max_val)
  end

  defp get_value(pad, key) when is_map(pad) do
    Map.get(pad, key) || Map.get(pad, Atom.to_string(key)) || 0.0
  end
end
