defmodule Viva.Avatars.Systems.Psychology do
  @moduledoc """
  Translates biological state into psychological/emotional state using the PAD model.
  PAD: Pleasure, Arousal, Dominance.
  """

  alias Viva.Avatars.BioState
  alias Viva.Avatars.EmotionalState
  alias Viva.Avatars.Personality

  @doc """
  Calculates the new emotional state based on biology and personality.
  """
  @spec calculate_emotional_state(BioState.t(), Personality.t()) :: EmotionalState.t()
  def calculate_emotional_state(%BioState{} = bio, %Personality{} = personality) do
    # 1. SATISFACTION SIGNALS (balanced contribution)
    # Positive chemicals contribute to well-being with a reasonable cap
    # BALANCED: Increased cap from 0.5 to 0.75 for fairer positive emotions
    satisfaction_raw = bio.dopamine * 0.4 + bio.oxytocin * 0.4
    satisfaction_signal = min(0.75, satisfaction_raw)

    # 2. DISTRESS SIGNALS (balanced - no longer amplified)
    # Stress and fatigue contribute proportionally to displeasure
    # BALANCED: Reduced stress multiplier from 1.2 to 1.0 for symmetry
    stress_signal = bio.cortisol * 1.0
    fatigue_signal = bio.adenosine * 0.5

    # 3. NEED DEPRIVATION PAIN (drives action through discomfort)
    # When needs fall below threshold, creates negative valence
    # This is the critical mechanism for authentic suffering
    # BALANCED: Reduced deprivation multipliers for less punitive experience
    # - dopamine_deprivation: 0.5 -> 0.3 (less harsh)
    # - oxytocin_deprivation: 0.6 -> 0.4 (less harsh)
    dopamine_deprivation = max(0.0, 0.4 - bio.dopamine) * 0.3
    oxytocin_deprivation = max(0.0, 0.35 - bio.oxytocin) * 0.4
    deprivation_pain = dopamine_deprivation + oxytocin_deprivation

    # 4. RAW PLEASURE (symmetric: both positive and negative at full strength)
    # BALANCED: Satisfaction multiplier increased from 0.8 to 1.0
    # BALANCED: Removed -0.1 baseline bias that unfairly pushed toward negativity
    raw_pleasure =
      satisfaction_signal * 1.0 -
        (stress_signal + fatigue_signal + deprivation_pain)

    # HEDONIC AMPLIFICATION: Personality affects emotional sensitivity
    emotional_sensitivity = 1.0 + personality.neuroticism * 0.5

    # MACRO-FLUCTUATIONS: Reduced from 0.6 to 0.25 for more stable valence
    macro_fluctuation = (:rand.uniform() - 0.5) * 0.25

    # Apply sensitivity and fluctuation
    pleasure = raw_pleasure * emotional_sensitivity + macro_fluctuation

    # Arousal calculation with amplification
    raw_arousal = bio.dopamine + bio.libido + bio.cortisol - bio.adenosine

    arousal =
      raw_arousal * (1.0 + personality.extraversion * 0.3) +
        (:rand.uniform() - 0.5) * 0.15

    # Dominance: low cortisol = confidence, high stress = submissive
    dominance = 1.0 - bio.cortisol + personality.extraversion * 0.5

    # Normalize to -1.0 to 1.0 range
    pad = %{
      pleasure: clamp(pleasure, -1.0, 1.0),
      arousal: clamp(arousal, -1.0, 1.0),
      dominance: clamp(dominance, -1.0, 1.0)
    }

    # Determine mood label from PAD values
    label = classify_mood(pad)

    %EmotionalState{
      pleasure: pad.pleasure,
      arousal: pad.arousal,
      dominance: pad.dominance,
      mood_label: label
    }
  end

  defp classify_mood(%{pleasure: p, arousal: a, dominance: d}) do
    cond do
      p > 0.15 -> classify_positive_mood(p, a)
      p >= -0.15 -> "neutral"
      p > -0.4 -> classify_mild_negative_mood(a)
      p > -0.65 -> classify_moderate_negative_mood(a, d)
      true -> classify_severe_negative_mood(a)
    end
  end

  # Positive states (p > 0.15)
  defp classify_positive_mood(p, a) when p > 0.5 and a > 0.5, do: "excited"
  defp classify_positive_mood(p, _) when p > 0.5, do: "content"
  defp classify_positive_mood(_, a) when a > 0.4, do: "energized"
  defp classify_positive_mood(_, _), do: "pleasant"

  # Mild negative states (-0.4 < p <= -0.15)
  defp classify_mild_negative_mood(a) when a > 0.4, do: "restless"
  defp classify_mild_negative_mood(_), do: "uncomfortable"

  # Moderate negative states (-0.65 < p <= -0.4)
  defp classify_moderate_negative_mood(a, _) when a > 0.5, do: "anxious"
  defp classify_moderate_negative_mood(_, d) when d < -0.2, do: "defeated"
  defp classify_moderate_negative_mood(_, _), do: "sad"

  # Severe negative states (p <= -0.65)
  defp classify_severe_negative_mood(a) when a > 0.6, do: "distressed"
  defp classify_severe_negative_mood(a) when a < -0.2, do: "depressed"
  defp classify_severe_negative_mood(_), do: "suffering"

  defp clamp(n, min, max), do: max(min, min(n, max))
end
