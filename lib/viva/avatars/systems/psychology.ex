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
    # 1. Calculate raw PAD vectors from Hormones
    # REBALANCED: Normalize to neutral center (0.0) instead of positive bias
    # Positive chemicals: average of reward signals
    positive_chemicals = (bio.dopamine + bio.oxytocin) / 2.0

    # Negative chemicals: stress + fatigue contribution
    negative_chemicals = bio.cortisol + bio.adenosine * 0.5

    # Raw pleasure centered at 0.0 with defaults:
    # (0.5 + 0.3)/2 - (0.2 + 0.0) - 0.2 = 0.4 - 0.2 - 0.2 = 0.0
    raw_pleasure = positive_chemicals - negative_chemicals - 0.2

    # HEDONIC AMPLIFICATION: Personality affects emotional sensitivity
    # Neurotics have wider emotional swings, stable people more centered
    emotional_sensitivity = 1.0 + personality.neuroticism * 0.5

    # MACRO-FLUCTUATIONS: Larger random variations create hedonic variety
    # This helps create both positive AND negative experiences over time
    # Must be large enough to overcome Allostasis dampening
    macro_fluctuation = (:rand.uniform() - 0.5) * 0.6

    # Apply sensitivity and fluctuation
    pleasure = raw_pleasure * emotional_sensitivity + macro_fluctuation

    # Arousal calculation with amplification
    raw_arousal = bio.dopamine + bio.libido + bio.cortisol - bio.adenosine

    arousal =
      raw_arousal * (1.0 + personality.extraversion * 0.3) +
        (:rand.uniform() - 0.5) * 0.15

    # Dominance is boosted by Testosterone (not yet impl) or Confidence (Low Cortisol)
    dominance = 1.0 - bio.cortisol + personality.extraversion * 0.5

    # 2. Normalize to -1.0 to 1.0 range
    pad = %{
      pleasure: clamp(pleasure, -1.0, 1.0),
      arousal: clamp(arousal, -1.0, 1.0),
      dominance: clamp(dominance, -1.0, 1.0)
    }

    # 3. Determine label
    label = classify_mood(pad)

    %EmotionalState{
      pleasure: pad.pleasure,
      arousal: pad.arousal,
      dominance: pad.dominance,
      mood_label: label
    }
  end

  defp classify_mood(%{pleasure: p, arousal: a, dominance: _}) do
    cond do
      p > 0.5 -> if a > 0.5, do: "excited", else: "relaxed"
      p > 0.0 -> "happy"
      p < -0.5 -> if a > 0.5, do: "angry", else: "depressed"
      p < 0.0 -> if a > 0.0, do: "anxious", else: "sad"
      true -> "neutral"
    end
  end

  defp clamp(n, min, max), do: max(min, min(n, max))
end
