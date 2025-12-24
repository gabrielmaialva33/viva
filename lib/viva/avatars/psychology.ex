defmodule Viva.Avatars.Psychology do
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
    pleasure = bio.dopamine + bio.oxytocin - bio.cortisol
    arousal = bio.dopamine + bio.libido + bio.cortisol - bio.adenosine
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
