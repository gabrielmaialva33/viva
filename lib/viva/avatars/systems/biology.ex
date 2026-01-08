defmodule Viva.Avatars.Systems.Biology do
  @moduledoc """
  Governs the physiological rules of the avatar.
  Handles hormone decay, circadian rhythm, and homeostatic interactions.
  """

  alias Viva.Avatars.BioState

  # Decay rates per minute (simulated) - BALANCED to prevent total collapse
  @decay_dopamine 0.03  # Reduced from 0.08 - was causing dopamine death spiral
  @decay_oxytocin 0.03  # Reduced from 0.05 - social connection persists longer
  # Stress lingers but decays faster now
  @decay_cortisol 0.04
  # Sleep pressure builds slowly
  @build_adenosine 0.01

  # Basal tonus - minimum continuous gain to prevent total collapse
  @basal_dopamine 0.02  # Small continuous dopamine from being alive
  @basal_oxytocin 0.01  # Small social baseline

  # Floor values - never go below these (survival minimum)
  @floor_dopamine 0.15
  @floor_oxytocin 0.10

  @doc """
  Advances the biological state by one tick (minute).
  """
  @spec tick(BioState.t(), Viva.Avatars.Personality.t()) :: BioState.t()
  def tick(%BioState{} = bio, personality) do
    bio
    |> decay_hormones(personality)
    |> accumulate_fatigue()
    |> apply_homeostatic_pressure()
  end

  @doc """
  Apply stimulus effects to biology.
  Stimuli directly modulate neurochemistry for Global Workspace integration.
  """
  @spec apply_stimulus(BioState.t(), map(), Viva.Avatars.Personality.t()) :: BioState.t()
  def apply_stimulus(%BioState{} = bio, stimulus, personality) do
    type = Map.get(stimulus, :type, :ambient)
    intensity = Map.get(stimulus, :intensity, 0.5)
    valence = Map.get(stimulus, :valence, 0.0)

    # Base modulation scaled by intensity
    base_mod = intensity * 0.15

    {dopamine_delta, cortisol_delta, oxytocin_delta} =
      case type do
        :social ->
          # Social: boost oxytocin, dopamine if positive
          oxy = base_mod * (1.0 + personality.agreeableness * 0.3)
          dop = if valence > 0, do: base_mod * 0.8, else: 0.0
          cort = if valence < -0.3, do: base_mod * 0.5, else: 0.0
          {dop, cort, oxy}

        :novelty ->
          # Novelty: boost dopamine (reward), slight cortisol (arousal)
          dop = base_mod * (1.0 + personality.openness * 0.3)
          cort = base_mod * 0.3
          {dop, cort, 0.0}

        :threat ->
          # Threat: spike cortisol, suppress dopamine
          cort = base_mod * 1.5 * (1.0 + personality.neuroticism * 0.5)
          dop = -base_mod * 0.5
          {dop, cort, 0.0}

        :rest ->
          # Rest: reduce cortisol, slight dopamine recovery
          cort = -base_mod * 0.8
          dop = base_mod * 0.3
          {dop, cort, 0.0}

        :achievement ->
          # Achievement: big dopamine boost
          dop = base_mod * 1.2
          {dop, 0.0, base_mod * 0.3}

        :insight ->
          # Insight: dopamine and slight oxytocin (self-connection)
          dop = base_mod * 1.0
          oxy = base_mod * 0.4
          {dop, 0.0, oxy}

        _ ->
          # Ambient: small random fluctuations
          fluct = (:rand.uniform() - 0.5) * 0.05
          {fluct, fluct * 0.5, 0.0}
      end

    # Valence directly affects pleasure chemistry
    valence_dopamine = valence * 0.08
    valence_cortisol = if valence < -0.3, do: abs(valence) * 0.06, else: 0.0

    %{
      bio
      | dopamine: clamp(bio.dopamine + dopamine_delta + valence_dopamine, 0.0, 1.0),
        cortisol: clamp(bio.cortisol + cortisol_delta + valence_cortisol, 0.0, 1.0),
        oxytocin: clamp(bio.oxytocin + oxytocin_delta, 0.0, 1.0)
    }
  end

  defp clamp(value, min_val, max_val), do: max(min_val, min(max_val, value))

  defp decay_hormones(bio, personality) do
    # Extraverts burn dopamine faster (need more stimulation)
    dopamine_decay = @decay_dopamine * (1.0 + personality.extraversion * 0.3)

    # Neurotics hold onto cortisol longer (slower decay)
    cortisol_decay = @decay_cortisol * (1.0 - personality.neuroticism * 0.5)

    # Calculate net dopamine change: decay - basal tonus (being alive generates some dopamine)
    # Net change is decay minus basal gain, with floor protection
    new_dopamine = bio.dopamine - dopamine_decay + @basal_dopamine
    new_oxytocin = bio.oxytocin - @decay_oxytocin + @basal_oxytocin

    %{
      bio
      | dopamine: max(@floor_dopamine, new_dopamine),
        oxytocin: max(@floor_oxytocin, new_oxytocin),
        cortisol: max(0.0, bio.cortisol - cortisol_decay)
    }
  end

  defp accumulate_fatigue(bio) do
    # Adenosine builds up linearly until sleep
    %{bio | adenosine: min(1.0, bio.adenosine + @build_adenosine)}
  end

  defp apply_homeostatic_pressure(bio) do
    # Hormonal interactions

    # High Cortisol kills Libido and suppresses Dopamine
    new_libido = if bio.cortisol > 0.6, do: 0.0, else: bio.libido

    # High Adenosine (tiredness) suppresses dopamine slightly
    # Reduced multiplier from 0.3 to 0.15 to prevent total collapse
    fatigue_factor = 1.0 - bio.adenosine * 0.15

    # Apply fatigue but respect the floor
    fatigued_dopamine = max(@floor_dopamine, bio.dopamine * fatigue_factor)

    %{bio | libido: new_libido, dopamine: fatigued_dopamine}
  end
end
