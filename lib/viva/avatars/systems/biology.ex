defmodule Viva.Avatars.Systems.Biology do
  @moduledoc """
  Governs the physiological rules of the avatar.
  Handles hormone decay, circadian rhythm, and homeostatic interactions.
  """

  alias Viva.Avatars.BioState

  # Decay rates per minute (simulated)
  @decay_dopamine 0.05
  @decay_oxytocin 0.02
  # Stress lingers
  @decay_cortisol 0.01
  # Sleep pressure builds slowly
  @build_adenosine 0.005

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

  defp decay_hormones(bio, personality) do
    # Extraverts burn dopamine faster (need more stimulation)
    dopamine_decay = @decay_dopamine * (1.0 + personality.extraversion * 0.5)

    # Neurotics hold onto cortisol longer (slower decay)
    cortisol_decay = @decay_cortisol * (1.0 - personality.neuroticism * 0.5)

    %{
      bio
      | dopamine: max(0.0, bio.dopamine - dopamine_decay),
        oxytocin: max(0.0, bio.oxytocin - @decay_oxytocin),
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

    # High Adenosine (tiredness) suppresses everything slightly
    fatigue_factor = 1.0 - bio.adenosine * 0.3

    %{bio | libido: new_libido, dopamine: bio.dopamine * fatigue_factor}
  end
end
