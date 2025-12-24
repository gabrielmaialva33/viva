defmodule Viva.Avatars.Neurochemistry do
  @moduledoc """
  Manages the neurochemical reactions to events, interactions, and stimuli.
  Instead of simple "+10 social", we simulate hormonal bursts.
  """

  alias Viva.Avatars.BioState

  @doc """
  Applies a neurochemical effect to the biological state based on an event.

  ## Events
  - :interaction_start - Burst of dopamine and oxytocin
  - :interaction_end - Drop in arousal, slight adenosine increase
  - :rejection - Cortisol spike, dopamine drop
  - :accomplishment - Dopamine spike
  - :resting - Adenosine reduction, cortisol reduction
  """
  @spec apply_effect(BioState.t(), atom()) :: BioState.t()
  def apply_effect(%BioState{} = bio, event) do
    case event do
      :interaction_start ->
        bio
        |> boost(:dopamine, 0.15)
        |> boost(:oxytocin, 0.2)
        # Socializing costs energy
        |> boost(:adenosine, 0.05)
        # Excitement
        |> boost(:libido, 0.05)

      :interaction_ongoing ->
        bio
        |> boost(:oxytocin, 0.05)
        |> boost(:adenosine, 0.02)

      :interaction_end ->
        # Slight crash after interaction
        dampen(bio, :dopamine, 0.05)

      :thought_generated ->
        # Thinking is tiring but rewarding
        bio
        |> boost(:dopamine, 0.02)
        |> boost(:adenosine, 0.01)

      :deep_sleep_tick ->
        bio
        |> dampen(:adenosine, 0.05)
        |> dampen(:cortisol, 0.05)
        # Cleanup
        |> dampen(:dopamine, 0.02)

      :stress_event ->
        bio
        |> boost(:cortisol, 0.3)
        |> dampen(:dopamine, 0.2)
        |> dampen(:oxytocin, 0.1)

      _ ->
        bio
    end
  end

  # Helpers

  defp boost(bio, hormone, amount) do
    current = Map.get(bio, hormone)
    new_val = min(1.0, current + amount)
    Map.put(bio, hormone, new_val)
  end

  defp dampen(bio, hormone, amount) do
    current = Map.get(bio, hormone)
    new_val = max(0.0, current - amount)
    Map.put(bio, hormone, new_val)
  end
end
