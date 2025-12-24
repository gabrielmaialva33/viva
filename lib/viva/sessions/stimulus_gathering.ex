defmodule Viva.Sessions.StimulusGathering do
  @moduledoc """
  Handles gathering environmental stimuli for avatar perception.
  Extracted from LifeProcess to reduce module dependencies.
  """

  @type process_state :: map()
  @type stimulus :: map()

  @doc """
  Gather current environmental stimulus based on avatar's situation.
  """
  @spec gather(process_state()) :: stimulus()
  def gather(process_state) do
    base_stimulus = %{
      type: determine_type(process_state),
      source: determine_source(process_state),
      intensity: calculate_intensity(process_state),
      valence: calculate_valence(process_state),
      novelty: calculate_novelty(process_state),
      threat_level: 0.0
    }

    # Add interaction-specific details if in conversation
    if process_state.current_conversation do
      Map.merge(base_stimulus, %{
        social_context: :conversation,
        partner_id: process_state.current_conversation
      })
    else
      base_stimulus
    end
  end

  # === Private Functions ===

  defp determine_type(state) do
    cond do
      state.current_conversation -> :social
      state.owner_online? -> :social_ambient
      state.state.current_activity == :sleeping -> :rest
      true -> :ambient
    end
  end

  defp determine_source(state) do
    cond do
      state.current_conversation -> "conversation_partner"
      state.owner_online? -> "owner_presence"
      true -> "environment"
    end
  end

  defp calculate_intensity(state) do
    base =
      cond do
        state.current_conversation -> 0.7
        state.owner_online? -> 0.5
        state.state.current_activity == :sleeping -> 0.1
        true -> 0.3
      end

    # Modulate by arousal
    arousal_factor = state.state.emotional.arousal * 0.2
    min(base + arousal_factor, 1.0)
  end

  defp calculate_valence(state) do
    cond do
      state.current_conversation -> 0.3
      state.owner_online? -> 0.4
      true -> 0.0
    end
  end

  defp calculate_novelty(state) do
    # Higher novelty if owner just connected or conversation just started
    if state.owner_online? and state.tick_count < 3 do
      0.7
    else
      0.3
    end
  end
end
