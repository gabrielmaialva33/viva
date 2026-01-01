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
    base_type =
      cond do
        state.current_conversation ->
          :social

        state.owner_online? ->
          :social_ambient

        state.state.current_activity == :sleeping ->
          :rest

        true ->
          # DYNAMIC: Vary ambient type based on internal state for richer experience
          generate_dynamic_ambient_type(state)
      end

    # RECURRENT PROCESSING: Desire feeds back into perception type
    # When strongly wanting something, attention naturally shifts toward related stimuli
    apply_desire_feedback(base_type, state.state.current_desire)
  end

  # Generate varied ambient stimulus types based on bio and emotional state
  defp generate_dynamic_ambient_type(state) do
    bio = state.state.bio
    emotional = state.state.emotional
    roll = :rand.uniform()

    cond do
      # High adenosine = fatigue → rest-related perception
      bio.adenosine > 0.6 and roll < 0.4 -> :rest
      # Low dopamine + high arousal = threat perception
      bio.dopamine < 0.3 and emotional.arousal > 0.4 and roll < 0.3 -> :threat
      # High openness personality + curiosity → novelty
      bio.dopamine > 0.5 and roll < 0.3 -> :novelty
      # High oxytocin = social longing
      bio.oxytocin > 0.4 and roll < 0.3 -> :social_ambient
      # Default ambient with small chance of insight
      roll < 0.1 -> :insight
      true -> :ambient
    end
  end

  # RPT Feedback: Desires influence what type of stimuli we perceive
  defp apply_desire_feedback(:ambient, :wants_rest), do: :rest
  defp apply_desire_feedback(:ambient, :wants_attention), do: :social_ambient
  defp apply_desire_feedback(:ambient, :wants_stimulation), do: :novelty
  defp apply_desire_feedback(base_type, _), do: base_type

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
        state.owner_online? -> 0.6
        state.state.current_activity == :sleeping -> 0.2
        # INCREASED base ambient intensity
        true -> 0.45
      end

    # Modulate by arousal
    arousal_factor = state.state.emotional.arousal * 0.25

    # DYNAMIC: Random micro-fluctuations create varied experiences
    random_variation = (:rand.uniform() - 0.5) * 0.25

    min(max(base + arousal_factor + random_variation, 0.2), 1.0)
  end

  defp calculate_valence(state) do
    base =
      cond do
        state.current_conversation -> 0.35
        state.owner_online? -> 0.45
        true -> 0.0
      end

    # DYNAMIC: Ambient valence fluctuates based on internal state and random variation
    # This creates hedonic variety (positive AND negative experiences)
    internal_valence_shift = (state.state.emotional.pleasure - 0.5) * 0.15

    # Random fluctuation to create positive and negative experiences
    random_valence = (:rand.uniform() - 0.5) * 0.4

    # Bio-based valence: high cortisol = negative, high dopamine = positive
    bio = state.state.bio
    bio_valence = (bio.dopamine - bio.cortisol) * 0.2

    (base + internal_valence_shift + random_valence + bio_valence)
    |> max(-0.8)
    |> min(0.8)
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
