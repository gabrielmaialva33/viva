defmodule Viva.Sessions.ThoughtEngine do
  @moduledoc """
  Handles spontaneous thought generation for avatars.
  Extracted from LifeProcess to reduce module dependencies.
  """

  alias Viva.Avatars.InternalState
  alias Viva.Avatars.Systems.Consciousness

  # Probability of spontaneous thought
  @prob_spontaneous_thought 0.1

  @type process_state :: map()

  @doc """
  Maybe generate a spontaneous thought based on probability and state.
  """
  @spec maybe_think(process_state()) :: process_state()
  def maybe_think(process_state) do
    # Probability increases if desire is strong
    adjusted_prob =
      if process_state.state.current_desire != :none,
        do: @prob_spontaneous_thought * 2.0,
        else: @prob_spontaneous_thought

    should_think =
      :rand.uniform() < adjusted_prob ||
        (process_state.owner_online? && :rand.uniform() < 0.3)

    if should_think do
      generate_thought(process_state)
    else
      process_state
    end
  end

  @doc """
  Generate a thought and publish it via EventBus.
  """
  @spec generate_thought(process_state()) :: process_state()
  def generate_thought(process_state) do
    avatar_id = process_state.avatar_id
    prompt = build_thought_prompt(process_state)

    payload = %{
      type: :spontaneous_thought,
      avatar_id: avatar_id,
      prompt: prompt,
      timestamp: DateTime.utc_now()
    }

    # Fire and forget via EventBus
    event_bus().publish_thought(payload)

    process_state
  end

  # === Private Functions ===

  defp event_bus do
    Application.get_env(:viva, :event_bus, Viva.Infrastructure.EventBus)
  end

  defp build_thought_prompt(process_state) do
    avatar = process_state.avatar
    internal = process_state.state

    # Translate bio-state to feelings
    energy_desc = if internal.bio.adenosine > 0.7, do: "tired", else: "energetic"
    social_desc = if internal.bio.oxytocin > 0.7, do: "loved", else: "lonely"

    # Get experience narrative from consciousness (enriched context)
    experience_narrative =
      Consciousness.synthesize_experience_narrative(
        internal.consciousness,
        internal.sensory,
        internal.emotional
      )

    # Get current qualia if available
    qualia_narrative = InternalState.qualia_narrative(internal)

    # Get metacognitive observation if available
    meta_observation = InternalState.meta_observation(internal)

    """
    You are #{avatar.name}. Generate a single spontaneous thought.

    CURRENT EXPERIENCE:
    #{experience_narrative}

    #{if qualia_narrative, do: "WHAT YOU'RE SENSING:\n#{qualia_narrative}\n", else: ""}
    #{if meta_observation, do: "SELF-AWARENESS:\n#{meta_observation}\n", else: ""}
    Context:
    - Mood: #{describe_mood(internal.emotional.mood_label)}
    - Feeling: #{energy_desc} and #{social_desc}
    - Desire: #{internal.current_desire}
    - Dominant Emotion: #{InternalState.dominant_emotion(internal)}

    Generate ONE brief, authentic thought (max 2 sentences).
    The thought should reflect what you're currently experiencing.
    Be genuine to your personality. No quotes.
    """
  end

  defp describe_mood(mood_label), do: mood_label || "neutral"
end
