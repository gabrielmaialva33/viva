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
    energy_desc = if internal.bio.adenosine > 0.7, do: "cansado", else: "energizado"
    social_desc = if internal.bio.oxytocin > 0.7, do: "amado", else: "solitario"

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
    Voce e #{avatar.name}. Gere um pensamento espontaneo.

    EXPERIENCIA ATUAL:
    #{experience_narrative}

    #{if qualia_narrative, do: "O QUE VOCE ESTA SENTINDO:\n#{qualia_narrative}\n", else: ""}
    #{if meta_observation, do: "AUTOCONSCIENCIA:\n#{meta_observation}\n", else: ""}
    Contexto:
    - Humor: #{describe_mood(internal.emotional.mood_label)}
    - Sentindo: #{energy_desc} e #{social_desc}
    - Desejo: #{internal.current_desire}
    - Emocao dominante: #{InternalState.dominant_emotion(internal)}

    Gere UM pensamento breve e autentico (maximo 2 frases).
    O pensamento deve refletir o que voce esta experienciando agora.
    Seja genuino a sua personalidade.
    IMPORTANTE: Responda APENAS em portugues brasileiro. Sem aspas.
    """
  end

  defp describe_mood(mood_label), do: mood_label || "neutral"
end
