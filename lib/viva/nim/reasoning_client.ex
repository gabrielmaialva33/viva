defmodule Viva.Nim.ReasoningClient do
  @moduledoc """
  Advanced reasoning client for complex autonomous decisions.

  Uses `deepseek-ai/deepseek-r1-0528` for:
  - Deep compatibility analysis
  - Relationship conflict resolution
  - Autonomous decision making
  - Complex emotional reasoning

  ## Features

  - Chain-of-thought reasoning
  - Reduced hallucinations
  - Context-aware decisions
  - Structured output
  """
  require Logger

  alias Viva.Nim
  alias Viva.Avatars.{Personality, InternalState, Enneagram}

  @doc """
  Perform deep reasoning with chain-of-thought.

  ## Options

  - `:max_tokens` - Maximum tokens (default: 2000)
  - `:temperature` - Sampling temperature (default: 0.3)
  - `:structured` - Return structured JSON (default: false)
  """
  def reason(prompt, opts \\ []) do
    model = Keyword.get(opts, :model, Nim.model(:reasoning))
    structured = Keyword.get(opts, :structured, false)

    system_prompt =
      if structured do
        """
        You are an advanced reasoning system. Think step by step.
        Always respond with valid JSON only, no markdown.
        """
      else
        """
        You are an advanced reasoning system. Think step by step.
        Show your reasoning process, then provide your conclusion.
        """
      end

    body = %{
      model: model,
      messages: [
        %{role: "system", content: system_prompt},
        %{role: "user", content: prompt}
      ],
      max_tokens: Keyword.get(opts, :max_tokens, 2000),
      temperature: Keyword.get(opts, :temperature, 0.3),
      top_p: 0.95
    }

    case Nim.request("/chat/completions", body, timeout: 180_000) do
      {:ok, %{"choices" => [%{"message" => %{"content" => content}} | _]}} ->
        if structured do
          parse_json_response(content)
        else
          {:ok, content}
        end

      {:error, reason} ->
        Logger.error("Reasoning error: #{inspect(reason)}")
        {:error, reason}
    end
  end

  @doc """
  Deep analysis of compatibility between two avatars.
  Returns detailed breakdown with reasoning.
  """
  def deep_analyze_compatibility(avatar_a, avatar_b) do
    enneagram_a = Enneagram.get_type(avatar_a.personality.enneagram_type)
    enneagram_b = Enneagram.get_type(avatar_b.personality.enneagram_type)
    temp_a = Personality.temperament(avatar_a.personality)
    temp_b = Personality.temperament(avatar_b.personality)

    prompt = """
    Analyze the deep compatibility between these two individuals:

    PERSON A (#{avatar_a.name}):
    - Age: #{avatar_a.age}
    - Enneagram: Type #{enneagram_a.number} - #{enneagram_a.name}
      * Basic fear: #{enneagram_a.basic_fear}
      * Basic desire: #{enneagram_a.basic_desire}
      * Core motivation: #{enneagram_a.motivation}
    - Temperament: #{temp_a}
    - Big Five:
      * Openness: #{avatar_a.personality.openness}
      * Conscientiousness: #{avatar_a.personality.conscientiousness}
      * Extraversion: #{avatar_a.personality.extraversion}
      * Agreeableness: #{avatar_a.personality.agreeableness}
      * Neuroticism: #{avatar_a.personality.neuroticism}
    - Attachment: #{avatar_a.personality.attachment_style}
    - Love language: #{avatar_a.personality.love_language}
    - Interests: #{Enum.join(avatar_a.personality.interests, ", ")}
    - Values: #{Enum.join(avatar_a.personality.values, ", ")}

    PERSON B (#{avatar_b.name}):
    - Age: #{avatar_b.age}
    - Enneagram: Type #{enneagram_b.number} - #{enneagram_b.name}
      * Basic fear: #{enneagram_b.basic_fear}
      * Basic desire: #{enneagram_b.basic_desire}
      * Core motivation: #{enneagram_b.motivation}
    - Temperament: #{temp_b}
    - Big Five:
      * Openness: #{avatar_b.personality.openness}
      * Conscientiousness: #{avatar_b.personality.conscientiousness}
      * Extraversion: #{avatar_b.personality.extraversion}
      * Agreeableness: #{avatar_b.personality.agreeableness}
      * Neuroticism: #{avatar_b.personality.neuroticism}
    - Attachment: #{avatar_b.personality.attachment_style}
    - Love language: #{avatar_b.personality.love_language}
    - Interests: #{Enum.join(avatar_b.personality.interests, ", ")}
    - Values: #{Enum.join(avatar_b.personality.values, ", ")}

    Analyze their compatibility across these dimensions and return JSON:
    {
      "overall_score": 0.0-1.0,
      "romantic_potential": 0.0-1.0,
      "friendship_potential": 0.0-1.0,
      "conflict_risk": 0.0-1.0,
      "growth_potential": 0.0-1.0,
      "dimensions": {
        "emotional_compatibility": {"score": 0.0-1.0, "reasoning": "..."},
        "communication_style": {"score": 0.0-1.0, "reasoning": "..."},
        "values_alignment": {"score": 0.0-1.0, "reasoning": "..."},
        "attachment_dynamics": {"score": 0.0-1.0, "reasoning": "..."},
        "growth_areas": {"score": 0.0-1.0, "reasoning": "..."}
      },
      "strengths": ["..."],
      "challenges": ["..."],
      "advice": "...",
      "predicted_dynamics": "..."
    }
    """

    reason(prompt, structured: true, max_tokens: 3000)
  end

  @doc """
  Resolve a relationship conflict between avatars.
  """
  def resolve_relationship_conflict(relationship, conflict_context) do
    prompt = """
    Two individuals are experiencing conflict in their relationship.

    RELATIONSHIP STATUS:
    - Type: #{relationship.status}
    - Trust level: #{relationship.trust}
    - Affection level: #{relationship.affection}
    - Familiarity: #{relationship.familiarity}
    - Time together: #{relationship.total_interactions} interactions

    CONFLICT CONTEXT:
    #{conflict_context}

    Analyze this conflict and provide resolution guidance:
    {
      "conflict_type": "...",
      "root_cause": "...",
      "each_perspective": {
        "person_a": "...",
        "person_b": "..."
      },
      "misunderstandings": ["..."],
      "resolution_steps": ["..."],
      "healing_actions": {
        "person_a_should": ["..."],
        "person_b_should": ["..."]
      },
      "predicted_outcome": "...",
      "trust_impact": -1.0 to 1.0,
      "affection_impact": -1.0 to 1.0
    }
    """

    reason(prompt, structured: true)
  end

  @doc """
  Make an autonomous decision for an avatar.
  """
  def make_autonomous_decision(avatar, situation, options) do
    enneagram = Enneagram.get_type(avatar.personality.enneagram_type)
    internal = avatar.internal_state

    prompt = """
    You are #{avatar.name}, making a decision based on your personality.

    YOUR PSYCHOLOGY:
    - Enneagram Type #{enneagram.number} (#{enneagram.name})
    - Core fear: #{enneagram.basic_fear}
    - Core desire: #{enneagram.basic_desire}
    - When healthy: #{enneagram.behavior_when_healthy}
    - When stressed: #{enneagram.behavior_when_stressed}

    YOUR CURRENT STATE:
    - Mood: #{internal.mood}
    - Energy: #{internal.energy}%
    - Dominant emotion: #{InternalState.dominant_emotion(internal)}
    - Current desire: #{internal.current_desire}

    SITUATION:
    #{situation}

    OPTIONS:
    #{format_options(options)}

    Based on your personality and current state, which option would you choose?
    Return JSON:
    {
      "chosen_option": 1 to N,
      "reasoning": "Why this choice fits your personality...",
      "emotional_reasoning": "How your current feelings influenced this...",
      "confidence": 0.0-1.0,
      "predicted_outcome": "...",
      "alternative_consideration": "..."
    }
    """

    reason(prompt, structured: true)
  end

  @doc """
  Analyze emotional state and predict behavior.
  """
  def analyze_emotional_trajectory(avatar, recent_events) do
    internal = avatar.internal_state
    enneagram = Enneagram.get_type(avatar.personality.enneagram_type)

    prompt = """
    Analyze the emotional trajectory of #{avatar.name}.

    PERSONALITY:
    - Enneagram Type #{enneagram.number}: #{enneagram.name}
    - Vice: #{enneagram.vice}
    - Virtue: #{enneagram.virtue}
    - Stress behavior: #{enneagram.behavior_when_stressed}
    - Growth behavior: #{enneagram.behavior_when_healthy}

    CURRENT STATE:
    - Mood: #{internal.mood}
    - Energy: #{internal.energy}%
    - Social need: #{internal.social}%
    - Dominant emotion: #{InternalState.dominant_emotion(internal)}
    - Wellbeing: #{InternalState.wellbeing(internal)}

    RECENT EVENTS:
    #{format_events(recent_events)}

    Analyze and predict:
    {
      "current_assessment": {
        "stress_level": 0.0-1.0,
        "health_level": 0.0-1.0,
        "dominant_pattern": "..."
      },
      "trajectory": "improving/stable/declining",
      "predicted_needs": ["..."],
      "recommended_actions": ["..."],
      "risk_factors": ["..."],
      "growth_opportunities": ["..."]
    }
    """

    reason(prompt, structured: true)
  end

  @doc """
  Generate a life decision for the avatar.
  Used for major autonomous choices.
  """
  def life_decision(avatar, decision_type, context) do
    prompt = """
    #{avatar.name} needs to make an important life decision.

    DECISION TYPE: #{decision_type}
    CONTEXT: #{context}

    PERSONALITY SUMMARY:
    #{personality_summary(avatar)}

    What would #{avatar.name} decide, and why?
    Consider their values, fears, and desires.

    Return JSON:
    {
      "decision": "...",
      "reasoning": "...",
      "emotional_weight": 0.0-1.0,
      "confidence": 0.0-1.0,
      "potential_regret": "...",
      "growth_aspect": "..."
    }
    """

    reason(prompt, structured: true)
  end


  defp parse_json_response(response) do
    cleaned =
      response
      |> String.replace(~r/```json\n?/, "")
      |> String.replace(~r/```\n?/, "")
      |> String.trim()

    case Jason.decode(cleaned) do
      {:ok, data} -> {:ok, data}
      {:error, _} -> {:error, :invalid_json}
    end
  end

  defp format_options(options) when is_list(options) do
    options
    |> Enum.with_index(1)
    |> Enum.map(fn {option, idx} -> "#{idx}. #{option}" end)
    |> Enum.join("\n")
  end

  defp format_events(events) when is_list(events) do
    events
    |> Enum.map(fn event ->
      "- #{event[:description] || event}"
    end)
    |> Enum.join("\n")
  end

  defp format_events(_), do: "No recent events"

  defp personality_summary(avatar) do
    enneagram = Enneagram.get_type(avatar.personality.enneagram_type)
    temp = Personality.temperament(avatar.personality)

    """
    - Type #{enneagram.number} (#{enneagram.name}): #{enneagram.motivation}
    - Temperament: #{temp}
    - Values: #{Enum.join(avatar.personality.values, ", ")}
    - Attachment: #{avatar.personality.attachment_style}
    """
  end
end
