defmodule Viva.Avatars.Systems.Consciousness do
  @moduledoc """
  The Consciousness Engine.

  Integrates perception, emotion, thought, and memory into unified
  subjective experience. Manages the stream of consciousness,
  self-model, and metacognition.

  This is the "global workspace" where attention, emotion, thought,
  and memory merge into coherent awareness.
  """

  require Logger

  alias Viva.Avatars.BioState
  alias Viva.Avatars.ConsciousnessState
  alias Viva.Avatars.EmotionalState
  alias Viva.Avatars.Personality
  alias Viva.Avatars.SelfModel
  alias Viva.Avatars.SensoryState

  # Maximum moments in experience stream (working memory limit)
  @max_stream_length 10

  # Maximum items in peripheral awareness
  @max_peripheral 5

  # === Public API ===

  @doc """
  Main integration function: Create unified moment of experience.
  Called every tick to update consciousness state.
  """
  @spec integrate(
          ConsciousnessState.t(),
          SensoryState.t(),
          BioState.t(),
          EmotionalState.t(),
          String.t() | nil,
          Personality.t()
        ) :: ConsciousnessState.t()
  def integrate(consciousness, sensory, bio, emotional, thought, personality) do
    # 1. Determine stream tempo based on arousal and fatigue
    tempo = calculate_tempo(emotional.arousal, bio.adenosine)

    # 2. Create this moment's experience
    moment = create_moment(sensory, emotional, thought)

    # 3. Update stream (add new moment)
    new_stream = update_stream(consciousness.experience_stream, moment, tempo)

    # 4. Compete for global workspace (what gets conscious attention)
    {focal, peripheral} =
      update_workspace(
        consciousness.focal_content,
        sensory.current_qualia,
        emotional,
        thought
      )

    # 5. Calculate presence/dissociation
    presence = calculate_presence(emotional, bio, sensory.surprise_level)

    # 6. Calculate experience intensity
    intensity = calculate_intensity(sensory, emotional, presence)

    # 7. Calculate flow state
    flow = calculate_flow(sensory, emotional, consciousness)

    # 8. Maybe generate metacognitive observation
    {meta_awareness, meta_obs} = maybe_metacognate(consciousness, emotional, personality)

    # 9. Check self-congruence
    congruence = check_self_congruence(moment, consciousness.self_model)

    # 10. Calculate temporal focus
    temporal = determine_temporal_focus(thought, emotional)

    %{
      consciousness
      | experience_stream: new_stream,
        stream_tempo: tempo,
        focal_content: focal,
        peripheral_content: peripheral,
        presence_level: presence,
        experience_intensity: intensity,
        flow_state: flow,
        meta_awareness: meta_awareness,
        meta_observation: meta_obs,
        self_congruence: congruence,
        temporal_focus: temporal
    }
  end

  @doc """
  Calculate stream tempo based on arousal and fatigue.
  """
  @spec calculate_tempo(float(), float()) :: atom()
  def calculate_tempo(arousal, adenosine) do
    effective_arousal = arousal - adenosine * 0.5

    cond do
      adenosine > 0.9 -> :frozen
      effective_arousal < -0.5 -> :slow
      effective_arousal > 0.7 -> :racing
      effective_arousal > 0.3 -> :fast
      effective_arousal < -0.2 -> :slow
      true -> :normal
    end
  end

  @doc """
  Create a moment of experience from current state.
  """
  @spec create_moment(SensoryState.t(), EmotionalState.t(), String.t() | nil) :: map()
  def create_moment(sensory, emotional, thought) do
    %{
      timestamp: DateTime.utc_now(),
      qualia: sensory.current_qualia,
      emotion: %{
        pleasure: emotional.pleasure,
        arousal: emotional.arousal,
        dominance: emotional.dominance,
        mood: emotional.mood_label
      },
      thought: thought,
      attention: sensory.attention_focus,
      surprise: sensory.surprise_level,
      intensity: calculate_moment_intensity(sensory, emotional)
    }
  end

  @doc """
  Update stream of consciousness with new moment.
  """
  @spec update_stream(list(map()), map(), atom()) :: list(map())
  def update_stream(stream, moment, tempo) do
    # Tempo affects how many moments we keep
    limit =
      case tempo do
        :frozen -> 3
        :slow -> 5
        :normal -> @max_stream_length
        :fast -> 8
        # Racing mind = less coherent memory
        :racing -> 6
      end

    Enum.take([moment | stream], limit)
  end

  @doc """
  Update global workspace - compete for conscious attention.
  """
  @spec update_workspace(map(), map(), EmotionalState.t(), String.t() | nil) ::
          {map(), list(map())}
  def update_workspace(current_focal, qualia, emotional, thought) do
    # Build candidates for focal attention
    candidates =
      [
        %{type: :perception, content: qualia, salience: qualia[:intensity] || 0.5},
        %{type: :emotion, content: emotional.mood_label, salience: abs(emotional.pleasure)},
        %{type: :thought, content: thought, salience: if(thought, do: 0.7, else: 0.0)}
      ]
      |> Enum.filter(& &1.content)
      |> Enum.sort_by(& &1.salience, :desc)

    case candidates do
      [winner | rest] ->
        peripheral = Enum.take(rest, @max_peripheral)
        {%{type: winner.type, content: winner.content, source: :workspace}, peripheral}

      [] ->
        {current_focal, []}
    end
  end

  @doc """
  Calculate presence level (vs dissociation).
  """
  @spec calculate_presence(EmotionalState.t(), BioState.t(), float()) :: float()
  def calculate_presence(emotional, bio, surprise) do
    # Baseline presence
    base = 0.7

    # Extreme stress can cause dissociation
    stress_penalty = if bio.cortisol > 0.8, do: -0.3, else: 0.0

    # Overwhelming surprise can cause dissociation
    surprise_penalty = if surprise > 0.9, do: -0.2, else: 0.0

    # High arousal generally increases presence
    arousal_bonus = emotional.arousal * 0.15

    # Fatigue decreases presence
    fatigue_penalty = bio.adenosine * -0.2

    (base + stress_penalty + surprise_penalty + arousal_bonus + fatigue_penalty)
    |> max(0.1)
    |> min(1.0)
  end

  @doc """
  Calculate overall experience intensity.
  """
  @spec calculate_intensity(SensoryState.t(), EmotionalState.t(), float()) :: float()
  def calculate_intensity(sensory, emotional, presence) do
    sensory_component = sensory.attention_intensity * 0.3
    emotional_component = abs(emotional.pleasure) * 0.3 + emotional.arousal * 0.2
    presence_component = presence * 0.2

    min(sensory_component + emotional_component + presence_component, 1.0)
  end

  @doc """
  Calculate flow state (absorption in activity).
  """
  @spec calculate_flow(SensoryState.t(), EmotionalState.t(), ConsciousnessState.t()) :: float()
  def calculate_flow(sensory, emotional, consciousness) do
    # Flow requires: focused attention, positive mood, present focus, moderate arousal
    attention_score = sensory.attention_intensity

    # Positive mood helps flow
    mood_score = (emotional.pleasure + 1) / 2

    # Present focus is needed for flow
    temporal_score = if consciousness.temporal_focus == :present, do: 1.0, else: 0.5

    # Moderate arousal is optimal (inverted-U)
    arousal_optimal = 1.0 - abs(emotional.arousal - 0.4) * 2
    arousal_score = max(arousal_optimal, 0.0)

    (attention_score * 0.3 + mood_score * 0.2 + temporal_score * 0.2 + arousal_score * 0.3)
    |> min(1.0)
    |> max(0.0)
  end

  @doc """
  Maybe generate metacognitive observation.
  """
  @spec maybe_metacognate(ConsciousnessState.t(), EmotionalState.t(), Personality.t()) ::
          {float(), String.t() | nil}
  def maybe_metacognate(consciousness, emotional, personality) do
    # Openness and neuroticism increase metacognition
    meta_tendency = personality.openness * 0.5 + personality.neuroticism * 0.3
    current_meta = consciousness.meta_awareness

    # Strong emotions trigger metacognition
    emotional_trigger = abs(emotional.pleasure) > 0.6 or emotional.arousal > 0.7

    new_meta =
      if emotional_trigger do
        min(current_meta + 0.2, 1.0)
      else
        max(current_meta - 0.05, meta_tendency * 0.5)
      end

    observation =
      if new_meta > 0.5 and emotional_trigger do
        generate_meta_observation(emotional, consciousness.focal_content)
      else
        consciousness.meta_observation
      end

    {new_meta, observation}
  end

  @doc """
  Check if current moment is congruent with self-model.
  """
  @spec check_self_congruence(map(), SelfModel.t() | nil) :: float()
  def check_self_congruence(_, nil), do: 0.7

  def check_self_congruence(moment, self_model) do
    # Check if current behavior/emotion aligns with self-image
    base_congruence = 0.7

    # Low self-esteem = more self-doubt, less congruence
    esteem_factor = self_model.self_esteem * 0.2

    # Strong emotions can feel incongruent if unexpected
    emotion_intensity = moment.emotion[:arousal] || 0.0
    intensity_penalty = if emotion_intensity > 0.8, do: -0.1, else: 0.0

    # Small random variation for realism
    variation = (:rand.uniform() - 0.5) * 0.1

    (base_congruence + esteem_factor + intensity_penalty + variation)
    |> max(0.3)
    |> min(1.0)
  end

  @doc """
  Determine temporal focus based on thought content and emotion.
  """
  @spec determine_temporal_focus(String.t() | nil, EmotionalState.t()) :: atom()
  def determine_temporal_focus(nil, _), do: :present

  def determine_temporal_focus(thought, emotional) do
    thought_lower = String.downcase(thought)

    cond do
      String.contains?(thought_lower, ["remember", "was", "used to", "back when", "yesterday"]) ->
        :past

      String.contains?(thought_lower, [
        "will",
        "going to",
        "plan",
        "future",
        "tomorrow",
        "hope",
        "worry"
      ]) ->
        :future

      # Negative mood tends to ruminate on past
      emotional.pleasure < -0.5 ->
        :past

      # Anxiety focuses on future
      emotional.arousal > 0.7 and emotional.pleasure < 0 ->
        :future

      true ->
        :present
    end
  end

  @doc """
  Synthesize the current experience into a narrative for LLM prompts.
  """
  @spec synthesize_experience_narrative(
          ConsciousnessState.t(),
          SensoryState.t(),
          EmotionalState.t()
        ) :: String.t()
  def synthesize_experience_narrative(consciousness, sensory, emotional) do
    presence_desc = presence_description(consciousness.presence_level)
    tempo_desc = tempo_description(consciousness.stream_tempo)

    focal_desc =
      # Handle both atom and string keys (Ecto serialization converts atoms to strings)
      case consciousness.focal_content do
        %{content: %{narrative: n}} when is_binary(n) -> n
        %{"content" => %{"narrative" => n}} when is_binary(n) -> n
        %{content: c} when is_binary(c) -> c
        %{"content" => c} when is_binary(c) -> c
        _ -> "nothing in particular"
      end

    qualia_desc = sensory.current_qualia[:narrative] || "no distinct sensation"

    meta_part =
      if consciousness.meta_observation do
        " #{consciousness.meta_observation}"
      else
        ""
      end

    temporal_part =
      case consciousness.temporal_focus do
        :past -> "My mind is drawn to the past."
        :future -> "My thoughts reach toward what's coming."
        :present -> ""
      end

    String.trim("""
    CURRENT EXPERIENCE:
    Right now, my awareness is #{presence_desc} and my thoughts are #{tempo_desc}.
    I'm primarily focused on: #{focal_desc}.
    Sensation: #{qualia_desc}
    Emotional state: #{emotional.mood_label || "neutral"} (pleasure: #{Float.round(emotional.pleasure, 2)}, arousal: #{Float.round(emotional.arousal, 2)})#{meta_part}
    #{temporal_part}
    """)
  end

  @doc """
  Tick function for consciousness updates.
  """
  @spec tick(ConsciousnessState.t(), Personality.t()) :: ConsciousnessState.t()
  def tick(consciousness, personality) do
    consciousness
    |> decay_meta_awareness(personality)
    |> drift_toward_present()
    |> age_experience_stream()
  end

  # === Private Functions ===

  defp calculate_moment_intensity(sensory, emotional) do
    (sensory.attention_intensity + abs(emotional.pleasure)) / 2
  end

  defp generate_meta_observation(emotional, focal) do
    mood_desc = emotional.mood_label || "something"
    # Handle both atom and string keys (Ecto serialization converts atoms to strings)
    content = focal[:content] || focal["content"]
    focus_desc = if content, do: " while #{describe_focus(focal)}", else: ""

    "I notice I'm feeling #{mood_desc}#{focus_desc}."
  end

  # Handle both atom and string keys from Ecto serialization
  defp describe_focus(focal) do
    type = focal[:type] || focal["type"]
    content = focal[:content] || focal["content"]

    case {type, content} do
      {:perception, %{narrative: n}} when is_binary(n) ->
        "experiencing: #{String.slice(n, 0, 40)}..."

      {"perception", %{"narrative" => n}} when is_binary(n) ->
        "experiencing: #{String.slice(n, 0, 40)}..."

      {:thought, t} when is_binary(t) ->
        "thinking about #{String.slice(t, 0, 30)}..."

      {"thought", t} when is_binary(t) ->
        "thinking about #{String.slice(t, 0, 30)}..."

      {:emotion, c} when is_binary(c) ->
        "feeling #{c}"

      {"emotion", c} when is_binary(c) ->
        "feeling #{c}"

      _ ->
        "noticing my inner state"
    end
  end

  defp presence_description(p) when p > 0.8, do: "fully present and grounded"
  defp presence_description(p) when p > 0.6, do: "mostly present"
  defp presence_description(p) when p > 0.4, do: "somewhat distracted"
  defp presence_description(p) when p > 0.2, do: "feeling distant"
  defp presence_description(_), do: "dissociated and foggy"

  defp tempo_description(:frozen), do: "completely stopped"
  defp tempo_description(:slow), do: "moving slowly and heavily"
  defp tempo_description(:normal), do: "flowing naturally"
  defp tempo_description(:fast), do: "moving quickly"
  defp tempo_description(:racing), do: "racing uncontrollably"

  defp decay_meta_awareness(consciousness, personality) do
    baseline = personality.openness * 0.3
    new_meta = consciousness.meta_awareness * 0.95
    %{consciousness | meta_awareness: max(new_meta, baseline)}
  end

  defp drift_toward_present(consciousness) do
    # Natural tendency to return to present moment
    if :rand.uniform() < 0.3 do
      %{consciousness | temporal_focus: :present}
    else
      consciousness
    end
  end

  defp age_experience_stream(consciousness) do
    # Keep only recent experiences
    now = DateTime.utc_now()

    # Remove experiences older than 10 minutes (600 seconds)
    recent_stream =
      Enum.filter(consciousness.experience_stream, fn exp ->
        DateTime.diff(now, exp.timestamp, :second) < 600
      end)

    %{consciousness | experience_stream: recent_stream}
  end
end
