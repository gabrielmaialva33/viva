defmodule Viva.Avatars.Systems.Dreams do
  @moduledoc """
  The Dream Engine.

  Processes and consolidates experiences during sleep.
  Updates self-model, strengthens important memories,
  and generates dream content.

  Dreams are probabilistic based on the emotional intensity
  of the day's experiences.
  """

  require Logger

  alias Viva.AI.LLM.LlmClient
  alias Viva.Avatars.ConsciousnessState
  alias Viva.Avatars.Memory
  alias Viva.Avatars.SelfModel

  # Minimum emotional intensity to potentially trigger a dream
  @dream_intensity_threshold 0.3

  # Base probability of dreaming (modified by emotional intensity)
  @base_dream_probability 0.4

  # === Public API ===

  @doc """
  Decides if the avatar should dream based on emotional intensity.
  Returns true probabilistically based on the day's emotional content.
  """
  @spec should_dream?(list(map())) :: boolean()
  def should_dream?([]), do: false

  def should_dream?(day_experiences) do
    intensity = calculate_emotional_intensity(day_experiences)

    cond do
      # High emotional day = always dream
      intensity > 0.7 -> true
      # Low emotional day = rarely dream
      intensity < @dream_intensity_threshold -> :rand.uniform() < 0.1
      # Normal = probabilistic based on intensity
      true -> :rand.uniform() < intensity * @base_dream_probability + 0.2
    end
  end

  @doc """
  Calculate the average emotional intensity of experiences.
  """
  @spec calculate_emotional_intensity(list(map())) :: float()
  def calculate_emotional_intensity([]), do: 0.0

  def calculate_emotional_intensity(experiences) do
    total =
      Enum.reduce(experiences, 0.0, fn exp, acc ->
        acc + calculate_experience_intensity(exp)
      end)

    total / length(experiences)
  end

  @doc """
  Main dream cycle entry point.
  Called when avatar falls asleep and should_dream? returns true.
  """
  @spec process_dream_cycle(binary(), ConsciousnessState.t(), list(map())) ::
          {:ok, ConsciousnessState.t(), Memory.t() | nil} | {:error, term()}
  def process_dream_cycle(avatar_id, consciousness, day_experiences) do
    Logger.info("Starting dream cycle for avatar #{avatar_id}")

    # 1. Identify emotionally significant experiences
    significant = identify_significant_experiences(day_experiences)

    # 2. Consolidate memories (mark for strengthening)
    consolidation_notes = consolidate_memories(significant)

    # 3. Process unresolved emotions
    emotional_residue = extract_emotional_residue(day_experiences)

    # 4. Update self-model based on day's experiences
    updated_self_model = update_self_model(consciousness.self_model, day_experiences)

    # 5. Generate dream content
    dream_content = generate_dream_content(avatar_id, emotional_residue, significant)

    # 6. Create dream memory (if dream was generated)
    dream_memory = create_dream_memory(avatar_id, dream_content, consolidation_notes)

    updated_consciousness = %{consciousness | self_model: updated_self_model}

    {:ok, updated_consciousness, dream_memory}
  rescue
    e ->
      Logger.error("Dream cycle failed: #{inspect(e)}")
      {:error, e}
  end

  @doc """
  Identify the most emotionally significant experiences.
  """
  @spec identify_significant_experiences(list(map())) :: list(map())
  def identify_significant_experiences(experiences) do
    experiences
    |> Enum.map(fn exp ->
      intensity = calculate_experience_intensity(exp)
      Map.put(exp, :significance, intensity)
    end)
    |> Enum.filter(&(&1.significance > 0.5))
    |> Enum.sort_by(& &1.significance, :desc)
    |> Enum.take(5)
  end

  @doc """
  Consolidate memories based on day's experiences.
  Returns notes about what was consolidated for inclusion in dream memory.
  """
  @spec consolidate_memories(list(map())) :: String.t()
  def consolidate_memories([]), do: "No significant experiences to consolidate."

  def consolidate_memories(significant_experiences) do
    summaries =
      significant_experiences
      |> Enum.take(3)
      |> Enum.map_join("; ", &summarize_experience/1)

    "Consolidated memories: #{summaries}"
  end

  @doc """
  Extract unresolved emotional content for dream processing.
  """
  @spec extract_emotional_residue(list(map())) :: map()
  def extract_emotional_residue(experiences) do
    # Find experiences with strong negative emotions that weren't resolved
    unresolved =
      Enum.filter(experiences, fn exp ->
        emotion = Map.get(exp, :emotion, %{})
        pleasure = Map.get(emotion, :pleasure, 0.0)
        # Negative experiences
        pleasure < -0.3
      end)

    intensity =
      if Enum.empty?(unresolved) do
        0.0
      else
        Enum.reduce(unresolved, 0.0, fn exp, acc ->
          emotion = Map.get(exp, :emotion, %{})
          acc + abs(Map.get(emotion, :pleasure, 0.0))
        end) / length(unresolved)
      end

    %{
      negative_experiences: Enum.take(unresolved, 3),
      dominant_negative_theme: identify_theme(unresolved),
      intensity: intensity
    }
  end

  @doc """
  Update self-model based on day's experiences.
  """
  @spec update_self_model(SelfModel.t() | nil, list(map())) :: SelfModel.t()
  def update_self_model(nil, _), do: SelfModel.new()

  def update_self_model(self_model, experiences) do
    # Learn behavioral patterns (simplified)
    new_patterns = detect_new_patterns(experiences, self_model.behavioral_patterns)

    # Adjust self-esteem based on day's outcomes
    esteem_delta = calculate_esteem_delta(experiences)

    clamped_delta =
      esteem_delta
      |> max(-0.1)
      |> min(0.1)

    new_esteem =
      (self_model.self_esteem + clamped_delta)
      |> max(0.1)
      |> min(0.9)

    # Update emotional patterns
    new_emotional_patterns = update_emotional_patterns(experiences, self_model.emotional_patterns)

    %{
      self_model
      | behavioral_patterns: new_patterns,
        self_esteem: new_esteem,
        emotional_patterns: new_emotional_patterns
    }
  end

  @doc """
  Generate dream content from emotional residue.
  Uses LLM to create surreal, symbolic dream narratives.
  """
  @spec generate_dream_content(binary(), map(), list(map())) :: String.t() | nil
  def generate_dream_content(_, emotional_residue, significant_experiences) do
    if emotional_residue.intensity < 0.2 and Enum.empty?(significant_experiences) do
      # Light sleep, no memorable dream
      nil
    else
      prompt = build_dream_prompt(emotional_residue, significant_experiences)

      case LlmClient.generate(prompt, max_tokens: 200, temperature: 1.0) do
        {:ok, content} ->
          String.trim(content)

        {:error, reason} ->
          Logger.warning("Failed to generate dream content: #{inspect(reason)}")
          generate_fallback_dream(emotional_residue)
      end
    end
  end

  @doc """
  Create a memory record for the dream.
  """
  @spec create_dream_memory(binary(), String.t() | nil, String.t()) :: Memory.t() | nil
  def create_dream_memory(_, nil, _), do: nil

  def create_dream_memory(avatar_id, dream_content, consolidation_notes) do
    %Memory{
      avatar_id: avatar_id,
      content: dream_content,
      summary: "A dream during sleep",
      type: :dream,
      importance: 0.4,
      strength: 0.6,
      context: %{
        source: "dream_cycle",
        consolidation: consolidation_notes
      }
    }
  end

  @doc """
  Light processing for when avatar sleeps but doesn't dream.
  Just does passive recovery without full dream cycle.
  """
  @spec light_sleep_processing(ConsciousnessState.t()) :: ConsciousnessState.t()
  def light_sleep_processing(consciousness) do
    # Just clear the experience stream and reset temporal focus
    %{
      consciousness
      | experience_stream: [],
        temporal_focus: :present,
        meta_observation: nil,
        focal_content: %{type: nil, content: nil, source: nil}
    }
  end

  # === Private Functions ===

  defp calculate_experience_intensity(experience) do
    emotion = Map.get(experience, :emotion, %{})
    pleasure = Map.get(emotion, :pleasure, 0.0)
    arousal = Map.get(emotion, :arousal, 0.0)
    surprise = Map.get(experience, :surprise, 0.0)

    min(abs(pleasure) * 0.4 + arousal * 0.3 + surprise * 0.3, 1.0)
  end

  defp summarize_experience(experience) do
    emotion = Map.get(experience, :emotion, %{})
    mood = Map.get(emotion, :mood, "neutral")
    qualia = Map.get(experience, :qualia, %{})
    narrative = Map.get(qualia, :narrative, "an experience")

    # Truncate narrative if too long
    short_narrative =
      if String.length(narrative) > 50 do
        String.slice(narrative, 0, 47) <> "..."
      else
        narrative
      end

    "#{short_narrative} (felt #{mood})"
  end

  defp identify_theme([]), do: nil

  defp identify_theme(experiences) do
    emotions =
      Enum.map(experiences, fn e ->
        get_in(e, [:emotion, :mood]) || "unknown"
      end)

    emotions
    |> Enum.frequencies()
    |> Enum.max_by(fn {_, count} -> count end, fn -> {"unknown", 0} end)
    |> elem(0)
  end

  defp detect_new_patterns(experiences, existing_patterns) do
    # Simplified pattern detection
    # In a full implementation, this would use more sophisticated analysis

    # Look for repeated emotional responses
    emotion_counts =
      experiences
      |> Enum.map(fn e -> get_in(e, [:emotion, :mood]) end)
      |> Enum.reject(&is_nil/1)
      |> Enum.frequencies()

    # If any emotion appeared 3+ times, that's a pattern
    new_pattern =
      emotion_counts
      |> Enum.filter(fn {_, count} -> count >= 3 end)
      |> Enum.map(fn {emotion, count} ->
        %{
          trigger: "daily experiences",
          response: emotion,
          frequency: count
        }
      end)

    # Merge with existing, keeping max 10 patterns
    (new_pattern ++ existing_patterns)
    |> Enum.uniq_by(& &1.response)
    |> Enum.take(10)
  end

  defp calculate_esteem_delta(experiences) do
    # Positive experiences boost esteem, negative ones reduce it
    Enum.reduce(experiences, 0.0, fn exp, acc ->
      pleasure = get_in(exp, [:emotion, :pleasure]) || 0.0
      # Small adjustments
      acc + pleasure * 0.01
    end)
  end

  defp update_emotional_patterns(experiences, existing) do
    # Track how we typically react to situations
    # This is simplified - would need more context about situations

    # Find the dominant emotional response
    dominant =
      experiences
      |> Enum.map(fn e -> get_in(e, [:emotion, :mood]) end)
      |> Enum.reject(&is_nil/1)
      |> Enum.frequencies()
      |> Enum.max_by(fn {_, c} -> c end, fn -> nil end)

    case dominant do
      nil ->
        existing

      {emotion, _} ->
        new_pattern = %{situation: "general daily life", typical_emotion: emotion}

        # Add or update pattern, keep max 5
        updated_list = [
          new_pattern | Enum.reject(existing, &(&1.situation == "general daily life"))
        ]

        Enum.take(updated_list, 5)
    end
  end

  defp build_dream_prompt(residue, experiences) do
    theme = residue.dominant_negative_theme || "wandering"

    exp_summaries =
      experiences
      |> Enum.take(3)
      |> Enum.map_join(", ", fn e ->
        get_in(e, [:qualia, :narrative]) || "something happened"
      end)

    intensity_desc =
      cond do
        residue.intensity > 0.7 -> "intense and vivid"
        residue.intensity > 0.4 -> "moderately vivid"
        true -> "subtle and fleeting"
      end

    """
    Generate a brief, surreal dream that processes emotional experiences.

    Dominant emotional theme: #{theme}
    Key experiences from the day: #{exp_summaries}
    Dream intensity: #{intensity_desc}

    The dream should be:
    - Symbolic and metaphorical (not literal replays)
    - Slightly disjointed like real dreams
    - Processing the emotional content in abstract ways
    - 2-3 sentences maximum
    - Written in first person ("I...")

    Write only the dream content, no explanation or quotes.
    """
  end

  defp generate_fallback_dream(residue) do
    theme = residue.dominant_negative_theme || "wandering"

    case theme do
      "anxious" -> "I find myself running through endless corridors, doors leading to more doors."
      "sad" -> "Rain falls silently in an empty room. I watch it without understanding why."
      "angry" -> "Something breaks, shatters into countless pieces. I can't tell what it was."
      "stressed" -> "Papers swirl around me, each one demanding attention I cannot give."
      _ -> "I drift through familiar places that feel strange, as if seen for the first time."
    end
  end
end
