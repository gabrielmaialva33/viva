defmodule Viva.Embodied.Perception do
  @moduledoc """
  VIVA's perception system - transforms sensory input into memories and emotions.

  This module bridges the gap between raw senses (vision, reading, hearing)
  and VIVA's inner world (HRR memories, PAD emotions).

  Pipeline:
  1. Sense → Raw data from NIMs
  2. Percept → Structured perception
  3. Memory → HRR vector binding
  4. Emotion → PAD state update
  """

  use GenServer
  require Logger

  alias Viva.Embodied.Senses

  # Perception state
  defstruct [
    :last_percept,
    :perception_history,
    :attention_focus,
    :emotional_baseline,
    perception_count: 0
  ]

  # ============================================================================
  # PUBLIC API
  # ============================================================================

  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @doc """
  Perceive an image and integrate it into VIVA's consciousness.
  Returns the percept and emotional response.
  """
  def perceive(image_path) do
    GenServer.call(__MODULE__, {:perceive, image_path}, 60_000)
  end

  @doc """
  Perceive the current screen.
  """
  def perceive_screen do
    GenServer.call(__MODULE__, :perceive_screen, 60_000)
  end

  @doc """
  Listen to audio and integrate it.
  """
  def hear(audio_path) do
    GenServer.call(__MODULE__, {:hear, audio_path}, 60_000)
  end

  @doc """
  Get current attention focus.
  """
  def attention do
    GenServer.call(__MODULE__, :get_attention)
  end

  @doc """
  Get recent perception history.
  """
  def history(limit \\ 10) do
    GenServer.call(__MODULE__, {:get_history, limit})
  end

  @doc """
  Get emotional impact of recent perceptions.
  """
  def emotional_summary do
    GenServer.call(__MODULE__, :emotional_summary)
  end

  # ============================================================================
  # GENSERVER CALLBACKS
  # ============================================================================

  @impl true
  def init(_opts) do
    Logger.info("[Perception] Starting VIVA's perception system")

    state = %__MODULE__{
      last_percept: nil,
      perception_history: [],
      attention_focus: :idle,
      emotional_baseline: %{valence: 0.0, arousal: 0.3, dominance: 0.5}
    }

    {:ok, state}
  end

  @impl true
  def handle_call({:perceive, image_path}, _from, state) do
    case Senses.perceive(image_path) do
      {:ok, percept} ->
        # Process the percept
        processed = process_percept(percept, state)

        # Update state
        new_state = %{state |
          last_percept: processed,
          perception_history: [processed | Enum.take(state.perception_history, 99)],
          attention_focus: processed.attention_focus,
          perception_count: state.perception_count + 1
        }

        # Create HRR memory vector
        memory_vector = create_memory_vector(processed)

        # Calculate emotional impact
        emotional_impact = calculate_emotional_impact(processed, state.emotional_baseline)

        response = %{
          percept: processed,
          memory_vector: memory_vector,
          emotional_impact: emotional_impact,
          attention: processed.attention_focus
        }

        {:reply, {:ok, response}, new_state}

      {:error, reason} ->
        {:reply, {:error, reason}, state}
    end
  end

  @impl true
  def handle_call(:perceive_screen, _from, state) do
    case Senses.perceive_screen() do
      {:ok, percept} ->
        processed = process_percept(percept, state)

        new_state = %{state |
          last_percept: processed,
          perception_history: [processed | Enum.take(state.perception_history, 99)],
          attention_focus: processed.attention_focus,
          perception_count: state.perception_count + 1
        }

        memory_vector = create_memory_vector(processed)
        emotional_impact = calculate_emotional_impact(processed, state.emotional_baseline)

        response = %{
          percept: processed,
          memory_vector: memory_vector,
          emotional_impact: emotional_impact,
          attention: processed.attention_focus
        }

        {:reply, {:ok, response}, new_state}

      {:error, reason} ->
        {:reply, {:error, reason}, state}
    end
  end

  @impl true
  def handle_call({:hear, audio_path}, _from, state) do
    case Senses.hear(audio_path) do
      {:ok, audio_result} ->
        # Think about what was heard
        {:ok, thought} = Senses.think("I heard: #{audio_result.text}")

        processed = %{
          type: :auditory,
          content: audio_result.text,
          language: audio_result.language,
          thought: thought,
          timestamp: DateTime.utc_now(),
          attention_focus: determine_attention(:auditory, audio_result.text)
        }

        new_state = %{state |
          last_percept: processed,
          perception_history: [processed | Enum.take(state.perception_history, 99)],
          attention_focus: processed.attention_focus,
          perception_count: state.perception_count + 1
        }

        memory_vector = create_memory_vector(processed)
        emotional_impact = calculate_emotional_impact(processed, state.emotional_baseline)

        {:reply, {:ok, %{
          percept: processed,
          memory_vector: memory_vector,
          emotional_impact: emotional_impact
        }}, new_state}

      {:error, reason} ->
        {:reply, {:error, reason}, state}
    end
  end

  @impl true
  def handle_call(:get_attention, _from, state) do
    {:reply, state.attention_focus, state}
  end

  @impl true
  def handle_call({:get_history, limit}, _from, state) do
    {:reply, Enum.take(state.perception_history, limit), state}
  end

  @impl true
  def handle_call(:emotional_summary, _from, state) do
    summary = state.perception_history
    |> Enum.take(10)
    |> Enum.reduce(%{valence: 0.0, arousal: 0.0, dominance: 0.0, count: 0}, fn percept, acc ->
      emotion = get_in(percept, [:thought, :emotion]) || %{valence: 0, arousal: 0, dominance: 0}
      %{
        valence: acc.valence + Map.get(emotion, :valence, 0),
        arousal: acc.arousal + Map.get(emotion, :arousal, 0),
        dominance: acc.dominance + Map.get(emotion, :dominance, 0),
        count: acc.count + 1
      }
    end)

    avg = if summary.count > 0 do
      %{
        valence: summary.valence / summary.count,
        arousal: summary.arousal / summary.count,
        dominance: summary.dominance / summary.count
      }
    else
      state.emotional_baseline
    end

    {:reply, %{average: avg, perception_count: state.perception_count}, state}
  end

  # ============================================================================
  # PROCESSING FUNCTIONS
  # ============================================================================

  defp process_percept(raw_percept, state) do
    # Determine what to focus attention on
    attention = determine_attention_from_percept(raw_percept)

    # Check for novelty (compare with recent perceptions)
    novelty = calculate_novelty(raw_percept, state.perception_history)

    # Salience - how important is this perception?
    salience = calculate_salience(raw_percept, novelty)

    %{
      type: :visual,
      visual: raw_percept.visual,
      textual: raw_percept.textual,
      thought: raw_percept.thought,
      timestamp: raw_percept.timestamp,
      source: raw_percept.source,
      attention_focus: attention,
      novelty: novelty,
      salience: salience
    }
  end

  defp determine_attention_from_percept(percept) do
    cond do
      # Code gets high attention
      percept.textual.has_code ->
        :code_analysis

      # Errors get immediate attention
      String.contains?(percept.textual.text || "", ["error", "Error", "ERROR"]) ->
        :error_detection

      # Scene type determines attention
      percept.visual.scene_type == :workspace ->
        :work_observation

      percept.visual.scene_type == :communication ->
        :social_interaction

      percept.visual.scene_type == :entertainment ->
        :passive_observation

      true ->
        :general_awareness
    end
  end

  defp determine_attention(type, content) do
    case type do
      :auditory ->
        cond do
          String.contains?(content, ["VIVA", "viva", "hey"]) -> :direct_address
          String.contains?(content, ["help", "?", "how"]) -> :help_request
          true -> :listening
        end

      _ ->
        :general_awareness
    end
  end

  defp calculate_novelty(percept, history) do
    if Enum.empty?(history) do
      1.0  # First perception is novel
    else
      # Compare with recent perceptions
      recent = Enum.take(history, 5)

      similarities = Enum.map(recent, fn past ->
        compare_percepts(percept, past)
      end)

      avg_similarity = Enum.sum(similarities) / length(similarities)
      1.0 - avg_similarity  # Novelty is inverse of similarity
    end
  end

  defp compare_percepts(a, b) do
    # Simple comparison based on scene type and dominant visual
    scene_match = if a.visual.scene_type == get_in(b, [:visual, :scene_type]), do: 0.3, else: 0.0
    dominant_match = if a.visual.dominant == get_in(b, [:visual, :dominant]), do: 0.3, else: 0.0

    # Text similarity (jaccard)
    text_a = MapSet.new(String.split(a.textual.text || "", ~r/\s+/))
    text_b = MapSet.new(String.split(get_in(b, [:textual, :text]) || "", ~r/\s+/))

    text_sim = if MapSet.size(text_a) > 0 and MapSet.size(text_b) > 0 do
      intersection = MapSet.intersection(text_a, text_b) |> MapSet.size()
      union = MapSet.union(text_a, text_b) |> MapSet.size()
      0.4 * (intersection / max(union, 1))
    else
      0.0
    end

    scene_match + dominant_match + text_sim
  end

  defp calculate_salience(percept, novelty) do
    # Salience factors
    novelty_weight = novelty * 0.3

    # Emotional intensity
    emotion = percept.thought.emotion
    emotional_intensity = abs(emotion.valence) * 0.3 + emotion.arousal * 0.2

    # Action suggestion importance
    action_weight = case percept.thought.action_suggestion do
      :alert -> 0.3
      :offer_help -> 0.2
      :celebrate -> 0.15
      :empathize -> 0.15
      _ -> 0.0
    end

    min(novelty_weight + emotional_intensity + action_weight, 1.0)
  end

  # ============================================================================
  # MEMORY INTEGRATION (HRR Vectors)
  # ============================================================================

  @doc """
  Create an HRR memory vector from a percept.
  This can be stored in VIVA's holographic memory.
  """
  def create_memory_vector(percept) do
    # HRR dimension (must match VIVA's memory system)
    dim = 512

    # Create component vectors
    visual_vec = hash_to_vector(percept.visual.dominant, dim)
    scene_vec = hash_to_vector(to_string(percept.visual.scene_type), dim)
    text_vec = text_to_vector(get_text(percept), dim)
    emotion_vec = emotion_to_vector(percept.thought.emotion, dim)
    time_vec = time_to_vector(percept.timestamp, dim)

    # Bind components using circular convolution (simplified as element-wise for now)
    # In full HRR: convolution in frequency domain
    memory = 0..(dim - 1)
    |> Enum.map(fn i ->
      (Enum.at(visual_vec, i, 0.0) +
       Enum.at(scene_vec, i, 0.0) * 0.8 +
       Enum.at(text_vec, i, 0.0) * 0.6 +
       Enum.at(emotion_vec, i, 0.0) * 0.4 +
       Enum.at(time_vec, i, 0.0) * 0.2) / 3.0
    end)

    # Normalize
    norm = :math.sqrt(Enum.sum(Enum.map(memory, fn x -> x * x end)))
    if norm > 0 do
      Enum.map(memory, fn x -> x / norm end)
    else
      memory
    end
  end

  defp get_text(percept) do
    cond do
      Map.has_key?(percept, :textual) -> percept.textual.text || ""
      Map.has_key?(percept, :content) -> percept.content || ""
      true -> ""
    end
  end

  defp hash_to_vector(string, dim) do
    # Create pseudo-random vector from string hash
    seed = :erlang.phash2(string)
    :rand.seed(:exsplus, {seed, seed, seed})

    1..dim
    |> Enum.map(fn _ -> :rand.normal() end)
    |> normalize_vector()
  end

  defp text_to_vector(text, dim) do
    # Bag of words to vector
    words = String.split(text, ~r/\s+/)
    |> Enum.take(100)

    if Enum.empty?(words) do
      List.duplicate(0.0, dim)
    else
      # Sum word vectors
      word_vecs = Enum.map(words, &hash_to_vector(&1, dim))

      0..(dim - 1)
      |> Enum.map(fn i ->
        Enum.sum(Enum.map(word_vecs, fn v -> Enum.at(v, i, 0.0) end)) / length(words)
      end)
      |> normalize_vector()
    end
  end

  defp emotion_to_vector(emotion, dim) do
    # Encode PAD as periodic functions
    v = Map.get(emotion, :valence, 0.0)
    a = Map.get(emotion, :arousal, 0.5)
    d = Map.get(emotion, :dominance, 0.5)

    0..(dim - 1)
    |> Enum.map(fn i ->
      phase = 2 * :math.pi() * i / dim
      v * :math.sin(phase) + a * :math.cos(phase * 2) + d * :math.sin(phase * 3)
    end)
    |> normalize_vector()
  end

  defp time_to_vector(timestamp, dim) do
    # Encode time cyclically (hour of day, day of week)
    hour = timestamp.hour / 24.0 * 2 * :math.pi()
    day = Date.day_of_week(timestamp) / 7.0 * 2 * :math.pi()

    0..(dim - 1)
    |> Enum.map(fn i ->
      phase = 2 * :math.pi() * i / dim
      :math.sin(hour + phase) * 0.5 + :math.cos(day + phase) * 0.5
    end)
    |> normalize_vector()
  end

  defp normalize_vector(vec) do
    norm = :math.sqrt(Enum.sum(Enum.map(vec, fn x -> x * x end)))
    if norm > 0 do
      Enum.map(vec, fn x -> x / norm end)
    else
      vec
    end
  end

  # ============================================================================
  # EMOTIONAL IMPACT
  # ============================================================================

  defp calculate_emotional_impact(percept, baseline) do
    thought_emotion = percept.thought.emotion

    # Delta from baseline
    delta = %{
      valence: thought_emotion.valence - baseline.valence,
      arousal: thought_emotion.arousal - baseline.arousal,
      dominance: thought_emotion.dominance - baseline.dominance
    }

    # Intensity of change
    intensity = :math.sqrt(
      delta.valence * delta.valence +
      delta.arousal * delta.arousal +
      delta.dominance * delta.dominance
    )

    %{
      current: thought_emotion,
      delta: delta,
      intensity: intensity,
      significant: intensity > 0.3
    }
  end
end
