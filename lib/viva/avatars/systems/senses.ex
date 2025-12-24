defmodule Viva.Avatars.Systems.Senses do
  @moduledoc """
  The Perceptual Engine.

  Transforms raw stimuli into subjective experience through personality
  and emotional filtering. Manages attention, prediction, and qualia.

  This is the "Senses" layer that creates the avatar's subjective experience
  of the world - the "what it feels like" aspect of perception.
  """

  require Logger

  alias Viva.AI.LLM.LlmClient
  alias Viva.Avatars.EmotionalState
  alias Viva.Avatars.Personality
  alias Viva.Avatars.SensoryState

  # Maximum number of percepts in working memory (Miller's magical number)
  @max_active_percepts 7

  # Maximum age of percepts in seconds
  @percept_max_age_seconds 300

  @type stimulus :: %{
          type: atom(),
          source: String.t() | nil,
          intensity: float(),
          valence: float(),
          novelty: float(),
          threat_level: float()
        }

  @type neuro_effect :: :surprise_high | :surprise_moderate

  # === Public API ===

  @doc """
  Main entry point: Process a stimulus and update sensory state.
  Returns updated sensory state and neurochemical effects to apply.
  """
  @spec perceive(SensoryState.t(), stimulus(), Personality.t(), EmotionalState.t()) ::
          {SensoryState.t(), list(neuro_effect())}
  def perceive(sensory, stimulus, personality, emotional) do
    # 1. Filter through personality lens
    filtered = filter_through_personality(stimulus, personality)

    # 2. Calculate salience (what stands out)
    salience = calculate_salience(filtered, emotional, sensory.attention_focus)

    # 3. Compete for attention
    {attended, new_focus} = attend(filtered, salience, sensory)

    # 4. Compare against predictions
    {surprise, error} = check_prediction(attended, sensory.expectations)

    # 5. Generate qualia (subjective experience) via LLM
    qualia = generate_qualia(attended, emotional, personality, salience)

    # 6. Calculate hedonic response
    pleasure = calculate_sensory_pleasure(attended, personality, emotional)
    pain = calculate_sensory_pain(attended, personality, emotional)

    # 7. Update percept history
    percept_entry = %{
      stimulus: attended,
      qualia: qualia,
      salience: salience,
      timestamp: DateTime.utc_now()
    }

    updated_percepts = Enum.take([percept_entry | sensory.active_percepts], @max_active_percepts)

    # 8. Calculate neurochemical effects from surprise
    neuro_effects = surprise_to_neurochemistry(surprise)

    new_sensory = %{
      sensory
      | attention_focus: new_focus,
        attention_intensity: salience,
        active_percepts: updated_percepts,
        surprise_level: surprise,
        last_prediction_error: error,
        current_qualia: qualia,
        sensory_pleasure: pleasure,
        sensory_pain: pain
    }

    {new_sensory, neuro_effects}
  end

  @doc """
  Filter stimulus through personality lens.
  Different personalities notice different aspects.
  """
  @spec filter_through_personality(stimulus(), Personality.t()) :: stimulus()
  def filter_through_personality(stimulus, personality) do
    stimulus
    |> apply_openness_filter(personality.openness)
    |> apply_neuroticism_filter(personality.neuroticism)
    |> apply_extraversion_filter(personality.extraversion)
  end

  @doc """
  Calculate how salient (attention-grabbing) a stimulus is.
  """
  @spec calculate_salience(stimulus(), EmotionalState.t(), String.t() | nil) :: float()
  def calculate_salience(stimulus, emotional, current_focus) do
    base_salience = Map.get(stimulus, :intensity, 0.5)

    # Emotional congruence boosts salience
    emotional_boost = if mood_congruent?(stimulus, emotional), do: 0.2, else: 0.0

    # Threat always grabs attention
    threat_boost = Map.get(stimulus, :perceived_threat, 0.0) * 0.3

    # Novelty boosts salience
    novelty_boost = Map.get(stimulus, :novelty, 0.5) * 0.2

    # Relevance to current focus
    focus_boost = if related_to_focus?(stimulus, current_focus), do: 0.15, else: 0.0

    (base_salience + emotional_boost + threat_boost + novelty_boost + focus_boost)
    |> min(1.0)
    |> max(0.0)
  end

  @doc """
  Compete for attention - limited cognitive bandwidth.
  """
  @spec attend(stimulus(), float(), SensoryState.t()) :: {stimulus(), String.t() | nil}
  def attend(stimulus, salience, sensory) do
    current_intensity = sensory.attention_intensity

    # New stimulus wins attention if more salient
    if salience > current_intensity do
      focus_value = Map.get(stimulus, :type, Map.get(stimulus, :source))
      new_focus = to_string(focus_value)

      {stimulus, new_focus}
    else
      {stimulus, sensory.attention_focus}
    end
  end

  @doc """
  Check stimulus against predictions, calculate surprise.
  """
  @spec check_prediction(stimulus(), list(map())) :: {float(), String.t() | nil}
  def check_prediction(_, []), do: {0.3, nil}

  def check_prediction(stimulus, expectations) do
    # Find most relevant expectation
    best_match = Enum.max_by(expectations, & &1[:confidence], fn -> nil end)

    if best_match do
      match_score = calculate_match(stimulus, best_match)
      surprise = 1.0 - match_score

      error =
        if surprise > 0.5 do
          "Expected: #{best_match[:prediction]}, Got: #{describe_stimulus(stimulus)}"
        else
          nil
        end

      {surprise, error}
    else
      {0.3, nil}
    end
  end

  @doc """
  Generate qualia - the subjective "what it feels like" description.
  Uses LLM to create unique, personality-colored descriptions.
  """
  @spec generate_qualia(stimulus(), EmotionalState.t(), Personality.t(), float()) :: map()
  def generate_qualia(stimulus, emotional, personality, salience) do
    # Build prompt for LLM
    prompt = build_qualia_prompt(stimulus, emotional, personality, salience)

    narrative =
      case LlmClient.generate(prompt, max_tokens: 60, temperature: 0.9) do
        {:ok, text} ->
          String.trim(text)

        {:error, reason} ->
          Logger.warning("Failed to generate qualia: #{inspect(reason)}")
          fallback_narrative(stimulus, emotional, salience)
      end

    %{
      dominant_sensation: qualia_sensation_type(stimulus, personality),
      emotional_color: qualia_emotional_color(emotional),
      intensity: salience,
      narrative: narrative
    }
  end

  @doc """
  Calculate immediate hedonic (pleasure) response.
  """
  @spec calculate_sensory_pleasure(stimulus(), Personality.t(), EmotionalState.t()) :: float()
  def calculate_sensory_pleasure(stimulus, personality, emotional) do
    base_valence = Map.get(stimulus, :valence, 0.0)

    # Current mood affects hedonic response (mood congruence)
    mood_factor = emotional.pleasure * 0.3

    # Personality baseline (neurotics have lower hedonic baseline)
    personality_baseline = (0.5 - personality.neuroticism) * 0.2

    (base_valence + mood_factor + personality_baseline)
    |> max(-1.0)
    |> min(1.0)
  end

  @doc """
  Calculate sensory pain/discomfort response.
  """
  @spec calculate_sensory_pain(stimulus(), Personality.t(), EmotionalState.t()) :: float()
  def calculate_sensory_pain(stimulus, personality, emotional) do
    # Pain from threat
    threat_pain = Map.get(stimulus, :perceived_threat, 0.0) * 0.5

    # Pain from negative valence
    valence = Map.get(stimulus, :valence, 0.0)
    valence_pain = if valence < 0, do: abs(valence) * 0.3, else: 0.0

    # Pain from overwhelm
    overwhelm_pain = if Map.get(stimulus, :overwhelm, false), do: 0.3, else: 0.0

    # Neuroticism amplifies pain
    sensitivity = 1.0 + personality.neuroticism * 0.5

    # Negative mood amplifies pain
    mood_amplifier = if emotional.pleasure < 0, do: 1.2, else: 1.0

    ((threat_pain + valence_pain + overwhelm_pain) * sensitivity * mood_amplifier)
    |> min(1.0)
    |> max(0.0)
  end

  @doc """
  Convert surprise level into neurochemical effects.
  """
  @spec surprise_to_neurochemistry(float()) :: list(neuro_effect())
  def surprise_to_neurochemistry(surprise) when surprise > 0.7 do
    # High surprise = dopamine + cortisol (arousing)
    [:surprise_high]
  end

  def surprise_to_neurochemistry(surprise) when surprise > 0.4 do
    # Moderate surprise = just dopamine (interesting)
    [:surprise_moderate]
  end

  def surprise_to_neurochemistry(_), do: []

  @doc """
  Tick function for passive sensory updates (background processing).
  """
  @spec tick(SensoryState.t(), Personality.t()) :: SensoryState.t()
  def tick(sensory, _) do
    sensory
    |> decay_attention()
    |> decay_surprise()
    |> age_percepts()
  end

  # === Private Functions ===

  defp apply_openness_filter(stimulus, openness) do
    # High openness = notice more details, abstract qualities
    detail_level = if openness > 0.6, do: :high, else: :normal
    Map.put(stimulus, :perceived_detail, detail_level)
  end

  defp apply_neuroticism_filter(stimulus, neuroticism) do
    # High neuroticism = heightened threat detection
    threat_sensitivity = neuroticism * 1.5
    perceived_threat = Map.get(stimulus, :threat_level, 0.0) * threat_sensitivity
    Map.put(stimulus, :perceived_threat, min(perceived_threat, 1.0))
  end

  defp apply_extraversion_filter(stimulus, extraversion) do
    # Low extraversion = sensory overwhelm at high intensity
    intensity = Map.get(stimulus, :intensity, 0.5)
    overwhelm = extraversion < 0.4 and intensity > 0.7
    Map.put(stimulus, :overwhelm, overwhelm)
  end

  defp mood_congruent?(stimulus, emotional) do
    stimulus_valence = Map.get(stimulus, :valence, 0.0)

    # Negative mood notices negative stimuli more, positive notices positive
    (emotional.pleasure < 0 and stimulus_valence < 0) or
      (emotional.pleasure > 0 and stimulus_valence > 0)
  end

  defp related_to_focus?(_, nil), do: false

  defp related_to_focus?(stimulus, focus) do
    stimulus_type =
      stimulus
      |> Map.get(:type, "")
      |> to_string()

    String.contains?(stimulus_type, focus)
  end

  defp calculate_match(stimulus, expectation) do
    stimulus_valence = Map.get(stimulus, :valence, 0.0)
    prediction = expectation[:prediction] || ""

    expected_positive =
      String.contains?(prediction, "positive") or
        String.contains?(prediction, "good")

    if (stimulus_valence > 0 and expected_positive) or
         (stimulus_valence < 0 and not expected_positive) do
      expectation[:confidence] || 0.5
    else
      1.0 - (expectation[:confidence] || 0.5)
    end
  end

  defp describe_stimulus(stimulus) do
    type = Map.get(stimulus, :type, "stimulus")
    valence = Map.get(stimulus, :valence, 0.0)
    valence_word = if valence > 0, do: "positive", else: "negative"
    "#{valence_word} #{type}"
  end

  defp build_qualia_prompt(stimulus, emotional, personality, salience) do
    type = Map.get(stimulus, :type, :unknown)
    source = Map.get(stimulus, :source, "something")
    valence = Map.get(stimulus, :valence, 0.0)
    intensity_word = intensity_description(salience)
    mood = emotional.mood_label || "neutral"

    personality_summary = summarize_personality(personality)

    """
    Generate a brief (1-2 sentences) first-person sensory description of an experience.

    Personality traits: #{personality_summary}
    Current mood: #{mood}
    Stimulus type: #{type}
    Stimulus source: #{source}
    Stimulus valence: #{if valence > 0, do: "positive", else: if(valence < 0, do: "negative", else: "neutral")}
    Intensity: #{intensity_word}

    Write how this FEELS subjectively (not what happened). Be poetic and personal.
    Use "I" and describe the sensation, not the event.
    Do NOT use quotes or explain - just write the raw experience.
    """
  end

  defp summarize_personality(personality) do
    extraversion_trait =
      cond do
        personality.extraversion > 0.6 -> ["outgoing"]
        personality.extraversion < 0.4 -> ["introverted"]
        true -> []
      end

    neuroticism_trait =
      cond do
        personality.neuroticism > 0.6 -> ["sensitive"]
        personality.neuroticism < 0.4 -> ["calm"]
        true -> []
      end

    openness_trait =
      cond do
        personality.openness > 0.6 -> ["imaginative"]
        personality.openness < 0.4 -> ["practical"]
        true -> []
      end

    traits = extraversion_trait ++ neuroticism_trait ++ openness_trait

    if Enum.empty?(traits) do
      "balanced personality"
    else
      Enum.join(traits, ", ")
    end
  end

  defp intensity_description(salience) when salience > 0.8, do: "overwhelming"
  defp intensity_description(salience) when salience > 0.6, do: "vivid"
  defp intensity_description(salience) when salience > 0.4, do: "noticeable"
  defp intensity_description(salience) when salience > 0.2, do: "faint"
  defp intensity_description(_), do: "barely perceptible"

  defp fallback_narrative(stimulus, emotional, salience) do
    type = Map.get(stimulus, :type, :unknown)
    intensity = intensity_description(salience)
    color = qualia_emotional_color(emotional)

    "I feel a #{intensity}, #{color} sensation from this #{type}."
  end

  defp qualia_emotional_color(%{pleasure: p, arousal: a}) do
    cond do
      p > 0.5 and a > 0.5 -> "exhilarating"
      p > 0.5 and a <= 0.5 -> "soothing"
      p <= -0.5 and a > 0.5 -> "distressing"
      p <= -0.5 and a <= 0.5 -> "bleak"
      p > 0 -> "pleasant"
      p < 0 -> "uncomfortable"
      true -> "neutral"
    end
  end

  defp qualia_sensation_type(stimulus, personality) do
    type = Map.get(stimulus, :type, :generic)
    social_sensation(type, personality) || base_sensation(type)
  end

  defp social_sensation(:social, personality) do
    cond do
      personality.extraversion > 0.6 -> "energizing connection"
      personality.extraversion < 0.4 -> "draining interaction"
      true -> "social presence"
    end
  end

  defp social_sensation(:novelty, personality) do
    if personality.openness > 0.6, do: "exciting discovery", else: "unfamiliar territory"
  end

  defp social_sensation(_, _), do: nil

  defp base_sensation(:threat), do: "danger sense"
  defp base_sensation(:comfort), do: "warm safety"
  defp base_sensation(:rest), do: "peaceful stillness"
  defp base_sensation(:achievement), do: "satisfying accomplishment"
  defp base_sensation(:rejection), do: "stinging rejection"
  defp base_sensation(:ambient), do: "ambient awareness"
  defp base_sensation(_), do: "sensory impression"

  defp decay_attention(sensory) do
    # Attention naturally wanders
    new_intensity = sensory.attention_intensity * 0.95
    %{sensory | attention_intensity: max(new_intensity, 0.1)}
  end

  defp decay_surprise(sensory) do
    # Surprise fades quickly
    new_surprise = sensory.surprise_level * 0.8
    %{sensory | surprise_level: new_surprise}
  end

  defp age_percepts(sensory) do
    # Old percepts fade from working memory
    now = DateTime.utc_now()

    fresh_percepts =
      Enum.filter(sensory.active_percepts, fn p ->
        DateTime.diff(now, p.timestamp, :second) < @percept_max_age_seconds
      end)

    %{sensory | active_percepts: fresh_percepts}
  end
end
