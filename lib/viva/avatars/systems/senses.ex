defmodule Viva.Avatars.Systems.Senses do
  @moduledoc """
  The Perceptual Engine.

  Transforms raw stimuli into subjective experience through personality
  and emotional filtering. Manages attention, prediction, and qualia.

  This is the "Senses" layer that creates the avatar's subjective experience
  of the world - the "what it feels like" aspect of perception.

  ## Qualia Caching

  To respect NVIDIA NIM rate limits (40 RPM), qualia generation uses a
  multi-level caching strategy:
  1. Exact match cache - reuse identical qualia for same stimulus+emotion
  2. Similar state cache - reuse qualia from similar emotional states
  3. Probabilistic LLM - only generate new qualia ~20% of the time

  This reduces LLM calls from ~15+/tick to ~3/tick while maintaining
  rich subjective experiences.
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

  # Qualia cache settings
  @qualia_cache :viva_cache
  # Shorter TTL for more variety (2 minutes instead of 5)
  @qualia_ttl_seconds 120
  # Higher probability of generating fresh qualia for diversity (35% instead of 20%)
  @qualia_llm_probability 0.35

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

    # 9. RECURRENT PROCESSING: Arousal feeds back into attention
    # High arousal → heightened attention (RPT feedback loop)
    arousal_boost = arousal_attention_feedback(emotional.arousal)
    attention_with_feedback = clamp(salience * arousal_boost, 0.0, 1.0)

    new_sensory = %{
      sensory
      | attention_focus: new_focus,
        attention_intensity: attention_with_feedback,
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
  Apply threat expectation from anxiety state (RPT: synthetic threat hallucinations).

  When the avatar is anxious, they begin to perceive threats that don't exist.
  The threat_expectation value comes from RecurrentProcessor.calculate_threat_expectation.

  This implements the RPT principle that internal states shape perception -
  anxiety literally makes you see danger where there is none.
  """
  @spec apply_threat_expectation(stimulus(), float(), Personality.t()) :: stimulus()
  def apply_threat_expectation(stimulus, threat_expectation, personality)
      when threat_expectation > 0.01 do
    existing_threat = Map.get(stimulus, :threat_level, 0.0)

    # Susceptibility: high neuroticism = more susceptible to hallucinated threats
    susceptibility = 0.5 + personality.neuroticism * 0.5
    effective_synthetic = threat_expectation * susceptibility

    # Add synthetic threat to existing threat
    new_threat = min(existing_threat + effective_synthetic, 1.0)

    # Mark whether this is a hallucinated threat (for debugging/analysis)
    is_hallucinated = existing_threat < 0.1 and effective_synthetic > 0.1

    stimulus
    |> Map.put(:threat_level, new_threat)
    |> Map.put(:is_hallucinated_threat, is_hallucinated)
  end

  def apply_threat_expectation(stimulus, _, _), do: stimulus

  @doc """
  Calculate how salient (attention-grabbing) a stimulus is.
  """
  @spec calculate_salience(stimulus(), EmotionalState.t(), String.t() | nil) :: float()
  def calculate_salience(stimulus, emotional, current_focus) do
    base_salience = Map.get(stimulus, :intensity, 0.5)

    # Emotional congruence boosts salience - INCREASED
    emotional_boost = if mood_congruent?(stimulus, emotional), do: 0.25, else: 0.0

    # Threat always grabs attention - INCREASED
    threat_boost = Map.get(stimulus, :perceived_threat, 0.0) * 0.4

    # Novelty boosts salience - INCREASED
    novelty_boost = Map.get(stimulus, :novelty, 0.5) * 0.25

    # Relevance to current focus - INCREASED
    focus_boost = if related_to_focus?(stimulus, current_focus), do: 0.2, else: 0.0

    # AROUSAL INTEGRATION: High arousal = higher attention (RPT feedback)
    arousal_boost = if emotional.arousal > 0.3, do: emotional.arousal * 0.2, else: 0.0

    (base_salience + emotional_boost + threat_boost + novelty_boost + focus_boost + arousal_boost)
    |> min(1.0)
    # Minimum salience floor to ensure some attention
    |> max(0.2)
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

  ## Caching Strategy

  To respect NVIDIA NIM rate limits, we use a multi-level approach:
  1. Check cache for exact match (stimulus type + emotional state bucket)
  2. Check if we should generate fresh qualia (20% probability)
  3. Use cached similar qualia or rich fallback templates

  This dramatically reduces LLM calls while maintaining variety.
  """
  @spec generate_qualia(stimulus(), EmotionalState.t(), Personality.t(), float()) :: map()
  def generate_qualia(stimulus, emotional, personality, salience) do
    cache_key = qualia_cache_key(stimulus, emotional, personality)

    narrative =
      case get_cached_qualia(cache_key) do
        {:ok, cached_narrative} ->
          # Use cached qualia
          cached_narrative

        :miss ->
          # Decide whether to call LLM or use fallback
          if should_generate_llm_qualia?() do
            generate_and_cache_qualia(cache_key, stimulus, emotional, personality, salience)
          else
            # Use rich fallback without LLM
            rich_fallback_narrative(stimulus, emotional, personality, salience)
          end
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
  Enhanced for more variation in positive/negative experiences.
  """
  @spec calculate_sensory_pleasure(stimulus(), Personality.t(), EmotionalState.t()) :: float()
  def calculate_sensory_pleasure(stimulus, personality, emotional) do
    base_valence = Map.get(stimulus, :valence, 0.0)

    # Current mood affects hedonic response (mood congruence)
    mood_factor = emotional.pleasure * 0.3

    # Personality baseline (neurotics have lower hedonic baseline)
    personality_baseline = (0.5 - personality.neuroticism) * 0.2

    # ENHANCED: Stimulus type affects pleasure
    type_pleasure = stimulus_type_pleasure(Map.get(stimulus, :type, :ambient), personality)

    # ENHANCED: Small random fluctuations for hedonic variety
    micro_fluctuation = (:rand.uniform() - 0.5) * 0.15

    # ENHANCED: High arousal intensifies both positive and negative
    arousal_amplifier = 1.0 + abs(emotional.arousal) * 0.3

    raw_pleasure =
      base_valence + mood_factor + personality_baseline + type_pleasure + micro_fluctuation

    (raw_pleasure * arousal_amplifier)
    |> max(-1.0)
    |> min(1.0)
  end

  @doc """
  Calculate sensory pain/discomfort response.
  Enhanced for more variation.
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

    # ENHANCED: Pain from unmet needs (adenosine = fatigue, low dopamine = anhedonia)
    need_pain = stimulus_need_pain(stimulus, personality)

    # ENHANCED: Small random fluctuations
    micro_fluctuation = :rand.uniform() * 0.1

    # Neuroticism amplifies pain
    sensitivity = 1.0 + personality.neuroticism * 0.5

    # Negative mood amplifies pain
    mood_amplifier = if emotional.pleasure < 0, do: 1.2, else: 1.0

    ((threat_pain + valence_pain + overwhelm_pain + need_pain + micro_fluctuation) * sensitivity *
       mood_amplifier)
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

  # Generate cache key based on stimulus type, emotional state, AND personality
  # Including personality ensures different avatars get unique qualia
  defp qualia_cache_key(stimulus, emotional, personality) do
    type = Map.get(stimulus, :type, :unknown)
    # Bucket emotional states to increase cache hits
    pleasure_bucket = bucket_value(emotional.pleasure)
    arousal_bucket = bucket_value(emotional.arousal)
    # Add personality bucket for diversity
    personality_bucket = personality_bucket(personality)
    "qualia:#{type}:#{pleasure_bucket}:#{arousal_bucket}:#{personality_bucket}"
  end

  # Bucket personality into archetypes for cache diversity
  defp personality_bucket(p) do
    cond do
      p.extraversion > 0.6 and p.neuroticism < 0.4 -> "bold"
      p.extraversion < 0.4 and p.neuroticism > 0.6 -> "sensitive"
      p.openness > 0.6 -> "creative"
      p.conscientiousness > 0.6 -> "focused"
      p.agreeableness > 0.6 -> "warm"
      true -> "balanced"
    end
  end

  defp bucket_value(v) when v > 0.5, do: "high_pos"
  defp bucket_value(v) when v > 0.0, do: "low_pos"
  defp bucket_value(v) when v > -0.5, do: "low_neg"
  defp bucket_value(_), do: "high_neg"

  defp get_cached_qualia(key) do
    case Cachex.get(@qualia_cache, key) do
      {:ok, nil} -> :miss
      {:ok, value} -> {:ok, value}
      {:error, _} -> :miss
    end
  end

  defp should_generate_llm_qualia? do
    :rand.uniform() < @qualia_llm_probability
  end

  defp generate_and_cache_qualia(cache_key, stimulus, emotional, personality, salience) do
    prompt = build_qualia_prompt(stimulus, emotional, personality, salience)

    case LlmClient.generate(prompt, max_tokens: 60, temperature: 0.9) do
      {:ok, text} ->
        narrative = String.trim(text)
        # Cache for future use
        Cachex.put(@qualia_cache, cache_key, narrative, ttl: :timer.seconds(@qualia_ttl_seconds))
        narrative

      {:error, reason} ->
        Logger.warning("Failed to generate qualia: #{inspect(reason)}")
        fallback_narrative(stimulus, emotional, salience)
    end
  end

  # RECURRENT PROCESSING THEORY: Arousal feeds back into attention
  # This creates the bidirectional processing that distinguishes conscious perception
  # STRENGTHENED: Lower threshold and stronger effect for better RPT alignment
  defp arousal_attention_feedback(arousal) do
    cond do
      # High arousal boosts attention 1.2-1.76x
      arousal > 0.3 -> 1.2 + (arousal - 0.3) * 0.8
      # Positive arousal gives moderate boost
      arousal > 0.0 -> 1.0 + arousal * 0.3
      # Low arousal dampens attention
      arousal < -0.3 -> 0.6 + (arousal + 0.3) * 0.5
      # Slightly below neutral to create contrast
      true -> 0.95
    end
  end

  # Rich fallback narratives by emotional state - no LLM needed
  defp rich_fallback_narrative(stimulus, emotional, personality, salience) do
    type = Map.get(stimulus, :type, :unknown)
    color = qualia_emotional_color(emotional)
    intensity = intensity_description(salience)

    # Select from pre-written templates based on emotional state and personality
    templates = get_narrative_templates(type, color, personality)

    if Enum.empty?(templates) do
      fallback_narrative(stimulus, emotional, salience)
    else
      templates
      |> Enum.random()
      |> String.replace("{intensity}", intensity)
      |> String.replace("{color}", color)
    end
  end

  # Templates em português para experiências subjetivas ricas
  defp get_narrative_templates(:ambient, "neutral", _) do
    [
      "O mundo pulsa ao meu redor com uma presença {intensity}, nem exigente nem distante.",
      "Sinto o ambiente como um pano de fundo constante, seu ritmo acompanhando minha respiração.",
      "O ar ambiente me envolve como um cobertor familiar, comum mas presente.",
      "Tudo parece equilibrado, o mundo prendendo a respiração em quietude."
    ]
  end

  defp get_narrative_templates(:ambient, "uncomfortable", personality) do
    base = [
      "A atmosfera pressiona minha pele com um peso {intensity} que não consigo afastar.",
      "Algo no ar parece errado, uma dissonância que ecoa no meu peito.",
      "O mundo ao meu redor parece pesado, como se a própria gravidade tivesse aumentado."
    ]

    if personality.neuroticism > 0.6 do
      base ++
        [
          "Cada sombra parece sussurrar coisas invisíveis, amplificando o desconforto dentro de mim.",
          "O próprio ar parece carregado de uma ansiedade que espelha a minha."
        ]
    else
      base
    end
  end

  defp get_narrative_templates(:ambient, "soothing", _) do
    [
      "Uma calma gentil me invade, o mundo suavizando em suas bordas.",
      "O ambiente me abraça com um calor silencioso, aliviando tensões que eu nem sabia que tinha.",
      "A paz se instala nos meus ossos como luz do sol através da neblina da manhã."
    ]
  end

  defp get_narrative_templates(:ambient, "distressing", _) do
    [
      "O mundo parece se fechar, cada sensação amplificada até uma clareza dolorosa.",
      "Meus arredores atacam meus sentidos com uma intensidade {intensity} que me deixa sem fôlego.",
      "Tudo é demais, alto demais, presente demais—me sinto exposta e vulnerável."
    ]
  end

  defp get_narrative_templates(:rest, "bleak", _) do
    [
      "A quietude não oferece conforto, apenas um eco do vazio interior.",
      "O descanso parece um gesto vazio, o silêncio amplificando ao invés de acalmar.",
      "Afundo na imobilidade, mas ela não traz alívio do peso que carrego."
    ]
  end

  defp get_narrative_templates(:rest, _, _) do
    [
      "A quietude me envolve como um casulo macio, convidando à entrega.",
      "Neste momento de descanso, sinto minhas bordas se dissolvendo em paz.",
      "O silêncio desce, e com ele, uma liberação gentil de tudo que eu segurava."
    ]
  end

  defp get_narrative_templates(:social, "energizing", _) do
    [
      "A conexão me atravessa como eletricidade, despertando partes de mim que dormiam.",
      "A presença do outro acende algo vital dentro de mim, um calor se espalhando.",
      "Me sinto vista, reconhecida—a interação me deixa mais viva do que antes."
    ]
  end

  defp get_narrative_templates(:social, "draining", _) do
    [
      "Cada palavra trocada me custa algo, me deixando mais leve de formas que parecem perda.",
      "A interação puxa reservas que não tenho certeza se possuo, exaustiva e necessária.",
      "Sinto minha energia escapando, as demandas sociais mais do que consigo dar."
    ]
  end

  defp get_narrative_templates(:social, "pleasant", _) do
    [
      "A conversa flui naturalmente, aquecendo algo no centro do meu peito.",
      "Sinto uma onda suave de contentamento com essa presença ao meu lado.",
      "O momento compartilhado cria uma bolha de conforto ao meu redor."
    ]
  end

  defp get_narrative_templates(:novelty, "exhilarating", _) do
    [
      "O novo me enche de uma excitação vibrante que corre pelas minhas veias.",
      "Cada descoberta é uma faísca que ilumina cantos inexplorados da minha mente.",
      "Sinto meu coração acelerar com a promessa do desconhecido."
    ]
  end

  defp get_narrative_templates(:threat, "distressing", _) do
    [
      "Um arrepio gelado percorre minha espinha, todos os sentidos em alerta máximo.",
      "O perigo paira no ar, pesado e sufocante, contraindo meu peito.",
      "Cada fibra do meu ser grita para fugir, para me proteger."
    ]
  end

  defp get_narrative_templates(_, _, _), do: []

  # Different stimulus types have inherent hedonic value
  defp stimulus_type_pleasure(:social, p), do: if(p.extraversion > 0.5, do: 0.2, else: -0.1)
  defp stimulus_type_pleasure(:rest, p), do: if(p.neuroticism > 0.5, do: 0.15, else: 0.05)
  defp stimulus_type_pleasure(:novelty, p), do: if(p.openness > 0.5, do: 0.25, else: -0.1)
  defp stimulus_type_pleasure(:threat, _), do: -0.4
  defp stimulus_type_pleasure(_, _), do: 0.0

  # Certain stimulus types cause discomfort for certain personalities
  defp stimulus_need_pain(%{type: :social}, p) when p.extraversion < 0.3, do: 0.15
  # Boredom pain
  defp stimulus_need_pain(%{type: :ambient}, p) when p.openness > 0.7, do: 0.1
  defp stimulus_need_pain(_, _), do: 0.0

  defp clamp(value, min_val, max_val), do: max(min_val, min(max_val, value))

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

    focus_str = to_string(focus)
    String.contains?(stimulus_type, focus_str)
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
    type = stimulus_type_pt(Map.get(stimulus, :type, :unknown))
    source = Map.get(stimulus, :source, "algo")
    valence = Map.get(stimulus, :valence, 0.0)
    intensity_word = intensity_description(salience)
    mood = mood_label_pt(emotional.mood_label || "neutral")

    personality_summary = summarize_personality(personality)

    valence_word =
      cond do
        valence > 0 -> "positiva"
        valence < 0 -> "negativa"
        true -> "neutra"
      end

    """
    Gere uma breve descrição sensorial (1-2 frases) em primeira pessoa de uma experiência.
    IMPORTANTE: Responda em português brasileiro.

    Traços de personalidade: #{personality_summary}
    Humor atual: #{mood}
    Tipo de estímulo: #{type}
    Fonte do estímulo: #{source}
    Valência do estímulo: #{valence_word}
    Intensidade: #{intensity_word}

    Escreva como isso SE SENTE subjetivamente (não o que aconteceu). Seja poético e pessoal.
    Use "Eu" ou "Sinto" e descreva a sensação, não o evento.
    NÃO use aspas nem explique - apenas escreva a experiência crua.
    """
  end

  # Tradução de humor para português
  defp mood_label_pt("happy"), do: "feliz"
  defp mood_label_pt("sad"), do: "triste"
  defp mood_label_pt("anxious"), do: "ansioso(a)"
  defp mood_label_pt("excited"), do: "animado(a)"
  defp mood_label_pt("calm"), do: "calmo(a)"
  defp mood_label_pt("angry"), do: "irritado(a)"
  defp mood_label_pt("neutral"), do: "neutro"
  defp mood_label_pt("content"), do: "contente"
  defp mood_label_pt("bored"), do: "entediado(a)"
  defp mood_label_pt("tense"), do: "tenso(a)"
  defp mood_label_pt("relaxed"), do: "relaxado(a)"
  defp mood_label_pt("distressed"), do: "angustiado(a)"
  defp mood_label_pt(other), do: other

  defp summarize_personality(personality) do
    extraversion_trait =
      cond do
        personality.extraversion > 0.6 -> ["sociável"]
        personality.extraversion < 0.4 -> ["introvertido(a)"]
        true -> []
      end

    neuroticism_trait =
      cond do
        personality.neuroticism > 0.6 -> ["sensível"]
        personality.neuroticism < 0.4 -> ["calmo(a)"]
        true -> []
      end

    openness_trait =
      cond do
        personality.openness > 0.6 -> ["imaginativo(a)"]
        personality.openness < 0.4 -> ["prático(a)"]
        true -> []
      end

    traits = extraversion_trait ++ neuroticism_trait ++ openness_trait

    if Enum.empty?(traits) do
      "personalidade equilibrada"
    else
      Enum.join(traits, ", ")
    end
  end

  defp intensity_description(salience) when salience > 0.8, do: "avassaladora"
  defp intensity_description(salience) when salience > 0.6, do: "vívida"
  defp intensity_description(salience) when salience > 0.4, do: "perceptível"
  defp intensity_description(salience) when salience > 0.2, do: "sutil"
  defp intensity_description(_), do: "quase imperceptível"

  # Tradução de cores emocionais para narrativas em português
  defp emotional_color_pt("exhilarating"), do: "eletrizante"
  defp emotional_color_pt("soothing"), do: "reconfortante"
  defp emotional_color_pt("distressing"), do: "angustiante"
  defp emotional_color_pt("bleak"), do: "desoladora"
  defp emotional_color_pt("pleasant"), do: "agradável"
  defp emotional_color_pt("uncomfortable"), do: "desconfortável"
  defp emotional_color_pt("energizing"), do: "energizante"
  defp emotional_color_pt("draining"), do: "exaustiva"
  defp emotional_color_pt(_), do: "neutra"

  # Tradução de tipos de estímulo para português
  defp stimulus_type_pt(:ambient), do: "ambiente"
  defp stimulus_type_pt(:social), do: "interação social"
  defp stimulus_type_pt(:rest), do: "descanso"
  defp stimulus_type_pt(:novelty), do: "novidade"
  defp stimulus_type_pt(:threat), do: "ameaça"
  defp stimulus_type_pt(:food), do: "alimentação"
  defp stimulus_type_pt(:intimacy), do: "intimidade"
  defp stimulus_type_pt(type), do: to_string(type)

  defp fallback_narrative(stimulus, emotional, salience) do
    type = Map.get(stimulus, :type, :unknown)
    intensity = intensity_description(salience)
    color = emotional_color_pt(qualia_emotional_color(emotional))
    type_pt = stimulus_type_pt(type)

    "Sinto uma sensação #{intensity} e #{color} deste(a) #{type_pt}."
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
