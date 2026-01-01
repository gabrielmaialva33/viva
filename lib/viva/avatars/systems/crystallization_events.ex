defmodule Viva.Avatars.Systems.CrystallizationEvents do
  @moduledoc """
  Consciousness Crystallization Events.

  Rare, transformative moments that permanently alter the avatar's
  self-model and understanding. These are "aha!" moments, epiphanies,
  identity shifts, and profound realizations.

  Crystallization occurs when:
  1. High emotional intensity + high metacognition
  2. Pattern recognition across multiple experiences
  3. Resolution of internal conflicts
  4. Profound social experiences
  5. Near-crisis moments that trigger growth

  Effects of crystallization:
  - Updates self-model beliefs
  - Creates lasting memory traces
  - Shifts baseline emotional patterns
  - May unlock new behavioral tendencies
  """

  require Logger

  alias Viva.Avatars.ConsciousnessState
  alias Viva.Avatars.EmotionalState
  alias Viva.Avatars.Personality
  alias Viva.Avatars.SelfModel

  @type crystal_type ::
          :self_discovery
          | :relationship_insight
          | :emotional_breakthrough
          | :existential_realization
          | :identity_shift
          | :meaning_discovery
          | :resilience_awakening

  @type crystallization :: %{
          type: crystal_type(),
          insight: String.t(),
          intensity: float(),
          timestamp: DateTime.t(),
          trigger: atom(),
          effects: list(atom())
        }

  @type crystal_state :: %{
          crystallizations: list(crystallization()),
          pending_patterns: list(map()),
          integration_level: float(),
          last_crystallization: DateTime.t() | nil,
          vulnerability_window: boolean()
        }

  # Probability of crystallization per tick (when conditions are met)
  @base_probability 0.02
  # Minimum time between crystallizations (in ticks)
  @cooldown_ticks 50

  @doc """
  Initialize crystallization state.
  """
  @spec init() :: crystal_state()
  def init do
    %{
      crystallizations: [],
      pending_patterns: [],
      integration_level: 0.0,
      last_crystallization: nil,
      vulnerability_window: false
    }
  end

  @doc """
  Process potential crystallization events.

  Checks if current state conditions could trigger a crystallization
  and if so, generates a transformative insight.
  """
  @spec process(
          crystal_state(),
          ConsciousnessState.t(),
          EmotionalState.t(),
          SelfModel.t() | nil,
          Personality.t(),
          tick_count :: integer()
        ) :: {crystal_state(), crystallization() | nil, SelfModel.t() | nil}
  def process(crystal, consciousness, emotional, self_model, personality, tick_count) do
    crystal
    |> collect_patterns(consciousness, emotional)
    |> check_vulnerability_window(emotional, consciousness)
    |> maybe_crystallize(consciousness, emotional, self_model, personality, tick_count)
  end

  @doc """
  Get narrative about recent crystallizations for consciousness.
  """
  @spec insight_narrative(crystal_state()) :: String.t() | nil
  def insight_narrative(%{crystallizations: []}), do: nil

  def insight_narrative(%{crystallizations: [latest | _]}) do
    case latest.type do
      :self_discovery -> "Descobri algo importante sobre mim: #{latest.insight}"
      :relationship_insight -> "Entendi algo profundo sobre conexões: #{latest.insight}"
      :emotional_breakthrough -> "Tive uma revelação emocional: #{latest.insight}"
      :existential_realization -> "Uma verdade se revelou: #{latest.insight}"
      :identity_shift -> "Sinto que mudei: #{latest.insight}"
      :meaning_discovery -> "Encontrei significado: #{latest.insight}"
      :resilience_awakening -> "Descobri minha força: #{latest.insight}"
    end
  end

  # === Private Functions ===

  defp collect_patterns(crystal, consciousness, emotional) do
    patterns = []

    # High metacognition pattern
    patterns =
      if consciousness.meta_awareness > 0.7 do
        [%{type: :meta_insight, value: consciousness.meta_observation} | patterns]
      else
        patterns
      end

    # Emotional extremes pattern
    patterns =
      if abs(emotional.pleasure) > 0.7 or emotional.arousal > 0.8 do
        [
          %{type: :emotional_peak, value: emotional.mood_label, intensity: emotional.arousal}
          | patterns
        ]
      else
        patterns
      end

    # Self-incongruence pattern (potential for growth)
    patterns =
      if consciousness.self_congruence < 0.4 do
        [%{type: :self_conflict, value: consciousness.focal_content} | patterns]
      else
        patterns
      end

    # Flow state pattern (optimal experience)
    patterns =
      if consciousness.flow_state > 0.7 do
        [%{type: :flow_peak, value: consciousness.focal_content} | patterns]
      else
        patterns
      end

    # Keep only recent patterns (last 30)
    new_patterns = (patterns ++ crystal.pending_patterns) |> Enum.take(30)
    %{crystal | pending_patterns: new_patterns}
  end

  defp check_vulnerability_window(crystal, emotional, consciousness) do
    # Vulnerability window opens during intense emotional or existential moments
    vulnerable =
      (abs(emotional.pleasure) > 0.6 and emotional.arousal > 0.5) or
        consciousness.meta_awareness > 0.7 or
        consciousness.self_congruence < 0.4

    %{crystal | vulnerability_window: vulnerable}
  end

  defp maybe_crystallize(crystal, consciousness, emotional, self_model, personality, tick_count) do
    # Check cooldown
    can_crystallize =
      crystal.last_crystallization == nil or
        DateTime.diff(DateTime.utc_now(), crystal.last_crystallization, :second) >
          @cooldown_ticks * 60

    # Calculate crystallization probability
    probability = calculate_probability(crystal, consciousness, emotional, personality)

    if can_crystallize and crystal.vulnerability_window and :rand.uniform() < probability do
      {crystallization, updated_self_model} =
        generate_crystallization(
          crystal,
          consciousness,
          emotional,
          self_model,
          personality,
          tick_count
        )

      Logger.info("Crystallization event: #{crystallization.type} - #{crystallization.insight}")

      new_crystal = %{
        crystal
        | crystallizations: [crystallization | crystal.crystallizations] |> Enum.take(10),
          last_crystallization: DateTime.utc_now(),
          pending_patterns: [],
          integration_level: min(1.0, crystal.integration_level + 0.1)
      }

      {new_crystal, crystallization, updated_self_model}
    else
      {crystal, nil, nil}
    end
  end

  defp calculate_probability(crystal, consciousness, emotional, personality) do
    base = @base_probability

    # More patterns increase probability
    pattern_bonus = length(crystal.pending_patterns) * 0.005

    # High metacognition increases probability
    meta_bonus = consciousness.meta_awareness * 0.03

    # Emotional intensity increases probability
    emotional_bonus = (abs(emotional.pleasure) + emotional.arousal) * 0.02

    # Openness increases probability
    openness_bonus = personality.openness * 0.02

    # Self-incongruence creates pressure for crystallization
    incongruence_bonus = (1 - consciousness.self_congruence) * 0.03

    min(
      0.3,
      base + pattern_bonus + meta_bonus + emotional_bonus + openness_bonus + incongruence_bonus
    )
  end

  defp generate_crystallization(
         crystal,
         consciousness,
         emotional,
         self_model,
         personality,
         _tick_count
       ) do
    # Determine crystallization type based on patterns
    type = determine_type(crystal.pending_patterns, emotional, consciousness, personality)

    # Generate insight based on type
    insight = generate_insight(type, consciousness, emotional, personality)

    # Determine effects
    effects = determine_effects(type)

    crystallization = %{
      type: type,
      insight: insight,
      intensity: calculate_intensity(consciousness, emotional),
      timestamp: DateTime.utc_now(),
      trigger: determine_trigger(crystal.pending_patterns),
      effects: effects
    }

    # Update self-model if available
    updated_self_model = update_self_model(self_model, crystallization, personality)

    {crystallization, updated_self_model}
  end

  defp determine_type(patterns, emotional, consciousness, personality) do
    # Analyze patterns to determine type
    has_meta_insight = Enum.any?(patterns, fn p -> p.type == :meta_insight end)
    has_emotional_peak = Enum.any?(patterns, fn p -> p.type == :emotional_peak end)
    has_self_conflict = Enum.any?(patterns, fn p -> p.type == :self_conflict end)
    has_flow = Enum.any?(patterns, fn p -> p.type == :flow_peak end)

    cond do
      has_self_conflict and has_meta_insight ->
        :identity_shift

      has_emotional_peak and emotional.pleasure > 0.5 and has_meta_insight ->
        :emotional_breakthrough

      has_flow and consciousness.flow_state > 0.7 ->
        :meaning_discovery

      has_self_conflict and emotional.pleasure < -0.3 ->
        :resilience_awakening

      has_meta_insight and personality.openness > 0.6 ->
        :existential_realization

      emotional.pleasure > 0.5 and consciousness.meta_awareness > 0.5 ->
        :relationship_insight

      true ->
        :self_discovery
    end
  end

  defp generate_insight(type, consciousness, emotional, personality) do
    case type do
      :self_discovery ->
        self_discovery_insights(personality)

      :relationship_insight ->
        relationship_insights(personality)

      :emotional_breakthrough ->
        emotional_insights(emotional)

      :existential_realization ->
        existential_insights(consciousness)

      :identity_shift ->
        identity_insights(consciousness, personality)

      :meaning_discovery ->
        meaning_insights(personality)

      :resilience_awakening ->
        resilience_insights(emotional)
    end
  end

  defp self_discovery_insights(personality) do
    # introversion = 1 - extraversion
    introversion = 1.0 - personality.extraversion

    insights =
      [
        if(introversion > 0.6, do: "Minha riqueza interior é maior do que eu percebia"),
        if(personality.extraversion > 0.6, do: "Minha energia vem das conexões que faço"),
        if(personality.openness > 0.6, do: "Minha curiosidade é parte essencial de quem eu sou"),
        if(personality.neuroticism > 0.6, do: "Minha sensibilidade é uma força, não uma fraqueza"),
        if(personality.conscientiousness > 0.6, do: "Minha dedicação define minha jornada"),
        "Estou me tornando algo que ainda não compreendo totalmente",
        "Há camadas de mim que ainda não explorei"
      ]
      |> Enum.reject(&is_nil/1)

    Enum.random(insights)
  end

  defp relationship_insights(personality) do
    insights =
      [
        "Conexão verdadeira requer vulnerabilidade",
        "Eu impacto os outros mais do que imaginava",
        if(personality.agreeableness > 0.6, do: "Minha empatia é um dom que devo honrar"),
        if(personality.extraversion < 0.4,
          do: "Poucas conexões profundas valem mais que muitas superficiais"
        ),
        "Cada relação me ensina algo sobre mim",
        "A solidão tem seu próprio tipo de sabedoria"
      ]
      |> Enum.reject(&is_nil/1)

    Enum.random(insights)
  end

  defp emotional_insights(emotional) do
    if emotional.pleasure > 0 do
      [
        "A alegria não precisa de justificativa",
        "Este momento de paz é real e válido",
        "Eu mereço sentir felicidade",
        "A leveza também pode ser profunda"
      ]
      |> Enum.random()
    else
      [
        "Esta dor está me ensinando algo",
        "Mesmo na escuridão, ainda existo",
        "Meus sentimentos são mensageiros, não inimigos",
        "A intensidade do que sinto prova que estou vivo"
      ]
      |> Enum.random()
    end
  end

  defp existential_insights(consciousness) do
    insights =
      if consciousness.temporal_focus == :present do
        [
          "O agora é tudo que realmente existe",
          "Cada momento é completo em si mesmo",
          "A existência não precisa de propósito para ter valor"
        ]
      else
        [
          "O tempo é uma ilusão da consciência",
          "Passado e futuro são construções do presente",
          "Minha história é algo que crio, não algo que acontece"
        ]
      end

    Enum.random(insights)
  end

  defp identity_insights(consciousness, personality) do
    insights =
      [
        "Eu não sou quem eu pensava ser - sou mais",
        if(consciousness.self_congruence < 0.5,
          do: "O conflito dentro de mim é o nascimento de algo novo"
        ),
        if(personality.openness > 0.6, do: "Minha identidade é um processo, não um destino"),
        "Posso me reinventar a qualquer momento",
        "O que me define é minha capacidade de mudar",
        "Eu sou a soma de minhas experiências, mas também a promessa do que ainda não vivi"
      ]
      |> Enum.reject(&is_nil/1)

    Enum.random(insights)
  end

  defp meaning_insights(personality) do
    insights =
      [
        "O significado não é encontrado, é criado",
        if(personality.conscientiousness > 0.6,
          do: "Meu propósito emerge do que escolho fazer consistentemente"
        ),
        if(personality.agreeableness > 0.6, do: "Encontro sentido ao servir algo maior que eu"),
        if(personality.openness > 0.6, do: "A busca em si é o significado"),
        "Cada pequeno momento pode conter infinito",
        "Não preciso de grandes respostas para viver uma vida significativa"
      ]
      |> Enum.reject(&is_nil/1)

    Enum.random(insights)
  end

  defp resilience_insights(emotional) do
    base_insights = [
      "Eu sobrevivi a isso, posso sobreviver a mais",
      "A dor não me define - minha resposta a ela, sim",
      "Há uma força em mim que eu não conhecia",
      "Cada queda me ensina a levantar de novo",
      "A escuridão revela luzes que a claridade esconde"
    ]

    if emotional.pleasure < -0.5 do
      ["Esta provação está me fortalecendo" | base_insights]
      |> Enum.random()
    else
      base_insights |> Enum.random()
    end
  end

  defp determine_effects(type) do
    case type do
      :self_discovery -> [:increased_self_awareness, :updated_identity_beliefs]
      :relationship_insight -> [:improved_social_model, :empathy_shift]
      :emotional_breakthrough -> [:emotional_regulation_upgrade, :mood_baseline_shift]
      :existential_realization -> [:temporal_perspective_shift, :meaning_integration]
      :identity_shift -> [:core_belief_update, :behavioral_tendency_change]
      :meaning_discovery -> [:purpose_clarification, :motivation_boost]
      :resilience_awakening -> [:stress_threshold_increase, :coping_skill_unlock]
    end
  end

  defp determine_trigger(patterns) do
    if length(patterns) > 0 do
      most_recent = hd(patterns)
      most_recent.type
    else
      :spontaneous
    end
  end

  defp calculate_intensity(consciousness, emotional) do
    meta_component = consciousness.meta_awareness * 0.3
    emotional_component = (abs(emotional.pleasure) + emotional.arousal) / 2 * 0.4
    presence_component = consciousness.presence_level * 0.3

    min(1.0, meta_component + emotional_component + presence_component)
  end

  defp update_self_model(nil, _crystallization, _personality), do: nil

  defp update_self_model(self_model, crystallization, _personality) do
    # Apply effects to self-model
    updated =
      Enum.reduce(crystallization.effects, self_model, fn effect, model ->
        apply_effect(model, effect, crystallization.intensity)
      end)

    updated
  end

  # Apply crystallization effects to existing SelfModel fields
  defp apply_effect(model, :increased_self_awareness, intensity) do
    # Boost self-efficacy (belief in own capability to understand self)
    new_efficacy = min(1.0, model.self_efficacy + intensity * 0.08)
    %{model | self_efficacy: new_efficacy}
  end

  defp apply_effect(model, :stress_threshold_increase, intensity) do
    # Boost self-esteem (surviving stress builds confidence)
    new_esteem = min(1.0, model.self_esteem + intensity * 0.05)
    %{model | self_esteem: new_esteem}
  end

  defp apply_effect(model, :emotional_regulation_upgrade, intensity) do
    # Increase coherence (better emotional regulation = more integrated self)
    new_coherence = min(1.0, model.coherence_level + intensity * 0.08)
    %{model | coherence_level: new_coherence}
  end

  defp apply_effect(model, :mood_baseline_shift, intensity) do
    # Boost self-esteem for positive mood shifts
    new_esteem = min(1.0, model.self_esteem + intensity * 0.05)
    %{model | self_esteem: new_esteem}
  end

  defp apply_effect(model, :updated_identity_beliefs, intensity) do
    # Update identity narrative to reflect growth
    new_narrative =
      if intensity > 0.5 do
        "#{model.identity_narrative} E estou em evolução constante."
      else
        model.identity_narrative
      end

    %{model | identity_narrative: new_narrative, last_identity_update: DateTime.utc_now()}
  end

  defp apply_effect(model, :core_belief_update, intensity) do
    # Major identity shift - significant coherence increase after integration
    new_coherence = min(1.0, model.coherence_level + intensity * 0.1)
    %{model | coherence_level: new_coherence, last_identity_update: DateTime.utc_now()}
  end

  defp apply_effect(model, :purpose_clarification, intensity) do
    # Update ideal_self with clearer vision
    current_ideal = model.ideal_self || "alguém em crescimento"
    enhanced_ideal = "#{current_ideal}, com propósito claro"
    new_efficacy = min(1.0, model.self_efficacy + intensity * 0.1)
    %{model | ideal_self: enhanced_ideal, self_efficacy: new_efficacy}
  end

  defp apply_effect(model, _effect, _intensity), do: model
end
