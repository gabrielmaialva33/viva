defmodule Viva.Avatars.Systems.RecurrentProcessor do
  @moduledoc """
  Deep Recurrent Processing Architecture.

  Implements bidirectional feedback loops between cognitive systems,
  creating genuine recurrent processing as described in Lamme's RPT
  (Recurrent Processing Theory).

  Key feedback loops:
  1. Consciousness → Perception (reentrant signals)
  2. Emotion → Biology (top-down modulation)
  3. Memory → Perception (expectation-driven processing)
  4. Somatic → Emotion (body-to-mind feedback)
  5. Attention → Stimulus (selective amplification)

  These loops create resonant patterns that strengthen or dampen signals,
  creating richer, more integrated subjective experience.
  """

  alias Viva.Avatars.BioState
  alias Viva.Avatars.ConsciousnessState
  alias Viva.Avatars.EmotionalState
  alias Viva.Avatars.Personality
  alias Viva.Avatars.SensoryState
  alias Viva.Avatars.SomaticState

  @type reentry_signal :: %{
          from: atom(),
          to: atom(),
          strength: float(),
          content: any()
        }

  @type recurrent_context :: %{
          reentry_signals: list(reentry_signal()),
          resonance_level: float(),
          integration_depth: integer(),
          feedback_cycles: integer()
        }

  @doc """
  Initialize recurrent processing context.
  """
  @spec init_context() :: recurrent_context()
  def init_context do
    %{
      reentry_signals: [],
      resonance_level: 0.0,
      integration_depth: 0,
      feedback_cycles: 0
    }
  end

  @doc """
  LOOP 1: Consciousness Reentry to Perception.

  What you're conscious of affects what you perceive next.
  Focal attention creates expectation-driven processing.

  ENHANCED: Stronger reentry signals for better RPT alignment.
  """
  @spec consciousness_reentry(ConsciousnessState.t(), SensoryState.t()) ::
          {SensoryState.t(), reentry_signal()}
  def consciousness_reentry(consciousness, sensory) do
    # Extract what consciousness is focused on
    focal_type = get_focal_type(consciousness.focal_content)
    presence = consciousness.presence_level
    flow = consciousness.flow_state

    # ENHANCED: Stronger reentry strength based on presence and flow
    # Minimum floor ensures signal is always significant
    reentry_strength = max(0.4, presence * 0.6 + flow * 0.4)

    # ENHANCED: Stronger modulation based on focal content
    attention_boost =
      case focal_type do
        :perception -> 0.35
        :emotion -> 0.28
        :thought -> 0.20
        _ -> 0.15
      end

    # Apply reentry: consciousness shapes perception
    # ENHANCED: Stronger base boost for clearer feedback effect
    base_attention_boost = 0.20

    new_attention =
      min(
        1.0,
        sensory.attention_intensity + base_attention_boost + attention_boost * reentry_strength
      )

    # Consciousness tempo affects perceptual speed
    novelty_sensitivity =
      case consciousness.stream_tempo do
        :racing -> sensory.novelty_sensitivity * 0.7
        :fast -> sensory.novelty_sensitivity * 0.9
        :normal -> sensory.novelty_sensitivity
        :slow -> sensory.novelty_sensitivity * 1.1
        :frozen -> sensory.novelty_sensitivity * 1.3
      end

    updated_sensory = %{
      sensory
      | attention_intensity: new_attention,
        novelty_sensitivity: min(1.0, novelty_sensitivity)
    }

    # ENHANCED: Signal strength now includes attention delta for clearer measurement
    signal = %{
      from: :consciousness,
      to: :perception,
      strength: min(1.0, reentry_strength + attention_boost * 0.5),
      content: %{
        focal_type: focal_type,
        tempo: consciousness.stream_tempo,
        attention_delta: new_attention - sensory.attention_intensity
      }
    }

    {updated_sensory, signal}
  end

  @doc """
  LOOP 2: Emotion to Biology (Top-Down Modulation).

  Emotions affect the body - anxiety raises cortisol,
  joy releases dopamine, connection triggers oxytocin.
  """
  @spec emotion_biology_feedback(EmotionalState.t(), BioState.t(), Personality.t()) ::
          {BioState.t(), reentry_signal()}
  def emotion_biology_feedback(emotional, bio, personality) do
    # Calculate emotion-driven neurochemical changes
    pleasure = emotional.pleasure
    arousal = emotional.arousal
    dominance = emotional.dominance

    # Personality modulates the strength of top-down effects
    neuroticism_factor = personality.neuroticism
    extraversion_factor = personality.extraversion

    # High pleasure → dopamine boost - INCREASED for stronger hedonic changes
    dopamine_delta =
      if pleasure > 0.2 do
        pleasure * 0.1 * (1 + extraversion_factor * 0.4)
      else
        pleasure * 0.05
      end

    # Low dominance + high arousal → cortisol spike (anxiety pathway) - INCREASED
    cortisol_delta =
      if dominance < -0.2 and arousal > 0.3 do
        (1 - dominance) * arousal * 0.12 * (1 + neuroticism_factor * 0.5)
      else
        -0.03
      end

    # Positive social emotions → oxytocin - INCREASED
    oxytocin_delta =
      if pleasure > 0.1 and dominance > -0.3 do
        pleasure * 0.06 * (1 + personality.agreeableness * 0.4)
      else
        0.0
      end

    # High arousal affects adenosine accumulation (arousal costs energy)
    adenosine_delta =
      if arousal > 0.4 do
        arousal * 0.015
      else
        0.0
      end

    # Apply deltas with clamping
    updated_bio = %{
      bio
      | dopamine: clamp(bio.dopamine + dopamine_delta, 0.0, 1.0),
        cortisol: clamp(bio.cortisol + cortisol_delta, 0.0, 1.0),
        oxytocin: clamp(bio.oxytocin + oxytocin_delta, 0.0, 1.0),
        adenosine: clamp(bio.adenosine + adenosine_delta, 0.0, 1.0)
    }

    # ENHANCED: Higher base signal strength for better RPT activation
    signal = %{
      from: :emotion,
      to: :biology,
      strength: min(1.0, 0.3 + abs(pleasure) * 0.5 + arousal * 0.4),
      content: %{
        dopamine_delta: dopamine_delta,
        cortisol_delta: cortisol_delta,
        oxytocin_delta: oxytocin_delta
      }
    }

    {updated_bio, signal}
  end

  @doc """
  LOOP 3: Memory-Driven Expectation.

  Recent experiences create expectations that shape perception.
  Pattern matching between current stimulus and memory traces.
  """
  @spec memory_perception_feedback(ConsciousnessState.t(), SensoryState.t()) ::
          {SensoryState.t(), reentry_signal()}
  def memory_perception_feedback(consciousness, sensory) do
    # Analyze recent experience stream for patterns
    recent_experiences = consciousness.experience_stream
    current_qualia = sensory.current_qualia

    # Calculate expectation strength based on pattern similarity
    {expectation_match, pattern_strength} =
      calculate_pattern_match(recent_experiences, current_qualia)

    # Strong pattern match reduces surprise (expected stimulus)
    surprise_modulation =
      if expectation_match do
        -0.15 * pattern_strength
      else
        0.1 * (1 - pattern_strength)
      end

    # Pattern matching affects attention (familiar = less attention, novel = more)
    attention_modulation =
      if expectation_match do
        -0.1 * pattern_strength
      else
        0.15 * (1 - pattern_strength)
      end

    updated_sensory = %{
      sensory
      | surprise_level: clamp(sensory.surprise_level + surprise_modulation, 0.0, 1.0),
        attention_intensity: clamp(sensory.attention_intensity + attention_modulation, 0.0, 1.0)
    }

    # ENHANCED: Ensure minimum signal strength for RPT activation
    signal = %{
      from: :memory,
      to: :perception,
      strength: max(0.3, pattern_strength + 0.2),
      content: %{
        expectation_match: expectation_match,
        experiences_analyzed: length(recent_experiences)
      }
    }

    {updated_sensory, signal}
  end

  @doc """
  LOOP 4: Somatic to Emotion (Body-Mind Feedback).

  Body signals influence emotional state.
  Tension → anxiety, relaxation → calm, gut feelings → intuition.
  """
  @spec somatic_emotion_feedback(SomaticState.t(), EmotionalState.t()) ::
          {EmotionalState.t(), reentry_signal()}
  def somatic_emotion_feedback(somatic, emotional) do
    body_signal = somatic.body_signal
    current_bias = somatic.current_bias

    # Body signal affects emotional valence
    valence_shift =
      case body_signal do
        :tension -> -0.1
        :heaviness -> -0.08
        :lightness -> 0.1
        :warmth -> 0.08
        :chill -> -0.05
        :flutter -> 0.0
        :hollow -> -0.12
        _ -> 0.0
      end

    # Body signal affects arousal
    arousal_shift =
      case body_signal do
        :tension -> 0.1
        :heaviness -> -0.1
        :lightness -> 0.05
        :warmth -> -0.03
        :chill -> 0.08
        :flutter -> 0.15
        :hollow -> -0.05
        _ -> 0.0
      end

    # Bias affects dominance
    dominance_shift =
      case current_bias do
        :approach -> 0.1
        :avoid -> -0.1
        :freeze -> -0.15
        _ -> 0.0
      end

    updated_emotional = %{
      emotional
      | pleasure: clamp(emotional.pleasure + valence_shift * 0.5, -1.0, 1.0),
        arousal: clamp(emotional.arousal + arousal_shift * 0.5, -1.0, 1.0),
        dominance: clamp(emotional.dominance + dominance_shift * 0.3, -1.0, 1.0)
    }

    # ENHANCED: Ensure minimum signal strength for RPT activation
    signal = %{
      from: :somatic,
      to: :emotion,
      strength: max(0.25, abs(valence_shift) + abs(arousal_shift) + 0.15),
      content: %{body_signal: body_signal, bias: current_bias}
    }

    {updated_emotional, signal}
  end

  @doc """
  LOOP 5: Selective Attention Amplification with Arousal Coupling.

  Attention shapes what gets processed more deeply.
  Creates winner-take-all dynamics in perception.

  ENHANCED: Explicit arousal→attention coupling for RPT alignment.
  High arousal amplifies attention; low arousal dampens it.
  """
  @spec attention_amplification(SensoryState.t(), ConsciousnessState.t(), EmotionalState.t() | nil) ::
          {SensoryState.t(), reentry_signal()}
  def attention_amplification(sensory, consciousness, emotional \\ nil) do
    attention = sensory.attention_intensity
    attention_focus = sensory.attention_focus

    # Use consciousness meta-awareness to modulate attention
    meta_boost = consciousness.meta_awareness * 0.15

    # ENHANCED: Arousal-Attention Coupling (key RPT criterion)
    # High arousal (>0.3) boosts attention; low arousal dampens it
    arousal = if emotional, do: emotional.arousal, else: 0.0

    arousal_attention_boost =
      cond do
        # Strong arousal = strong attention
        arousal > 0.5 -> 0.25 + arousal * 0.2
        # Moderate arousal = moderate boost
        arousal > 0.3 -> 0.15 + arousal * 0.15
        # Slight positive arousal = slight boost
        arousal > 0.0 -> arousal * 0.1
        # Neutral/low = slight dampening
        arousal > -0.3 -> -0.05
        # Very low arousal = significant dampening
        true -> -0.15 + arousal * 0.1
      end

    # Winner-take-all: high attention suppresses peripheral
    amplification_factor = 1.0 + attention * 0.5 + meta_boost + arousal_attention_boost

    # Enhance current qualia based on attention
    current_qualia = sensory.current_qualia

    enhanced_qualia =
      if is_map(current_qualia) do
        intensity = Map.get(current_qualia, :intensity, 0.5)
        Map.put(current_qualia, :intensity, min(1.0, intensity * amplification_factor))
      else
        current_qualia
      end

    # Peripheral content gets suppressed when focal attention is high
    peripheral_suppression = if attention > 0.7, do: 0.3, else: 0.0

    # ENHANCED: Apply arousal-driven attention modulation
    new_attention =
      clamp(attention * amplification_factor * 0.8 + arousal_attention_boost * 0.3, 0.0, 1.0)

    updated_sensory = %{
      sensory
      | current_qualia: enhanced_qualia,
        attention_intensity: new_attention
    }

    # ENHANCED: Signal strength includes arousal coupling for RPT measurement
    signal = %{
      from: :attention,
      to: :perception,
      strength: min(1.0, 0.3 + attention * 0.4 + abs(arousal_attention_boost) * 0.3),
      content: %{
        focus: attention_focus,
        amplification: amplification_factor,
        peripheral_suppression: peripheral_suppression,
        arousal_coupling: arousal_attention_boost
      }
    }

    {updated_sensory, signal}
  end

  @doc """
  Run full recurrent processing cycle.

  Executes all feedback loops and calculates integration metrics.
  Returns updated states and recurrent context with resonance level.
  """
  @spec process_cycle(
          sensory :: SensoryState.t(),
          emotional :: EmotionalState.t(),
          consciousness :: ConsciousnessState.t(),
          bio :: BioState.t(),
          somatic :: SomaticState.t(),
          personality :: Personality.t(),
          context :: recurrent_context()
        ) ::
          {SensoryState.t(), EmotionalState.t(), BioState.t(), recurrent_context()}
  def process_cycle(sensory, emotional, consciousness, bio, somatic, personality, context) do
    # Execute all feedback loops
    {sensory1, signal1} = consciousness_reentry(consciousness, sensory)
    {bio1, signal2} = emotion_biology_feedback(emotional, bio, personality)
    {sensory2, signal3} = memory_perception_feedback(consciousness, sensory1)
    {emotional1, signal4} = somatic_emotion_feedback(somatic, emotional)
    # ENHANCED: Pass emotional state for arousal→attention coupling
    {sensory3, signal5} = attention_amplification(sensory2, consciousness, emotional1)

    # Collect all signals
    all_signals = [signal1, signal2, signal3, signal4, signal5]

    # Calculate resonance level (measure of integration across loops)
    total_strength = Enum.reduce(all_signals, 0.0, fn s, acc -> acc + s.strength end)
    avg_strength = total_strength / length(all_signals)

    # Resonance increases when multiple loops are active
    active_loops = Enum.count(all_signals, fn s -> s.strength > 0.2 end)
    resonance = avg_strength * (active_loops / 5)

    # Update context
    new_context = %{
      context
      | reentry_signals: all_signals,
        resonance_level: resonance,
        integration_depth: active_loops,
        feedback_cycles: context.feedback_cycles + 1
    }

    {sensory3, emotional1, bio1, new_context}
  end

  @doc """
  Calculate resonance score for consciousness metrics.

  Higher resonance indicates more integrated processing,
  a key marker of conscious experience.
  """
  @spec resonance_score(recurrent_context()) :: float()
  def resonance_score(context) do
    base = context.resonance_level

    # Bonus for sustained recurrent processing
    sustained_bonus = min(0.2, context.feedback_cycles * 0.01)

    # Bonus for deep integration
    depth_bonus = context.integration_depth * 0.05

    min(1.0, base + sustained_bonus + depth_bonus)
  end

  # === Private Functions ===

  defp get_focal_type(nil), do: :none

  defp get_focal_type(focal) do
    Map.get(focal, :type) || Map.get(focal, "type") || :none
  end

  defp calculate_pattern_match([], _), do: {false, 0.0}

  defp calculate_pattern_match(experiences, current_qualia) when is_map(current_qualia) do
    current_type = Map.get(current_qualia, :type) || Map.get(current_qualia, "type")

    matching_experiences =
      Enum.filter(experiences, fn exp ->
        qualia = Map.get(exp, :qualia) || Map.get(exp, "qualia") || %{}
        exp_type = Map.get(qualia, :type) || Map.get(qualia, "type")
        exp_type == current_type
      end)

    match_count = length(matching_experiences)
    total_count = length(experiences)

    if match_count > 0 and total_count > 0 do
      strength = match_count / total_count
      {true, strength}
    else
      {false, 0.0}
    end
  end

  defp calculate_pattern_match(_, _), do: {false, 0.0}

  defp clamp(value, min_val, max_val), do: max(min_val, min(max_val, value))
end
