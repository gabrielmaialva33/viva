//// Awareness - Integrates sensory perception with Soul
////
//// This module connects:
////   - Percepts (what VIVA perceives)
////   - Soul (emotional core)
////   - Memory (HRR storage)
////
//// Flow:
////   Sense → Percept → Awareness → Soul (emotion + memory)

import gleam/float
import gleam/list
import gleam/option.{type Option, None, Some}
import viva/embodied/percept.{
  type AttentionFocus, type Percept, CodeAnalysis, DirectAddress, ErrorDetection,
  GeneralAwareness, HelpRequest, Idle, Listening, PassiveObservation,
  SocialInteraction, WorkObservation,
}
import viva/embodied/sense.{
  type Emotion, Alert, Celebrate, Empathize, Observe, OfferHelp, Rest,
}
import viva_emotion/stimulus.{type Stimulus}
import viva_glyph/glyph.{type Glyph}

// =============================================================================
// AWARENESS TYPES
// =============================================================================

/// Result of processing a percept through awareness
pub type AwarenessResult {
  AwarenessResult(
    /// Original percept
    percept: Percept,
    /// Stimulus to send to Soul
    stimulus: Stimulus,
    /// Intensity of the stimulus (0-1)
    intensity: Float,
    /// Context glyph for memory
    context_glyph: Glyph,
    /// HRR memory vector (512-dim)
    memory_vector: List(Float),
    /// Whether this requires immediate attention
    urgent: Bool,
    /// Suggested response
    response: AwarenessResponse,
  )
}

/// What VIVA should do in response to perception
pub type AwarenessResponse {
  /// Just observe, no action needed
  ObserveOnly
  /// Offer assistance
  OfferAssistance(reason: String)
  /// Alert user to something
  AlertUser(message: String)
  /// Express emotion
  Express(emotion_type: String)
  /// Rest/idle
  DoNothing
}

// =============================================================================
// MAIN PROCESSING
// =============================================================================

/// Process a percept into awareness result
pub fn process(percept: Percept) -> AwarenessResult {
  // Convert percept to stimulus
  let stimulus = percept_to_stimulus(percept)

  // Calculate intensity from salience and novelty
  let intensity = calculate_intensity(percept)

  // Generate context glyph from percept
  let context_glyph = percept_to_context_glyph(percept)

  // Get HRR memory vector
  let memory_vector = percept.to_memory_vector(percept)

  // Determine if urgent
  let urgent = is_urgent(percept)

  // Determine response
  let response = determine_response(percept)

  AwarenessResult(
    percept: percept,
    stimulus: stimulus,
    intensity: intensity,
    context_glyph: context_glyph,
    memory_vector: memory_vector,
    urgent: urgent,
    response: response,
  )
}

/// Process multiple percepts, returning the most salient
pub fn process_many(percepts: List(Percept)) -> Option(AwarenessResult) {
  case percepts {
    [] -> None
    [single] -> Some(process(single))
    _ -> {
      // Find most salient percept
      let sorted =
        percepts
        |> list.sort(fn(a, b) {
          float.compare(combined_importance(b), combined_importance(a))
        })

      case sorted {
        [most_important, ..] -> Some(process(most_important))
        [] -> None
      }
    }
  }
}

fn combined_importance(percept: Percept) -> Float {
  percept.salience *. 0.6 +. percept.novelty *. 0.4
}

// =============================================================================
// STIMULUS CONVERSION
// =============================================================================

/// Convert percept to emotional stimulus
fn percept_to_stimulus(percept: Percept) -> Stimulus {
  // First check action suggestion
  case percept.thought.action {
    Alert -> stimulus.NearDeath
    // Most urgent - danger
    OfferHelp -> stimulus.Companionship
    // Social engagement
    Celebrate -> stimulus.Success
    // Achievement
    Empathize -> stimulus.Companionship
    // Social connection
    Rest -> stimulus.Safety
    // Calm state
    Observe -> {
      // Determine from attention focus
      attention_to_stimulus(percept.attention)
    }
  }
}

fn attention_to_stimulus(attention: AttentionFocus) -> Stimulus {
  case attention {
    ErrorDetection -> stimulus.Threat
    // Error = threat
    DirectAddress -> stimulus.Acceptance
    // Someone addressing us
    HelpRequest -> stimulus.Companionship
    // Someone needs us
    CodeAnalysis -> stimulus.LucidInsight
    // Mental engagement
    WorkObservation -> stimulus.LucidInsight
    // Observing work
    SocialInteraction -> stimulus.Companionship
    // Social
    PassiveObservation -> stimulus.Safety
    // Calm
    Listening -> stimulus.Safety
    // Calm
    GeneralAwareness -> stimulus.Safety
    // Calm
    Idle -> stimulus.Safety
    // Calm
  }
}

// =============================================================================
// INTENSITY CALCULATION
// =============================================================================

fn calculate_intensity(percept: Percept) -> Float {
  // Base from salience and novelty
  let base = percept.salience *. 0.5 +. percept.novelty *. 0.3

  // Emotional intensity from thought
  let emotional = sense.emotion_intensity(percept.thought.emotion)

  // Action urgency
  let action_boost = case percept.thought.action {
    Alert -> 0.3
    OfferHelp -> 0.15
    Celebrate -> 0.1
    Empathize -> 0.1
    Observe -> 0.0
    Rest -> 0.0
  }

  // Attention boost
  let attention_boost = case percept.attention {
    ErrorDetection -> 0.2
    DirectAddress -> 0.15
    HelpRequest -> 0.1
    _ -> 0.0
  }

  // Combine and clamp
  float_min(1.0, base +. emotional *. 0.2 +. action_boost +. attention_boost)
}

// =============================================================================
// CONTEXT GLYPH
// =============================================================================

/// Convert percept to context glyph for memory
fn percept_to_context_glyph(percept: Percept) -> Glyph {
  // Token 1: Attention focus (0-9 → 0-255)
  let t1 = attention_to_token(percept.attention)

  // Token 2: Content type (visual/auditory/internal)
  let t2 = case percept.vision, percept.hearing {
    Some(_), _ -> 200
    // Visual
    _, Some(_) -> 100
    // Auditory
    _, _ -> 50
    // Internal
  }

  // Token 3: Novelty (0-255)
  let t3 = float_to_int(percept.novelty *. 255.0)

  // Token 4: Salience (0-255)
  let t4 = float_to_int(percept.salience *. 255.0)

  glyph.new([t1, t2, t3, t4])
}

fn attention_to_token(attention: AttentionFocus) -> Int {
  case attention {
    CodeAnalysis -> 255
    ErrorDetection -> 230
    WorkObservation -> 200
    SocialInteraction -> 170
    DirectAddress -> 150
    HelpRequest -> 130
    PassiveObservation -> 100
    Listening -> 70
    GeneralAwareness -> 40
    Idle -> 0
  }
}

// =============================================================================
// URGENCY & RESPONSE
// =============================================================================

fn is_urgent(percept: Percept) -> Bool {
  case percept.attention {
    ErrorDetection -> True
    DirectAddress -> True
    HelpRequest -> True
    _ -> percept.thought.action == Alert
  }
}

fn determine_response(percept: Percept) -> AwarenessResponse {
  case percept.thought.action {
    Alert -> AlertUser(percept.thought.content)
    OfferHelp -> OfferAssistance(percept.thought.content)
    Celebrate -> Express("celebration")
    Empathize -> Express("empathy")
    Rest -> DoNothing
    Observe -> {
      case percept.attention {
        ErrorDetection -> AlertUser("Error detected")
        DirectAddress -> OfferAssistance("You called me")
        HelpRequest -> OfferAssistance("You need help")
        _ -> ObserveOnly
      }
    }
  }
}

// =============================================================================
// EMOTIONAL INTEGRATION
// =============================================================================

/// Convert percept emotion to PAD delta for Soul
pub fn to_pad_delta(percept: Percept) -> #(Float, Float, Float) {
  let emotion = percept.thought.emotion

  // Scale by intensity
  let intensity = calculate_intensity(percept)

  // Valence → Pleasure
  let pleasure = emotion.valence *. intensity *. 0.5

  // Arousal stays arousal
  let arousal = { emotion.arousal -. 0.5 } *. intensity *. 0.3

  // Dominance stays dominance
  let dominance = { emotion.dominance -. 0.5 } *. intensity *. 0.3

  #(pleasure, arousal, dominance)
}

/// Get emotional summary from percept
pub fn emotional_summary(percept: Percept) -> String {
  let emotion = percept.thought.emotion

  let valence_desc = case emotion.valence {
    v if v >. 0.5 -> "positive"
    v if v <. -0.5 -> "negative"
    _ -> "neutral"
  }

  let arousal_desc = case emotion.arousal {
    a if a >. 0.7 -> "excited"
    a if a <. 0.3 -> "calm"
    _ -> "moderate"
  }

  let dominance_desc = case emotion.dominance {
    d if d >. 0.7 -> "in control"
    d if d <. 0.3 -> "overwhelmed"
    _ -> "balanced"
  }

  valence_desc <> ", " <> arousal_desc <> ", " <> dominance_desc
}

// =============================================================================
// ATTENTION FILTERING
// =============================================================================

/// Filter percepts by attention priority
pub fn filter_by_attention(
  percepts: List(Percept),
  min_attention: AttentionFocus,
) -> List(Percept) {
  list.filter(percepts, fn(p) {
    attention_priority(p.attention) >= attention_priority(min_attention)
  })
}

fn attention_priority(attention: AttentionFocus) -> Int {
  case attention {
    ErrorDetection -> 10
    DirectAddress -> 9
    HelpRequest -> 8
    CodeAnalysis -> 7
    SocialInteraction -> 6
    WorkObservation -> 5
    PassiveObservation -> 4
    Listening -> 3
    GeneralAwareness -> 2
    Idle -> 1
  }
}

/// Is this percept worth storing in memory?
pub fn worth_remembering(percept: Percept) -> Bool {
  // High salience or novelty
  percept.salience >. 0.5
  || percept.novelty >. 0.7
  // Or emotionally significant
  || sense.emotion_intensity(percept.thought.emotion) >. 0.5
  // Or requires action
  || percept.thought.action == Alert
  || percept.thought.action == OfferHelp
}

// =============================================================================
// CONTINUOUS AWARENESS
// =============================================================================

/// State for continuous awareness processing
pub type AwarenessState {
  AwarenessState(
    /// Recent percepts (for novelty calculation)
    recent: List(Percept),
    /// Running attention focus
    current_attention: AttentionFocus,
    /// Emotional momentum (smoothed)
    emotional_momentum: Emotion,
    /// Tick counter
    tick: Int,
  )
}

/// Create new awareness state
pub fn new_state() -> AwarenessState {
  AwarenessState(
    recent: [],
    current_attention: Idle,
    emotional_momentum: sense.neutral_emotion(),
    tick: 0,
  )
}

/// Process percept and update state
pub fn process_with_state(
  state: AwarenessState,
  percept: Percept,
) -> #(AwarenessState, AwarenessResult) {
  // Calculate novelty with recent history
  let percept_with_novelty = percept.with_novelty(percept, state.recent)

  // Process
  let result = process(percept_with_novelty)

  // Update emotional momentum (exponential moving average)
  let new_momentum =
    sense.blend_emotions(state.emotional_momentum, percept.thought.emotion, 0.3)

  // Update state
  let new_state =
    AwarenessState(
      recent: [percept_with_novelty, ..list.take(state.recent, 19)],
      current_attention: percept_with_novelty.attention,
      emotional_momentum: new_momentum,
      tick: state.tick + 1,
    )

  #(new_state, result)
}

/// Get current emotional momentum
pub fn get_momentum(state: AwarenessState) -> Emotion {
  state.emotional_momentum
}

/// Get attention trend (most common recent attention)
pub fn attention_trend(state: AwarenessState) -> AttentionFocus {
  case state.recent {
    [] -> Idle
    _ -> state.current_attention
    // Simplified: just use current
  }
}

// =============================================================================
// FFI HELPERS
// =============================================================================

fn float_min(a: Float, b: Float) -> Float {
  case a <. b {
    True -> a
    False -> b
  }
}

@external(erlang, "erlang", "trunc")
fn float_to_int(f: Float) -> Int
