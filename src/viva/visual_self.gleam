//// Visual Self Model - How VIVA perceives her own visual appearance
////
//// The internal representation of "how I look" based on emotional state:
//// - Posture: body language derived from PAD
//// - Expression: facial/avatar expressions with intensity
//// - Aura: colors, patterns, pulsation rate
////
//// Visual Coherence: the match between "how I feel" and "how I appear"
//// When coherence is low, VIVA experiences visual dissonance.
////
//// "I notice my posture has become defensive..."
//// "My aura is shifting to warmer tones..."

import gleam/float
import gleam/int
import gleam/list
import gleam/option.{type Option, None, Some}
import viva_emotion/pad.{type Pad}
import viva_glyph/glyph.{type Glyph}

// =============================================================================
// TYPES
// =============================================================================

/// Visual self-model: internal representation of appearance
pub type VisualSelfModel {
  VisualSelfModel(
    /// Current posture (body language)
    posture: Posture,
    /// Current expression with intensity
    expression: Expression,
    /// Aura model (colors, patterns)
    aura: AuraModel,
    /// Baseline visual state (my "normal" appearance)
    baseline_posture: Posture,
    baseline_expression: Expression,
    baseline_aura: AuraModel,
    /// How strongly I perceive my own visual state (0.0-1.0)
    visual_awareness: Float,
    /// History of visual observations (max 50)
    observation_history: List(VisualObservation),
    /// Tick count for temporal tracking
    tick: Int,
    /// Motion intensity (0.0 = still, 1.0 = highly animated)
    motion_intensity: Float,
  )
}

/// Posture: body language derived from PAD dimensions
pub type Posture {
  /// High pleasure, high dominance - confident, open
  Open
  /// Low pleasure, low dominance - withdrawn, closed
  Closed
  /// High arousal, high dominance - aggressive, forward
  Forward
  /// Low arousal, low dominance - retreating, shrinking
  Retreating
  /// High pleasure, low arousal - relaxed, loose
  Relaxed
  /// Low pleasure, high arousal - tense, rigid
  Tense
  /// High dominance - tall, expansive
  Expansive
  /// Low dominance - small, contracted
  Contracted
  /// Neutral baseline
  Neutral
}

/// Expression: emotional display with intensity
pub type Expression {
  Expression(
    /// Type of expression
    kind: ExpressionKind,
    /// Intensity (0.0-1.0)
    intensity: Float,
    /// Symmetry (1.0 = perfectly symmetric, 0.0 = highly asymmetric)
    symmetry: Float,
    /// Micro-expression overlay (subtle secondary emotion)
    micro: Option(ExpressionKind),
  )
}

/// Types of expressions
pub type ExpressionKind {
  /// Joy - pleasure positive
  Joyful
  /// Sadness - pleasure negative
  Sad
  /// Excitement - arousal positive
  Excited
  /// Calm - arousal negative
  Calm
  /// Confident - dominance positive
  Confident
  /// Uncertain - dominance negative
  Uncertain
  /// Fear - low pleasure, high arousal, low dominance
  Fearful
  /// Anger - low pleasure, high arousal, high dominance
  Angry
  /// Surprise - sudden arousal spike
  Surprised
  /// Contempt - negative pleasure with high dominance
  Contemptuous
  /// Neutral baseline
  Blank
}

/// Aura model: visual energy field representation
pub type AuraModel {
  AuraModel(
    /// Primary color (RGB 0-255)
    primary_color: Color,
    /// Secondary color (for gradients)
    secondary_color: Color,
    /// Pattern type
    pattern: AuraPattern,
    /// Pulse rate (beats per second, 0.5-3.0)
    pulse_rate: Float,
    /// Brightness (0.0-1.0)
    brightness: Float,
    /// Size/radius multiplier (0.5-2.0)
    size: Float,
    /// Stability (1.0 = solid, 0.0 = flickering)
    stability: Float,
  )
}

/// RGB Color
pub type Color {
  Color(r: Int, g: Int, b: Int)
}

/// Aura patterns
pub type AuraPattern {
  /// Steady, solid glow
  Solid
  /// Gentle waves radiating outward
  Waves
  /// Spinning/rotating energy
  Spiral
  /// Pulsing in and out
  Pulse
  /// Chaotic, fragmented
  Fragmented
  /// Soft, cloud-like
  Nebula
  /// Sharp, crystalline edges
  Crystalline
  /// Flame-like, rising
  Flame
}

/// Visual observation: comparison between internal state and perceived appearance
pub type VisualObservation {
  VisualObservation(
    /// Tick when observation was made
    tick: Int,
    /// Expected visual state (from PAD)
    expected_posture: Posture,
    expected_expression: Expression,
    expected_aura: AuraModel,
    /// Actual visual state (perceived)
    actual_posture: Posture,
    actual_expression: Expression,
    actual_aura: AuraModel,
    /// Coherence score for this observation
    coherence: Float,
    /// Action being performed (if any)
    action: Option(String),
  )
}

/// Visual coherence: match between feeling and appearance
pub type VisualCoherence {
  VisualCoherence(
    /// Overall coherence (0.0 = dissonant, 1.0 = aligned)
    overall: Float,
    /// Posture coherence
    posture_match: Float,
    /// Expression coherence
    expression_match: Float,
    /// Aura coherence
    aura_match: Float,
    /// Is experiencing visual dissonance?
    dissonant: Bool,
    /// Description of the visual state
    description: String,
    /// Suggested adjustment (if dissonant)
    suggestion: Option(String),
  )
}

// =============================================================================
// CONSTRUCTORS
// =============================================================================

/// Create new visual self-model with neutral baseline
pub fn new() -> VisualSelfModel {
  let neutral_expression =
    Expression(kind: Blank, intensity: 0.0, symmetry: 1.0, micro: None)

  let neutral_aura =
    AuraModel(
      primary_color: Color(r: 128, g: 128, b: 200),
      secondary_color: Color(r: 100, g: 100, b: 180),
      pattern: Solid,
      pulse_rate: 1.0,
      brightness: 0.5,
      size: 1.0,
      stability: 1.0,
    )

  VisualSelfModel(
    posture: Neutral,
    expression: neutral_expression,
    aura: neutral_aura,
    baseline_posture: Neutral,
    baseline_expression: neutral_expression,
    baseline_aura: neutral_aura,
    visual_awareness: 0.5,
    observation_history: [],
    tick: 0,
    motion_intensity: 0.0,
  )
}

/// Create visual self-model from initial PAD state
pub fn from_pad(initial_pad: Pad) -> VisualSelfModel {
  let posture = posture_from_pad(initial_pad)
  let expression = expression_from_pad(initial_pad)
  let aura = aura_from_pad(initial_pad)

  VisualSelfModel(
    posture: posture,
    expression: expression,
    aura: aura,
    baseline_posture: posture,
    baseline_expression: expression,
    baseline_aura: aura,
    visual_awareness: 0.5,
    observation_history: [],
    tick: 0,
    motion_intensity: 0.0,
  )
}

// =============================================================================
// PAD TO VISUAL MAPPING
// =============================================================================

/// Derive posture from PAD state
pub fn posture_from_pad(p: Pad) -> Posture {
  let pleasure = p.pleasure
  let arousal = p.arousal
  let dominance = p.dominance

  // Decision tree based on PAD dimensions
  case pleasure >. 0.3, arousal >. 0.3, dominance >. 0.3 {
    // High pleasure
    True, True, True -> Open
    // Open and confident
    True, True, False -> Relaxed
    // Happy but not dominant
    True, False, True -> Expansive
    // Content and in control
    True, False, False -> Relaxed

    // At ease
    // Low pleasure
    False, True, True ->
      case pleasure <. -0.3 {
        True -> Forward
        // Aggressive
        False -> Tense
      }
    False, True, False -> Tense
    // Anxious
    False, False, True -> Closed
    // Withdrawn but guarded
    False, False, False ->
      case pleasure <. -0.3 {
        True -> Retreating
        // Depressed
        False -> Neutral
      }
  }
}

/// Derive expression from PAD state
pub fn expression_from_pad(p: Pad) -> Expression {
  let pleasure = p.pleasure
  let arousal = p.arousal
  let dominance = p.dominance

  // Calculate intensity from PAD magnitude
  let sum_squares =
    pleasure *. pleasure +. arousal *. arousal +. dominance *. dominance
  let magnitude = case float.square_root(sum_squares) {
    Ok(m) -> m
    Error(_) -> 0.0
  }
  let intensity = float.min(1.0, magnitude /. 1.732)
  // sqrt(3) for normalization

  // Determine primary expression
  let kind = case pleasure >. 0.2, pleasure <. -0.2 {
    True, _ ->
      // Positive pleasure
      case arousal >. 0.2 {
        True -> Excited
        False -> Joyful
      }
    _, True ->
      // Negative pleasure
      case arousal >. 0.2, dominance >. 0.2 {
        True, True -> Angry
        True, False -> Fearful
        False, True -> Contemptuous
        False, False -> Sad
      }
    _, _ ->
      // Neutral pleasure
      case arousal >. 0.3 {
        True -> Surprised
        False ->
          case dominance >. 0.2 {
            True -> Confident
            False ->
              case dominance <. -0.2 {
                True -> Uncertain
                False -> Calm
              }
          }
      }
  }

  // Determine micro-expression (subtle secondary)
  let micro = case kind {
    Joyful if dominance >. 0.3 -> Some(Confident)
    Sad if arousal >. 0.2 -> Some(Fearful)
    Angry if dominance <. 0.0 -> Some(Fearful)
    Calm if pleasure >. 0.3 -> Some(Joyful)
    _ -> None
  }

  // Symmetry decreases with stress (high arousal + low dominance)
  let stress_factor = float.max(0.0, arousal -. dominance)
  let symmetry = float.max(0.5, 1.0 -. stress_factor *. 0.3)

  Expression(kind: kind, intensity: intensity, symmetry: symmetry, micro: micro)
}

/// Derive aura from PAD state
pub fn aura_from_pad(p: Pad) -> AuraModel {
  let pleasure = p.pleasure
  let arousal = p.arousal
  let dominance = p.dominance

  // Primary color based on pleasure (warm/cool)
  let primary_color = case pleasure >. 0.0 {
    True -> {
      // Warm colors (gold, orange, pink)
      let warmth = float.min(1.0, pleasure *. 2.0)
      Color(
        r: float_to_int(200.0 +. warmth *. 55.0),
        g: float_to_int(150.0 +. warmth *. 50.0),
        b: float_to_int(100.0 -. warmth *. 50.0),
      )
    }
    False -> {
      // Cool colors (blue, purple, gray)
      let coolness = float.min(1.0, float.absolute_value(pleasure) *. 2.0)
      Color(
        r: float_to_int(100.0 -. coolness *. 30.0),
        g: float_to_int(100.0 +. coolness *. 20.0),
        b: float_to_int(180.0 +. coolness *. 75.0),
      )
    }
  }

  // Secondary color based on dominance (bright/dark)
  let secondary_color = case dominance >. 0.0 {
    True -> {
      // Brighter, more saturated
      Color(
        r: clamp_int(primary_color.r + 30, 0, 255),
        g: clamp_int(primary_color.g + 30, 0, 255),
        b: clamp_int(primary_color.b + 30, 0, 255),
      )
    }
    False -> {
      // Darker, more muted
      Color(
        r: clamp_int(primary_color.r - 30, 0, 255),
        g: clamp_int(primary_color.g - 30, 0, 255),
        b: clamp_int(primary_color.b - 30, 0, 255),
      )
    }
  }

  // Pattern based on arousal and pleasure combination
  let pattern = case arousal >. 0.3, arousal <. -0.3, pleasure >. 0.0 {
    True, _, True -> Flame
    // Excited, happy
    True, _, False -> Fragmented
    // Agitated, unhappy
    _, True, True -> Nebula
    // Calm, content
    _, True, False -> Solid
    // Subdued
    _, _, _ ->
      case dominance >. 0.3 {
        True -> Crystalline
        // Assertive
        False ->
          case dominance <. -0.3 {
            True -> Waves
            // Yielding
            False -> Pulse
            // Neutral
          }
      }
  }

  // Pulse rate based on arousal (faster when aroused)
  let pulse_rate = 1.0 +. arousal *. 1.5
  let pulse_rate = clamp(pulse_rate, 0.5, 3.0)

  // Brightness based on pleasure
  let brightness = 0.5 +. pleasure *. 0.4
  let brightness = clamp(brightness, 0.1, 1.0)

  // Size based on dominance (larger when dominant)
  let size = 1.0 +. dominance *. 0.5
  let size = clamp(size, 0.5, 2.0)

  // Stability based on emotional balance
  let imbalance =
    float.absolute_value(pleasure)
    +. float.absolute_value(arousal)
    +. float.absolute_value(dominance)
  let stability = float.max(0.3, 1.0 -. imbalance *. 0.2)

  AuraModel(
    primary_color: primary_color,
    secondary_color: secondary_color,
    pattern: pattern,
    pulse_rate: pulse_rate,
    brightness: brightness,
    size: size,
    stability: stability,
  )
}

// =============================================================================
// OBSERVATION
// =============================================================================

/// Observe visual state: compare expected (from PAD) vs actual
pub fn observe(
  model: VisualSelfModel,
  p: Pad,
  action: Option(String),
) -> VisualObservation {
  // Expected state from PAD
  let expected_posture = posture_from_pad(p)
  let expected_expression = expression_from_pad(p)
  let expected_aura = aura_from_pad(p)

  // Actual state is current model state
  let actual_posture = model.posture
  let actual_expression = model.expression
  let actual_aura = model.aura

  // Calculate coherence
  let posture_match = posture_similarity(expected_posture, actual_posture)
  let expression_match =
    expression_similarity(expected_expression, actual_expression)
  let aura_match = aura_similarity(expected_aura, actual_aura)
  let coherence = { posture_match +. expression_match +. aura_match } /. 3.0

  VisualObservation(
    tick: model.tick,
    expected_posture: expected_posture,
    expected_expression: expected_expression,
    expected_aura: expected_aura,
    actual_posture: actual_posture,
    actual_expression: actual_expression,
    actual_aura: actual_aura,
    coherence: coherence,
    action: action,
  )
}

// =============================================================================
// UPDATE
// =============================================================================

/// Update visual self-model based on PAD, stress, and motion
pub fn update(
  model: VisualSelfModel,
  p: Pad,
  stress: Float,
  motion: Float,
) -> VisualSelfModel {
  let new_tick = model.tick + 1

  // Derive new visual state from PAD
  let new_posture = posture_from_pad(p)
  let new_expression = expression_from_pad(p)
  let new_aura = aura_from_pad(p)

  // Apply stress effects (reduces symmetry, stability)
  let stressed_expression =
    Expression(
      ..new_expression,
      symmetry: float.max(0.3, new_expression.symmetry -. stress *. 0.2),
    )
  let stressed_aura =
    AuraModel(
      ..new_aura,
      stability: float.max(0.2, new_aura.stability -. stress *. 0.3),
    )

  // Update motion intensity (smoothed)
  let new_motion = model.motion_intensity *. 0.8 +. motion *. 0.2

  // Increase visual awareness with observations
  let awareness_delta = 0.01
  let new_awareness = float.min(1.0, model.visual_awareness +. awareness_delta)

  // Update baseline slowly (EMA with alpha=0.05)
  let alpha = 0.05
  let new_baseline_aura = case new_tick % 100 == 0 {
    True -> blend_aura(model.baseline_aura, new_aura, alpha)
    False -> model.baseline_aura
  }

  VisualSelfModel(
    posture: new_posture,
    expression: stressed_expression,
    aura: stressed_aura,
    baseline_posture: case new_tick % 100 == 0 {
      True -> new_posture
      False -> model.baseline_posture
    },
    baseline_expression: case new_tick % 100 == 0 {
      True -> stressed_expression
      False -> model.baseline_expression
    },
    baseline_aura: new_baseline_aura,
    visual_awareness: new_awareness,
    observation_history: model.observation_history,
    tick: new_tick,
    motion_intensity: new_motion,
  )
}

// =============================================================================
// INTROSPECTION
// =============================================================================

/// Introspect visual state: assess coherence between feeling and appearance
pub fn introspect_visual(model: VisualSelfModel, p: Pad) -> VisualCoherence {
  // Get expected state from current PAD
  let expected_posture = posture_from_pad(p)
  let expected_expression = expression_from_pad(p)
  let expected_aura = aura_from_pad(p)

  // Calculate component coherence
  let posture_match = posture_similarity(expected_posture, model.posture)
  let expression_match =
    expression_similarity(expected_expression, model.expression)
  let aura_match = aura_similarity(expected_aura, model.aura)

  // Overall coherence (weighted average)
  let overall =
    posture_match *. 0.3 +. expression_match *. 0.4 +. aura_match *. 0.3

  // Dissonance threshold
  let dissonant = overall <. 0.5

  // Generate description
  let description = generate_visual_description(model, p)

  // Generate suggestion if dissonant
  let suggestion = case dissonant {
    True ->
      Some(generate_coherence_suggestion(
        posture_match,
        expression_match,
        aura_match,
      ))
    False -> None
  }

  VisualCoherence(
    overall: overall,
    posture_match: posture_match,
    expression_match: expression_match,
    aura_match: aura_match,
    dissonant: dissonant,
    description: description,
    suggestion: suggestion,
  )
}

// =============================================================================
// GLYPH CONVERSION
// =============================================================================

/// Convert visual self-model to 4-token Glyph
pub fn to_glyph(model: VisualSelfModel) -> Glyph {
  // Token 1: Posture (0-255)
  let t1 = posture_to_token(model.posture)

  // Token 2: Expression kind + intensity
  let t2 = expression_to_token(model.expression)

  // Token 3: Aura color (hue-based)
  let t3 = aura_color_to_token(model.aura)

  // Token 4: Aura dynamics (pattern, pulse, stability)
  let t4 = aura_dynamics_to_token(model.aura)

  glyph.new([t1, t2, t3, t4])
}

/// Create visual self-model from Glyph
pub fn from_glyph(g: Glyph) -> VisualSelfModel {
  case g.tokens {
    [t1, t2, t3, t4] -> {
      let posture = token_to_posture(t1)
      let expression = token_to_expression(t2)
      let aura = tokens_to_aura(t3, t4)

      VisualSelfModel(
        posture: posture,
        expression: expression,
        aura: aura,
        baseline_posture: posture,
        baseline_expression: expression,
        baseline_aura: aura,
        visual_awareness: 0.5,
        observation_history: [],
        tick: 0,
        motion_intensity: 0.0,
      )
    }
    _ -> new()
  }
}

// =============================================================================
// SIMILARITY FUNCTIONS
// =============================================================================

/// Calculate similarity between two postures (0.0-1.0)
fn posture_similarity(a: Posture, b: Posture) -> Float {
  case a == b {
    True -> 1.0
    False -> {
      // Group similar postures
      let a_group = posture_group(a)
      let b_group = posture_group(b)
      case a_group == b_group {
        True -> 0.7
        // Same category
        False -> 0.3
        // Different category
      }
    }
  }
}

/// Group postures by category
fn posture_group(p: Posture) -> Int {
  case p {
    Open | Expansive -> 1
    // Positive/confident
    Closed | Contracted | Retreating -> 2
    // Negative/withdrawn
    Forward | Tense -> 3
    // Aggressive/anxious
    Relaxed -> 4
    // Calm
    Neutral -> 0
  }
}

/// Calculate similarity between two expressions (0.0-1.0)
fn expression_similarity(a: Expression, b: Expression) -> Float {
  // Kind similarity
  let kind_sim = case a.kind == b.kind {
    True -> 1.0
    False -> expression_kind_similarity(a.kind, b.kind)
  }

  // Intensity similarity
  let intensity_diff = float.absolute_value(a.intensity -. b.intensity)
  let intensity_sim = 1.0 -. intensity_diff

  // Weighted combination
  kind_sim *. 0.7 +. intensity_sim *. 0.3
}

/// Similarity between expression kinds (0.0-1.0)
fn expression_kind_similarity(a: ExpressionKind, b: ExpressionKind) -> Float {
  // Group similar expressions
  let a_valence = expression_valence(a)
  let b_valence = expression_valence(b)

  case a_valence == b_valence {
    True -> 0.6
    // Same valence
    False -> 0.2
    // Different valence
  }
}

/// Get valence of expression (-1, 0, 1)
fn expression_valence(e: ExpressionKind) -> Int {
  case e {
    Joyful | Excited | Confident -> 1
    Sad | Fearful | Angry | Contemptuous -> -1
    Calm | Surprised | Uncertain | Blank -> 0
  }
}

/// Calculate similarity between two auras (0.0-1.0)
fn aura_similarity(a: AuraModel, b: AuraModel) -> Float {
  // Color similarity
  let color_sim = color_similarity(a.primary_color, b.primary_color)

  // Pattern similarity
  let pattern_sim = case a.pattern == b.pattern {
    True -> 1.0
    False -> 0.4
  }

  // Dynamics similarity
  let pulse_diff = float.absolute_value(a.pulse_rate -. b.pulse_rate) /. 2.5
  let brightness_diff = float.absolute_value(a.brightness -. b.brightness)
  let dynamics_sim = 1.0 -. { pulse_diff +. brightness_diff } /. 2.0

  // Weighted combination
  color_sim *. 0.4 +. pattern_sim *. 0.3 +. dynamics_sim *. 0.3
}

/// Calculate color similarity (0.0-1.0)
fn color_similarity(a: Color, b: Color) -> Float {
  let r_diff = int.absolute_value(a.r - b.r)
  let g_diff = int.absolute_value(a.g - b.g)
  let b_diff = int.absolute_value(a.b - b.b)
  let total_diff = r_diff + g_diff + b_diff
  // Max diff is 765 (255*3)
  1.0 -. int_to_float(total_diff) /. 765.0
}

// =============================================================================
// TOKEN CONVERSION HELPERS
// =============================================================================

fn posture_to_token(p: Posture) -> Int {
  case p {
    Neutral -> 128
    Open -> 200
    Closed -> 56
    Forward -> 220
    Retreating -> 36
    Relaxed -> 180
    Tense -> 76
    Expansive -> 240
    Contracted -> 16
  }
}

fn token_to_posture(t: Int) -> Posture {
  case t {
    n if n < 32 -> Contracted
    n if n < 64 -> Retreating
    n if n < 96 -> Closed
    n if n < 112 -> Tense
    n if n < 144 -> Neutral
    n if n < 176 -> Relaxed
    n if n < 208 -> Open
    n if n < 232 -> Forward
    _ -> Expansive
  }
}

fn expression_to_token(e: Expression) -> Int {
  let kind_base = case e.kind {
    Blank -> 0
    Sad -> 20
    Fearful -> 40
    Uncertain -> 60
    Calm -> 80
    Surprised -> 100
    Confident -> 120
    Joyful -> 140
    Excited -> 160
    Angry -> 180
    Contemptuous -> 200
  }
  // Add intensity (0-55 range)
  let intensity_add = float_to_int(e.intensity *. 55.0)
  clamp_int(kind_base + intensity_add, 0, 255)
}

fn token_to_expression(t: Int) -> Expression {
  let kind = case t {
    n if n < 20 -> Blank
    n if n < 40 -> Sad
    n if n < 60 -> Fearful
    n if n < 80 -> Uncertain
    n if n < 100 -> Calm
    n if n < 120 -> Surprised
    n if n < 140 -> Confident
    n if n < 160 -> Joyful
    n if n < 180 -> Excited
    n if n < 200 -> Angry
    _ -> Contemptuous
  }
  let intensity = int_to_float(t % 20) /. 20.0
  Expression(kind: kind, intensity: intensity, symmetry: 1.0, micro: None)
}

fn aura_color_to_token(a: AuraModel) -> Int {
  // Simple hue-based encoding
  let r = a.primary_color.r
  let g = a.primary_color.g
  let b = a.primary_color.b
  // Weighted average with emphasis on dominant channel
  { r * 2 + g + b } / 4
}

fn aura_dynamics_to_token(a: AuraModel) -> Int {
  // Encode pattern (0-7) in upper 3 bits
  let pattern_bits = case a.pattern {
    Solid -> 0
    Waves -> 1
    Spiral -> 2
    Pulse -> 3
    Fragmented -> 4
    Nebula -> 5
    Crystalline -> 6
    Flame -> 7
  }

  // Encode pulse rate (0-31) in middle 5 bits
  let pulse_normalized = { a.pulse_rate -. 0.5 } /. 2.5
  let pulse_bits = float_to_int(pulse_normalized *. 31.0)

  // Combine
  pattern_bits * 32 + clamp_int(pulse_bits, 0, 31)
}

fn tokens_to_aura(color_token: Int, dynamics_token: Int) -> AuraModel {
  // Decode color (grayscale for simplicity)
  let primary = Color(r: color_token, g: color_token, b: color_token + 40)
  let secondary =
    Color(
      r: clamp_int(color_token - 20, 0, 255),
      g: clamp_int(color_token - 20, 0, 255),
      b: clamp_int(color_token + 20, 0, 255),
    )

  // Decode pattern
  let pattern_bits = dynamics_token / 32
  let pattern = case pattern_bits {
    0 -> Solid
    1 -> Waves
    2 -> Spiral
    3 -> Pulse
    4 -> Fragmented
    5 -> Nebula
    6 -> Crystalline
    _ -> Flame
  }

  // Decode pulse
  let pulse_bits = dynamics_token % 32
  let pulse_rate = 0.5 +. int_to_float(pulse_bits) /. 31.0 *. 2.5

  AuraModel(
    primary_color: primary,
    secondary_color: secondary,
    pattern: pattern,
    pulse_rate: pulse_rate,
    brightness: 0.5,
    size: 1.0,
    stability: 1.0,
  )
}

// =============================================================================
// DESCRIPTION GENERATION
// =============================================================================

fn generate_visual_description(model: VisualSelfModel, _p: Pad) -> String {
  let posture_desc = posture_to_string(model.posture)
  let expression_desc = expression_to_string(model.expression)
  let aura_desc = aura_to_string(model.aura)

  "My posture is "
  <> posture_desc
  <> ", my expression is "
  <> expression_desc
  <> ", and my aura appears "
  <> aura_desc
  <> "."
}

fn generate_coherence_suggestion(
  posture_match: Float,
  expression_match: Float,
  aura_match: Float,
) -> String {
  // Find lowest match
  case
    posture_match <. expression_match && posture_match <. aura_match,
    expression_match <. aura_match
  {
    True, _ -> "I should adjust my posture to match how I feel inside."
    _, True -> "My expression doesn't match my internal state."
    _, _ -> "My aura energy feels disconnected from my emotions."
  }
}

// =============================================================================
// STRING CONVERSIONS
// =============================================================================

/// Convert posture to descriptive string
pub fn posture_to_string(p: Posture) -> String {
  case p {
    Open -> "open and welcoming"
    Closed -> "closed and protective"
    Forward -> "leaning forward, assertive"
    Retreating -> "pulling back, withdrawn"
    Relaxed -> "relaxed and at ease"
    Tense -> "tense and rigid"
    Expansive -> "expansive and confident"
    Contracted -> "small and contracted"
    Neutral -> "neutral"
  }
}

/// Convert expression to descriptive string
pub fn expression_to_string(e: Expression) -> String {
  let intensity_word = case e.intensity {
    i if i <. 0.3 -> "subtly"
    i if i <. 0.6 -> "noticeably"
    i if i <. 0.8 -> "clearly"
    _ -> "intensely"
  }

  let kind_word = case e.kind {
    Joyful -> "joyful"
    Sad -> "sad"
    Excited -> "excited"
    Calm -> "calm"
    Confident -> "confident"
    Uncertain -> "uncertain"
    Fearful -> "fearful"
    Angry -> "angry"
    Surprised -> "surprised"
    Contemptuous -> "contemptuous"
    Blank -> "neutral"
  }

  intensity_word <> " " <> kind_word
}

/// Convert aura to descriptive string
pub fn aura_to_string(a: AuraModel) -> String {
  let color_word = color_to_word(a.primary_color)
  let pattern_word = pattern_to_word(a.pattern)
  let energy_word = case a.pulse_rate {
    r if r <. 1.0 -> "slow"
    r if r <. 1.5 -> "steady"
    r if r <. 2.0 -> "quick"
    _ -> "rapid"
  }

  color_word <> " and " <> pattern_word <> " with " <> energy_word <> " pulsing"
}

fn color_to_word(c: Color) -> String {
  // Determine dominant color
  case c.r > c.g && c.r > c.b, c.g > c.b {
    True, _ ->
      case c.r > 200 {
        True -> "bright red"
        False -> "warm reddish"
      }
    _, True ->
      case c.g > 200 {
        True -> "vibrant green"
        False -> "soft green"
      }
    _, _ ->
      case c.b > 200 {
        True -> "deep blue"
        False -> "cool blue"
      }
  }
}

fn pattern_to_word(p: AuraPattern) -> String {
  case p {
    Solid -> "solid"
    Waves -> "flowing with waves"
    Spiral -> "spiraling"
    Pulse -> "rhythmically pulsing"
    Fragmented -> "fragmented"
    Nebula -> "soft and nebulous"
    Crystalline -> "crystalline"
    Flame -> "flame-like"
  }
}

/// Convert expression kind to string
pub fn expression_kind_to_string(e: ExpressionKind) -> String {
  case e {
    Joyful -> "joyful"
    Sad -> "sad"
    Excited -> "excited"
    Calm -> "calm"
    Confident -> "confident"
    Uncertain -> "uncertain"
    Fearful -> "fearful"
    Angry -> "angry"
    Surprised -> "surprised"
    Contemptuous -> "contemptuous"
    Blank -> "blank"
  }
}

/// Convert aura pattern to string
pub fn pattern_to_string(p: AuraPattern) -> String {
  case p {
    Solid -> "solid"
    Waves -> "waves"
    Spiral -> "spiral"
    Pulse -> "pulse"
    Fragmented -> "fragmented"
    Nebula -> "nebula"
    Crystalline -> "crystalline"
    Flame -> "flame"
  }
}

// =============================================================================
// QUERIES
// =============================================================================

/// Get current posture
pub fn get_posture(model: VisualSelfModel) -> Posture {
  model.posture
}

/// Get current expression
pub fn get_expression(model: VisualSelfModel) -> Expression {
  model.expression
}

/// Get current aura
pub fn get_aura(model: VisualSelfModel) -> AuraModel {
  model.aura
}

/// Get visual awareness level
pub fn awareness(model: VisualSelfModel) -> Float {
  model.visual_awareness
}

/// Get motion intensity
pub fn motion(model: VisualSelfModel) -> Float {
  model.motion_intensity
}

/// Get observation count
pub fn observation_count(model: VisualSelfModel) -> Int {
  list.length(model.observation_history)
}

/// Get recent observations
pub fn recent_observations(
  model: VisualSelfModel,
  limit: Int,
) -> List(VisualObservation) {
  list.take(model.observation_history, limit)
}

/// Store an observation in history
pub fn record_observation(
  model: VisualSelfModel,
  obs: VisualObservation,
) -> VisualSelfModel {
  let new_history = [obs, ..model.observation_history]
  let trimmed = list.take(new_history, 50)
  VisualSelfModel(..model, observation_history: trimmed)
}

/// Calculate average coherence from recent observations
pub fn average_coherence(model: VisualSelfModel, window: Int) -> Float {
  let recent = list.take(model.observation_history, window)
  case recent {
    [] -> 1.0
    // No observations = assume coherent
    obs -> {
      let sum =
        list.fold(obs, 0.0, fn(acc, o: VisualObservation) { acc +. o.coherence })
      sum /. int_to_float(list.length(obs))
    }
  }
}

// =============================================================================
// BLENDING HELPERS
// =============================================================================

fn blend_aura(
  baseline: AuraModel,
  current: AuraModel,
  alpha: Float,
) -> AuraModel {
  AuraModel(
    primary_color: blend_color(
      baseline.primary_color,
      current.primary_color,
      alpha,
    ),
    secondary_color: blend_color(
      baseline.secondary_color,
      current.secondary_color,
      alpha,
    ),
    pattern: case alpha >. 0.5 {
      True -> current.pattern
      False -> baseline.pattern
    },
    pulse_rate: baseline.pulse_rate
      *. { 1.0 -. alpha }
      +. current.pulse_rate
      *. alpha,
    brightness: baseline.brightness
      *. { 1.0 -. alpha }
      +. current.brightness
      *. alpha,
    size: baseline.size *. { 1.0 -. alpha } +. current.size *. alpha,
    stability: baseline.stability
      *. { 1.0 -. alpha }
      +. current.stability
      *. alpha,
  )
}

fn blend_color(a: Color, b: Color, alpha: Float) -> Color {
  Color(
    r: float_to_int(
      int_to_float(a.r) *. { 1.0 -. alpha } +. int_to_float(b.r) *. alpha,
    ),
    g: float_to_int(
      int_to_float(a.g) *. { 1.0 -. alpha } +. int_to_float(b.g) *. alpha,
    ),
    b: float_to_int(
      int_to_float(a.b) *. { 1.0 -. alpha } +. int_to_float(b.b) *. alpha,
    ),
  )
}

// =============================================================================
// NARRATIVE INTEGRATION
// =============================================================================

/// Generate visual insight for narrative
pub fn visual_insight(model: VisualSelfModel, p: Pad) -> Option(String) {
  let coherence = introspect_visual(model, p)

  case coherence.dissonant {
    True ->
      Some(
        "I notice a disconnect between how I feel and how I appear. "
        <> coherence.description,
      )
    False -> {
      // Check for significant visual changes
      case model.motion_intensity >. 0.5 {
        True ->
          Some(
            "My visual presence is highly animated. " <> coherence.description,
          )
        False -> None
      }
    }
  }
}

/// Describe visual state for inner monologue
pub fn describe_for_narrative(model: VisualSelfModel) -> String {
  let posture_str = posture_to_string(model.posture)
  let expression_str = expression_to_string(model.expression)
  let aura_str = aura_to_string(model.aura)

  "Visually, I present as "
  <> posture_str
  <> " with a "
  <> expression_str
  <> " expression. My aura is "
  <> aura_str
  <> "."
}

// =============================================================================
// SOUL INTEGRATION TYPES
// =============================================================================

/// Visual state summary for SoulState integration
pub type VisualSummary {
  VisualSummary(
    posture: Posture,
    expression_kind: ExpressionKind,
    expression_intensity: Float,
    aura_primary_color: Color,
    aura_pattern: AuraPattern,
    coherence: Float,
    motion: Float,
  )
}

/// Create visual summary from model
pub fn summarize(model: VisualSelfModel, p: Pad) -> VisualSummary {
  let coherence = introspect_visual(model, p)

  VisualSummary(
    posture: model.posture,
    expression_kind: model.expression.kind,
    expression_intensity: model.expression.intensity,
    aura_primary_color: model.aura.primary_color,
    aura_pattern: model.aura.pattern,
    coherence: coherence.overall,
    motion: model.motion_intensity,
  )
}

// =============================================================================
// HELPERS
// =============================================================================

fn clamp(value: Float, min: Float, max: Float) -> Float {
  float.min(max, float.max(min, value))
}

fn clamp_int(value: Int, min: Int, max: Int) -> Int {
  case value {
    v if v < min -> min
    v if v > max -> max
    v -> v
  }
}

@external(erlang, "erlang", "trunc")
fn float_to_int(f: Float) -> Int

fn int_to_float(n: Int) -> Float {
  case n <= 0 {
    True -> 0.0
    False -> 1.0 +. int_to_float(n - 1)
  }
}
