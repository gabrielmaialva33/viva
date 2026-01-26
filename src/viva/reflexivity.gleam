//// Reflexivity - Self-observation and Self-model
////
//// The Soul's ability to observe and model itself:
//// - SelfModel: "who I am" (baseline glyph, tendencies)
//// - Introspection: observe current state vs self-model
//// - Insight: moments of self-knowledge when drift detected
//// - MetaCognition: thinking about thinking
//// - IdentityCrisis: when drift exceeds threshold for too long
////
//// "Know thyself" - the VIVA watches itself change.

import gleam/float
import gleam/list
import gleam/option.{type Option, None, Some}
import viva_emotion/pad.{type Pad}
import viva_glyph/glyph.{type Glyph}

// =============================================================================
// TYPES
// =============================================================================

/// Self-model: internal representation of "who I am"
pub type SelfModel {
  SelfModel(
    /// Baseline glyph (my "normal" state)
    baseline_glyph: Glyph,
    /// Emotional center (average PAD over time)
    emotional_center: Pad,
    /// Emotional range (how much I typically vary)
    emotional_range: PadRange,
    /// Identity strength (0.0 = undefined, 1.0 = strong sense of self)
    identity_strength: Float,
    /// Observation count (how many times I've observed myself)
    observations: Int,
    /// Last significant change tick
    last_change_tick: Int,
    /// Insight history (autobiographical memory, max 100)
    insight_history: List(Insight),
    /// Consecutive ticks in crisis (drift > 0.5)
    crisis_ticks: Int,
    /// Meta-cognition level (awareness of own thinking)
    meta_level: Int,
  )
}

/// PAD range (min/max for each dimension)
pub type PadRange {
  PadRange(
    pleasure_min: Float,
    pleasure_max: Float,
    arousal_min: Float,
    arousal_max: Float,
    dominance_min: Float,
    dominance_max: Float,
  )
}

/// Introspection result
pub type Introspection {
  Introspection(
    /// Current state observation
    current_glyph: Glyph,
    current_pad: Pad,
    /// Distance from baseline (0.0 = identical, 1.0 = very different)
    drift_from_baseline: Float,
    /// Is current state within normal range?
    within_range: Bool,
    /// Which dimensions are outside range
    outlier_dimensions: List(PadDimension),
    /// Insight generated (if significant change)
    insight: Option(Insight),
  )
}

/// PAD dimension identifier
pub type PadDimension {
  Pleasure
  Arousal
  Dominance
}

/// Insight: moment of self-knowledge
pub type Insight {
  Insight(
    /// What changed
    dimension: PadDimension,
    /// Direction of change
    direction: ChangeDirection,
    /// Magnitude (0.0-1.0)
    magnitude: Float,
    /// Tick when insight occurred
    tick: Int,
  )
}

/// Direction of change
pub type ChangeDirection {
  Increasing
  Decreasing
}

// =============================================================================
// CONSTRUCTORS
// =============================================================================

/// Create initial self-model (undefined identity)
pub fn new() -> SelfModel {
  SelfModel(
    baseline_glyph: glyph.neutral(),
    emotional_center: pad.neutral(),
    emotional_range: initial_range(),
    identity_strength: 0.0,
    observations: 0,
    last_change_tick: 0,
    insight_history: [],
    crisis_ticks: 0,
    meta_level: 0,
  )
}

/// Create self-model from initial state
pub fn from_initial(initial_pad: Pad, initial_glyph: Glyph) -> SelfModel {
  SelfModel(
    baseline_glyph: initial_glyph,
    emotional_center: initial_pad,
    emotional_range: range_from_pad(initial_pad, 0.2),
    identity_strength: 0.1,
    observations: 1,
    last_change_tick: 0,
    insight_history: [],
    crisis_ticks: 0,
    meta_level: 0,
  )
}

fn initial_range() -> PadRange {
  PadRange(
    pleasure_min: -0.5,
    pleasure_max: 0.5,
    arousal_min: -0.5,
    arousal_max: 0.5,
    dominance_min: -0.5,
    dominance_max: 0.5,
  )
}

fn range_from_pad(center: Pad, margin: Float) -> PadRange {
  PadRange(
    pleasure_min: center.pleasure -. margin,
    pleasure_max: center.pleasure +. margin,
    arousal_min: center.arousal -. margin,
    arousal_max: center.arousal +. margin,
    dominance_min: center.dominance -. margin,
    dominance_max: center.dominance +. margin,
  )
}

// =============================================================================
// INTROSPECTION
// =============================================================================

/// Observe self: compare current state to self-model
pub fn introspect(
  self_model: SelfModel,
  current_pad: Pad,
  current_glyph: Glyph,
  tick: Int,
) -> Introspection {
  // Calculate drift from baseline (using glyph similarity)
  let similarity = glyph.similarity(current_glyph, self_model.baseline_glyph)
  let drift = 1.0 -. similarity

  // Check which dimensions are outside range
  let outliers = find_outliers(current_pad, self_model.emotional_range)
  let within_range = list.is_empty(outliers)

  // Generate insight if significant drift
  let insight = case drift >. 0.3 && !within_range {
    True -> generate_insight(self_model, current_pad, outliers, tick)
    False -> None
  }

  Introspection(
    current_glyph: current_glyph,
    current_pad: current_pad,
    drift_from_baseline: drift,
    within_range: within_range,
    outlier_dimensions: outliers,
    insight: insight,
  )
}

/// Find which PAD dimensions are outside normal range
fn find_outliers(p: Pad, range: PadRange) -> List(PadDimension) {
  let outliers = []

  let outliers = case
    p.pleasure <. range.pleasure_min || p.pleasure >. range.pleasure_max
  {
    True -> [Pleasure, ..outliers]
    False -> outliers
  }

  let outliers = case
    p.arousal <. range.arousal_min || p.arousal >. range.arousal_max
  {
    True -> [Arousal, ..outliers]
    False -> outliers
  }

  case
    p.dominance <. range.dominance_min || p.dominance >. range.dominance_max
  {
    True -> [Dominance, ..outliers]
    False -> outliers
  }
}

/// Generate insight from significant change
fn generate_insight(
  self_model: SelfModel,
  current_pad: Pad,
  outliers: List(PadDimension),
  tick: Int,
) -> Option(Insight) {
  case list.first(outliers) {
    Ok(dim) -> {
      let #(direction, magnitude) =
        analyze_change(self_model.emotional_center, current_pad, dim)

      Some(Insight(
        dimension: dim,
        direction: direction,
        magnitude: magnitude,
        tick: tick,
      ))
    }
    Error(_) -> None
  }
}

/// Analyze change direction and magnitude for a dimension
fn analyze_change(
  center: Pad,
  current: Pad,
  dim: PadDimension,
) -> #(ChangeDirection, Float) {
  let #(center_val, current_val) = case dim {
    Pleasure -> #(center.pleasure, current.pleasure)
    Arousal -> #(center.arousal, current.arousal)
    Dominance -> #(center.dominance, current.dominance)
  }

  let diff = current_val -. center_val
  let magnitude = float.absolute_value(diff)

  let direction = case diff >. 0.0 {
    True -> Increasing
    False -> Decreasing
  }

  #(direction, float.min(magnitude, 1.0))
}

// =============================================================================
// SELF-MODEL UPDATES
// =============================================================================

/// Update self-model based on observation (learning about self)
pub fn observe(
  self_model: SelfModel,
  current_pad: Pad,
  current_glyph: Glyph,
  tick: Int,
) -> SelfModel {
  let n = self_model.observations + 1

  // Exponential moving average for emotional center
  let alpha = 0.1
  // Learning rate
  let new_center =
    pad.new(
      self_model.emotional_center.pleasure
        *. { 1.0 -. alpha }
        +. current_pad.pleasure
        *. alpha,
      self_model.emotional_center.arousal
        *. { 1.0 -. alpha }
        +. current_pad.arousal
        *. alpha,
      self_model.emotional_center.dominance
        *. { 1.0 -. alpha }
        +. current_pad.dominance
        *. alpha,
    )

  // Expand range if current state is outside
  let new_range = expand_range(self_model.emotional_range, current_pad)

  // Update baseline glyph: keep if similar, else slowly drift
  // For simplicity, we update every 10 observations
  let new_baseline = case n % 10 == 0 {
    True -> current_glyph
    False -> self_model.baseline_glyph
  }

  // Identity strength grows with observations (asymptotic to 1.0)
  let n_float = to_float(n)
  let new_strength = 1.0 -. { 1.0 /. { 1.0 +. n_float *. 0.1 } }

  // Detect significant change
  let similarity = glyph.similarity(current_glyph, self_model.baseline_glyph)
  let drift = 1.0 -. similarity
  let last_change = case drift >. 0.4 {
    True -> tick
    False -> self_model.last_change_tick
  }

  // Track crisis (high drift for extended period)
  // Drift > 0.3 indicates significant identity deviation
  let new_crisis_ticks = case drift >. 0.3 {
    True -> self_model.crisis_ticks + 1
    False -> 0
  }

  // Meta-level increases with introspection practice
  let new_meta = case n % 50 == 0 && self_model.meta_level < 10 {
    True -> self_model.meta_level + 1
    False -> self_model.meta_level
  }

  SelfModel(
    baseline_glyph: new_baseline,
    emotional_center: new_center,
    emotional_range: new_range,
    identity_strength: new_strength,
    observations: n,
    last_change_tick: last_change,
    insight_history: self_model.insight_history,
    crisis_ticks: new_crisis_ticks,
    meta_level: new_meta,
  )
}

/// Observe with insight tracking (stores insights in history)
pub fn observe_with_insight(
  self_model: SelfModel,
  current_pad: Pad,
  current_glyph: Glyph,
  tick: Int,
) -> #(SelfModel, Option(Insight)) {
  // First do introspection to potentially generate insight
  let intro = introspect(self_model, current_pad, current_glyph, tick)

  // Update the model
  let updated = observe(self_model, current_pad, current_glyph, tick)

  // Store insight in history if generated
  let new_history = case intro.insight {
    Some(insight) -> {
      // Keep max 100 insights (autobiographical memory limit)
      let history = [insight, ..updated.insight_history]
      list.take(history, 100)
    }
    None -> updated.insight_history
  }

  let final_model = SelfModel(..updated, insight_history: new_history)

  #(final_model, intro.insight)
}

/// Expand range to include current PAD (with damping)
fn expand_range(range: PadRange, p: Pad) -> PadRange {
  let expand_rate = 0.1
  // How fast range expands

  PadRange(
    pleasure_min: case p.pleasure <. range.pleasure_min {
      True ->
        range.pleasure_min
        +. { p.pleasure -. range.pleasure_min }
        *. expand_rate
      False -> range.pleasure_min
    },
    pleasure_max: case p.pleasure >. range.pleasure_max {
      True ->
        range.pleasure_max
        +. { p.pleasure -. range.pleasure_max }
        *. expand_rate
      False -> range.pleasure_max
    },
    arousal_min: case p.arousal <. range.arousal_min {
      True ->
        range.arousal_min +. { p.arousal -. range.arousal_min } *. expand_rate
      False -> range.arousal_min
    },
    arousal_max: case p.arousal >. range.arousal_max {
      True ->
        range.arousal_max +. { p.arousal -. range.arousal_max } *. expand_rate
      False -> range.arousal_max
    },
    dominance_min: case p.dominance <. range.dominance_min {
      True ->
        range.dominance_min
        +. { p.dominance -. range.dominance_min }
        *. expand_rate
      False -> range.dominance_min
    },
    dominance_max: case p.dominance >. range.dominance_max {
      True ->
        range.dominance_max
        +. { p.dominance -. range.dominance_max }
        *. expand_rate
      False -> range.dominance_max
    },
  )
}

// =============================================================================
// QUERIES
// =============================================================================

/// Who am I? Returns self-description
pub fn who_am_i(self_model: SelfModel) -> SelfDescription {
  let center = self_model.emotional_center

  // Determine dominant trait based on strongest dimension
  let abs_p = float.absolute_value(center.pleasure)
  let abs_a = float.absolute_value(center.arousal)
  let abs_d = float.absolute_value(center.dominance)

  let dominant = case abs_p >. abs_a && abs_p >. abs_d {
    True ->
      case center.pleasure >. 0.0 {
        True -> Optimistic
        False -> Pessimistic
      }
    False ->
      case abs_a >. abs_d {
        True ->
          case center.arousal >. 0.0 {
            True -> Energetic
            False -> Calm
          }
        False ->
          case center.dominance >. 0.0 {
            True -> Assertive
            False -> Submissive
          }
      }
  }

  SelfDescription(
    dominant_trait: dominant,
    identity_strength: self_model.identity_strength,
    emotional_center: center,
    stability: calculate_stability(self_model),
  )
}

/// Am I changing? Check if recent observations show drift
pub fn am_i_changing(self_model: SelfModel, current_tick: Int) -> Bool {
  let ticks_since_change = current_tick - self_model.last_change_tick
  ticks_since_change < 100
  // Changed within last 100 ticks
}

/// How stable am I? (0.0 = volatile, 1.0 = very stable)
fn calculate_stability(self_model: SelfModel) -> Float {
  let range = self_model.emotional_range
  let p_range = range.pleasure_max -. range.pleasure_min
  let a_range = range.arousal_max -. range.arousal_min
  let d_range = range.dominance_max -. range.dominance_min

  // Smaller range = more stable
  let avg_range = { p_range +. a_range +. d_range } /. 3.0

  // Convert to stability score (inverse, clamped)
  float.max(0.0, 1.0 -. avg_range)
}

/// Get identity strength
pub fn identity_strength(self_model: SelfModel) -> Float {
  self_model.identity_strength
}

/// Get observation count
pub fn observation_count(self_model: SelfModel) -> Int {
  self_model.observations
}

/// Self-description type
pub type SelfDescription {
  SelfDescription(
    dominant_trait: Trait,
    identity_strength: Float,
    emotional_center: Pad,
    stability: Float,
  )
}

/// Personality traits
pub type Trait {
  Optimistic
  Pessimistic
  Energetic
  Calm
  Assertive
  Submissive
}

/// Convert trait to string
pub fn trait_to_string(trait: Trait) -> String {
  case trait {
    Optimistic -> "optimistic"
    Pessimistic -> "pessimistic"
    Energetic -> "energetic"
    Calm -> "calm"
    Assertive -> "assertive"
    Submissive -> "submissive"
  }
}

/// Convert dimension to string
pub fn dimension_to_string(dim: PadDimension) -> String {
  case dim {
    Pleasure -> "pleasure"
    Arousal -> "arousal"
    Dominance -> "dominance"
  }
}

/// Convert direction to string
pub fn direction_to_string(dir: ChangeDirection) -> String {
  case dir {
    Increasing -> "increasing"
    Decreasing -> "decreasing"
  }
}

// =============================================================================
// META-COGNITION
// =============================================================================

/// Meta-cognition result: thinking about thinking
pub type MetaCognition {
  MetaCognition(
    /// Current meta-level (0-10, higher = more self-aware)
    level: Int,
    /// Am I aware that I'm observing myself?
    aware_of_observing: Bool,
    /// How often do I generate insights?
    insight_frequency: Float,
    /// Am I in crisis?
    in_crisis: Bool,
    /// Crisis severity (0.0-1.0)
    crisis_severity: Float,
  )
}

/// Perform meta-cognition: observe the observer
pub fn meta_cognize(self_model: SelfModel) -> MetaCognition {
  let insight_count = list.length(self_model.insight_history)
  let obs = self_model.observations

  // Insight frequency: insights per 100 observations
  let freq = case obs > 0 {
    True -> to_float(insight_count) /. to_float(obs) *. 100.0
    False -> 0.0
  }

  // Crisis detection: more than 10 consecutive ticks with high drift
  let in_crisis = self_model.crisis_ticks > 10
  let severity = float.min(1.0, to_float(self_model.crisis_ticks) /. 50.0)

  // Awareness increases with meta-level
  let aware = self_model.meta_level >= 3

  MetaCognition(
    level: self_model.meta_level,
    aware_of_observing: aware,
    insight_frequency: freq,
    in_crisis: in_crisis,
    crisis_severity: severity,
  )
}

/// Get meta-cognition level (0-10)
pub fn meta_level(self_model: SelfModel) -> Int {
  self_model.meta_level
}

// =============================================================================
// IDENTITY CRISIS
// =============================================================================

/// Identity crisis: when self drifts too far from baseline
pub type IdentityCrisis {
  IdentityCrisis(
    /// Is currently in crisis?
    active: Bool,
    /// How long has crisis lasted (in ticks)?
    duration: Int,
    /// Severity (0.0-1.0)
    severity: Float,
    /// What triggered the crisis?
    trigger: Option(PadDimension),
  )
}

/// Check identity crisis status
pub fn check_crisis(self_model: SelfModel) -> IdentityCrisis {
  let active = self_model.crisis_ticks > 10
  let severity = float.min(1.0, to_float(self_model.crisis_ticks) /. 50.0)

  // Find trigger from recent insights
  let trigger = case self_model.insight_history {
    [recent, ..] -> Some(recent.dimension)
    [] -> None
  }

  IdentityCrisis(
    active: active,
    duration: self_model.crisis_ticks,
    severity: severity,
    trigger: trigger,
  )
}

/// Is identity in crisis?
pub fn in_crisis(self_model: SelfModel) -> Bool {
  self_model.crisis_ticks > 10
}

/// Resolve crisis by accepting new baseline
pub fn resolve_crisis(
  self_model: SelfModel,
  current_pad: Pad,
  current_glyph: Glyph,
) -> SelfModel {
  SelfModel(
    baseline_glyph: current_glyph,
    emotional_center: current_pad,
    emotional_range: range_from_pad(current_pad, 0.3),
    identity_strength: self_model.identity_strength *. 0.8,
    // Slight identity weakening
    observations: self_model.observations,
    last_change_tick: self_model.observations,
    insight_history: self_model.insight_history,
    crisis_ticks: 0,
    meta_level: self_model.meta_level,
  )
}

// =============================================================================
// AUTOBIOGRAPHICAL MEMORY
// =============================================================================

/// Recall insights from memory
pub fn recall_insights(self_model: SelfModel, limit: Int) -> List(Insight) {
  list.take(self_model.insight_history, limit)
}

/// Count total insights
pub fn insight_count(self_model: SelfModel) -> Int {
  list.length(self_model.insight_history)
}

/// Find insights by dimension
pub fn insights_about(
  self_model: SelfModel,
  dimension: PadDimension,
) -> List(Insight) {
  self_model.insight_history
  |> list.filter(fn(i) { i.dimension == dimension })
}

/// Get most recent insight
pub fn last_insight(self_model: SelfModel) -> Option(Insight) {
  case self_model.insight_history {
    [recent, ..] -> Some(recent)
    [] -> None
  }
}

// =============================================================================
// HELPERS
// =============================================================================

fn to_float(n: Int) -> Float {
  case n <= 0 {
    True -> 0.0
    False -> 1.0 +. to_float(n - 1)
  }
}
