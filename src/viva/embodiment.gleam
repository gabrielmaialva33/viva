//// Embodiment - Physical body for VIVA
////
//// A body has needs that decay over time:
//// - Energy (depletes with activity)
//// - Satiety (hunger)
//// - Rest (fatigue)
////
//// Unmet needs create suffering (negative PAD delta).
//// The body grounds the Soul in physical reality.

import gleam/float
import viva_math/vector.{type Vec3}

// =============================================================================
// TYPES
// =============================================================================

/// Physical body state
pub type Body {
  Body(
    /// Energy level (0.0 = exhausted, 1.0 = full)
    energy: Float,
    /// Satiety level (0.0 = starving, 1.0 = satisfied)
    satiety: Float,
    /// Rest level (0.0 = exhausted, 1.0 = well-rested)
    rest: Float,
    /// Body configuration
    config: BodyConfig,
    /// Accumulated stress (from unmet needs)
    stress: Float,
  )
}

/// Body configuration (metabolism rates)
pub type BodyConfig {
  BodyConfig(
    /// Energy decay per tick (default 0.001)
    energy_decay: Float,
    /// Satiety decay per tick (default 0.0005)
    satiety_decay: Float,
    /// Rest decay per tick (default 0.0002)
    rest_decay: Float,
    /// Stress accumulation rate when needs unmet
    stress_rate: Float,
    /// Stress decay when needs met
    stress_recovery: Float,
  )
}

/// Stimulus that can affect the body
pub type BodyStimulus {
  /// Receive energy (activity, motivation)
  Energize(amount: Float)
  /// Consume food (nourishment)
  Feed(amount: Float)
  /// Rest/sleep
  Rest(amount: Float)
  /// External stress (environment)
  Stress(amount: Float)
  /// Physical pain
  Pain(intensity: Float)
  /// Physical pleasure
  Comfort(intensity: Float)
}

/// Body needs assessment
pub type NeedsState {
  NeedsState(
    /// Overall wellbeing (0.0 = suffering, 1.0 = thriving)
    wellbeing: Float,
    /// Is energy critical? (< 0.2)
    energy_critical: Bool,
    /// Is hunger critical? (< 0.2)
    hunger_critical: Bool,
    /// Is fatigue critical? (< 0.2)
    fatigue_critical: Bool,
    /// Is stress high? (> 0.7)
    stressed: Bool,
  )
}

// =============================================================================
// CONSTRUCTORS
// =============================================================================

/// Create new body with default config
pub fn new() -> Body {
  new_with_config(default_config())
}

/// Create body with custom config
pub fn new_with_config(config: BodyConfig) -> Body {
  Body(energy: 0.8, satiety: 0.8, rest: 0.8, config: config, stress: 0.0)
}

/// Default body configuration
pub fn default_config() -> BodyConfig {
  BodyConfig(
    energy_decay: 0.001,
    satiety_decay: 0.0005,
    rest_decay: 0.0002,
    stress_rate: 0.01,
    stress_recovery: 0.005,
  )
}

/// Fast metabolism (decays faster, needs more care)
pub fn fast_metabolism() -> BodyConfig {
  BodyConfig(
    energy_decay: 0.002,
    satiety_decay: 0.001,
    rest_decay: 0.0005,
    stress_rate: 0.02,
    stress_recovery: 0.01,
  )
}

/// Slow metabolism (hardy, resilient)
pub fn slow_metabolism() -> BodyConfig {
  BodyConfig(
    energy_decay: 0.0005,
    satiety_decay: 0.00025,
    rest_decay: 0.0001,
    stress_rate: 0.005,
    stress_recovery: 0.003,
  )
}

// =============================================================================
// SIMULATION
// =============================================================================

/// Tick the body (decay needs, accumulate/recover stress)
pub fn tick(body: Body) -> Body {
  // Decay needs
  let new_energy = float.max(0.0, body.energy -. body.config.energy_decay)
  let new_satiety = float.max(0.0, body.satiety -. body.config.satiety_decay)
  let new_rest = float.max(0.0, body.rest -. body.config.rest_decay)

  // Assess if needs are unmet
  let energy_unmet = new_energy <. 0.3
  let satiety_unmet = new_satiety <. 0.3
  let rest_unmet = new_rest <. 0.3

  let unmet_count =
    case energy_unmet {
      True -> 1.0
      False -> 0.0
    }
    +. case satiety_unmet {
      True -> 1.0
      False -> 0.0
    }
    +. case rest_unmet {
      True -> 1.0
      False -> 0.0
    }

  // Stress accumulates when needs unmet, recovers otherwise
  let stress_delta = case unmet_count >. 0.0 {
    True -> unmet_count *. body.config.stress_rate
    False -> 0.0 -. body.config.stress_recovery
  }
  let new_stress = clamp(body.stress +. stress_delta, 0.0, 1.0)

  Body(
    ..body,
    energy: new_energy,
    satiety: new_satiety,
    rest: new_rest,
    stress: new_stress,
  )
}

/// Apply stimulus to body
pub fn apply_stimulus(body: Body, stimulus: BodyStimulus) -> Body {
  case stimulus {
    Energize(amount) ->
      Body(..body, energy: clamp(body.energy +. amount, 0.0, 1.0))

    Feed(amount) ->
      Body(..body, satiety: clamp(body.satiety +. amount, 0.0, 1.0))

    Rest(amount) -> Body(..body, rest: clamp(body.rest +. amount, 0.0, 1.0))

    Stress(amount) ->
      Body(..body, stress: clamp(body.stress +. amount, 0.0, 1.0))

    Pain(intensity) -> {
      // Pain depletes energy and adds stress
      let energy_loss = intensity *. 0.1
      let stress_gain = intensity *. 0.2
      Body(
        ..body,
        energy: clamp(body.energy -. energy_loss, 0.0, 1.0),
        stress: clamp(body.stress +. stress_gain, 0.0, 1.0),
      )
    }

    Comfort(intensity) -> {
      // Comfort restores and reduces stress
      let rest_gain = intensity *. 0.1
      let stress_loss = intensity *. 0.15
      Body(
        ..body,
        rest: clamp(body.rest +. rest_gain, 0.0, 1.0),
        stress: clamp(body.stress -. stress_loss, 0.0, 1.0),
      )
    }
  }
}

// =============================================================================
// NEEDS ASSESSMENT
// =============================================================================

/// Assess current needs state
pub fn assess_needs(body: Body) -> NeedsState {
  let energy_critical = body.energy <. 0.2
  let hunger_critical = body.satiety <. 0.2
  let fatigue_critical = body.rest <. 0.2
  let stressed = body.stress >. 0.7

  // Wellbeing is average of needs minus stress
  let needs_avg = { body.energy +. body.satiety +. body.rest } /. 3.0
  let wellbeing = float.max(0.0, needs_avg -. body.stress *. 0.5)

  NeedsState(
    wellbeing: wellbeing,
    energy_critical: energy_critical,
    hunger_critical: hunger_critical,
    fatigue_critical: fatigue_critical,
    stressed: stressed,
  )
}

/// Convert body state to PAD delta
/// Unmet needs create negative emotions
pub fn to_pad_delta(body: Body) -> Vec3 {
  let needs = assess_needs(body)

  // Pleasure: high wellbeing = positive, low = negative
  let pleasure = { needs.wellbeing -. 0.5 } *. 0.4

  // Arousal: stress increases arousal, fatigue decreases
  let arousal = body.stress *. 0.3 -. { 1.0 -. body.rest } *. 0.2

  // Dominance: unmet needs reduce sense of control
  let critical_count =
    case needs.energy_critical {
      True -> 1.0
      False -> 0.0
    }
    +. case needs.hunger_critical {
      True -> 1.0
      False -> 0.0
    }
    +. case needs.fatigue_critical {
      True -> 1.0
      False -> 0.0
    }
  let dominance = 0.0 -. critical_count *. 0.1

  vector.Vec3(x: pleasure, y: arousal, z: dominance)
}

// =============================================================================
// QUERIES
// =============================================================================

/// Get overall wellbeing (0.0-1.0)
pub fn wellbeing(body: Body) -> Float {
  assess_needs(body).wellbeing
}

/// Get energy level
pub fn energy(body: Body) -> Float {
  body.energy
}

/// Get satiety level
pub fn satiety(body: Body) -> Float {
  body.satiety
}

/// Get rest level
pub fn rest(body: Body) -> Float {
  body.rest
}

/// Get stress level
pub fn stress(body: Body) -> Float {
  body.stress
}

/// Is body in critical state? (any need critical)
pub fn is_critical(body: Body) -> Bool {
  let needs = assess_needs(body)
  needs.energy_critical || needs.hunger_critical || needs.fatigue_critical
}

/// Is body suffering? (low wellbeing + high stress)
pub fn is_suffering(body: Body) -> Bool {
  let needs = assess_needs(body)
  needs.wellbeing <. 0.3 && needs.stressed
}

// =============================================================================
// HELPERS
// =============================================================================

fn clamp(value: Float, min: Float, max: Float) -> Float {
  float.min(max, float.max(min, value))
}
