//// Motor Imprinting - Motor patterns and reflexes
////
//// Tracks learned motor patterns during critical period.

import gleam/list
import gleam/option.{type Option}
import viva/imprint/types.{type ImprintEvent, MotorLearned}

// =============================================================================
// TYPES
// =============================================================================

/// Motor imprint state
pub type MotorImprint {
  MotorImprint(
    /// Learned motor patterns
    patterns: List(MotorPattern),
    /// Reflexes (stimulus -> action)
    reflexes: List(Reflex),
    /// Number of patterns learned
    pattern_count: Int,
  )
}

/// A motor pattern
pub type MotorPattern {
  MotorPattern(
    /// Action name
    action: String,
    /// Stimulus before
    before_light: Int,
    before_sound: Int,
    before_touch: Bool,
    /// Stimulus after
    after_light: Int,
    after_sound: Int,
    after_touch: Bool,
    /// Effect (pleasure delta)
    effect: Float,
    /// Confidence
    confidence: Float,
    /// Observations
    observations: Int,
  )
}

/// A reflex (learned automatic response)
pub type Reflex {
  Reflex(
    /// Trigger stimulus
    light_threshold: Int,
    sound_threshold: Int,
    touch_required: Bool,
    /// Response action
    action: String,
    /// How often it was reinforced
    strength: Float,
  )
}

// =============================================================================
// CONSTRUCTORS
// =============================================================================

/// Create empty motor imprint
pub fn new() -> MotorImprint {
  MotorImprint(patterns: [], reflexes: [], pattern_count: 0)
}

// =============================================================================
// OPERATIONS
// =============================================================================

/// Learn a motor pattern (action -> effect)
pub fn learn(
  imprint: MotorImprint,
  action_name: String,
  before_light: Int,
  before_sound: Int,
  before_touch: Bool,
  after_light: Int,
  after_sound: Int,
  after_touch: Bool,
  pleasure_delta: Float,
  intensity: Float,
  _current_tick: Int,
) -> #(MotorImprint, List(ImprintEvent)) {
  // Check if pattern exists
  let existing = list.find(imprint.patterns, fn(p) { p.action == action_name })

  let #(new_patterns, event) = case existing {
    Ok(pattern) -> {
      // Update existing pattern
      let updated =
        MotorPattern(
          ..pattern,
          effect: { pattern.effect +. pleasure_delta *. intensity } /. 2.0,
          confidence: min_float(pattern.confidence +. intensity *. 0.1, 1.0),
          observations: pattern.observations + 1,
        )
      let patterns =
        list.map(imprint.patterns, fn(p) {
          case p.action == action_name {
            True -> updated
            False -> p
          }
        })
      #(patterns, [])
    }
    Error(_) -> {
      // New pattern
      let pattern =
        MotorPattern(
          action: action_name,
          before_light: before_light,
          before_sound: before_sound,
          before_touch: before_touch,
          after_light: after_light,
          after_sound: after_sound,
          after_touch: after_touch,
          effect: pleasure_delta,
          confidence: intensity *. 0.3,
          observations: 1,
        )
      let effect_desc = case pleasure_delta >. 0.0 {
        True -> "positive"
        False -> "negative"
      }
      #([pattern, ..imprint.patterns], [
        MotorLearned(action: action_name, effect: effect_desc),
      ])
    }
  }

  // Maybe learn reflex if strong stimulus-response
  let new_reflexes = case abs_float(pleasure_delta) >. 0.5 {
    True ->
      maybe_add_reflex(
        imprint.reflexes,
        before_light,
        before_sound,
        before_touch,
        action_name,
        intensity,
      )
    False -> imprint.reflexes
  }

  #(
    MotorImprint(
      patterns: new_patterns,
      reflexes: new_reflexes,
      pattern_count: list.length(new_patterns),
    ),
    event,
  )
}

/// Get reflex action for stimulus
pub fn get_reflex(
  imprint: MotorImprint,
  light: Int,
  sound: Int,
  touch: Bool,
) -> Option(String) {
  imprint.reflexes
  |> list.find(fn(r) {
    light >= r.light_threshold
    && sound >= r.sound_threshold
    && { !r.touch_required || touch }
    && r.strength >. 0.5
  })
  |> option.from_result
  |> option.map(fn(r) { r.action })
}

/// Get pattern count
pub fn pattern_count(imprint: MotorImprint) -> Int {
  imprint.pattern_count
}

// =============================================================================
// HELPERS
// =============================================================================

fn maybe_add_reflex(
  reflexes: List(Reflex),
  light: Int,
  sound: Int,
  touch: Bool,
  action: String,
  intensity: Float,
) -> List(Reflex) {
  // Check if similar reflex exists
  let existing = list.find(reflexes, fn(r) { r.action == action })

  case existing {
    Ok(r) -> {
      let updated =
        Reflex(..r, strength: min_float(r.strength +. intensity *. 0.2, 1.0))
      list.map(reflexes, fn(r2) {
        case r2.action == action {
          True -> updated
          False -> r2
        }
      })
    }
    Error(_) -> {
      let reflex =
        Reflex(
          light_threshold: light / 2,
          sound_threshold: sound / 2,
          touch_required: touch,
          action: action,
          strength: intensity *. 0.3,
        )
      [reflex, ..reflexes]
    }
  }
}

fn min_float(a: Float, b: Float) -> Float {
  case a <. b {
    True -> a
    False -> b
  }
}

fn abs_float(a: Float) -> Float {
  case a <. 0.0 {
    True -> 0.0 -. a
    False -> a
  }
}
