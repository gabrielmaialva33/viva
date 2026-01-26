//// Survival Imprinting - Survival needs and signals
////
//// Tracks survival-related learning during critical period.
//// Learns danger/safety signals from sensory context.

import gleam/list
import gleam/option.{type Option, None}
import viva/imprint/types.{type ImprintEvent, DangerLearned, SafetyLearned}

// =============================================================================
// TYPES
// =============================================================================

/// Survival imprint state
pub type SurvivalImprint {
  SurvivalImprint(
    /// Learned danger signals
    danger_signals: List(DangerSignal),
    /// Learned safety signals
    safety_signals: List(SafetySignal),
    /// Energy threshold learned as "hungry"
    hunger_threshold: Float,
    /// Safety count (times felt safe)
    safe_count: Int,
    /// Danger count (times felt danger)
    danger_count: Int,
  )
}

/// A learned danger signal
pub type DangerSignal {
  DangerSignal(
    /// Light threshold that triggers danger
    light_threshold: Int,
    /// Sound threshold that triggers danger
    sound_threshold: Int,
    /// Touch component
    touch_involved: Bool,
    /// Energy level when danger was learned
    energy_context: Float,
    /// Danger intensity (how bad)
    intensity: Float,
    /// Times observed
    observations: Int,
  )
}

/// A learned safety signal
pub type SafetySignal {
  SafetySignal(
    /// Light level associated with safety
    light_level: Int,
    /// Sound level associated with safety
    sound_level: Int,
    /// Entity present when safe
    entity_present: Option(String),
    /// Comfort level
    comfort: Float,
    /// Times observed
    observations: Int,
  )
}

// =============================================================================
// CONSTRUCTORS
// =============================================================================

/// Create new survival imprint
pub fn new() -> SurvivalImprint {
  SurvivalImprint(
    danger_signals: [],
    safety_signals: [],
    hunger_threshold: 0.3,
    safe_count: 0,
    danger_count: 0,
  )
}

// =============================================================================
// OPERATIONS
// =============================================================================

/// Evaluate survival context during critical period
pub fn evaluate(
  imprint: SurvivalImprint,
  body_energy: Float,
  light: Int,
  sound: Int,
  touch: Bool,
  pleasure: Float,
  intensity: Float,
  _current_tick: Int,
) -> #(SurvivalImprint, List(ImprintEvent)) {
  let events = []

  // Detect danger: low pleasure + high arousal stimuli
  let is_danger_context =
    pleasure <. 0.3 && { light > 800 || sound > 800 || touch }

  // Detect safety: high pleasure + calm stimuli
  let is_safe_context = pleasure >. 0.6 && light < 500 && sound < 500

  let #(new_danger, danger_events) = case is_danger_context {
    True -> learn_danger(imprint, light, sound, touch, body_energy, intensity)
    False -> #(imprint.danger_signals, [])
  }

  let #(new_safety, safety_events) = case is_safe_context {
    True -> learn_safety(imprint, light, sound, None, pleasure, intensity)
    False -> #(imprint.safety_signals, [])
  }

  // Update hunger threshold if low energy + low pleasure
  let new_hunger = case body_energy <. 0.4 && pleasure <. 0.4 {
    True -> { imprint.hunger_threshold +. body_energy } /. 2.0
    False -> imprint.hunger_threshold
  }

  let new_imprint =
    SurvivalImprint(
      danger_signals: new_danger,
      safety_signals: new_safety,
      hunger_threshold: new_hunger,
      danger_count: imprint.danger_count
        + case is_danger_context {
          True -> 1
          False -> 0
        },
      safe_count: imprint.safe_count
        + case is_safe_context {
          True -> 1
          False -> 0
        },
    )

  #(new_imprint, list.flatten([events, danger_events, safety_events]))
}

/// Check if stimulus matches a danger signal
pub fn is_danger_signal(
  imprint: SurvivalImprint,
  light: Int,
  sound: Int,
  touch: Bool,
  energy: Float,
) -> Bool {
  list.any(imprint.danger_signals, fn(d) {
    light >= d.light_threshold
    && sound >= d.sound_threshold
    && { !d.touch_involved || touch }
    && energy <. d.energy_context +. 0.2
    && d.intensity >. 0.3
  })
}

/// Check if context matches a safety signal
pub fn is_safety_signal(
  imprint: SurvivalImprint,
  light: Int,
  sound: Int,
  entity_present: Option(String),
) -> Bool {
  list.any(imprint.safety_signals, fn(s) {
    abs_int(light - s.light_level) < 200
    && abs_int(sound - s.sound_level) < 200
    && {
      option.is_none(s.entity_present) || s.entity_present == entity_present
    }
    && s.comfort >. 0.5
  })
}

/// Get safety count
pub fn safety_count(imprint: SurvivalImprint) -> Int {
  imprint.safe_count
}

/// Get danger count
pub fn danger_count(imprint: SurvivalImprint) -> Int {
  imprint.danger_count
}

// =============================================================================
// HELPERS
// =============================================================================

fn learn_danger(
  imprint: SurvivalImprint,
  light: Int,
  sound: Int,
  touch: Bool,
  energy: Float,
  intensity: Float,
) -> #(List(DangerSignal), List(ImprintEvent)) {
  // Check if similar danger exists
  let existing =
    list.find(imprint.danger_signals, fn(d) {
      abs_int(light - d.light_threshold) < 100
      && abs_int(sound - d.sound_threshold) < 100
    })

  case existing {
    Ok(d) -> {
      let updated =
        DangerSignal(
          ..d,
          intensity: min_float(d.intensity +. intensity *. 0.1, 1.0),
          observations: d.observations + 1,
        )
      let signals =
        list.map(imprint.danger_signals, fn(d2) {
          case abs_int(light - d2.light_threshold) < 100 {
            True -> updated
            False -> d2
          }
        })
      #(signals, [])
    }
    Error(_) -> {
      let signal =
        DangerSignal(
          light_threshold: light,
          sound_threshold: sound,
          touch_involved: touch,
          energy_context: energy,
          intensity: intensity *. 0.5,
          observations: 1,
        )
      let trigger = case touch {
        True -> "touch"
        False ->
          case light > sound {
            True -> "bright_light"
            False -> "loud_sound"
          }
      }
      #([signal, ..imprint.danger_signals], [
        DangerLearned(trigger: trigger, intensity: intensity),
      ])
    }
  }
}

fn learn_safety(
  imprint: SurvivalImprint,
  light: Int,
  sound: Int,
  entity: Option(String),
  comfort: Float,
  intensity: Float,
) -> #(List(SafetySignal), List(ImprintEvent)) {
  // Check if similar safety exists
  let existing =
    list.find(imprint.safety_signals, fn(s) {
      abs_int(light - s.light_level) < 100
      && abs_int(sound - s.sound_level) < 100
    })

  case existing {
    Ok(s) -> {
      let updated =
        SafetySignal(
          ..s,
          comfort: { s.comfort +. comfort } /. 2.0,
          observations: s.observations + 1,
        )
      let signals =
        list.map(imprint.safety_signals, fn(s2) {
          case abs_int(light - s2.light_level) < 100 {
            True -> updated
            False -> s2
          }
        })
      #(signals, [])
    }
    Error(_) -> {
      let signal =
        SafetySignal(
          light_level: light,
          sound_level: sound,
          entity_present: entity,
          comfort: comfort,
          observations: 1,
        )
      let trigger = "calm_environment"
      #([signal, ..imprint.safety_signals], [
        SafetyLearned(trigger: trigger, comfort: comfort *. intensity),
      ])
    }
  }
}

fn abs_int(a: Int) -> Int {
  case a < 0 {
    True -> 0 - a
    False -> a
  }
}

fn min_float(a: Float, b: Float) -> Float {
  case a <. b {
    True -> a
    False -> b
  }
}
