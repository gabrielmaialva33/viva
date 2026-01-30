//// VIVA Imprinting System - Critical Period Learning
////
//// Like biological imprinting in newborns:
//// - Critical period in first ~1000 ticks after spawn
//// - Accelerated learning (3x intensity)
//// - Establishes foundational associations:
////   - Sensory: stimulus -> valence
////   - Motor: action -> effect
////   - Social: entity -> attachment
////   - Survival: need -> urgency
////
//// After critical period, learning continues but slower.

import gleam/float
import gleam/int
import gleam/list
import gleam/option.{type Option}
import viva/memory/imprint/motor.{type MotorImprint}
import viva/memory/imprint/sensory.{type SensoryImprint}
import viva/memory/imprint/social.{type SocialImprint}
import viva/memory/imprint/survival.{type SurvivalImprint}
import viva/memory/imprint/types.{type ImprintEvent, CriticalPeriodEnded}

// =============================================================================
// TYPES
// =============================================================================

/// Configuration for imprinting system
pub type ImprintConfig {
  ImprintConfig(
    /// Duration of critical period in ticks (default: 1000 ~16s @ 60fps)
    critical_period: Int,
    /// Learning intensity multiplier during critical period (default: 3.0)
    intensity_multiplier: Float,
    /// Enable sensory imprinting
    sensory_enabled: Bool,
    /// Enable motor imprinting
    motor_enabled: Bool,
    /// Enable social imprinting
    social_enabled: Bool,
    /// Enable survival imprinting
    survival_enabled: Bool,
    /// Name of primary attachment figure (default: "Gabriel")
    creator_name: String,
  )
}

/// Main imprinting state
pub type ImprintState {
  ImprintState(
    /// Is critical period active?
    active: Bool,
    /// Tick when critical period started
    started_at: Int,
    /// Configuration
    config: ImprintConfig,
    /// Sensory associations
    sensory: SensoryImprint,
    /// Motor patterns and reflexes
    motor: MotorImprint,
    /// Social recognition and attachment
    social: SocialImprint,
    /// Survival needs and signals
    survival: SurvivalImprint,
    /// Current learning intensity (3.0 during critical, 1.0 after)
    current_intensity: Float,
    /// Statistics
    total_observations: Int,
  )
}

// ImprintEvent imported from viva/imprint/types

/// Summary after imprinting completes
pub type ImprintSummary {
  ImprintSummary(
    /// Total observations during critical period
    total_observations: Int,
    /// Sensory associations count
    sensory_count: Int,
    /// Motor patterns count
    motor_count: Int,
    /// Known entities count
    social_count: Int,
    /// Primary attachment strength
    attachment_strength: Float,
    /// Danger signals count
    danger_count: Int,
    /// Safety signals count
    safety_count: Int,
  )
}

// =============================================================================
// CONSTRUCTORS
// =============================================================================

/// Default configuration
pub fn default_config() -> ImprintConfig {
  ImprintConfig(
    critical_period: 1000,
    intensity_multiplier: 3.0,
    sensory_enabled: True,
    motor_enabled: True,
    social_enabled: True,
    survival_enabled: True,
    creator_name: "Gabriel",
  )
}

/// Create new imprinting state
pub fn new(config: ImprintConfig) -> ImprintState {
  ImprintState(
    active: False,
    started_at: 0,
    config: config,
    sensory: sensory.new(256),
    motor: motor.new(),
    social: social.new(config.creator_name),
    survival: survival.new(),
    current_intensity: 1.0,
    total_observations: 0,
  )
}

/// Create with default config
pub fn new_default() -> ImprintState {
  new(default_config())
}

// =============================================================================
// LIFECYCLE
// =============================================================================

/// Start critical period
pub fn start(state: ImprintState, tick: Int) -> ImprintState {
  ImprintState(
    ..state,
    active: True,
    started_at: tick,
    current_intensity: state.config.intensity_multiplier,
  )
}

/// Check if still in critical period
pub fn is_critical_period(state: ImprintState, current_tick: Int) -> Bool {
  state.active && current_tick - state.started_at < state.config.critical_period
}

/// Get elapsed ticks in critical period
pub fn elapsed(state: ImprintState, current_tick: Int) -> Int {
  case state.active {
    True -> current_tick - state.started_at
    False -> 0
  }
}

/// Get progress through critical period (0.0 to 1.0)
pub fn progress(state: ImprintState, current_tick: Int) -> Float {
  case state.active {
    True -> {
      let elapsed_ticks = current_tick - state.started_at
      let progress_raw =
        int_to_float(elapsed_ticks)
        /. int_to_float(state.config.critical_period)
      float.min(1.0, progress_raw)
    }
    False -> 1.0
  }
}

// =============================================================================
// TICK PROCESSING
// =============================================================================

/// Process a single tick of imprinting
/// Returns updated state and any events
pub fn tick(
  state: ImprintState,
  pleasure: Float,
  arousal: Float,
  _dominance: Float,
  light: Int,
  sound: Int,
  touch: Bool,
  entity_present: Option(String),
  body_energy: Float,
  current_tick: Int,
) -> #(ImprintState, List(ImprintEvent)) {
  // Check if critical period ended
  case is_critical_period(state, current_tick) {
    False ->
      case state.active {
        // Just ended - emit event and deactivate
        True -> {
          let summary = complete(state)
          let new_state =
            ImprintState(..state, active: False, current_intensity: 1.0)
          #(new_state, [CriticalPeriodEnded(summary.total_observations)])
        }
        // Already inactive
        False -> #(state, [])
      }

    // Still in critical period - process learning
    True -> {
      let events = []
      let intensity = state.current_intensity

      // 1. Sensory imprinting
      let #(new_sensory, sensory_events) = case state.config.sensory_enabled {
        True ->
          sensory.observe(
            state.sensory,
            light,
            sound,
            touch,
            pleasure,
            intensity,
            current_tick,
          )
        False -> #(state.sensory, [])
      }
      let events = list.append(events, sensory_events)

      // 2. Social imprinting
      let #(new_social, social_events) = case state.config.social_enabled {
        True ->
          social.observe_presence(
            state.social,
            entity_present,
            pleasure,
            arousal,
            intensity,
            current_tick,
          )
        False -> #(state.social, [])
      }
      let events = list.append(events, social_events)

      // 3. Survival imprinting
      let #(new_survival, survival_events) = case
        state.config.survival_enabled
      {
        True ->
          survival.evaluate(
            state.survival,
            body_energy,
            light,
            sound,
            touch,
            pleasure,
            intensity,
            current_tick,
          )
        False -> #(state.survival, [])
      }
      let events = list.append(events, survival_events)

      let new_state =
        ImprintState(
          ..state,
          sensory: new_sensory,
          social: new_social,
          survival: new_survival,
          total_observations: state.total_observations + 1,
        )

      #(new_state, events)
    }
  }
}

/// Process motor learning separately (needs before/after sensation)
pub fn learn_motor(
  state: ImprintState,
  action_name: String,
  before_light: Int,
  before_sound: Int,
  before_touch: Bool,
  after_light: Int,
  after_sound: Int,
  after_touch: Bool,
  pleasure_delta: Float,
  current_tick: Int,
) -> #(ImprintState, List(ImprintEvent)) {
  case state.config.motor_enabled && state.active {
    False -> #(state, [])
    True -> {
      let #(new_motor, events) =
        motor.learn(
          state.motor,
          action_name,
          before_light,
          before_sound,
          before_touch,
          after_light,
          after_sound,
          after_touch,
          pleasure_delta,
          state.current_intensity,
          current_tick,
        )
      #(ImprintState(..state, motor: new_motor), events)
    }
  }
}

// =============================================================================
// QUERIES
// =============================================================================

/// Get expected valence for a sensation
pub fn expected_valence(
  state: ImprintState,
  light: Int,
  sound: Int,
  touch: Bool,
) -> Option(Float) {
  sensory.query_valence(state.sensory, light, sound, touch)
}

/// Get reflex action for a stimulus
pub fn get_reflex(
  state: ImprintState,
  light: Int,
  sound: Int,
  touch: Bool,
) -> Option(String) {
  motor.get_reflex(state.motor, light, sound, touch)
}

/// Get attachment strength to an entity
pub fn attachment_to(state: ImprintState, entity: String) -> Float {
  social.attachment_strength(state.social, entity)
}

/// Check if stimulus is dangerous
pub fn is_danger(
  state: ImprintState,
  light: Int,
  sound: Int,
  touch: Bool,
  energy: Float,
) -> Bool {
  survival.is_danger_signal(state.survival, light, sound, touch, energy)
}

/// Check if context is safe
pub fn is_safe(
  state: ImprintState,
  light: Int,
  sound: Int,
  entity_present: Option(String),
) -> Bool {
  survival.is_safety_signal(state.survival, light, sound, entity_present)
}

// =============================================================================
// SUMMARY
// =============================================================================

/// Generate summary of imprinting
pub fn complete(state: ImprintState) -> ImprintSummary {
  ImprintSummary(
    total_observations: state.total_observations,
    sensory_count: sensory.association_count(state.sensory),
    motor_count: motor.pattern_count(state.motor),
    social_count: social.entity_count(state.social),
    attachment_strength: social.primary_attachment_strength(state.social),
    danger_count: survival.danger_count(state.survival),
    safety_count: survival.safety_count(state.survival),
  )
}

/// Describe imprinting state for debugging
pub fn describe(state: ImprintState, current_tick: Int) -> String {
  let status = case state.active {
    True ->
      "ACTIVE (tick "
      <> int_to_string(elapsed(state, current_tick))
      <> "/"
      <> int_to_string(state.config.critical_period)
      <> ")"
    False -> "INACTIVE"
  }

  "Imprinting: "
  <> status
  <> "\n"
  <> "  Intensity: "
  <> float_to_string(state.current_intensity)
  <> "x\n"
  <> "  Observations: "
  <> int_to_string(state.total_observations)
  <> "\n"
  <> "  Sensory: "
  <> int_to_string(sensory.association_count(state.sensory))
  <> " associations\n"
  <> "  Motor: "
  <> int_to_string(motor.pattern_count(state.motor))
  <> " patterns\n"
  <> "  Social: "
  <> int_to_string(social.entity_count(state.social))
  <> " entities\n"
  <> "  Survival: "
  <> int_to_string(survival.danger_count(state.survival))
  <> " dangers, "
  <> int_to_string(survival.safety_count(state.survival))
  <> " safeties"
}

// =============================================================================
// HELPERS
// =============================================================================

fn int_to_float(i: Int) -> Float {
  case i >= 0 {
    True -> do_int_to_float_positive(i, 0.0)
    False -> 0.0 -. do_int_to_float_positive(-i, 0.0)
  }
}

fn do_int_to_float_positive(i: Int, acc: Float) -> Float {
  case i {
    0 -> acc
    _ -> do_int_to_float_positive(i - 1, acc +. 1.0)
  }
}

fn int_to_string(i: Int) -> String {
  int.to_string(i)
}

fn float_to_string(f: Float) -> String {
  float.to_string(f)
}
