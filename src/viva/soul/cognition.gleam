//// Cognition - Grouped cognitive operations
////
//// Extracted from soul.gleam tick handler to reduce complexity.
//// Single function that processes all cognitive subsystems.
////
//// Before: soul.gleam tick had 10 sequential operations
//// After: soul.gleam calls cognition.tick() for cognitive part

import viva/memory/imprint.{type ImprintState}
import viva/memory/narrative.{type NarrativeMemory}
import viva/soul/inner_life.{type InnerLife}
import viva/soul/reflexivity.{type SelfModel}
import gleam/option.{type Option}
import viva_emotion/pad.{type Pad}
import viva_glyph/glyph.{type Glyph}

// =============================================================================
// TYPES
// =============================================================================

/// Grouped cognitive state (for future refactoring)
pub type CognitiveState {
  CognitiveState(
    self_model: SelfModel,
    narrative: NarrativeMemory,
    imprint: ImprintState,
    inner_life: InnerLife,
  )
}

/// Sensory input for cognitive processing
pub type SensoryInput {
  SensoryInput(
    light: Int,
    sound: Int,
    touch: Bool,
    entity: Option(String),
  )
}

/// Result of cognitive tick
pub type CognitionResult {
  CognitionResult(
    self_model: SelfModel,
    narrative: NarrativeMemory,
    imprint: ImprintState,
    inner_life: InnerLife,
    inner_voice: String,
  )
}

// =============================================================================
// CONSTRUCTORS
// =============================================================================

/// Create cognitive state from components
pub fn new(
  self_model: SelfModel,
  narrative: NarrativeMemory,
  imprint: ImprintState,
  inner_life: InnerLife,
) -> CognitiveState {
  CognitiveState(
    self_model: self_model,
    narrative: narrative,
    imprint: imprint,
    inner_life: inner_life,
  )
}

/// Extract cognitive state from soul state components
pub fn from_components(
  self_model: SelfModel,
  narrative: NarrativeMemory,
  imprint: ImprintState,
  inner_life: InnerLife,
) -> CognitiveState {
  new(self_model, narrative, imprint, inner_life)
}

// =============================================================================
// MAIN TICK
// =============================================================================

/// Process all cognitive subsystems in one call
///
/// This consolidates:
/// - Narrative recording (temporal links)
/// - Imprinting (critical period learning)
/// - Inner life (verbalized self-reflection)
///
/// Note: self_model is updated inside inner_life.tick()
pub fn tick(
  cognitive: CognitiveState,
  current_pad: Pad,
  current_glyph: Glyph,
  previous_glyph: Glyph,
  sensation: SensoryInput,
  body_energy: Float,
  tick_count: Int,
) -> CognitionResult {
  // 1. Narrative: record temporal link if glyph changed
  let new_narrative = case glyph.equals(previous_glyph, current_glyph) {
    True -> narrative.tick(cognitive.narrative)
    False -> {
      cognitive.narrative
      |> narrative.record_preceded(previous_glyph, current_glyph, tick_count)
      |> narrative.tick()
    }
  }

  // 2. Imprinting: process critical period learning
  let #(new_imprint, _imprint_events) =
    imprint.tick(
      cognitive.imprint,
      current_pad.pleasure,
      current_pad.arousal,
      current_pad.dominance,
      sensation.light,
      sensation.sound,
      sensation.touch,
      sensation.entity,
      body_energy,
      tick_count,
    )

  // 3. Inner life: verbalized self-reflection (includes reflexivity.observe)
  let #(new_inner_life, inner_voice) =
    inner_life.tick(cognitive.inner_life, current_pad, current_glyph)

  CognitionResult(
    self_model: new_inner_life.self_model,
    narrative: new_narrative,
    imprint: new_imprint,
    inner_life: new_inner_life,
    inner_voice: inner_voice,
  )
}

// =============================================================================
// ACCESSORS
// =============================================================================

/// Get self model from cognitive state
pub fn get_self_model(cognitive: CognitiveState) -> SelfModel {
  cognitive.self_model
}

/// Get narrative from cognitive state
pub fn get_narrative(cognitive: CognitiveState) -> NarrativeMemory {
  cognitive.narrative
}

/// Get imprint state
pub fn get_imprint(cognitive: CognitiveState) -> ImprintState {
  cognitive.imprint
}

/// Get inner life
pub fn get_inner_life(cognitive: CognitiveState) -> InnerLife {
  cognitive.inner_life
}
