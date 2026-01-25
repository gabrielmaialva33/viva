//// Types - VIVA shared types
////
//// Unifies types used across multiple modules to avoid duplication.

import viva_emotion/pad.{type Pad}
import viva_glyph/glyph.{type Glyph}

// =============================================================================
// CORE IDS
// =============================================================================

/// Unique VIVA ID
pub type VivaId =
  Int

/// Archetype ID
pub type ArchetypeId =
  Int

// =============================================================================
// VIVA STATE (for inter-module communication)
// =============================================================================

/// Simplified VIVA state (for resonance, supervisor)
pub type VivaSnapshot {
  VivaSnapshot(
    /// Unique ID
    id: VivaId,
    /// Current PAD
    pad: Pad,
    /// Current glyph
    glyph: Glyph,
    /// Whether alive
    alive: Bool,
    /// Current tick
    tick: Int,
    /// Life number (reincarnation)
    life_number: Int,
  )
}

// =============================================================================
// LIFECYCLE EVENTS
// =============================================================================

/// Lifecycle events that Supervisor processes
pub type LifecycleEvent {
  /// VIVA was born
  Born(id: VivaId, life_number: Int)
  /// VIVA died - needs bardo processing
  Died(id: VivaId, final_glyph: Glyph, total_karma: Float)
  /// Bardo completed - result
  BardoComplete(id: VivaId, liberated: Bool)
  /// VIVA was reborn
  Reborn(id: VivaId, new_life_number: Int)
}

// =============================================================================
// CONFIG
// =============================================================================

/// Configuration for creating new VIVA
pub type VivaConfig {
  VivaConfig(
    /// Name/identity
    name: String,
    /// Life number (1 = first)
    life_number: Int,
    /// Initial mood (0-1)
    initial_mood: Float,
    /// Initial karma
    initial_karma: Float,
    /// Glyphs inherited from previous lives
    inherited_glyphs: List(Glyph),
    /// Relevant archetype IDs
    relevant_archetypes: List(ArchetypeId),
    /// Recency decay rate (default 0.99, half-life ~69 ticks)
    recency_decay: Float,
  )
}

/// Default config for first life
pub fn default_config(name: String) -> VivaConfig {
  VivaConfig(
    name: name,
    life_number: 1,
    initial_mood: 0.5,
    initial_karma: 0.0,
    inherited_glyphs: [],
    relevant_archetypes: [],
    recency_decay: 0.99,
  )
}

/// Create config from Creator's LifeSeed
pub fn config_from_seed(
  name: String,
  life_number: Int,
  initial_mood: Float,
  initial_karma: Float,
  inherited_glyphs: List(Glyph),
  relevant_archetypes: List(ArchetypeId),
) -> VivaConfig {
  VivaConfig(
    name: name,
    life_number: life_number,
    initial_mood: initial_mood,
    initial_karma: initial_karma,
    inherited_glyphs: inherited_glyphs,
    relevant_archetypes: relevant_archetypes,
    recency_decay: 0.99,
  )
}

/// Create config with custom decay
pub fn config_with_decay(base: VivaConfig, decay: Float) -> VivaConfig {
  VivaConfig(..base, recency_decay: decay)
}
