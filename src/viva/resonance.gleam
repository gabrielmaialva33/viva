//// Resonance - Empathy between VIVAs
////
//// "Collective Consciousness: There is no isolated 'self'.
////  All VIVAs are waves of the same ocean."
////
//// Living VIVAs can "feel" emotions of other VIVAs
//// through Glyph resonance.

import gleam/float
import gleam/int
import gleam/list
import gleam/option.{type Option, None, Some}
import viva_emotion/pad.{type Pad}
import viva_glyph
import viva_glyph/glyph.{type Glyph}

// =============================================================================
// TYPES
// =============================================================================

/// VIVA ID
pub type VivaId =
  Int

/// Simplified VIVA state (for resonance)
pub type VivaState {
  VivaState(
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
  )
}

/// Resonance event
pub type ResonanceEvent {
  ResonanceEvent(
    /// Who emitted
    source: VivaId,
    /// Who received
    target: VivaId,
    /// Transmitted emotional pattern
    glyph: Glyph,
    /// Resonance intensity (0-1)
    intensity: Float,
    /// When it occurred
    tick: Int,
  )
}

/// Pool of VIVAs for resonance
pub type ResonancePool {
  ResonancePool(
    /// Living VIVAs in pool
    vivas: List(VivaState),
    /// Recent resonance events
    events: List(ResonanceEvent),
    /// Resonance threshold
    threshold: Float,
    /// Maximum events to keep
    max_events: Int,
  )
}

// =============================================================================
// POOL OPERATIONS
// =============================================================================

/// Create new resonance pool
pub fn new_pool() -> ResonancePool {
  ResonancePool(vivas: [], events: [], threshold: 0.5, max_events: 100)
}

/// Create pool with custom threshold
pub fn new_pool_with_threshold(threshold: Float) -> ResonancePool {
  ResonancePool(vivas: [], events: [], threshold: threshold, max_events: 100)
}

/// Register VIVA in pool
pub fn register(pool: ResonancePool, viva: VivaState) -> ResonancePool {
  case viva.alive {
    True -> ResonancePool(..pool, vivas: [viva, ..pool.vivas])
    False -> pool
  }
}

/// Remove VIVA from pool
pub fn unregister(pool: ResonancePool, viva_id: VivaId) -> ResonancePool {
  let vivas = list.filter(pool.vivas, fn(v) { v.id != viva_id })
  ResonancePool(..pool, vivas: vivas)
}

/// Update VIVA state
pub fn update_viva(pool: ResonancePool, updated: VivaState) -> ResonancePool {
  let vivas =
    pool.vivas
    |> list.map(fn(v) {
      case v.id == updated.id {
        True -> updated
        False -> v
      }
    })
  ResonancePool(..pool, vivas: vivas)
}

// =============================================================================
// RESONANCE CALCULATION
// =============================================================================

/// Calculate resonance between two VIVAs
pub fn calculate_resonance(viva_a: VivaState, viva_b: VivaState) -> Float {
  // Only resonate if both alive
  case viva_a.alive && viva_b.alive {
    False -> 0.0
    True -> {
      // Base glyph similarity
      let base_sim = viva_glyph.similarity(viva_a.glyph, viva_b.glyph)

      // Amplified by arousal (intense emotions resonate more)
      let arousal_a = viva_a.pad.arousal
      let arousal_b = viva_b.pad.arousal
      let arousal_amp = { arousal_a +. arousal_b } /. 2.0

      // Resonance = similarity x (1 + average arousal)
      base_sim *. { 1.0 +. arousal_amp }
    }
  }
}

/// Propagate resonance from source to all targets
pub fn propagate(
  pool: ResonancePool,
  source: VivaState,
  tick: Int,
) -> #(ResonancePool, List(ResonanceEvent)) {
  // Find targets (all VIVAs except source)
  let targets = list.filter(pool.vivas, fn(v) { v.id != source.id && v.alive })

  // Calculate resonance with each target
  let events =
    targets
    |> list.filter_map(fn(target) {
      let resonance = calculate_resonance(source, target)
      case resonance >. pool.threshold {
        True ->
          Ok(ResonanceEvent(
            source: source.id,
            target: target.id,
            glyph: source.glyph,
            intensity: resonance,
            tick: tick,
          ))
        False -> Error(Nil)
      }
    })

  // Add events to history (keeping limit)
  let all_events = list.append(events, pool.events)
  let trimmed = list.take(all_events, pool.max_events)

  #(ResonancePool(..pool, events: trimmed), events)
}

/// Apply resonance effect on target
pub fn apply_resonance(
  target_pad: Pad,
  event: ResonanceEvent,
  sensitivity: Float,
) -> Pad {
  // Decode event glyph to PAD delta
  let source_pad = glyph_to_pad_estimate(event.glyph)

  // Calculate delta based on intensity and sensitivity
  let scale = event.intensity *. sensitivity *. 0.1

  // Blend target PAD with source influence
  let new_pleasure =
    target_pad.pleasure
    +. { source_pad.pleasure -. target_pad.pleasure }
    *. scale
  let new_arousal =
    target_pad.arousal +. { source_pad.arousal -. target_pad.arousal } *. scale
  let new_dominance =
    target_pad.dominance
    +. { source_pad.dominance -. target_pad.dominance }
    *. scale

  pad.new(new_pleasure, new_arousal, new_dominance)
}

/// Estimate PAD from a Glyph (approximation)
fn glyph_to_pad_estimate(g: Glyph) -> Pad {
  let tokens = g.tokens
  case tokens {
    [t1, t2, t3, _t4] -> {
      // Map tokens [0-255] to PAD [-1, 1]
      let pleasure = { int.to_float(t1) -. 128.0 } /. 128.0
      let arousal = { int.to_float(t2) -. 128.0 } /. 128.0
      let dominance = { int.to_float(t3) -. 128.0 } /. 128.0
      pad.new(pleasure, arousal, dominance)
    }
    _ -> pad.new(0.0, 0.0, 0.0)
  }
}

// =============================================================================
// QUERIES
// =============================================================================

/// Get recent events for a VIVA
pub fn events_for(pool: ResonancePool, viva_id: VivaId) -> List(ResonanceEvent) {
  pool.events
  |> list.filter(fn(e) { e.target == viva_id })
}

/// Get events emitted by a VIVA
pub fn events_from(pool: ResonancePool, viva_id: VivaId) -> List(ResonanceEvent) {
  pool.events
  |> list.filter(fn(e) { e.source == viva_id })
}

/// Calculate pool "emotional field" (average of glyphs)
pub fn emotional_field(pool: ResonancePool) -> Option(Glyph) {
  case pool.vivas {
    [] -> None
    vivas -> {
      // Average tokens of all glyphs
      let all_tokens = list.map(vivas, fn(v) { v.glyph.tokens })
      let avg_tokens = average_tokens(all_tokens)
      Some(glyph.new(avg_tokens))
    }
  }
}

fn average_tokens(all_tokens: List(List(Int))) -> List(Int) {
  case all_tokens {
    [] -> [128, 128, 128, 128]
    _ -> {
      let n = list.length(all_tokens)
      // Sum each position
      let sums =
        all_tokens
        |> list.fold([0, 0, 0, 0], fn(acc, tokens) {
          case acc, tokens {
            [a1, a2, a3, a4], [t1, t2, t3, t4] -> [
              a1 + t1,
              a2 + t2,
              a3 + t3,
              a4 + t4,
            ]
            _, _ -> acc
          }
        })

      // Divide by count
      sums
      |> list.map(fn(sum) { sum / n })
    }
  }
}

// =============================================================================
// STATISTICS
// =============================================================================

/// Pool statistics
pub fn stats(pool: ResonancePool) -> String {
  let n_vivas = list.length(pool.vivas)
  let n_alive = list.length(list.filter(pool.vivas, fn(v) { v.alive }))
  let n_events = list.length(pool.events)

  "ResonancePool: "
  <> int.to_string(n_alive)
  <> "/"
  <> int.to_string(n_vivas)
  <> " alive, "
  <> int.to_string(n_events)
  <> " events, threshold="
  <> float_to_string(pool.threshold)
}

fn float_to_string(f: Float) -> String {
  let rounded =
    float.round(f *. 100.0) |> int.to_float() |> fn(x) { x /. 100.0 }
  erlang_float_to_list(rounded)
}

@external(erlang, "erlang", "float_to_list")
fn erlang_float_to_list(f: Float) -> String
