//// VIVA Sensory Imprinting - Stimulus-Valence Associations
////
//// Learns associations between sensory inputs and emotional valence:
//// - Light levels → good/bad feeling
//// - Sound levels → good/bad feeling
//// - Touch → good/bad feeling
////
//// Uses HRR (Holographic Reduced Representations) for binding.

import gleam/dict.{type Dict}
import gleam/float
import gleam/int
import gleam/list
import gleam/option.{type Option, None, Some}
import viva/memory/imprint/types.{type ImprintEvent, SensoryLearned}
import viva/memory/hrr.{type HRR}

// =============================================================================
// TYPES
// =============================================================================

/// Type of sensory stimulus
pub type StimulusType {
  Light(level: Int)
  // 0-1023
  Sound(level: Int)
  // 0-1023
  Touch(active: Bool)
}

/// A learned sensory association
pub type SensoryAssociation {
  SensoryAssociation(
    /// Type of stimulus
    stimulus: StimulusType,
    /// HRR representation (for similarity queries)
    hrr: HRR,
    /// Learned valence (-1.0 bad to +1.0 good)
    valence: Float,
    /// Confidence (0.0 to 1.0, grows with observations)
    confidence: Float,
    /// Number of observations
    observations: Int,
    /// First observation tick
    first_seen: Int,
    /// Last observation tick
    last_seen: Int,
  )
}

/// Sensory imprinting state
pub type SensoryImprint {
  SensoryImprint(
    /// Light level associations (binned by 100s)
    light_associations: Dict(Int, SensoryAssociation),
    /// Sound level associations (binned by 100s)
    sound_associations: Dict(Int, SensoryAssociation),
    /// Touch associations
    touch_positive: Option(SensoryAssociation),
    touch_negative: Option(SensoryAssociation),
    /// HRR dimension
    hrr_dim: Int,
  )
}

// =============================================================================
// CONSTRUCTORS
// =============================================================================

/// Create new sensory imprinting state
pub fn new(hrr_dim: Int) -> SensoryImprint {
  SensoryImprint(
    light_associations: dict.new(),
    sound_associations: dict.new(),
    touch_positive: None,
    touch_negative: None,
    hrr_dim: hrr_dim,
  )
}

// =============================================================================
// OBSERVATION
// =============================================================================

/// Observe sensory input with associated valence
/// Returns updated state and any imprint events
pub fn observe(
  state: SensoryImprint,
  light: Int,
  sound: Int,
  touch: Bool,
  pleasure: Float,
  intensity_multiplier: Float,
  tick: Int,
) -> #(SensoryImprint, List(ImprintEvent)) {
  let events = []

  // Observe light
  let #(new_light_assoc, light_event) =
    observe_light(
      state.light_associations,
      light,
      pleasure,
      intensity_multiplier,
      state.hrr_dim,
      tick,
    )
  let events = case light_event {
    Some(e) -> [e, ..events]
    None -> events
  }

  // Observe sound
  let #(new_sound_assoc, sound_event) =
    observe_sound(
      state.sound_associations,
      sound,
      pleasure,
      intensity_multiplier,
      state.hrr_dim,
      tick,
    )
  let events = case sound_event {
    Some(e) -> [e, ..events]
    None -> events
  }

  // Observe touch
  let #(new_touch_pos, new_touch_neg, touch_event) =
    observe_touch(
      state.touch_positive,
      state.touch_negative,
      touch,
      pleasure,
      intensity_multiplier,
      state.hrr_dim,
      tick,
    )
  let events = case touch_event {
    Some(e) -> [e, ..events]
    None -> events
  }

  let new_state =
    SensoryImprint(
      ..state,
      light_associations: new_light_assoc,
      sound_associations: new_sound_assoc,
      touch_positive: new_touch_pos,
      touch_negative: new_touch_neg,
    )

  #(new_state, events)
}

/// Observe light level
fn observe_light(
  associations: Dict(Int, SensoryAssociation),
  level: Int,
  pleasure: Float,
  intensity: Float,
  hrr_dim: Int,
  tick: Int,
) -> #(Dict(Int, SensoryAssociation), Option(ImprintEvent)) {
  let bin = level / 100
  // Bin by 100s (0-10 bins)

  case dict.get(associations, bin) {
    // Update existing
    Ok(existing) -> {
      let new_assoc = update_association(existing, pleasure, intensity, tick)
      let new_dict = dict.insert(associations, bin, new_assoc)
      #(new_dict, None)
    }
    // Create new
    Error(_) -> {
      let new_assoc =
        SensoryAssociation(
          stimulus: Light(bin * 100),
          hrr: hrr.random(hrr_dim),
          valence: pleasure *. intensity,
          confidence: 0.1 *. intensity,
          observations: 1,
          first_seen: tick,
          last_seen: tick,
        )
      let new_dict = dict.insert(associations, bin, new_assoc)
      let event =
        SensoryLearned(
          stimulus_type: "light_" <> int_to_string(bin * 100),
          valence: new_assoc.valence,
        )
      #(new_dict, Some(event))
    }
  }
}

/// Observe sound level
fn observe_sound(
  associations: Dict(Int, SensoryAssociation),
  level: Int,
  pleasure: Float,
  intensity: Float,
  hrr_dim: Int,
  tick: Int,
) -> #(Dict(Int, SensoryAssociation), Option(ImprintEvent)) {
  let bin = level / 100

  case dict.get(associations, bin) {
    Ok(existing) -> {
      let new_assoc = update_association(existing, pleasure, intensity, tick)
      let new_dict = dict.insert(associations, bin, new_assoc)
      #(new_dict, None)
    }
    Error(_) -> {
      let new_assoc =
        SensoryAssociation(
          stimulus: Sound(bin * 100),
          hrr: hrr.random(hrr_dim),
          valence: pleasure *. intensity,
          confidence: 0.1 *. intensity,
          observations: 1,
          first_seen: tick,
          last_seen: tick,
        )
      let new_dict = dict.insert(associations, bin, new_assoc)
      let event =
        SensoryLearned(
          stimulus_type: "sound_" <> int_to_string(bin * 100),
          valence: new_assoc.valence,
        )
      #(new_dict, Some(event))
    }
  }
}

/// Observe touch
fn observe_touch(
  touch_pos: Option(SensoryAssociation),
  touch_neg: Option(SensoryAssociation),
  touched: Bool,
  pleasure: Float,
  intensity: Float,
  hrr_dim: Int,
  tick: Int,
) -> #(
  Option(SensoryAssociation),
  Option(SensoryAssociation),
  Option(ImprintEvent),
) {
  case touched {
    False -> #(touch_pos, touch_neg, None)
    True -> {
      case pleasure >. 0.0 {
        // Positive touch experience
        True -> {
          let new_assoc = case touch_pos {
            Some(existing) ->
              Some(update_association(existing, pleasure, intensity, tick))
            None ->
              Some(SensoryAssociation(
                stimulus: Touch(True),
                hrr: hrr.random(hrr_dim),
                valence: pleasure *. intensity,
                confidence: 0.1 *. intensity,
                observations: 1,
                first_seen: tick,
                last_seen: tick,
              ))
          }
          let event = case touch_pos {
            None ->
              Some(SensoryLearned(
                stimulus_type: "touch_positive",
                valence: pleasure,
              ))
            Some(_) -> None
          }
          #(new_assoc, touch_neg, event)
        }
        // Negative touch experience
        False -> {
          let new_assoc = case touch_neg {
            Some(existing) ->
              Some(update_association(existing, pleasure, intensity, tick))
            None ->
              Some(SensoryAssociation(
                stimulus: Touch(True),
                hrr: hrr.random(hrr_dim),
                valence: pleasure *. intensity,
                confidence: 0.1 *. intensity,
                observations: 1,
                first_seen: tick,
                last_seen: tick,
              ))
          }
          let event = case touch_neg {
            None ->
              Some(SensoryLearned(
                stimulus_type: "touch_negative",
                valence: pleasure,
              ))
            Some(_) -> None
          }
          #(touch_pos, new_assoc, event)
        }
      }
    }
  }
}

/// Update existing association with new observation
fn update_association(
  assoc: SensoryAssociation,
  pleasure: Float,
  intensity: Float,
  tick: Int,
) -> SensoryAssociation {
  // Exponential moving average for valence
  let alpha = 0.1 *. intensity
  let new_valence = assoc.valence *. { 1.0 -. alpha } +. pleasure *. alpha

  // Confidence grows asymptotically toward 1.0
  let new_confidence =
    float.min(
      1.0,
      assoc.confidence +. 0.05 *. intensity *. { 1.0 -. assoc.confidence },
    )

  SensoryAssociation(
    ..assoc,
    valence: new_valence,
    confidence: new_confidence,
    observations: assoc.observations + 1,
    last_seen: tick,
  )
}

// =============================================================================
// QUERIES
// =============================================================================

/// Query expected valence for given sensory input
pub fn query_valence(
  state: SensoryImprint,
  light: Int,
  sound: Int,
  touch: Bool,
) -> Option(Float) {
  let light_val = query_light_valence(state.light_associations, light)
  let sound_val = query_sound_valence(state.sound_associations, sound)
  let touch_val =
    query_touch_valence(state.touch_positive, state.touch_negative, touch)

  // Weighted average of available valences
  let vals =
    [light_val, sound_val, touch_val]
    |> list.filter_map(fn(v) {
      case v {
        Some(#(val, conf)) -> Ok(#(val, conf))
        None -> Error(Nil)
      }
    })

  case vals {
    [] -> None
    vals -> {
      let total_conf = list.fold(vals, 0.0, fn(acc, v) { acc +. v.1 })
      let weighted_sum = list.fold(vals, 0.0, fn(acc, v) { acc +. v.0 *. v.1 })
      Some(weighted_sum /. total_conf)
    }
  }
}

fn query_light_valence(
  associations: Dict(Int, SensoryAssociation),
  level: Int,
) -> Option(#(Float, Float)) {
  let bin = level / 100
  case dict.get(associations, bin) {
    Ok(assoc) -> Some(#(assoc.valence, assoc.confidence))
    Error(_) -> None
  }
}

fn query_sound_valence(
  associations: Dict(Int, SensoryAssociation),
  level: Int,
) -> Option(#(Float, Float)) {
  let bin = level / 100
  case dict.get(associations, bin) {
    Ok(assoc) -> Some(#(assoc.valence, assoc.confidence))
    Error(_) -> None
  }
}

fn query_touch_valence(
  touch_pos: Option(SensoryAssociation),
  touch_neg: Option(SensoryAssociation),
  touched: Bool,
) -> Option(#(Float, Float)) {
  case touched {
    False -> None
    True -> {
      // Return weighted average if both exist
      case touch_pos, touch_neg {
        Some(pos), Some(neg) -> {
          let total_conf = pos.confidence +. neg.confidence
          let weighted =
            pos.valence *. pos.confidence +. neg.valence *. neg.confidence
          Some(#(weighted /. total_conf, total_conf /. 2.0))
        }
        Some(pos), None -> Some(#(pos.valence, pos.confidence))
        None, Some(neg) -> Some(#(neg.valence, neg.confidence))
        None, None -> None
      }
    }
  }
}

// =============================================================================
// STATISTICS
// =============================================================================

/// Count total associations
pub fn association_count(state: SensoryImprint) -> Int {
  let light_count = dict.size(state.light_associations)
  let sound_count = dict.size(state.sound_associations)
  let touch_count = case state.touch_positive, state.touch_negative {
    Some(_), Some(_) -> 2
    Some(_), None -> 1
    None, Some(_) -> 1
    None, None -> 0
  }
  light_count + sound_count + touch_count
}

/// Get all associations as list (for debugging)
pub fn all_associations(state: SensoryImprint) -> List(SensoryAssociation) {
  let light_list = dict.values(state.light_associations)
  let sound_list = dict.values(state.sound_associations)
  let touch_list =
    [state.touch_positive, state.touch_negative]
    |> list.filter_map(fn(opt) {
      case opt {
        Some(a) -> Ok(a)
        None -> Error(Nil)
      }
    })

  list.flatten([light_list, sound_list, touch_list])
}

// =============================================================================
// HELPERS
// =============================================================================

fn int_to_string(i: Int) -> String {
  int.to_string(i)
}
