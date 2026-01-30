//// Social Imprinting - Social recognition and attachment
////
//// Tracks attachment figures and social preferences.

import gleam/list
import gleam/option.{type Option, None, Some}
import viva/memory/imprint/types.{type ImprintEvent, AttachmentChanged}

// =============================================================================
// TYPES
// =============================================================================

/// Social imprint state
pub type SocialImprint {
  SocialImprint(
    /// Attachment figures
    attachments: List(Attachment),
    /// Primary attachment (first bond)
    primary: String,
    /// Trust level (0.0 to 1.0)
    trust_level: Float,
  )
}

/// An attachment figure
pub type Attachment {
  Attachment(
    /// Name/identifier
    name: String,
    /// Attachment strength (0.0 to 1.0)
    strength: Float,
    /// Interactions count
    interactions: Int,
    /// Last seen tick
    last_seen: Int,
    /// Associated pleasure (average)
    pleasure_association: Float,
  )
}

// =============================================================================
// CONSTRUCTORS
// =============================================================================

/// Create new social imprint with primary attachment
pub fn new(creator_name: String) -> SocialImprint {
  let primary =
    Attachment(
      name: creator_name,
      strength: 0.5,
      interactions: 0,
      last_seen: 0,
      pleasure_association: 0.5,
    )
  SocialImprint(attachments: [primary], primary: creator_name, trust_level: 0.5)
}

// =============================================================================
// OPERATIONS
// =============================================================================

/// Observe presence of an entity during critical period
pub fn observe_presence(
  imprint: SocialImprint,
  entity_present: Option(String),
  pleasure: Float,
  _arousal: Float,
  intensity: Float,
  current_tick: Int,
) -> #(SocialImprint, List(ImprintEvent)) {
  case entity_present {
    None -> #(imprint, [])
    Some(entity) -> {
      // Find or create attachment
      let existing = list.find(imprint.attachments, fn(a) { a.name == entity })

      case existing {
        Ok(att) -> {
          // Update existing
          let delta = { pleasure -. 0.5 } *. intensity *. 0.1
          let new_strength = clamp(att.strength +. delta, 0.0, 1.0)
          let new_pleasure_assoc =
            { att.pleasure_association +. pleasure } /. 2.0

          let updated =
            Attachment(
              ..att,
              strength: new_strength,
              interactions: att.interactions + 1,
              last_seen: current_tick,
              pleasure_association: new_pleasure_assoc,
            )

          let attachments =
            list.map(imprint.attachments, fn(a) {
              case a.name == entity {
                True -> updated
                False -> a
              }
            })

          let events = case abs_float(delta) >. 0.1 {
            True -> [AttachmentChanged(entity: entity, strength: new_strength)]
            False -> []
          }

          #(SocialImprint(..imprint, attachments: attachments), events)
        }
        Error(_) -> {
          // New attachment
          let initial_strength = pleasure *. intensity *. 0.5
          let att =
            Attachment(
              name: entity,
              strength: clamp(initial_strength, 0.1, 0.8),
              interactions: 1,
              last_seen: current_tick,
              pleasure_association: pleasure,
            )

          #(
            SocialImprint(..imprint, attachments: [att, ..imprint.attachments]),
            [AttachmentChanged(entity: entity, strength: initial_strength)],
          )
        }
      }
    }
  }
}

/// Get attachment strength to an entity
pub fn attachment_strength(imprint: SocialImprint, entity: String) -> Float {
  imprint.attachments
  |> list.find(fn(a) { a.name == entity })
  |> option.from_result
  |> option.map(fn(a) { a.strength })
  |> option.unwrap(0.0)
}

/// Check if entity is primary attachment
pub fn is_primary(imprint: SocialImprint, entity: String) -> Bool {
  imprint.primary == entity
}

/// Get attachment count
pub fn attachment_count(imprint: SocialImprint) -> Int {
  list.length(imprint.attachments)
}

/// Get entity count (alias for attachment_count)
pub fn entity_count(imprint: SocialImprint) -> Int {
  list.length(imprint.attachments)
}

/// Get primary attachment strength
pub fn primary_attachment_strength(imprint: SocialImprint) -> Float {
  attachment_strength(imprint, imprint.primary)
}

/// Update trust level
pub fn update_trust(imprint: SocialImprint, delta: Float) -> SocialImprint {
  let new_trust = clamp(imprint.trust_level +. delta, 0.0, 1.0)
  SocialImprint(..imprint, trust_level: new_trust)
}

// =============================================================================
// HELPERS
// =============================================================================

fn clamp(value: Float, min: Float, max: Float) -> Float {
  case value <. min {
    True -> min
    False ->
      case value >. max {
        True -> max
        False -> value
      }
  }
}

fn abs_float(a: Float) -> Float {
  case a <. 0.0 {
    True -> 0.0 -. a
    False -> a
  }
}
