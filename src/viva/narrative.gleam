//// Narrative - Causal links between memories
////
//// Memories don't exist in isolation - they form stories:
//// "This happened BECAUSE of that"
//// "After X, I felt Y"
////
//// NarrativeLink connects glyphs with causal/temporal relations.
//// The Soul builds a web of experiences that explain itself.

import gleam/dict.{type Dict}
import gleam/float
import gleam/list
import gleam/option.{type Option, None, Some}
import viva_glyph/glyph.{type Glyph}

// =============================================================================
// TYPES
// =============================================================================

/// A link between two glyphs (cause → effect)
pub type NarrativeLink {
  NarrativeLink(
    /// Source glyph (cause/antecedent)
    cause: Glyph,
    /// Target glyph (effect/consequent)
    effect: Glyph,
    /// Type of relationship
    relation: Relation,
    /// Strength of association (0.0-1.0)
    strength: Float,
    /// How many times this link was observed
    occurrences: Int,
    /// Tick when first observed
    first_seen: Int,
    /// Tick when last observed
    last_seen: Int,
  )
}

/// Relation types
pub type Relation {
  /// A caused B (strong causal)
  Caused
  /// A preceded B (temporal, weaker)
  Preceded
  /// A is similar to B (associative)
  Associated
  /// A contrasts with B (opposite)
  Contrasted
}

/// Narrative memory: web of linked experiences
pub type NarrativeMemory {
  NarrativeMemory(
    /// Links indexed by cause glyph hash
    links_by_cause: Dict(Int, List(NarrativeLink)),
    /// Links indexed by effect glyph hash
    links_by_effect: Dict(Int, List(NarrativeLink)),
    /// Total link count
    link_count: Int,
    /// Configuration
    config: NarrativeConfig,
  )
}

/// Narrative configuration
pub type NarrativeConfig {
  NarrativeConfig(
    /// Max links to store
    max_links: Int,
    /// Min strength to keep link
    min_strength: Float,
    /// Strength decay per tick
    decay_rate: Float,
    /// Strength boost on repeat observation
    reinforce_rate: Float,
  )
}

/// Query result for narrative search
pub type NarrativeResult {
  NarrativeResult(
    link: NarrativeLink,
    relevance: Float,
  )
}

// =============================================================================
// CONSTRUCTORS
// =============================================================================

/// Create new narrative memory
pub fn new() -> NarrativeMemory {
  new_with_config(default_config())
}

/// Create with custom config
pub fn new_with_config(config: NarrativeConfig) -> NarrativeMemory {
  NarrativeMemory(
    links_by_cause: dict.new(),
    links_by_effect: dict.new(),
    link_count: 0,
    config: config,
  )
}

/// Default configuration
pub fn default_config() -> NarrativeConfig {
  NarrativeConfig(
    max_links: 1000,
    min_strength: 0.1,
    decay_rate: 0.001,
    reinforce_rate: 0.2,
  )
}

// =============================================================================
// RECORDING LINKS
// =============================================================================

/// Record a causal link: "cause led to effect"
pub fn record_caused(
  memory: NarrativeMemory,
  cause: Glyph,
  effect: Glyph,
  tick: Int,
) -> NarrativeMemory {
  record_link(memory, cause, effect, Caused, 0.5, tick)
}

/// Record a temporal link: "cause preceded effect"
pub fn record_preceded(
  memory: NarrativeMemory,
  cause: Glyph,
  effect: Glyph,
  tick: Int,
) -> NarrativeMemory {
  record_link(memory, cause, effect, Preceded, 0.3, tick)
}

/// Record an association: "these are similar"
pub fn record_associated(
  memory: NarrativeMemory,
  a: Glyph,
  b: Glyph,
  tick: Int,
) -> NarrativeMemory {
  // Associations are bidirectional
  memory
  |> record_link(a, b, Associated, 0.2, tick)
  |> record_link(b, a, Associated, 0.2, tick)
}

/// Record a contrast: "these are opposite"
pub fn record_contrasted(
  memory: NarrativeMemory,
  a: Glyph,
  b: Glyph,
  tick: Int,
) -> NarrativeMemory {
  memory
  |> record_link(a, b, Contrasted, 0.3, tick)
  |> record_link(b, a, Contrasted, 0.3, tick)
}

/// Internal: record or reinforce a link
fn record_link(
  memory: NarrativeMemory,
  cause: Glyph,
  effect: Glyph,
  relation: Relation,
  initial_strength: Float,
  tick: Int,
) -> NarrativeMemory {
  let cause_hash = glyph_hash(cause)
  let effect_hash = glyph_hash(effect)

  // Check if link exists
  let existing = find_link(memory, cause, effect, relation)

  case existing {
    Some(link) -> {
      // Reinforce existing link
      let new_strength = float.min(1.0, link.strength +. memory.config.reinforce_rate)
      let updated_link = NarrativeLink(
        ..link,
        strength: new_strength,
        occurrences: link.occurrences + 1,
        last_seen: tick,
      )
      update_link(memory, updated_link, cause_hash, effect_hash)
    }
    None -> {
      // Create new link
      let new_link = NarrativeLink(
        cause: cause,
        effect: effect,
        relation: relation,
        strength: initial_strength,
        occurrences: 1,
        first_seen: tick,
        last_seen: tick,
      )
      add_link(memory, new_link, cause_hash, effect_hash)
    }
  }
}

/// Find existing link
fn find_link(
  memory: NarrativeMemory,
  cause: Glyph,
  effect: Glyph,
  relation: Relation,
) -> Option(NarrativeLink) {
  let cause_hash = glyph_hash(cause)

  case dict.get(memory.links_by_cause, cause_hash) {
    Ok(links) -> {
      links
      |> list.find(fn(link) {
        glyph.equals(link.effect, effect) && link.relation == relation
      })
      |> option.from_result
    }
    Error(_) -> None
  }
}

/// Add new link
fn add_link(
  memory: NarrativeMemory,
  link: NarrativeLink,
  cause_hash: Int,
  effect_hash: Int,
) -> NarrativeMemory {
  // Check capacity
  let memory = case memory.link_count >= memory.config.max_links {
    True -> prune_weakest(memory)
    False -> memory
  }

  // Add to cause index
  let cause_links = case dict.get(memory.links_by_cause, cause_hash) {
    Ok(links) -> [link, ..links]
    Error(_) -> [link]
  }

  // Add to effect index
  let effect_links = case dict.get(memory.links_by_effect, effect_hash) {
    Ok(links) -> [link, ..links]
    Error(_) -> [link]
  }

  NarrativeMemory(
    ..memory,
    links_by_cause: dict.insert(memory.links_by_cause, cause_hash, cause_links),
    links_by_effect: dict.insert(memory.links_by_effect, effect_hash, effect_links),
    link_count: memory.link_count + 1,
  )
}

/// Update existing link
fn update_link(
  memory: NarrativeMemory,
  updated: NarrativeLink,
  cause_hash: Int,
  effect_hash: Int,
) -> NarrativeMemory {
  // Update in cause index
  let cause_links = case dict.get(memory.links_by_cause, cause_hash) {
    Ok(links) -> {
      list.map(links, fn(link) {
        case glyph.equals(link.effect, updated.effect) && link.relation == updated.relation {
          True -> updated
          False -> link
        }
      })
    }
    Error(_) -> [updated]
  }

  // Update in effect index
  let effect_links = case dict.get(memory.links_by_effect, effect_hash) {
    Ok(links) -> {
      list.map(links, fn(link) {
        case glyph.equals(link.cause, updated.cause) && link.relation == updated.relation {
          True -> updated
          False -> link
        }
      })
    }
    Error(_) -> [updated]
  }

  NarrativeMemory(
    ..memory,
    links_by_cause: dict.insert(memory.links_by_cause, cause_hash, cause_links),
    links_by_effect: dict.insert(memory.links_by_effect, effect_hash, effect_links),
  )
}

/// Remove weakest links to make room
fn prune_weakest(memory: NarrativeMemory) -> NarrativeMemory {
  // Simple strategy: remove all links below threshold
  let threshold = memory.config.min_strength *. 2.0

  let filtered_by_cause = dict.map_values(memory.links_by_cause, fn(_, links) {
    list.filter(links, fn(link) { link.strength >. threshold })
  })

  let filtered_by_effect = dict.map_values(memory.links_by_effect, fn(_, links) {
    list.filter(links, fn(link) { link.strength >. threshold })
  })

  // Recount
  let new_count = dict.fold(filtered_by_cause, 0, fn(acc, _, links) {
    acc + list.length(links)
  })

  NarrativeMemory(
    ..memory,
    links_by_cause: filtered_by_cause,
    links_by_effect: filtered_by_effect,
    link_count: new_count,
  )
}

// =============================================================================
// QUERYING
// =============================================================================

/// What caused this glyph? (look for effects matching this)
pub fn what_caused(memory: NarrativeMemory, effect: Glyph, limit: Int) -> List(NarrativeResult) {
  let effect_hash = glyph_hash(effect)

  case dict.get(memory.links_by_effect, effect_hash) {
    Ok(links) -> {
      links
      |> list.filter(fn(link) { link.relation == Caused || link.relation == Preceded })
      |> list.map(fn(link) {
        NarrativeResult(link: link, relevance: link.strength)
      })
      |> list.sort(fn(a, b) { float.compare(b.relevance, a.relevance) })
      |> list.take(limit)
    }
    Error(_) -> []
  }
}

/// What did this glyph cause? (look for causes matching this)
pub fn what_resulted(memory: NarrativeMemory, cause: Glyph, limit: Int) -> List(NarrativeResult) {
  let cause_hash = glyph_hash(cause)

  case dict.get(memory.links_by_cause, cause_hash) {
    Ok(links) -> {
      links
      |> list.filter(fn(link) { link.relation == Caused || link.relation == Preceded })
      |> list.map(fn(link) {
        NarrativeResult(link: link, relevance: link.strength)
      })
      |> list.sort(fn(a, b) { float.compare(b.relevance, a.relevance) })
      |> list.take(limit)
    }
    Error(_) -> []
  }
}

/// What is similar to this glyph?
pub fn what_associated(memory: NarrativeMemory, g: Glyph, limit: Int) -> List(NarrativeResult) {
  let hash = glyph_hash(g)

  case dict.get(memory.links_by_cause, hash) {
    Ok(links) -> {
      links
      |> list.filter(fn(link) { link.relation == Associated })
      |> list.map(fn(link) {
        NarrativeResult(link: link, relevance: link.strength)
      })
      |> list.sort(fn(a, b) { float.compare(b.relevance, a.relevance) })
      |> list.take(limit)
    }
    Error(_) -> []
  }
}

/// Get the narrative chain starting from a glyph (follow causes)
pub fn trace_causes(memory: NarrativeMemory, start: Glyph, depth: Int) -> List(Glyph) {
  trace_chain(memory, start, depth, True, [])
}

/// Get the narrative chain starting from a glyph (follow effects)
pub fn trace_effects(memory: NarrativeMemory, start: Glyph, depth: Int) -> List(Glyph) {
  trace_chain(memory, start, depth, False, [])
}

fn trace_chain(
  memory: NarrativeMemory,
  current: Glyph,
  depth: Int,
  backwards: Bool,
  visited: List(Glyph),
) -> List(Glyph) {
  case depth <= 0 {
    True -> list.reverse(visited)
    False -> {
      // Get next links
      let next_results = case backwards {
        True -> what_caused(memory, current, 1)
        False -> what_resulted(memory, current, 1)
      }

      case list.first(next_results) {
        Ok(result) -> {
          let next_glyph = case backwards {
            True -> result.link.cause
            False -> result.link.effect
          }
          // Avoid cycles
          case list.any(visited, fn(v) { glyph.equals(v, next_glyph) }) {
            True -> list.reverse(visited)
            False -> trace_chain(memory, next_glyph, depth - 1, backwards, [current, ..visited])
          }
        }
        Error(_) -> list.reverse([current, ..visited])
      }
    }
  }
}

// =============================================================================
// MAINTENANCE
// =============================================================================

/// Tick: decay all link strengths
pub fn tick(memory: NarrativeMemory) -> NarrativeMemory {
  let decay = memory.config.decay_rate
  let min_strength = memory.config.min_strength

  let decayed_by_cause = dict.map_values(memory.links_by_cause, fn(_, links) {
    links
    |> list.map(fn(link) {
      NarrativeLink(..link, strength: float.max(0.0, link.strength -. decay))
    })
    |> list.filter(fn(link) { link.strength >. min_strength })
  })

  let decayed_by_effect = dict.map_values(memory.links_by_effect, fn(_, links) {
    links
    |> list.map(fn(link) {
      NarrativeLink(..link, strength: float.max(0.0, link.strength -. decay))
    })
    |> list.filter(fn(link) { link.strength >. min_strength })
  })

  // Recount
  let new_count = dict.fold(decayed_by_cause, 0, fn(acc, _, links) {
    acc + list.length(links)
  })

  NarrativeMemory(
    ..memory,
    links_by_cause: decayed_by_cause,
    links_by_effect: decayed_by_effect,
    link_count: new_count,
  )
}

// =============================================================================
// STATS
// =============================================================================

/// Get link count
pub fn link_count(memory: NarrativeMemory) -> Int {
  memory.link_count
}

/// Get strongest links
pub fn strongest_links(memory: NarrativeMemory, limit: Int) -> List(NarrativeLink) {
  memory.links_by_cause
  |> dict.values()
  |> list.flatten()
  |> list.sort(fn(a, b) { float.compare(b.strength, a.strength) })
  |> list.take(limit)
}

/// Convert relation to string
pub fn relation_to_string(rel: Relation) -> String {
  case rel {
    Caused -> "caused"
    Preceded -> "preceded"
    Associated -> "associated"
    Contrasted -> "contrasted"
  }
}

// =============================================================================
// HELPERS
// =============================================================================

/// Simple hash for glyph (sum of tokens)
fn glyph_hash(g: Glyph) -> Int {
  list.fold(g.tokens, 0, fn(acc, t) { acc + t })
}
