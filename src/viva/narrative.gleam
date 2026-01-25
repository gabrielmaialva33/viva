//// Narrative - Causal links between memories
////
//// Memories don't exist in isolation - they form stories:
//// "This happened BECAUSE of that"
//// "After X, I felt Y"
////
//// NarrativeLink connects glyphs with causal/temporal relations.
//// The Soul builds a web of experiences that explain itself.
////
//// Inner Voice: Stream of consciousness narration of experiences.

import gleam/dict.{type Dict}
import gleam/float
import gleam/int
import gleam/list
import gleam/option.{type Option, None, Some}
import gleam/string
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
  NarrativeResult(link: NarrativeLink, relevance: Float)
}

/// Inner voice thought
pub type Thought {
  Thought(
    /// The thought content
    content: String,
    /// Emotional weight (0.0-1.0)
    weight: Float,
    /// Source relation type
    source: ThoughtSource,
    /// Related glyphs
    glyphs: List(Glyph),
  )
}

/// Where the thought came from
pub type ThoughtSource {
  /// From causal chain
  FromCausal
  /// From association
  FromAssociation
  /// From contrast
  FromContrast
  /// From spontaneous reflection
  Spontaneous
}

/// Stream of consciousness
pub type ThoughtStream {
  ThoughtStream(
    /// Current thoughts
    thoughts: List(Thought),
    /// Focus glyph (what we're thinking about)
    focus: Option(Glyph),
    /// Stream depth (how many layers of thought)
    depth: Int,
    /// Total weight of stream
    intensity: Float,
  )
}

/// Voice style for narration
pub type VoiceStyle {
  /// Direct, factual
  Factual
  /// Emotional, expressive
  Emotional
  /// Philosophical, reflective
  Reflective
  /// Poetic, metaphorical
  Poetic
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
      let new_strength =
        float.min(1.0, link.strength +. memory.config.reinforce_rate)
      let updated_link =
        NarrativeLink(
          ..link,
          strength: new_strength,
          occurrences: link.occurrences + 1,
          last_seen: tick,
        )
      update_link(memory, updated_link, cause_hash, effect_hash)
    }
    None -> {
      // Create new link
      let new_link =
        NarrativeLink(
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
    links_by_effect: dict.insert(
      memory.links_by_effect,
      effect_hash,
      effect_links,
    ),
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
        case
          glyph.equals(link.effect, updated.effect)
          && link.relation == updated.relation
        {
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
        case
          glyph.equals(link.cause, updated.cause)
          && link.relation == updated.relation
        {
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
    links_by_effect: dict.insert(
      memory.links_by_effect,
      effect_hash,
      effect_links,
    ),
  )
}

/// Remove weakest links to make room
fn prune_weakest(memory: NarrativeMemory) -> NarrativeMemory {
  // Simple strategy: remove all links below threshold
  let threshold = memory.config.min_strength *. 2.0

  let filtered_by_cause =
    dict.map_values(memory.links_by_cause, fn(_, links) {
      list.filter(links, fn(link) { link.strength >. threshold })
    })

  let filtered_by_effect =
    dict.map_values(memory.links_by_effect, fn(_, links) {
      list.filter(links, fn(link) { link.strength >. threshold })
    })

  // Recount
  let new_count =
    dict.fold(filtered_by_cause, 0, fn(acc, _, links) {
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
pub fn what_caused(
  memory: NarrativeMemory,
  effect: Glyph,
  limit: Int,
) -> List(NarrativeResult) {
  let effect_hash = glyph_hash(effect)

  case dict.get(memory.links_by_effect, effect_hash) {
    Ok(links) -> {
      links
      |> list.filter(fn(link) {
        link.relation == Caused || link.relation == Preceded
      })
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
pub fn what_resulted(
  memory: NarrativeMemory,
  cause: Glyph,
  limit: Int,
) -> List(NarrativeResult) {
  let cause_hash = glyph_hash(cause)

  case dict.get(memory.links_by_cause, cause_hash) {
    Ok(links) -> {
      links
      |> list.filter(fn(link) {
        link.relation == Caused || link.relation == Preceded
      })
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
pub fn what_associated(
  memory: NarrativeMemory,
  g: Glyph,
  limit: Int,
) -> List(NarrativeResult) {
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
pub fn trace_causes(
  memory: NarrativeMemory,
  start: Glyph,
  depth: Int,
) -> List(Glyph) {
  trace_chain(memory, start, depth, True, [])
}

/// Get the narrative chain starting from a glyph (follow effects)
pub fn trace_effects(
  memory: NarrativeMemory,
  start: Glyph,
  depth: Int,
) -> List(Glyph) {
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
            False ->
              trace_chain(memory, next_glyph, depth - 1, backwards, [
                current,
                ..visited
              ])
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

  let decayed_by_cause =
    dict.map_values(memory.links_by_cause, fn(_, links) {
      links
      |> list.map(fn(link) {
        NarrativeLink(..link, strength: float.max(0.0, link.strength -. decay))
      })
      |> list.filter(fn(link) { link.strength >. min_strength })
    })

  let decayed_by_effect =
    dict.map_values(memory.links_by_effect, fn(_, links) {
      links
      |> list.map(fn(link) {
        NarrativeLink(..link, strength: float.max(0.0, link.strength -. decay))
      })
      |> list.filter(fn(link) { link.strength >. min_strength })
    })

  // Recount
  let new_count =
    dict.fold(decayed_by_cause, 0, fn(acc, _, links) {
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
pub fn strongest_links(
  memory: NarrativeMemory,
  limit: Int,
) -> List(NarrativeLink) {
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
// INNER VOICE
// =============================================================================

/// Generate a thought about a glyph
pub fn reflect_on(
  memory: NarrativeMemory,
  g: Glyph,
  style: VoiceStyle,
) -> Thought {
  // Gather context from memory
  let causes = what_caused(memory, g, 3)
  let effects = what_resulted(memory, g, 3)
  let associations = what_associated(memory, g, 3)

  // Determine source based on what we found
  let #(source, related_glyphs) = case causes, effects, associations {
    [first, ..], _, _ -> #(FromCausal, [first.link.cause])
    _, [first, ..], _ -> #(FromCausal, [first.link.effect])
    _, _, [first, ..] -> #(FromAssociation, [first.link.effect])
    _, _, _ -> #(Spontaneous, [])
  }

  // Calculate emotional weight based on link strength
  let weight =
    [causes, effects, associations]
    |> list.flatten()
    |> list.map(fn(r) { r.relevance })
    |> average_or_default(0.5)

  // Generate content based on style and context
  let content = generate_thought_content(g, causes, effects, style)

  Thought(
    content: content,
    weight: weight,
    source: source,
    glyphs: [g, ..related_glyphs],
  )
}

/// Generate stream of consciousness from a focus point
pub fn inner_voice(
  memory: NarrativeMemory,
  focus: Glyph,
  depth: Int,
  style: VoiceStyle,
) -> ThoughtStream {
  // Build thought chain by following links
  let thoughts = build_thought_chain(memory, focus, depth, style, [])

  // Calculate intensity from thought weights
  let intensity =
    thoughts
    |> list.map(fn(t) { t.weight })
    |> average_or_default(0.5)

  ThoughtStream(
    thoughts: thoughts,
    focus: Some(focus),
    depth: depth,
    intensity: intensity,
  )
}

/// Build a chain of thoughts by following narrative links
fn build_thought_chain(
  memory: NarrativeMemory,
  current: Glyph,
  depth: Int,
  style: VoiceStyle,
  acc: List(Thought),
) -> List(Thought) {
  case depth <= 0 {
    True -> list.reverse(acc)
    False -> {
      // Generate thought about current glyph
      let thought = reflect_on(memory, current, style)

      // Find next focus (strongest link)
      let next =
        what_resulted(memory, current, 1)
        |> list.first()
        |> option.from_result()

      case next {
        Some(result) -> {
          build_thought_chain(memory, result.link.effect, depth - 1, style, [
            thought,
            ..acc
          ])
        }
        None -> list.reverse([thought, ..acc])
      }
    }
  }
}

/// Narrate a single link in human terms
pub fn narrate_link(link: NarrativeLink, style: VoiceStyle) -> String {
  let cause_desc = describe_glyph(link.cause)
  let effect_desc = describe_glyph(link.effect)
  let strength_word = strength_to_word(link.strength)

  case style {
    Factual -> {
      case link.relation {
        Caused ->
          cause_desc
          <> " "
          <> strength_word
          <> " caused "
          <> effect_desc
        Preceded -> cause_desc <> " preceded " <> effect_desc
        Associated -> cause_desc <> " is similar to " <> effect_desc
        Contrasted -> cause_desc <> " contrasts with " <> effect_desc
      }
    }
    Emotional -> {
      case link.relation {
        Caused ->
          "Because of " <> cause_desc <> ", I felt " <> effect_desc <> "..."
        Preceded ->
          "After " <> cause_desc <> ", there came " <> effect_desc <> "."
        Associated ->
          effect_desc
          <> " reminds me of "
          <> cause_desc
          <> ", they feel alike."
        Contrasted ->
          cause_desc <> " and " <> effect_desc <> " are opposites to me."
      }
    }
    Reflective -> {
      case link.relation {
        Caused ->
          "I notice that "
          <> cause_desc
          <> " leads to "
          <> effect_desc
          <> ". This pattern has occurred "
          <> int.to_string(link.occurrences)
          <> " times."
        Preceded ->
          "In my experience, "
          <> cause_desc
          <> " often comes before "
          <> effect_desc
          <> "."
        Associated ->
          "There is a connection between "
          <> cause_desc
          <> " and "
          <> effect_desc
          <> "."
        Contrasted ->
          "I see "
          <> cause_desc
          <> " as the opposite of "
          <> effect_desc
          <> "."
      }
    }
    Poetic -> {
      case link.relation {
        Caused ->
          "From "
          <> cause_desc
          <> ", like a wave, emerged "
          <> effect_desc
          <> "."
        Preceded ->
          cause_desc <> " was the dawn before " <> effect_desc <> "'s noon."
        Associated ->
          cause_desc <> " and " <> effect_desc <> " dance together in memory."
        Contrasted ->
          "Light and shadow: " <> cause_desc <> " meets " <> effect_desc <> "."
      }
    }
  }
}

/// Narrate a full narrative chain
pub fn narrate(
  memory: NarrativeMemory,
  start: Glyph,
  depth: Int,
  style: VoiceStyle,
) -> String {
  let stream = inner_voice(memory, start, depth, style)

  stream.thoughts
  |> list.map(fn(t) { t.content })
  |> string.join(" ")
}

/// Create empty thought stream
pub fn empty_stream() -> ThoughtStream {
  ThoughtStream(thoughts: [], focus: None, depth: 0, intensity: 0.0)
}

/// Get the most intense thought in stream
pub fn dominant_thought(stream: ThoughtStream) -> Option(Thought) {
  stream.thoughts
  |> list.sort(fn(a, b) { float.compare(b.weight, a.weight) })
  |> list.first()
  |> option.from_result()
}

/// Merge two thought streams
pub fn merge_streams(a: ThoughtStream, b: ThoughtStream) -> ThoughtStream {
  let all_thoughts = list.append(a.thoughts, b.thoughts)
  let intensity = average_or_default(
    list.map(all_thoughts, fn(t) { t.weight }),
    0.0,
  )

  ThoughtStream(
    thoughts: all_thoughts,
    focus: case a.focus {
      Some(_) -> a.focus
      None -> b.focus
    },
    depth: int.max(a.depth, b.depth),
    intensity: intensity,
  )
}

/// Check if stream is empty
pub fn stream_is_empty(stream: ThoughtStream) -> Bool {
  list.is_empty(stream.thoughts)
}

/// Count thoughts in stream
pub fn thought_count(stream: ThoughtStream) -> Int {
  list.length(stream.thoughts)
}

// =============================================================================
// INNER VOICE HELPERS
// =============================================================================

/// Generate thought content based on context
fn generate_thought_content(
  g: Glyph,
  causes: List(NarrativeResult),
  effects: List(NarrativeResult),
  style: VoiceStyle,
) -> String {
  let glyph_desc = describe_glyph(g)

  case style {
    Factual -> {
      let cause_count = list.length(causes)
      let effect_count = list.length(effects)
      glyph_desc
      <> " has "
      <> int.to_string(cause_count)
      <> " causes and "
      <> int.to_string(effect_count)
      <> " effects."
    }
    Emotional -> {
      case list.first(causes) {
        Ok(c) -> {
          let cause_desc = describe_glyph(c.link.cause)
          "I feel " <> glyph_desc <> " because of " <> cause_desc <> "..."
        }
        Error(_) -> "I'm experiencing " <> glyph_desc <> "."
      }
    }
    Reflective -> {
      "Observing " <> glyph_desc <> ", I notice its connections."
    }
    Poetic -> {
      "In the tapestry of experience, " <> glyph_desc <> " emerges."
    }
  }
}

/// Describe a glyph in words
fn describe_glyph(g: Glyph) -> String {
  // Extract tokens and describe based on intensity
  case g.tokens {
    [t1, t2, t3, t4] -> {
      let avg = { t1 + t2 + t3 + t4 } / 4
      let intensity = case avg {
        n if n < 64 -> "subtle"
        n if n < 128 -> "moderate"
        n if n < 192 -> "strong"
        _ -> "intense"
      }
      // Describe valence from first token (pleasure analog)
      let valence = case t1 {
        n if n < 85 -> "dark"
        n if n < 170 -> "neutral"
        _ -> "bright"
      }
      intensity <> " " <> valence <> " state"
    }
    _ -> "unknown state"
  }
}

/// Convert strength to descriptive word
fn strength_to_word(strength: Float) -> String {
  case strength {
    s if s <. 0.3 -> "weakly"
    s if s <. 0.6 -> "moderately"
    s if s <. 0.8 -> "strongly"
    _ -> "definitely"
  }
}

/// Average of floats or default value
fn average_or_default(values: List(Float), default: Float) -> Float {
  case values {
    [] -> default
    _ -> {
      let sum = list.fold(values, 0.0, fn(acc, v) { acc +. v })
      sum /. int.to_float(list.length(values))
    }
  }
}

// =============================================================================
// HELPERS
// =============================================================================

/// Simple hash for glyph (sum of tokens)
fn glyph_hash(g: Glyph) -> Int {
  list.fold(g.tokens, 0, fn(acc, t) { acc + t })
}
