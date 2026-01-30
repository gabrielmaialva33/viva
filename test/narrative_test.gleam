//// Narrative Tests
////
//// Tests for causal links and narrative memory.

import gleam/list
import gleam/option.{None, Some}
import gleam/string
import gleeunit/should
import viva/memory/narrative.{
  Associated, Caused, Contrasted, Emotional, Factual, FromAssociation,
  FromCausal, Poetic, Preceded, Reflective, Spontaneous,
}
import viva_glyph/glyph

// =============================================================================
// HELPERS
// =============================================================================

fn make_glyph(a: Int, b: Int, c: Int, d: Int) -> glyph.Glyph {
  glyph.new([a, b, c, d])
}

// =============================================================================
// CREATION
// =============================================================================

pub fn new_creates_empty_memory_test() {
  let mem = narrative.new()

  narrative.link_count(mem) |> should.equal(0)
}

pub fn new_with_config_respects_settings_test() {
  let config =
    narrative.NarrativeConfig(
      max_links: 500,
      min_strength: 0.2,
      decay_rate: 0.01,
      reinforce_rate: 0.3,
    )
  let mem = narrative.new_with_config(config)

  narrative.link_count(mem) |> should.equal(0)
}

// =============================================================================
// RECORDING LINKS
// =============================================================================

pub fn record_caused_creates_link_test() {
  let mem = narrative.new()
  let cause = make_glyph(100, 50, 50, 50)
  let effect = make_glyph(200, 100, 100, 100)

  let mem = narrative.record_caused(mem, cause, effect, 1)

  narrative.link_count(mem) |> should.equal(1)
}

pub fn record_preceded_creates_link_test() {
  let mem = narrative.new()
  let before = make_glyph(50, 50, 50, 50)
  let after = make_glyph(100, 100, 100, 100)

  let mem = narrative.record_preceded(mem, before, after, 1)

  narrative.link_count(mem) |> should.equal(1)
}

pub fn record_associated_creates_bidirectional_links_test() {
  let mem = narrative.new()
  let a = make_glyph(50, 50, 50, 50)
  let b = make_glyph(60, 60, 60, 60)

  let mem = narrative.record_associated(mem, a, b, 1)

  // Association creates 2 links (bidirectional)
  narrative.link_count(mem) |> should.equal(2)
}

pub fn record_contrasted_creates_bidirectional_links_test() {
  let mem = narrative.new()
  let a = make_glyph(0, 0, 0, 0)
  let b = make_glyph(255, 255, 255, 255)

  let mem = narrative.record_contrasted(mem, a, b, 1)

  // Contrast creates 2 links (bidirectional)
  narrative.link_count(mem) |> should.equal(2)
}

pub fn repeated_recording_reinforces_link_test() {
  let mem = narrative.new()
  let cause = make_glyph(100, 50, 50, 50)
  let effect = make_glyph(200, 100, 100, 100)

  // Record same link multiple times
  let mem = narrative.record_caused(mem, cause, effect, 1)
  let mem = narrative.record_caused(mem, cause, effect, 2)
  let mem = narrative.record_caused(mem, cause, effect, 3)

  // Still only 1 link (reinforced, not duplicated)
  narrative.link_count(mem) |> should.equal(1)

  // Strength should be higher
  let links = narrative.strongest_links(mem, 1)
  case list.first(links) {
    Ok(link) -> {
      { link.strength >. 0.5 } |> should.be_true()
      link.occurrences |> should.equal(3)
    }
    Error(_) -> should.fail()
  }
}

// =============================================================================
// QUERYING
// =============================================================================

pub fn what_caused_finds_causes_test() {
  let mem = narrative.new()
  let cause = make_glyph(100, 50, 50, 50)
  let effect = make_glyph(200, 100, 100, 100)

  let mem = narrative.record_caused(mem, cause, effect, 1)

  let results = narrative.what_caused(mem, effect, 10)

  list.length(results) |> should.equal(1)
  case list.first(results) {
    Ok(result) -> result.link.relation |> should.equal(Caused)
    Error(_) -> should.fail()
  }
}

pub fn what_resulted_finds_effects_test() {
  let mem = narrative.new()
  let cause = make_glyph(100, 50, 50, 50)
  let effect = make_glyph(200, 100, 100, 100)

  let mem = narrative.record_caused(mem, cause, effect, 1)

  let results = narrative.what_resulted(mem, cause, 10)

  list.length(results) |> should.equal(1)
  case list.first(results) {
    Ok(result) -> result.link.relation |> should.equal(Caused)
    Error(_) -> should.fail()
  }
}

pub fn what_associated_finds_associations_test() {
  let mem = narrative.new()
  let a = make_glyph(50, 50, 50, 50)
  let b = make_glyph(60, 60, 60, 60)

  let mem = narrative.record_associated(mem, a, b, 1)

  let results = narrative.what_associated(mem, a, 10)

  list.length(results) |> should.equal(1)
  case list.first(results) {
    Ok(result) -> result.link.relation |> should.equal(Associated)
    Error(_) -> should.fail()
  }
}

pub fn what_caused_returns_empty_for_unknown_glyph_test() {
  let mem = narrative.new()
  let unknown = make_glyph(1, 2, 3, 4)

  let results = narrative.what_caused(mem, unknown, 10)

  list.length(results) |> should.equal(0)
}

// =============================================================================
// TRACE CHAINS
// =============================================================================

pub fn trace_effects_follows_chain_test() {
  let mem = narrative.new()
  let g1 = make_glyph(10, 10, 10, 10)
  let g2 = make_glyph(20, 20, 20, 20)
  let g3 = make_glyph(30, 30, 30, 30)

  // g1 -> g2 -> g3
  let mem = narrative.record_caused(mem, g1, g2, 1)
  let mem = narrative.record_caused(mem, g2, g3, 2)

  let chain = narrative.trace_effects(mem, g1, 3)

  list.length(chain) |> should.equal(3)
}

pub fn trace_causes_follows_chain_backwards_test() {
  let mem = narrative.new()
  let g1 = make_glyph(10, 10, 10, 10)
  let g2 = make_glyph(20, 20, 20, 20)
  let g3 = make_glyph(30, 30, 30, 30)

  // g1 -> g2 -> g3
  let mem = narrative.record_caused(mem, g1, g2, 1)
  let mem = narrative.record_caused(mem, g2, g3, 2)

  let chain = narrative.trace_causes(mem, g3, 3)

  list.length(chain) |> should.equal(3)
}

// =============================================================================
// MAINTENANCE
// =============================================================================

pub fn tick_decays_strength_test() {
  let mem = narrative.new()
  let cause = make_glyph(100, 50, 50, 50)
  let effect = make_glyph(200, 100, 100, 100)

  let mem = narrative.record_caused(mem, cause, effect, 1)

  // Get initial strength
  let links_before = narrative.strongest_links(mem, 1)
  let strength_before = case list.first(links_before) {
    Ok(link) -> link.strength
    Error(_) -> 0.0
  }

  // Tick several times
  let mem =
    list.range(1, 10)
    |> list.fold(mem, fn(m, _) { narrative.tick(m) })

  // Strength should be lower
  let links_after = narrative.strongest_links(mem, 1)
  let strength_after = case list.first(links_after) {
    Ok(link) -> link.strength
    Error(_) -> 0.0
  }

  { strength_after <. strength_before } |> should.be_true()
}

pub fn tick_removes_weak_links_test() {
  let config =
    narrative.NarrativeConfig(
      max_links: 100,
      min_strength: 0.4,
      // High threshold
      decay_rate: 0.1,
      // Fast decay
      reinforce_rate: 0.2,
    )
  let mem = narrative.new_with_config(config)

  let cause = make_glyph(100, 50, 50, 50)
  let effect = make_glyph(200, 100, 100, 100)

  let mem = narrative.record_caused(mem, cause, effect, 1)
  narrative.link_count(mem) |> should.equal(1)

  // Tick until link decays below threshold
  let mem =
    list.range(1, 20)
    |> list.fold(mem, fn(m, _) { narrative.tick(m) })

  // Link should be removed
  narrative.link_count(mem) |> should.equal(0)
}

// =============================================================================
// STATS
// =============================================================================

pub fn strongest_links_returns_sorted_test() {
  let mem = narrative.new()

  // Create multiple links with different strengths
  let g1 = make_glyph(10, 10, 10, 10)
  let g2 = make_glyph(20, 20, 20, 20)
  let g3 = make_glyph(30, 30, 30, 30)
  let g4 = make_glyph(40, 40, 40, 40)

  let mem = narrative.record_caused(mem, g1, g2, 1)
  // Reinforce g3->g4 to make it stronger
  let mem = narrative.record_caused(mem, g3, g4, 1)
  let mem = narrative.record_caused(mem, g3, g4, 2)
  let mem = narrative.record_caused(mem, g3, g4, 3)

  let links = narrative.strongest_links(mem, 2)

  list.length(links) |> should.equal(2)

  // First should be stronger (g3->g4)
  case links {
    [first, second, ..] -> {
      { first.strength >=. second.strength } |> should.be_true()
    }
    _ -> should.fail()
  }
}

pub fn relation_to_string_test() {
  narrative.relation_to_string(Caused) |> should.equal("caused")
  narrative.relation_to_string(Preceded) |> should.equal("preceded")
  narrative.relation_to_string(Associated) |> should.equal("associated")
  narrative.relation_to_string(Contrasted) |> should.equal("contrasted")
}

// =============================================================================
// INNER VOICE
// =============================================================================

pub fn reflect_on_generates_thought_test() {
  let mem = narrative.new()
  let g = make_glyph(100, 100, 100, 100)

  let thought = narrative.reflect_on(mem, g, Factual)

  // Should generate content
  { string.length(thought.content) > 0 } |> should.be_true()
  // Without links, source is spontaneous
  thought.source |> should.equal(Spontaneous)
  // Should include the focus glyph
  { list.length(thought.glyphs) >= 1 } |> should.be_true()
}

pub fn reflect_on_with_links_finds_source_test() {
  let mem = narrative.new()
  let cause = make_glyph(50, 50, 50, 50)
  let effect = make_glyph(100, 100, 100, 100)

  let mem = narrative.record_caused(mem, cause, effect, 1)

  let thought = narrative.reflect_on(mem, effect, Emotional)

  // Should recognize causal source
  thought.source |> should.equal(FromCausal)
  // Should include related glyphs
  { list.length(thought.glyphs) >= 2 } |> should.be_true()
}

pub fn inner_voice_generates_stream_test() {
  let mem = narrative.new()
  let g1 = make_glyph(10, 10, 10, 10)
  let g2 = make_glyph(20, 20, 20, 20)
  let g3 = make_glyph(30, 30, 30, 30)

  let mem = narrative.record_caused(mem, g1, g2, 1)
  let mem = narrative.record_caused(mem, g2, g3, 2)

  let stream = narrative.inner_voice(mem, g1, 3, Reflective)

  // Should have thoughts
  { narrative.thought_count(stream) >= 1 } |> should.be_true()
  // Focus should be set
  stream.focus |> should.equal(Some(g1))
  // Intensity should be positive
  { stream.intensity >. 0.0 } |> should.be_true()
}

pub fn narrate_link_factual_test() {
  let cause = make_glyph(100, 100, 100, 100)
  let effect = make_glyph(200, 200, 200, 200)
  let link =
    narrative.NarrativeLink(
      cause: cause,
      effect: effect,
      relation: Caused,
      strength: 0.7,
      occurrences: 3,
      first_seen: 1,
      last_seen: 5,
    )

  let narration = narrative.narrate_link(link, Factual)

  string.contains(narration, "caused") |> should.be_true()
}

pub fn narrate_link_emotional_test() {
  let cause = make_glyph(50, 50, 50, 50)
  let effect = make_glyph(150, 150, 150, 150)
  let link =
    narrative.NarrativeLink(
      cause: cause,
      effect: effect,
      relation: Associated,
      strength: 0.5,
      occurrences: 1,
      first_seen: 1,
      last_seen: 1,
    )

  let narration = narrative.narrate_link(link, Emotional)

  string.contains(narration, "reminds me") |> should.be_true()
}

pub fn narrate_link_poetic_test() {
  let cause = make_glyph(0, 0, 0, 0)
  let effect = make_glyph(255, 255, 255, 255)
  let link =
    narrative.NarrativeLink(
      cause: cause,
      effect: effect,
      relation: Contrasted,
      strength: 0.8,
      occurrences: 2,
      first_seen: 1,
      last_seen: 3,
    )

  let narration = narrative.narrate_link(link, Poetic)

  string.contains(narration, "Light and shadow") |> should.be_true()
}

pub fn narrate_generates_text_test() {
  let mem = narrative.new()
  let g1 = make_glyph(10, 10, 10, 10)
  let g2 = make_glyph(20, 20, 20, 20)

  let mem = narrative.record_caused(mem, g1, g2, 1)

  let narration = narrative.narrate(mem, g1, 2, Factual)

  { string.length(narration) > 0 } |> should.be_true()
}

pub fn empty_stream_test() {
  let stream = narrative.empty_stream()

  narrative.stream_is_empty(stream) |> should.be_true()
  narrative.thought_count(stream) |> should.equal(0)
  stream.focus |> should.equal(None)
}

pub fn dominant_thought_returns_strongest_test() {
  let mem = narrative.new()
  let g1 = make_glyph(10, 10, 10, 10)
  let g2 = make_glyph(20, 20, 20, 20)
  let g3 = make_glyph(30, 30, 30, 30)

  // Build linked memory
  let mem = narrative.record_caused(mem, g1, g2, 1)
  let mem = narrative.record_caused(mem, g2, g3, 2)
  // Reinforce second link
  let mem = narrative.record_caused(mem, g2, g3, 3)
  let mem = narrative.record_caused(mem, g2, g3, 4)

  let stream = narrative.inner_voice(mem, g1, 3, Emotional)

  case narrative.dominant_thought(stream) {
    Some(thought) -> {
      { thought.weight >. 0.0 } |> should.be_true()
    }
    None -> should.fail()
  }
}

pub fn merge_streams_combines_thoughts_test() {
  let mem = narrative.new()
  let g1 = make_glyph(10, 10, 10, 10)
  let g2 = make_glyph(100, 100, 100, 100)

  let stream1 = narrative.inner_voice(mem, g1, 1, Factual)
  let stream2 = narrative.inner_voice(mem, g2, 1, Factual)

  let merged = narrative.merge_streams(stream1, stream2)

  let expected_count =
    narrative.thought_count(stream1) + narrative.thought_count(stream2)
  narrative.thought_count(merged) |> should.equal(expected_count)
}

pub fn voice_styles_generate_different_content_test() {
  let mem = narrative.new()
  let g = make_glyph(128, 128, 128, 128)

  let factual = narrative.reflect_on(mem, g, Factual)
  let emotional = narrative.reflect_on(mem, g, Emotional)
  let poetic = narrative.reflect_on(mem, g, Poetic)

  // Different styles should produce different content
  { factual.content != emotional.content } |> should.be_true()
  { emotional.content != poetic.content } |> should.be_true()
}

pub fn thought_source_from_association_test() {
  let mem = narrative.new()
  let a = make_glyph(50, 50, 50, 50)
  let b = make_glyph(60, 60, 60, 60)

  let mem = narrative.record_associated(mem, a, b, 1)

  let thought = narrative.reflect_on(mem, a, Factual)

  thought.source |> should.equal(FromAssociation)
}
