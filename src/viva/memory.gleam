//// Memory - The Karma Bank
////
//// 3-level memory system:
//// 1. Local - dies with VIVA
//// 2. Transcendent - survives Big Bounce
//// 3. Collective - eternal in Creator
////
//// DRE = Karma: intense emotional actions are "karmic weight"

import gleam/float
import gleam/int
import gleam/list
import gleam/option.{type Option}
import viva_glyph
import viva_glyph/glyph.{type Glyph}

// =============================================================================
// TYPES
// =============================================================================

/// Glyph-based memory
pub type GlyphMemory {
  GlyphMemory(
    /// Unique ID
    id: Int,
    /// Emotional state (4 tokens)
    glyph: Glyph,
    /// Context also as glyph
    context_glyph: Glyph,
    /// Emotional trajectory (derivative)
    trajectory: Option(Glyph),
    /// Karmic weight (DRE score)
    karma_weight: Float,
    /// Which life created it
    life_number: Int,
    /// Creation tick
    created_at: Int,
    /// Access counter (spaced repetition)
    access_count: Int,
  )
}

/// Memory levels
pub type MemoryLevel {
  /// Local memory - dies with VIVA
  Local(memory: GlyphMemory)
  /// Transcendent memory - survives Big Bounce
  Transcendent(memory: GlyphMemory, dre_score: Float)
  /// Collective memory - shared with Creator
  Collective(glyph: Glyph, source_count: Int, archetype_id: Option(Int))
}

/// Karma Bank (memories of a life)
pub type KarmaBank {
  KarmaBank(
    /// All memories
    memories: List(GlyphMemory),
    /// Total accumulated karma
    total_karma: Float,
    /// Clear light proximity (0-1)
    clear_light_proximity: Float,
    /// Next memory ID
    next_id: Int,
    /// Current life
    life_number: Int,
    /// Current tick
    tick: Int,
  )
}

/// Query for memory search
pub type GlyphQuery {
  GlyphQuery(
    /// Emotional glyph to search
    glyph: Glyph,
    /// Context to search
    context_glyph: Glyph,
  )
}

// =============================================================================
// KARMA BANK OPERATIONS
// =============================================================================

/// Create new KarmaBank
pub fn new(life_number: Int) -> KarmaBank {
  KarmaBank(
    memories: [],
    total_karma: 0.0,
    clear_light_proximity: 0.0,
    next_id: 1,
    life_number: life_number,
    tick: 0,
  )
}

/// Create KarmaBank with inherited memories (Big Bounce)
pub fn from_transcendent(
  life_number: Int,
  inherited: List(GlyphMemory),
  mood_carryover: Float,
) -> KarmaBank {
  // Inherit memories but reset access_count
  let memories =
    inherited
    |> list.map(fn(m) { GlyphMemory(..m, access_count: 0) })

  let initial_karma =
    memories
    |> list.fold(0.0, fn(acc, m) { acc +. m.karma_weight })
    |> fn(k) { k *. mood_carryover }

  KarmaBank(
    memories: memories,
    total_karma: initial_karma,
    clear_light_proximity: mood_carryover *. 0.1,
    next_id: list.length(memories) + 1,
    life_number: life_number,
    tick: 0,
  )
}

/// Store new memory
pub fn store(
  bank: KarmaBank,
  g: Glyph,
  context_g: Glyph,
  trajectory: Option(Glyph),
  intensity: Float,
) -> KarmaBank {
  // Calculate karmic weight based on intensity
  let karma_weight = calculate_karma(intensity)

  let memory =
    GlyphMemory(
      id: bank.next_id,
      glyph: g,
      context_glyph: context_g,
      trajectory: trajectory,
      karma_weight: karma_weight,
      life_number: bank.life_number,
      created_at: bank.tick,
      access_count: 0,
    )

  KarmaBank(
    ..bank,
    memories: [memory, ..bank.memories],
    total_karma: bank.total_karma +. karma_weight,
    next_id: bank.next_id + 1,
  )
}

/// Calculate karmic weight of an action
fn calculate_karma(intensity: Float) -> Float {
  // Karma = intensity^2 (more intense actions have more weight)
  float.min(intensity *. intensity, 1.0)
}

/// Search similar memories
pub fn recall(bank: KarmaBank, query: GlyphQuery, limit: Int) -> List(GlyphMemory) {
  bank.memories
  |> list.map(fn(m) {
    let emotion_sim = viva_glyph.similarity(m.glyph, query.glyph)
    let context_sim = viva_glyph.similarity(m.context_glyph, query.context_glyph)
    let combined = 0.6 *. emotion_sim +. 0.4 *. context_sim
    #(m, combined)
  })
  |> list.sort(fn(a, b) { float.compare(b.1, a.1) })
  |> list.take(limit)
  |> list.map(fn(pair) { pair.0 })
}

/// Search memories and increment access_count
pub fn recall_and_strengthen(
  bank: KarmaBank,
  query: GlyphQuery,
  limit: Int,
) -> #(KarmaBank, List(GlyphMemory)) {
  let recalled = recall(bank, query, limit)
  let recalled_ids = list.map(recalled, fn(m) { m.id })

  // Increment access_count of accessed memories
  let memories =
    bank.memories
    |> list.map(fn(m) {
      case list.contains(recalled_ids, m.id) {
        True -> GlyphMemory(..m, access_count: m.access_count + 1)
        False -> m
      }
    })

  #(KarmaBank(..bank, memories: memories), recalled)
}

/// Select transcendent memories (high DRE)
pub fn select_transcendent(bank: KarmaBank, threshold: Float) -> List(GlyphMemory) {
  bank.memories
  |> list.filter(fn(m) { m.karma_weight >. threshold })
  |> list.sort(fn(a, b) { float.compare(b.karma_weight, a.karma_weight) })
}

/// Update clear light proximity
pub fn update_clear_light(bank: KarmaBank, current_g: Glyph) -> KarmaBank {
  // Clear light = Glyph[0,0,0,0] (pure consciousness)
  let clear_light = glyph.neutral()
  // similarity goes from 0 to 1, where 1 = identical
  let proximity = viva_glyph.similarity(current_g, clear_light)

  KarmaBank(..bank, clear_light_proximity: proximity)
}

/// Advance tick
pub fn tick(bank: KarmaBank) -> KarmaBank {
  KarmaBank(..bank, tick: bank.tick + 1)
}

// =============================================================================
// DRE SCORING (Distinctiveness × Recency × Emotionality)
// =============================================================================

/// Calculate complete DRE score
pub fn dre_score(memory: GlyphMemory, current_tick: Int) -> Float {
  let distinctiveness = calc_distinctiveness(memory)
  let recency = calc_recency(memory, current_tick)
  let emotionality = memory.karma_weight

  distinctiveness *. recency *. emotionality
}

/// Distinctiveness: how unique the glyph is
fn calc_distinctiveness(memory: GlyphMemory) -> Float {
  // Use first token as uniqueness proxy
  let tokens = memory.glyph.tokens
  case tokens {
    [t1, ..] -> {
      // Extreme tokens (near 0 or 255) are more distinct
      let dist_from_center =
        int.absolute_value(t1 - 128)
        |> int.to_float()
      dist_from_center /. 128.0
    }
    _ -> 0.5
  }
}

/// Recency: recent memories have more weight
fn calc_recency(memory: GlyphMemory, current_tick: Int) -> Float {
  let age = int.to_float(current_tick - memory.created_at)
  // Exponential decay - 0.99^age (half-life ~69 ticks)
  case float.power(0.99, age) {
    Ok(value) -> value
    Error(_) -> 1.0
  }
}

// =============================================================================
// STATISTICS
// =============================================================================

/// Bank statistics
pub fn stats(bank: KarmaBank) -> String {
  let n_memories = list.length(bank.memories)
  let n_transcendent =
    bank.memories
    |> list.filter(fn(m) { m.karma_weight >. 0.7 })
    |> list.length()

  "KarmaBank: "
  <> int.to_string(n_memories)
  <> " memories, "
  <> int.to_string(n_transcendent)
  <> " transcendent, karma="
  <> float_to_string(bank.total_karma)
  <> ", light="
  <> float_to_string(bank.clear_light_proximity)
}

fn float_to_string(f: Float) -> String {
  let rounded = float.round(f *. 100.0) |> int.to_float() |> fn(x) { x /. 100.0 }
  erlang_float_to_list(rounded)
}

@external(erlang, "erlang", "float_to_list")
fn erlang_float_to_list(f: Float) -> String
