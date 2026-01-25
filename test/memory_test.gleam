//// Memory Tests
////
//// Testa KarmaBank, DRE scoring, e recall.

import gleam/list
import gleam/option.{None}
import gleeunit/should
import viva/memory
import viva_glyph/glyph

// =============================================================================
// KARMA BANK BASICS
// =============================================================================

pub fn new_karma_bank_starts_empty_test() {
  let bank = memory.new(1)

  bank.life_number |> should.equal(1)
  bank.total_karma |> should.equal(0.0)
  list.length(bank.memories) |> should.equal(0)
}

pub fn store_increases_karma_test() {
  let bank = memory.new(1)
  let g = glyph.new([100, 150, 200, 50])
  let ctx = glyph.neutral()

  let bank2 = memory.store(bank, g, ctx, None, 0.8)

  // Karma deve aumentar (intensidade 0.8 contribui)
  { bank2.total_karma >. 0.0 } |> should.be_true()
  list.length(bank2.memories) |> should.equal(1)
}

pub fn multiple_stores_accumulate_test() {
  let bank = memory.new(1)
  let ctx = glyph.neutral()

  // Armazena 3 memórias
  let bank = memory.store(bank, glyph.new([100, 100, 100, 100]), ctx, None, 0.5)
  let bank = memory.store(bank, glyph.new([150, 150, 150, 150]), ctx, None, 0.7)
  let bank = memory.store(bank, glyph.new([200, 200, 200, 200]), ctx, None, 0.9)

  list.length(bank.memories) |> should.equal(3)
  { bank.total_karma >. 1.0 } |> should.be_true()
}

// =============================================================================
// RECALL
// =============================================================================

pub fn recall_returns_similar_memories_test() {
  let bank = memory.new(1)
  let ctx = glyph.neutral()

  // Armazena memórias distintas
  let joy_glyph = glyph.new([200, 150, 180, 100])
  let sad_glyph = glyph.new([50, 80, 60, 100])
  let neutral_glyph = glyph.new([128, 128, 128, 50])

  let bank = memory.store(bank, joy_glyph, ctx, None, 0.8)
  let bank = memory.store(bank, sad_glyph, ctx, None, 0.8)
  let bank = memory.store(bank, neutral_glyph, ctx, None, 0.5)

  // Busca por algo similar a joy
  let query =
    memory.GlyphQuery(glyph: glyph.new([190, 140, 170, 90]), context_glyph: ctx)
  let results = memory.recall(bank, query, 2)

  // Deve retornar resultados (pelo menos 1)
  { list.length(results) >= 1 } |> should.be_true()
}

pub fn recall_empty_bank_returns_empty_test() {
  let bank = memory.new(1)
  let query =
    memory.GlyphQuery(
      glyph: glyph.new([100, 100, 100, 100]),
      context_glyph: glyph.neutral(),
    )

  let results = memory.recall(bank, query, 5)

  list.length(results) |> should.equal(0)
}

// =============================================================================
// CLEAR LIGHT PROXIMITY
// =============================================================================

pub fn clear_light_glyph_is_neutral_test() {
  // [0,0,0,0] = luz clara
  let clear_light = glyph.new([0, 0, 0, 0])
  let bank = memory.new(1)

  let bank = memory.update_clear_light(bank, clear_light)

  // Proximidade deve ser máxima (1.0) ou próxima
  { bank.clear_light_proximity >. 0.9 } |> should.be_true()
}

pub fn far_from_clear_light_low_proximity_test() {
  // [255,255,255,255] = máximo caos, longe da luz clara
  let chaos = glyph.new([255, 255, 255, 255])
  let bank = memory.new(1)

  let bank = memory.update_clear_light(bank, chaos)

  // Proximidade deve ser baixa
  { bank.clear_light_proximity <. 0.3 } |> should.be_true()
}

// =============================================================================
// FROM TRANSCENDENT
// =============================================================================

pub fn from_transcendent_starts_new_life_test() {
  // Memórias herdadas (vazias para simplificar)
  let inherited = []

  let bank = memory.from_transcendent(2, inherited, 0.7)

  bank.life_number |> should.equal(2)
}

// =============================================================================
// TICK DECAY
// =============================================================================

pub fn tick_updates_bank_test() {
  let bank = memory.new(1)
  let g = glyph.new([100, 100, 100, 100])
  let ctx = glyph.neutral()

  let bank = memory.store(bank, g, ctx, None, 0.8)

  // Simula 10 ticks
  let bank = tick_n_times(bank, 10)

  // Bank tick deve ter incrementado
  { bank.tick >= 10 } |> should.be_true()
}

fn tick_n_times(bank: memory.KarmaBank, n: Int) -> memory.KarmaBank {
  case n {
    0 -> bank
    _ -> tick_n_times(memory.tick(bank), n - 1)
  }
}
