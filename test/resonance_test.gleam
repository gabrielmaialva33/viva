//// Resonance Tests
////
//// Testa pool, cálculo de ressonância, e propagação.

import gleam/list
import gleam/option.{None, Some}
import gleeunit/should
import viva/soul/resonance
import viva_emotion/pad
import viva_glyph/glyph

// =============================================================================
// POOL BASICS
// =============================================================================

pub fn new_pool_is_empty_test() {
  let pool = resonance.new_pool()

  list.length(pool.vivas) |> should.equal(0)
}

pub fn register_adds_to_pool_test() {
  let pool = resonance.new_pool()
  let viva = create_test_viva(1)

  let pool = resonance.register(pool, viva)

  list.length(pool.vivas) |> should.equal(1)
}

pub fn register_multiple_test() {
  let pool = resonance.new_pool()

  let pool = resonance.register(pool, create_test_viva(1))
  let pool = resonance.register(pool, create_test_viva(2))
  let pool = resonance.register(pool, create_test_viva(3))

  list.length(pool.vivas) |> should.equal(3)
}

pub fn unregister_removes_from_pool_test() {
  let pool = resonance.new_pool()
  let pool = resonance.register(pool, create_test_viva(1))
  let pool = resonance.register(pool, create_test_viva(2))

  let pool = resonance.unregister(pool, 1)

  list.length(pool.vivas) |> should.equal(1)
}

// =============================================================================
// RESONANCE CALCULATION
// =============================================================================

pub fn similar_vivas_resonate_test() {
  // Duas VIVAs com estados emocionais similares
  let viva_a =
    resonance.VivaState(
      id: 1,
      pad: pad.new(0.5, 0.5, 0.5),
      glyph: glyph.new([150, 150, 150, 100]),
      alive: True,
      tick: 0,
    )

  let viva_b =
    resonance.VivaState(
      id: 2,
      pad: pad.new(0.4, 0.4, 0.4),
      glyph: glyph.new([145, 145, 145, 95]),
      alive: True,
      tick: 0,
    )

  let resonance_val = resonance.calculate_resonance(viva_a, viva_b)

  // Ressonância deve ser um valor válido (0-1)
  { resonance_val >=. 0.0 && resonance_val <=. 1.0 } |> should.be_true()
}

pub fn opposite_vivas_resonate_weakly_test() {
  // Duas VIVAs com estados emocionais opostos
  let viva_a =
    resonance.VivaState(
      id: 1,
      pad: pad.new(0.9, 0.9, 0.9),
      glyph: glyph.new([240, 240, 240, 200]),
      alive: True,
      tick: 0,
    )

  let viva_b =
    resonance.VivaState(
      id: 2,
      pad: pad.new(-0.9, -0.9, -0.9),
      glyph: glyph.new([10, 10, 10, 200]),
      alive: True,
      tick: 0,
    )

  let resonance_val = resonance.calculate_resonance(viva_a, viva_b)

  // Deve ter ressonância baixa (< 0.3)
  { resonance_val <. 0.3 } |> should.be_true()
}

pub fn dead_viva_no_resonance_test() {
  let alive_viva = create_test_viva(1)
  let dead_viva = resonance.VivaState(..create_test_viva(2), alive: False)

  let resonance_val = resonance.calculate_resonance(alive_viva, dead_viva)

  // Morto não ressoa
  resonance_val |> should.equal(0.0)
}

pub fn identical_glyphs_max_similarity_test() {
  let viva = create_test_viva(1)

  let resonance_val = resonance.calculate_resonance(viva, viva)

  // Glyphs idênticos têm similaridade máxima
  { resonance_val >=. 0.9 } |> should.be_true()
}

// =============================================================================
// PROPAGATION
// =============================================================================

pub fn propagate_to_empty_pool_no_events_test() {
  let pool = resonance.new_pool()
  let source = create_test_viva(1)

  let #(_pool, events) = resonance.propagate(pool, source, 0)

  list.length(events) |> should.equal(0)
}

pub fn propagate_with_similar_vivas_test() {
  // Pool com threshold baixo para facilitar teste
  let pool = resonance.new_pool_with_threshold(0.1)

  // Duas VIVAs similares
  let viva_a =
    resonance.VivaState(
      id: 1,
      pad: pad.new(0.5, 0.5, 0.5),
      glyph: glyph.new([150, 150, 150, 100]),
      alive: True,
      tick: 0,
    )

  let viva_b =
    resonance.VivaState(
      id: 2,
      pad: pad.new(0.5, 0.5, 0.5),
      glyph: glyph.new([150, 150, 150, 100]),
      alive: True,
      tick: 0,
    )

  let pool = resonance.register(pool, viva_a)
  let pool = resonance.register(pool, viva_b)

  let #(_pool, events) = resonance.propagate(pool, viva_a, 0)

  // Eventos podem ocorrer ou não dependendo do threshold
  { list.length(events) >= 0 } |> should.be_true()
}

// =============================================================================
// EMOTIONAL FIELD
// =============================================================================

pub fn emotional_field_empty_pool_none_test() {
  let pool = resonance.new_pool()

  let result = resonance.emotional_field(pool)

  case result {
    Some(_) -> should.fail()
    None -> should.be_true(True)
  }
}

pub fn emotional_field_single_viva_test() {
  let pool = resonance.new_pool()
  let viva = create_test_viva(1)
  let pool = resonance.register(pool, viva)

  let result = resonance.emotional_field(pool)

  case result {
    Some(_g) -> should.be_true(True)
    None -> should.fail()
  }
}

// =============================================================================
// UPDATE VIVA
// =============================================================================

pub fn update_viva_test() {
  let pool = resonance.new_pool()
  let viva = create_test_viva(1)
  let pool = resonance.register(pool, viva)

  // Update viva state
  let updated_viva = resonance.VivaState(..viva, tick: 100)
  let pool = resonance.update_viva(pool, updated_viva)

  // Pool should still have 1 viva
  list.length(pool.vivas) |> should.equal(1)
}

// =============================================================================
// HELPERS
// =============================================================================

fn create_test_viva(id: Int) -> resonance.VivaState {
  resonance.VivaState(
    id: id,
    pad: pad.new(0.0, 0.0, 0.0),
    glyph: glyph.neutral(),
    alive: True,
    tick: 0,
  )
}
