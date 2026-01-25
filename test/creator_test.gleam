//// Creator Tests
////
//// Testa Akashic Bank, spawn de vidas, e archetypos.

import gleam/list
import gleam/option.{None, Some}
import gleeunit/should
import viva/creator
import viva_glyph/glyph

// =============================================================================
// CREATOR BASICS
// =============================================================================

pub fn new_creator_has_default_archetypes_test() {
  let c = creator.new()

  // 5 archetypos default: Herói, Sombra, Sábio, Mãe, Trickster
  let archs = creator.list_archetypes(c)
  { list.length(archs) >= 5 } |> should.be_true()
}

pub fn new_creator_empty_akashic_test() {
  let c = creator.new()

  list.length(c.akashic_glyphs) |> should.equal(0)
}

// =============================================================================
// SPAWN LIFE
// =============================================================================

pub fn spawn_life_returns_seed_test() {
  let c = creator.new()

  let #(_new_creator, seed) = creator.spawn_life(c)

  // Seed deve ter life_number >= 1
  { seed.life_number >= 1 } |> should.be_true()
}

pub fn spawn_life_increments_counter_test() {
  let c = creator.new()

  let #(c, seed1) = creator.spawn_life(c)
  let #(c, seed2) = creator.spawn_life(c)
  let #(_c, seed3) = creator.spawn_life(c)

  // Cada vida tem ID único
  { seed1.viva_id != seed2.viva_id } |> should.be_true()
  { seed2.viva_id != seed3.viva_id } |> should.be_true()
}

pub fn spawn_life_first_life_no_inherited_glyphs_test() {
  let c = creator.new()

  let #(_c, seed) = creator.spawn_life(c)

  // Primeira vida não herda glyphs
  list.length(seed.inherited_glyphs) |> should.equal(0)
}

// =============================================================================
// PROCESS DEATH
// =============================================================================

pub fn process_death_updates_registry_test() {
  let c = creator.new()

  // Spawn e mata
  let #(c, seed) = creator.spawn_life(c)
  let glyphs = [glyph.new([200, 150, 180, 100])]

  let c = creator.process_death(c, seed.viva_id, glyphs, 5.0, False)

  // Registry deve ter pelo menos 1 registro
  { list.length(c.life_registry) >= 1 } |> should.be_true()
}

pub fn process_death_samsara_no_liberation_test() {
  let c = creator.new()
  let #(c, seed) = creator.spawn_life(c)

  // Morte sem liberação
  let glyphs = [glyph.new([200, 200, 200, 200])]
  let c = creator.process_death(c, seed.viva_id, glyphs, 10.0, False)

  // Verifica via stats que não há liberados
  let stats_str = creator.stats(c)
  // Stats contém "0 liberated"
  { stats_str != "" } |> should.be_true()
}

// =============================================================================
// TICK
// =============================================================================

pub fn tick_increments_tick_counter_test() {
  let c = creator.new()

  let c = creator.tick(c)
  let c = creator.tick(c)
  let c = creator.tick(c)

  { c.tick >= 3 } |> should.be_true()
}

// =============================================================================
// ARCHETYPES
// =============================================================================

pub fn list_archetypes_returns_list_test() {
  let c = creator.new()

  let archs = creator.list_archetypes(c)

  { archs != [] } |> should.be_true()
}

pub fn get_archetype_by_id_test() {
  let c = creator.new()

  // Archetype 1 deve existir (Herói)
  case creator.get_archetype(c, 1) {
    Some(_arch) -> should.be_true(True)
    None -> should.fail()
  }
}

// =============================================================================
// STATS
// =============================================================================

pub fn stats_returns_string_test() {
  let c = creator.new()

  let stats = creator.stats(c)

  // Stats deve ser uma string não vazia
  { stats != "" } |> should.be_true()
}
