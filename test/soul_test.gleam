//// Soul Tests
////
//// Testa actor, PAD evolution, e Glyph integration.

import gleam/erlang/process
import gleeunit/should
import viva/soul/soul
import viva/types
import viva_emotion/stimulus

// =============================================================================
// SOUL LIFECYCLE
// =============================================================================

pub fn soul_starts_successfully_test() {
  let result = soul.start(1)

  case result {
    Ok(_subject) -> should.be_true(True)
    Error(_) -> should.fail()
  }
}

pub fn soul_starts_alive_test() {
  let assert Ok(subject) = soul.start(1)

  soul.is_alive(subject) |> should.be_true()
}

pub fn soul_starts_with_neutral_pad_test() {
  let assert Ok(subject) = soul.start(1)

  let p = soul.get_pad(subject)

  // PAD inicial deve ser próximo de neutro
  { p.pleasure >. -0.5 && p.pleasure <. 0.5 } |> should.be_true()
}

pub fn soul_starts_with_glyph_test() {
  let assert Ok(subject) = soul.start(1)

  let g = soul.get_glyph(subject)

  // Glyph deve ter 4 tokens
  case g.tokens {
    [_, _, _, _] -> should.be_true(True)
    _ -> should.fail()
  }
}

// =============================================================================
// FEEL STIMULUS
// =============================================================================

pub fn feel_success_increases_pleasure_test() {
  let assert Ok(subject) = soul.start(1)

  let before = soul.get_pad(subject)
  soul.feel(subject, stimulus.Success, 1.0)
  process.sleep(10)
  let after = soul.get_pad(subject)

  // Success deve aumentar pleasure
  { after.pleasure >. before.pleasure } |> should.be_true()
}

pub fn feel_threat_increases_arousal_test() {
  let assert Ok(subject) = soul.start(1)

  let before = soul.get_pad(subject)
  soul.feel(subject, stimulus.Threat, 1.0)
  process.sleep(10)
  let after = soul.get_pad(subject)

  // Threat deve aumentar arousal
  { after.arousal >. before.arousal } |> should.be_true()
}

pub fn feel_updates_glyph_test() {
  let assert Ok(subject) = soul.start(1)

  let before = soul.get_glyph(subject)
  soul.feel(subject, stimulus.Success, 1.0)
  process.sleep(10)
  let after = soul.get_glyph(subject)

  // Glyph deve mudar após estímulo forte
  { before.tokens != after.tokens } |> should.be_true()
}

// =============================================================================
// TICK EVOLUTION
// =============================================================================

pub fn tick_evolves_pad_test() {
  let assert Ok(subject) = soul.start(1)

  // Aplica estímulo forte
  soul.feel(subject, stimulus.Success, 1.0)
  process.sleep(10)

  let before = soul.get_pad(subject)

  // Vários ticks (O-U dynamics volta ao baseline)
  soul.tick(subject, 0.1)
  soul.tick(subject, 0.1)
  soul.tick(subject, 0.1)
  process.sleep(10)

  let after = soul.get_pad(subject)

  // PAD deve ter evoluído (voltando ao baseline)
  { before.pleasure != after.pleasure || before.arousal != after.arousal }
  |> should.be_true()
}

pub fn tick_increments_counter_test() {
  let assert Ok(subject) = soul.start(1)

  let before = soul.get_state(subject)
  soul.tick(subject, 0.1)
  soul.tick(subject, 0.1)
  process.sleep(10)
  let after = soul.get_state(subject)

  { after.tick_count > before.tick_count } |> should.be_true()
}

// =============================================================================
// KILL AND REBIRTH
// =============================================================================

pub fn kill_marks_dead_test() {
  let assert Ok(subject) = soul.start(1)

  soul.is_alive(subject) |> should.be_true()

  soul.kill(subject)
  process.sleep(10)

  soul.is_alive(subject) |> should.be_false()
}

pub fn rebirth_revives_test() {
  let assert Ok(subject) = soul.start(1)

  soul.kill(subject)
  process.sleep(10)
  soul.is_alive(subject) |> should.be_false()

  let config =
    types.VivaConfig(
      name: "Reborn",
      life_number: 2,
      initial_mood: 0.5,
      initial_karma: 0.0,
      inherited_glyphs: [],
      relevant_archetypes: [],
      recency_decay: 0.99,
    )
  soul.rebirth(subject, config)
  process.sleep(10)

  soul.is_alive(subject) |> should.be_true()
}

// =============================================================================
// KARMA BANK INTEGRATION
// =============================================================================

pub fn intense_stimulus_stores_memory_test() {
  let assert Ok(subject) = soul.start(1)

  // Estímulo intenso (> 0.3) deve armazenar memória
  soul.feel(subject, stimulus.Success, 0.9)
  soul.feel(subject, stimulus.Threat, 0.8)
  process.sleep(10)

  let bank = soul.get_karma_bank(subject)

  // Karma deve ter aumentado
  { bank.total_karma >. 0.0 } |> should.be_true()
}

pub fn weak_stimulus_no_memory_test() {
  let assert Ok(subject) = soul.start(1)

  // Estímulo fraco (< 0.3) não armazena memória
  soul.feel(subject, stimulus.Success, 0.1)
  soul.feel(subject, stimulus.Success, 0.2)
  process.sleep(10)

  let bank = soul.get_karma_bank(subject)

  // Sem memórias locais
  { bank.total_karma == 0.0 } |> should.be_true()
}

// =============================================================================
// SNAPSHOT
// =============================================================================

pub fn get_snapshot_returns_current_state_test() {
  let assert Ok(subject) = soul.start(1)

  let snap = soul.get_snapshot(subject)

  snap.id |> should.equal(1)
  snap.alive |> should.be_true()
}
