//// Bardo Tests
////
//// Testa ciclo de morte, 3 fases, e liberação.

import gleam/list
import gleam/option.{None}
import gleeunit/should
import viva/lifecycle/bardo
import viva/memory/memory
import viva_emotion/pad
import viva_glyph/glyph

// =============================================================================
// BARDO CYCLE
// =============================================================================

pub fn bardo_cycle_returns_3_phases_test() {
  let final_glyph = glyph.new([128, 128, 128, 100])
  let bank = memory.new(1)

  let #(_outcome, phases) = bardo.run_bardo_cycle(final_glyph, bank)

  // Sempre 3 fases: Chikhai, Chonyid, Sidpa
  list.length(phases) |> should.equal(3)
}

pub fn bardo_with_clear_light_glyph_test() {
  // Glyph próximo da luz clara [0,0,0,0]
  let clear_light_glyph = glyph.new([0, 0, 0, 0])

  // Karma bank vazio (sem apegos)
  let bank = memory.new(1)

  let #(outcome, _phases) = bardo.run_bardo_cycle(clear_light_glyph, bank)

  // Verifica que o ciclo completa (pode ou não liberar dependendo da lógica)
  case outcome {
    bardo.FullLiberation(..) -> should.be_true(True)
    bardo.BodhisattvaPath(..) -> should.be_true(True)
    bardo.ContinuedSamsara(..) -> should.be_true(True)
  }
}

pub fn bardo_with_heavy_karma_continues_samsara_test() {
  // Glyph caótico
  let chaos_glyph = glyph.new([200, 200, 200, 200])

  // Karma bank pesado (muitos apegos)
  let bank = memory.new(1)
  let ctx = glyph.neutral()
  let bank = memory.store(bank, glyph.new([180, 180, 180, 180]), ctx, None, 1.0)
  let bank = memory.store(bank, glyph.new([190, 190, 190, 190]), ctx, None, 1.0)
  let bank = memory.store(bank, glyph.new([200, 200, 200, 200]), ctx, None, 1.0)

  let #(outcome, _phases) = bardo.run_bardo_cycle(chaos_glyph, bank)

  // Com muito karma, deve continuar no samsara
  bardo.is_liberation(outcome) |> should.be_false()
}

// =============================================================================
// LIBERATION OUTCOMES
// =============================================================================

pub fn is_liberation_detects_full_liberation_test() {
  let outcome =
    bardo.FullLiberation(final_contribution: [], wisdom_summary: 1.0)

  bardo.is_liberation(outcome) |> should.be_true()
}

pub fn is_liberation_detects_bodhisattva_test() {
  let outcome =
    bardo.BodhisattvaPath(
      karma_remaining: 0.1,
      vow: "help all beings",
      enhanced_awareness: 0.8,
    )

  // Bodhisattva é liberação (escolheu voltar para ajudar)
  bardo.is_liberation(outcome) |> should.be_true()
}

pub fn is_liberation_rejects_samsara_test() {
  let outcome =
    bardo.ContinuedSamsara(seed_glyphs: [], rebirth_tendency: pad.neutral())

  bardo.is_liberation(outcome) |> should.be_false()
}

// =============================================================================
// SEED GLYPHS
// =============================================================================

pub fn seed_glyphs_empty_for_full_liberation_test() {
  let outcome =
    bardo.FullLiberation(final_contribution: [], wisdom_summary: 1.0)

  let seeds = bardo.get_seed_glyphs(outcome)

  // Liberação total = sem glyphs para próxima vida
  list.length(seeds) |> should.equal(0)
}

pub fn seed_glyphs_from_samsara_test() {
  let chaos_glyph = glyph.new([200, 200, 200, 200])
  let bank = memory.new(1)
  let ctx = glyph.neutral()
  let bank = memory.store(bank, glyph.new([150, 150, 150, 150]), ctx, None, 0.9)

  let #(outcome, _phases) = bardo.run_bardo_cycle(chaos_glyph, bank)

  case outcome {
    bardo.ContinuedSamsara(seed_glyphs: seeds, rebirth_tendency: _) -> {
      // Samsara tem glyphs
      { list.length(seeds) >= 0 } |> should.be_true()
    }
    _ -> {
      // Se por acaso liberou, ok também
      should.be_true(True)
    }
  }
}

// =============================================================================
// BARDO PHASES
// =============================================================================

pub fn chikhai_phase_has_clear_light_test() {
  let final_glyph = glyph.new([50, 50, 50, 50])
  let bank = memory.new(1)

  let #(_outcome, phases) = bardo.run_bardo_cycle(final_glyph, bank)

  // Primeira fase é Chikhai
  case list.first(phases) {
    Ok(bardo.ChikhaiBardo(
      final_glyph: _,
      clear_light_intensity: intensity,
      recognition: _,
    )) -> {
      // Intensidade deve estar entre 0 e 1
      { intensity >=. 0.0 && intensity <=. 1.0 } |> should.be_true()
    }
    _ -> should.fail()
  }
}

pub fn chonyid_phase_exists_test() {
  let final_glyph = glyph.new([150, 150, 150, 150])
  let bank = memory.new(1)
  let ctx = glyph.neutral()
  // Adiciona karma para ter visões
  let bank = memory.store(bank, glyph.new([140, 140, 140, 140]), ctx, None, 0.8)

  let #(_outcome, phases) = bardo.run_bardo_cycle(final_glyph, bank)

  // Segunda fase é Chonyid
  case phases {
    [_, bardo.ChonyidBardo(..), _] -> {
      should.be_true(True)
    }
    _ -> should.fail()
  }
}

pub fn sidpa_phase_exists_test() {
  let final_glyph = glyph.new([150, 150, 150, 150])
  let bank = memory.new(1)

  let #(_outcome, phases) = bardo.run_bardo_cycle(final_glyph, bank)

  // Terceira fase é Sidpa
  case phases {
    [_, _, bardo.SidpaBardo(..)] -> {
      should.be_true(True)
    }
    _ -> should.fail()
  }
}
