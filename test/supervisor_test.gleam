//// Supervisor Tests
////
//// Testa lifecycle completo: spawn, tick, death, rebirth.

import gleam/erlang/process
import gleam/list
import gleeunit/should
import viva/supervisor

// =============================================================================
// SUPERVISOR BASICS
// =============================================================================

pub fn supervisor_starts_test() {
  let result = supervisor.start_without_telemetry()

  case result {
    Ok(_subject) -> should.be_true(True)
    Error(_) -> should.fail()
  }
}

pub fn supervisor_starts_empty_test() {
  let assert Ok(sup) = supervisor.start_without_telemetry()

  let alive = supervisor.list_alive(sup)

  list.length(alive) |> should.equal(0)
}

// =============================================================================
// SPAWN VIVA
// =============================================================================

pub fn spawn_viva_returns_id_test() {
  let assert Ok(sup) = supervisor.start_without_telemetry()

  let id = supervisor.spawn_viva(sup)

  { id >= 1 } |> should.be_true()
}

pub fn spawn_viva_adds_to_alive_test() {
  let assert Ok(sup) = supervisor.start_without_telemetry()

  let _id = supervisor.spawn_viva(sup)
  process.sleep(10)

  let alive = supervisor.list_alive(sup)

  list.length(alive) |> should.equal(1)
}

pub fn spawn_multiple_vivas_test() {
  let assert Ok(sup) = supervisor.start_without_telemetry()

  let _id1 = supervisor.spawn_viva(sup)
  let _id2 = supervisor.spawn_viva(sup)
  let _id3 = supervisor.spawn_viva(sup)
  process.sleep(10)

  let alive = supervisor.list_alive(sup)

  list.length(alive) |> should.equal(3)
}

pub fn spawned_vivas_have_unique_ids_test() {
  let assert Ok(sup) = supervisor.start_without_telemetry()

  let id1 = supervisor.spawn_viva(sup)
  let id2 = supervisor.spawn_viva(sup)

  { id1 != id2 } |> should.be_true()
}

// =============================================================================
// GLOBAL TICK
// =============================================================================

pub fn global_tick_increments_counter_test() {
  let assert Ok(sup) = supervisor.start_without_telemetry()
  let _id = supervisor.spawn_viva(sup)

  supervisor.global_tick(sup, 0.1)
  supervisor.global_tick(sup, 0.1)
  process.sleep(10)

  let state = supervisor.get_state(sup)

  { state.tick >= 2 } |> should.be_true()
}

pub fn global_tick_evolves_all_vivas_test() {
  let assert Ok(sup) = supervisor.start_without_telemetry()
  let _id1 = supervisor.spawn_viva(sup)
  let _id2 = supervisor.spawn_viva(sup)

  // 10 ticks
  tick_n_times(sup, 10)

  let state = supervisor.get_state(sup)

  state.tick |> should.equal(10)
}

fn tick_n_times(sup: process.Subject(supervisor.Message), n: Int) -> Nil {
  case n {
    0 -> Nil
    _ -> {
      supervisor.global_tick(sup, 0.1)
      tick_n_times(sup, n - 1)
    }
  }
}

// =============================================================================
// INTEROCEPTION
// =============================================================================

pub fn apply_interoception_no_crash_test() {
  let assert Ok(sup) = supervisor.start_without_telemetry()
  let _id = supervisor.spawn_viva(sup)

  // Não deve crashar
  supervisor.apply_interoception(sup)
  process.sleep(10)

  let alive = supervisor.list_alive(sup)
  list.length(alive) |> should.equal(1)
}

// =============================================================================
// KILL AND REBIRTH
// =============================================================================

pub fn kill_viva_processes_death_test() {
  let assert Ok(sup) = supervisor.start_without_telemetry()
  let id = supervisor.spawn_viva(sup)
  process.sleep(10)

  supervisor.kill_viva(sup, id)
  process.sleep(50)

  let state = supervisor.get_state(sup)

  // Deve ter eventos de morte
  { list.length(state.events) >= 2 } |> should.be_true()
}

pub fn kill_viva_rebirths_if_not_liberated_test() {
  let assert Ok(sup) = supervisor.start_without_telemetry()
  let id = supervisor.spawn_viva(sup)
  process.sleep(10)

  // Adiciona karma para garantir que não libera
  tick_n_times(sup, 5)

  supervisor.kill_viva(sup, id)
  process.sleep(50)

  let alive = supervisor.list_alive(sup)

  // VIVA deve ter renascido (ainda viva)
  list.length(alive) |> should.equal(1)
}

// =============================================================================
// STATS
// =============================================================================

pub fn get_stats_returns_info_test() {
  let assert Ok(sup) = supervisor.start_without_telemetry()
  let _id = supervisor.spawn_viva(sup)

  let stats = supervisor.get_stats(sup)

  // Stats deve conter info sobre VIVAs
  { stats != "" } |> should.be_true()
}

// =============================================================================
// EVENTS
// =============================================================================

pub fn spawn_creates_born_event_test() {
  let assert Ok(sup) = supervisor.start_without_telemetry()
  let _id = supervisor.spawn_viva(sup)
  process.sleep(10)

  let state = supervisor.get_state(sup)

  // Deve ter evento Born
  list.length(state.events) |> should.equal(1)
}

pub fn death_creates_multiple_events_test() {
  let assert Ok(sup) = supervisor.start_without_telemetry()
  let id = supervisor.spawn_viva(sup)
  process.sleep(10)

  supervisor.kill_viva(sup, id)
  process.sleep(50)

  let state = supervisor.get_state(sup)

  // Born + Died + BardoComplete + (Reborn se não liberou)
  { list.length(state.events) >= 3 } |> should.be_true()
}
