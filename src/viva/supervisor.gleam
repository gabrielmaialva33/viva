//// Supervisor - VIVA Lifecycle Orchestrator
////
//// Manages multiple VIVAs, coordinates:
//// - Birth (Creator -> Soul)
//// - Life (ticks, interoception, resonance)
//// - Death (Soul -> Bardo -> Creator)
//// - Rebirth (Creator -> Soul)

import gleam/dict.{type Dict}
import gleam/erlang/process.{type Subject}
import gleam/int
import gleam/json
import gleam/list
import gleam/option.{type Option, None, Some}
import gleam/otp/actor
import viva/bardo
import viva/creator.{type Creator}
import viva/interoception
import viva/resonance.{type ResonanceEvent, type ResonancePool}
import viva/soul
import viva/telemetry/server.{type Broadcaster}
import viva/types.{type LifecycleEvent, type VivaId}
import viva_glyph/glyph

// =============================================================================
// NATURAL DEATH CRITERIA
// =============================================================================

/// Maximum lifespan in ticks (~166 minutes at 60fps)
const max_lifespan: Int = 10_000

/// Pleasure threshold below which soul falls into despair
/// PAD range is -1.0 to 1.0
const despair_threshold: Float = -0.7

// =============================================================================
// TYPES
// =============================================================================

/// Supervisor state
pub type SupervisorState {
  SupervisorState(
    /// The Creator (collective bank)
    creator: Creator,
    /// Resonance pool
    resonance_pool: ResonancePool,
    /// Active souls (id -> subject)
    souls: Dict(VivaId, Subject(soul.Message)),
    /// Event history
    events: List(LifecycleEvent),
    /// Global tick
    tick: Int,
    /// Next available ID
    next_id: VivaId,
    /// Telemetry broadcaster
    telemetry: Option(Broadcaster),
  )
}

/// Supervisor messages
pub type Message {
  // === Lifecycle ===
  /// Spawn new VIVA
  SpawnViva(reply: Subject(VivaId))
  /// Kill a VIVA (initiates bardo)
  KillViva(id: VivaId)
  /// Process death (called internally after bardo)
  ProcessDeath(id: VivaId)

  // === Simulation ===
  /// Global tick (evolves all VIVAs)
  GlobalTick(dt: Float)
  /// Apply interoception to all VIVAs
  ApplyInteroception

  // === Queries ===
  /// List alive VIVAs
  ListAlive(reply: Subject(List(VivaId)))
  /// Get supervisor state
  GetState(reply: Subject(SupervisorState))
  /// Get statistics
  GetStats(reply: Subject(String))
}

// =============================================================================
// SUPERVISOR API
// =============================================================================

/// Start the Supervisor
pub fn start() -> Result(Subject(Message), actor.StartError) {
  start_with_telemetry(True)
}

/// Start the Supervisor without telemetry (for tests)
pub fn start_without_telemetry() -> Result(Subject(Message), actor.StartError) {
  start_with_telemetry(False)
}

fn start_with_telemetry(
  enable_telemetry: Bool,
) -> Result(Subject(Message), actor.StartError) {
  let state = init(enable_telemetry)
  let builder =
    actor.new(state)
    |> actor.on_message(handle_message)

  case actor.start(builder) {
    Ok(started) -> Ok(started.data)
    Error(e) -> Error(e)
  }
}

/// Spawn new VIVA
pub fn spawn_viva(sup: Subject(Message)) -> VivaId {
  process.call(sup, 5000, fn(reply) { SpawnViva(reply) })
}

/// Kill a VIVA
pub fn kill_viva(sup: Subject(Message), id: VivaId) -> Nil {
  process.send(sup, KillViva(id))
}

/// Global tick
pub fn global_tick(sup: Subject(Message), dt: Float) -> Nil {
  process.send(sup, GlobalTick(dt))
}

/// Apply interoception
pub fn apply_interoception(sup: Subject(Message)) -> Nil {
  process.send(sup, ApplyInteroception)
}

/// List alive VIVAs
pub fn list_alive(sup: Subject(Message)) -> List(VivaId) {
  process.call(sup, 1000, fn(reply) { ListAlive(reply) })
}

/// Get state
pub fn get_state(sup: Subject(Message)) -> SupervisorState {
  process.call(sup, 1000, fn(reply) { GetState(reply) })
}

/// Get statistics
pub fn get_stats(sup: Subject(Message)) -> String {
  process.call(sup, 1000, fn(reply) { GetStats(reply) })
}

// =============================================================================
// INTERNAL
// =============================================================================

fn init(enable_telemetry: Bool) -> SupervisorState {
  // Start telemetry (only if enabled)
  let telemetry = case enable_telemetry {
    True ->
      case server.start_broadcaster() {
        Ok(broadcaster) -> {
          // Start HTTP server on 8080
          let _ = server.start(8080, broadcaster)
          Some(broadcaster)
        }
        Error(_) -> None
      }
    False -> None
  }

  SupervisorState(
    creator: creator.new(),
    resonance_pool: resonance.new_pool(),
    souls: dict.new(),
    events: [],
    tick: 0,
    next_id: 1,
    telemetry: telemetry,
  )
}

fn handle_message(
  state: SupervisorState,
  message: Message,
) -> actor.Next(SupervisorState, Message) {
  case message {
    // === Lifecycle ===
    SpawnViva(reply) -> {
      let #(new_state, id) = do_spawn_viva(state)
      process.send(reply, id)
      actor.continue(new_state)
    }

    KillViva(id) -> {
      let new_state = do_kill_viva(state, id)
      actor.continue(new_state)
    }

    ProcessDeath(id) -> {
      let new_state = do_process_death(state, id)
      actor.continue(new_state)
    }

    // === Simulation ===
    GlobalTick(dt) -> {
      let new_state = do_global_tick(state, dt)
      actor.continue(new_state)
    }

    ApplyInteroception -> {
      let new_state = do_apply_interoception(state)
      actor.continue(new_state)
    }

    // === Queries ===
    ListAlive(reply) -> {
      let alive_ids = dict.keys(state.souls)
      process.send(reply, alive_ids)
      actor.continue(state)
    }

    GetState(reply) -> {
      process.send(reply, state)
      actor.continue(state)
    }

    GetStats(reply) -> {
      let stats = build_stats(state)
      process.send(reply, stats)
      actor.continue(state)
    }
  }
}

// =============================================================================
// LIFECYCLE OPERATIONS
// =============================================================================

/// Spawn new VIVA
fn do_spawn_viva(state: SupervisorState) -> #(SupervisorState, VivaId) {
  // Get seed from Creator
  let #(new_creator, seed) = creator.spawn_life(state.creator)

  // Create config from seed
  let config =
    types.config_from_seed(
      "VIVA-" <> int.to_string(seed.viva_id),
      seed.life_number,
      seed.initial_mood,
      seed.initial_karma,
      seed.inherited_glyphs,
      seed.relevant_archetypes,
    )

  // Start Soul
  let id = state.next_id
  let assert Ok(soul_subject) = soul.start_with_config(id, config)

  // Register event
  let event = types.Born(id: id, life_number: seed.life_number)

  // Update state
  let new_state =
    SupervisorState(
      ..state,
      creator: new_creator,
      souls: dict.insert(state.souls, id, soul_subject),
      events: [event, ..state.events],
      next_id: state.next_id + 1,
    )

  #(new_state, id)
}

/// Kill a VIVA (starts death process)
fn do_kill_viva(state: SupervisorState, id: VivaId) -> SupervisorState {
  case dict.get(state.souls, id) {
    Ok(soul_subject) -> {
      // Send death command
      soul.kill(soul_subject)

      // Schedule death processing (async)
      // In practice, should wait for bardo (49 ticks)
      // For simplicity, process immediately
      do_process_death(state, id)
    }
    Error(_) -> state
  }
}

/// Process death after bardo
fn do_process_death(state: SupervisorState, id: VivaId) -> SupervisorState {
  case dict.get(state.souls, id) {
    Ok(soul_subject) -> {
      // Collect final state
      let soul_state = soul.get_state(soul_subject)
      let final_glyph = soul_state.current_glyph
      let karma_bank = soul_state.karma_bank

      // Execute bardo cycle
      let #(outcome, _phases) = bardo.run_bardo_cycle(final_glyph, karma_bank)

      // Register event
      let liberated = bardo.is_liberation(outcome)
      let event =
        types.Died(
          id: id,
          final_glyph: final_glyph,
          total_karma: karma_bank.total_karma,
        )

      // Process in Creator
      let seed_glyphs = bardo.get_seed_glyphs(outcome)
      let new_creator =
        creator.process_death(
          state.creator,
          id,
          seed_glyphs,
          karma_bank.total_karma,
          liberated,
        )

      // Remove soul from dict
      let new_souls = dict.delete(state.souls, id)

      // If not liberated, rebirth
      let #(final_creator, final_souls, final_events) = case liberated {
        True -> {
          let bardo_event = types.BardoComplete(id: id, liberated: True)
          #(new_creator, new_souls, [bardo_event, event, ..state.events])
        }
        False -> {
          // Rebirth
          let #(rebirth_creator, seed) = creator.spawn_life(new_creator)
          let config =
            types.config_from_seed(
              "VIVA-" <> int.to_string(id) <> "-R",
              seed.life_number,
              seed.initial_mood,
              seed.initial_karma,
              seed.inherited_glyphs,
              seed.relevant_archetypes,
            )

          // Rebirth existing soul (reuse subject)
          soul.rebirth(soul_subject, config)

          let bardo_event = types.BardoComplete(id: id, liberated: False)
          let reborn_event =
            types.Reborn(id: id, new_life_number: seed.life_number)

          #(rebirth_creator, dict.insert(new_souls, id, soul_subject), [
            reborn_event,
            bardo_event,
            event,
            ..state.events
          ])
        }
      }

      SupervisorState(
        ..state,
        creator: final_creator,
        souls: final_souls,
        events: final_events,
      )
    }
    Error(_) -> state
  }
}

// =============================================================================
// SIMULATION OPERATIONS
// =============================================================================

/// Global tick - evolve all VIVAs
fn do_global_tick(state: SupervisorState, dt: Float) -> SupervisorState {
  // Tick each soul
  dict.each(state.souls, fn(_id, soul_subject) { soul.tick(soul_subject, dt) })

  // Propagate resonance and get events
  let #(new_pool, resonance_events) = propagate_all_resonance(state)

  // Apply resonance events to affected souls
  apply_resonance_events(state.souls, resonance_events)

  // Creator tick
  let new_creator = creator.tick(state.creator)

  // Check natural deaths (age or despair)
  let deaths = check_natural_deaths(state)
  let state_after_deaths =
    list.fold(
      deaths,
      SupervisorState(..state, creator: new_creator, resonance_pool: new_pool),
      fn(s, id) { do_kill_viva(s, id) },
    )

  // Broadcast telemetry
  broadcast_telemetry(state_after_deaths)

  SupervisorState(..state_after_deaths, tick: state.tick + 1)
}

fn broadcast_telemetry(state: SupervisorState) {
  case state.telemetry {
    Some(broadcaster) -> {
      // Collect basic data from all souls
      let souls_data =
        dict.map_values(state.souls, fn(id, subject) {
          let snap = soul.get_snapshot(subject)
          json.object([
            #("id", json.int(id)),
            #("pleasure", json.float(snap.pad.pleasure)),
            #("arousal", json.float(snap.pad.arousal)),
            #("dominance", json.float(snap.pad.dominance)),
            #("glyph", json.string(glyph.to_string(snap.glyph))),
          ])
        })

      let payload =
        json.object([
          #("tick", json.int(state.tick)),
          #("souls", json.preprocessed_array(dict.values(souls_data))),
          #("alive_count", json.int(dict.size(state.souls))),
        ])
        |> json.to_string

      server.broadcast(broadcaster, payload)
    }
    None -> Nil
  }
}

/// Apply resonance events to target souls
/// Each event carries emotional influence from source to target
fn apply_resonance_events(
  souls: Dict(VivaId, Subject(soul.Message)),
  events: List(ResonanceEvent),
) -> Nil {
  // Sensitivity factor for resonance influence (0.0-1.0)
  // Lower = souls are more emotionally independent
  // Higher = souls strongly influenced by each other
  let sensitivity = 0.5

  list.each(events, fn(event) {
    case dict.get(souls, event.target) {
      Ok(target_soul) -> {
        // Get target's current PAD
        let target_pad = soul.get_pad(target_soul)

        // Calculate new PAD with resonance influence
        let new_pad = resonance.apply_resonance(target_pad, event, sensitivity)

        // Apply delta to target soul
        let delta_p = new_pad.pleasure -. target_pad.pleasure
        let delta_a = new_pad.arousal -. target_pad.arousal
        let delta_d = new_pad.dominance -. target_pad.dominance

        soul.apply_delta(target_soul, delta_p, delta_a, delta_d)
      }
      Error(_) -> Nil
      // Target soul not found (may have died)
    }
  })
}

/// Check for natural deaths (age, despair, or body suffering)
/// Returns list of soul IDs that should die
fn check_natural_deaths(state: SupervisorState) -> List(VivaId) {
  state.souls
  |> dict.to_list()
  |> list.filter_map(fn(pair) {
    let #(id, soul_subject) = pair
    let snapshot = soul.get_snapshot(soul_subject)

    // Death by old age
    let too_old = snapshot.tick > max_lifespan

    // Death by despair (pleasure below threshold)
    let in_despair = snapshot.pad.pleasure <. despair_threshold

    // Death by body suffering (needs unmet + high stress)
    let body_suffering = soul.is_suffering(soul_subject)

    case too_old || in_despair || body_suffering {
      True -> Ok(id)
      False -> Error(Nil)
    }
  })
}

/// Apply interoception to all VIVAs
fn do_apply_interoception(state: SupervisorState) -> SupervisorState {
  case interoception.sense() {
    Ok(metrics) -> {
      let baseline = interoception.baseline()
      let intero_state = interoception.compute_free_energy(baseline, metrics)
      let delta = interoception.to_pad_delta(intero_state)

      // Apply delta to each soul
      dict.each(state.souls, fn(_id, soul_subject) {
        soul.apply_delta(soul_subject, delta.x, delta.y, delta.z)
      })

      state
    }
    Error(_) -> state
  }
}

/// Propagate resonance between all VIVAs
fn propagate_all_resonance(
  state: SupervisorState,
) -> #(ResonancePool, List(ResonanceEvent)) {
  // Collect snapshots from all souls
  let snapshots =
    state.souls
    |> dict.to_list()
    |> list.map(fn(pair) {
      let #(_id, soul_subject) = pair
      soul.get_snapshot(soul_subject)
    })

  // Convert snapshots to resonance VivaState
  let viva_states =
    snapshots
    |> list.map(fn(snap) {
      resonance.VivaState(
        id: snap.id,
        pad: snap.pad,
        glyph: snap.glyph,
        alive: snap.alive,
        tick: snap.tick,
      )
    })

  // Update pool with current states
  let pool =
    viva_states
    |> list.fold(resonance.new_pool(), fn(pool, viva_state) {
      resonance.register(pool, viva_state)
    })

  // Propagate resonance from each source
  let #(final_pool, all_events) =
    viva_states
    |> list.fold(#(pool, []), fn(acc, source) {
      let #(current_pool, events) = acc
      let #(new_pool, new_events) =
        resonance.propagate(current_pool, source, state.tick)
      #(new_pool, list.append(new_events, events))
    })

  #(final_pool, all_events)
}

// =============================================================================
// STATS
// =============================================================================

fn build_stats(state: SupervisorState) -> String {
  let n_alive = dict.size(state.souls)
  let n_events = list.length(state.events)
  let creator_stats = creator.stats(state.creator)

  "Supervisor: "
  <> int.to_string(n_alive)
  <> " VIVAs alive, "
  <> int.to_string(n_events)
  <> " events, tick="
  <> int.to_string(state.tick)
  <> "\n"
  <> creator_stats
}
