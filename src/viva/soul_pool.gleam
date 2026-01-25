//// Soul Pool - High-Performance Soul Management
////
//// Single actor managing multiple souls internally.
//// Eliminates per-soul message overhead.
////
//// Before: 100 souls = 100 actors = 100 messages/tick
//// After:  100 souls = 1 actor = 1 message/tick

import gleam/dict.{type Dict}
import gleam/erlang/process.{type Subject}
import gleam/float
import gleam/list
import gleam/option.{type Option, None, Some}
import gleam/otp/actor
import viva/types.{type VivaConfig, type VivaId}
import viva_emotion/pad.{type Pad}
import viva_glyph/glyph.{type Glyph}

// =============================================================================
// TYPES
// =============================================================================

/// Internal soul state (no actor overhead)
pub type SoulState {
  SoulState(
    id: VivaId,
    pad: Pad,
    glyph: Glyph,
    tick: Int,
    alive: Bool,
    // Dynamics
    mood_momentum: Pad,
    decay_rate: Float,
    // Config
    name: String,
    life_number: Int,
  )
}

/// Pool state
pub type PoolState {
  PoolState(souls: Dict(VivaId, SoulState), next_id: VivaId, global_tick: Int)
}

/// Pool messages
pub type Message {
  // Lifecycle
  Spawn(config: Option(VivaConfig), reply: Subject(VivaId))
  SpawnMany(count: Int, reply: Subject(List(VivaId)))
  Kill(id: VivaId)
  KillAll

  // Batch operations (the whole point)
  TickAll(dt: Float)
  TickMany(ids: List(VivaId), dt: Float)

  // Batch reads
  GetAllPads(reply: Subject(Dict(VivaId, Pad)))
  GetAllStates(reply: Subject(Dict(VivaId, SoulState)))
  GetAllSnapshots(reply: Subject(List(SoulSnapshot)))

  // Single operations (still available)
  GetPad(id: VivaId, reply: Subject(Option(Pad)))
  GetState(id: VivaId, reply: Subject(Option(SoulState)))
  ApplyDelta(id: VivaId, dp: Float, da: Float, dd: Float)
  ApplyDeltaAll(dp: Float, da: Float, dd: Float)

  // Queries
  Count(reply: Subject(Int))
  ListAlive(reply: Subject(List(VivaId)))
}

/// Snapshot for external use
pub type SoulSnapshot {
  SoulSnapshot(id: VivaId, pad: Pad, glyph: Glyph, tick: Int, alive: Bool)
}

// =============================================================================
// API
// =============================================================================

/// Start soul pool
pub fn start() -> Result(Subject(Message), actor.StartError) {
  let builder =
    actor.new(init())
    |> actor.on_message(handle_message)

  case actor.start(builder) {
    Ok(started) -> Ok(started.data)
    Error(e) -> Error(e)
  }
}

/// Spawn single soul
pub fn spawn(pool: Subject(Message)) -> VivaId {
  process.call(pool, 1000, fn(reply) { Spawn(None, reply) })
}

/// Spawn with config
pub fn spawn_with_config(pool: Subject(Message), config: VivaConfig) -> VivaId {
  process.call(pool, 1000, fn(reply) { Spawn(Some(config), reply) })
}

/// Spawn many souls at once
pub fn spawn_many(pool: Subject(Message), count: Int) -> List(VivaId) {
  process.call(pool, 5000, fn(reply) { SpawnMany(count, reply) })
}

/// Kill soul
pub fn kill(pool: Subject(Message), id: VivaId) -> Nil {
  process.send(pool, Kill(id))
}

/// Kill all souls
pub fn kill_all(pool: Subject(Message)) -> Nil {
  process.send(pool, KillAll)
}

/// Tick ALL souls (single message!)
pub fn tick_all(pool: Subject(Message), dt: Float) -> Nil {
  process.send(pool, TickAll(dt))
}

/// Tick specific souls
pub fn tick_many(pool: Subject(Message), ids: List(VivaId), dt: Float) -> Nil {
  process.send(pool, TickMany(ids, dt))
}

/// Get all PADs (single message!)
pub fn get_all_pads(pool: Subject(Message)) -> Dict(VivaId, Pad) {
  process.call(pool, 1000, fn(reply) { GetAllPads(reply) })
}

/// Get all states
pub fn get_all_states(pool: Subject(Message)) -> Dict(VivaId, SoulState) {
  process.call(pool, 1000, fn(reply) { GetAllStates(reply) })
}

/// Get all snapshots
pub fn get_all_snapshots(pool: Subject(Message)) -> List(SoulSnapshot) {
  process.call(pool, 1000, fn(reply) { GetAllSnapshots(reply) })
}

/// Get single PAD
pub fn get_pad(pool: Subject(Message), id: VivaId) -> Option(Pad) {
  process.call(pool, 1000, fn(reply) { GetPad(id, reply) })
}

/// Get single state
pub fn get_state(pool: Subject(Message), id: VivaId) -> Option(SoulState) {
  process.call(pool, 1000, fn(reply) { GetState(id, reply) })
}

/// Apply delta to single soul
pub fn apply_delta(
  pool: Subject(Message),
  id: VivaId,
  dp: Float,
  da: Float,
  dd: Float,
) -> Nil {
  process.send(pool, ApplyDelta(id, dp, da, dd))
}

/// Apply delta to ALL souls
pub fn apply_delta_all(
  pool: Subject(Message),
  dp: Float,
  da: Float,
  dd: Float,
) -> Nil {
  process.send(pool, ApplyDeltaAll(dp, da, dd))
}

/// Count alive souls
pub fn count(pool: Subject(Message)) -> Int {
  process.call(pool, 1000, fn(reply) { Count(reply) })
}

/// List alive soul IDs
pub fn list_alive(pool: Subject(Message)) -> List(VivaId) {
  process.call(pool, 1000, fn(reply) { ListAlive(reply) })
}

// =============================================================================
// INTERNAL - Message Handler
// =============================================================================

fn init() -> PoolState {
  PoolState(souls: dict.new(), next_id: 1, global_tick: 0)
}

fn handle_message(
  state: PoolState,
  message: Message,
) -> actor.Next(PoolState, Message) {
  case message {
    // === Lifecycle ===
    Spawn(config, reply) -> {
      let #(new_state, id) = do_spawn(state, config)
      process.send(reply, id)
      actor.continue(new_state)
    }

    SpawnMany(count, reply) -> {
      let #(new_state, ids) = do_spawn_many(state, count)
      process.send(reply, ids)
      actor.continue(new_state)
    }

    Kill(id) -> {
      actor.continue(do_kill(state, id))
    }

    KillAll -> {
      actor.continue(PoolState(..state, souls: dict.new()))
    }

    // === Batch Operations ===
    TickAll(dt) -> {
      actor.continue(do_tick_all(state, dt))
    }

    TickMany(ids, dt) -> {
      actor.continue(do_tick_many(state, ids, dt))
    }

    // === Batch Reads ===
    GetAllPads(reply) -> {
      let pads = dict.map_values(state.souls, fn(_, soul) { soul.pad })
      process.send(reply, pads)
      actor.continue(state)
    }

    GetAllStates(reply) -> {
      process.send(reply, state.souls)
      actor.continue(state)
    }

    GetAllSnapshots(reply) -> {
      let snapshots =
        state.souls
        |> dict.values()
        |> list.map(fn(soul) {
          SoulSnapshot(
            id: soul.id,
            pad: soul.pad,
            glyph: soul.glyph,
            tick: soul.tick,
            alive: soul.alive,
          )
        })
      process.send(reply, snapshots)
      actor.continue(state)
    }

    // === Single Operations ===
    GetPad(id, reply) -> {
      let pad = case dict.get(state.souls, id) {
        Ok(soul) -> Some(soul.pad)
        Error(_) -> None
      }
      process.send(reply, pad)
      actor.continue(state)
    }

    GetState(id, reply) -> {
      let soul_state = case dict.get(state.souls, id) {
        Ok(soul) -> Some(soul)
        Error(_) -> None
      }
      process.send(reply, soul_state)
      actor.continue(state)
    }

    ApplyDelta(id, dp, da, dd) -> {
      actor.continue(do_apply_delta(state, id, dp, da, dd))
    }

    ApplyDeltaAll(dp, da, dd) -> {
      actor.continue(do_apply_delta_all(state, dp, da, dd))
    }

    // === Queries ===
    Count(reply) -> {
      let alive_count =
        state.souls
        |> dict.values()
        |> list.filter(fn(s) { s.alive })
        |> list.length()
      process.send(reply, alive_count)
      actor.continue(state)
    }

    ListAlive(reply) -> {
      let alive_ids =
        state.souls
        |> dict.to_list()
        |> list.filter_map(fn(pair) {
          let #(id, soul) = pair
          case soul.alive {
            True -> Ok(id)
            False -> Error(Nil)
          }
        })
      process.send(reply, alive_ids)
      actor.continue(state)
    }
  }
}

// =============================================================================
// INTERNAL - Operations
// =============================================================================

fn do_spawn(
  state: PoolState,
  config: Option(VivaConfig),
) -> #(PoolState, VivaId) {
  let id = state.next_id

  let soul = case config {
    Some(cfg) -> {
      // initial_mood is 0-1, convert to pleasure dimension
      let initial_pleasure = cfg.initial_mood *. 2.0 -. 1.0
      // 0-1 -> -1 to 1
      SoulState(
        id: id,
        pad: pad.new(initial_pleasure, 0.0, 0.0),
        glyph: glyph.neutral(),
        tick: 0,
        alive: True,
        mood_momentum: pad.neutral(),
        decay_rate: 0.01,
        name: cfg.name,
        life_number: cfg.life_number,
      )
    }
    None ->
      SoulState(
        id: id,
        pad: pad.neutral(),
        glyph: glyph.neutral(),
        tick: 0,
        alive: True,
        mood_momentum: pad.neutral(),
        decay_rate: 0.01,
        name: "VIVA-" <> int_to_string(id),
        life_number: 1,
      )
  }

  let new_state =
    PoolState(
      ..state,
      souls: dict.insert(state.souls, id, soul),
      next_id: id + 1,
    )

  #(new_state, id)
}

fn do_spawn_many(state: PoolState, count: Int) -> #(PoolState, List(VivaId)) {
  do_spawn_many_acc(state, count, [])
}

fn do_spawn_many_acc(
  state: PoolState,
  remaining: Int,
  acc: List(VivaId),
) -> #(PoolState, List(VivaId)) {
  case remaining <= 0 {
    True -> #(state, list.reverse(acc))
    False -> {
      let #(new_state, id) = do_spawn(state, None)
      do_spawn_many_acc(new_state, remaining - 1, [id, ..acc])
    }
  }
}

fn do_kill(state: PoolState, id: VivaId) -> PoolState {
  case dict.get(state.souls, id) {
    Ok(soul) -> {
      let updated_soul = SoulState(..soul, alive: False)
      PoolState(..state, souls: dict.insert(state.souls, id, updated_soul))
    }
    Error(_) -> state
  }
}

fn do_tick_all(state: PoolState, dt: Float) -> PoolState {
  let new_souls =
    dict.map_values(state.souls, fn(_, soul) { tick_soul(soul, dt) })

  PoolState(..state, souls: new_souls, global_tick: state.global_tick + 1)
}

fn do_tick_many(state: PoolState, ids: List(VivaId), dt: Float) -> PoolState {
  let new_souls =
    list.fold(ids, state.souls, fn(souls, id) {
      case dict.get(souls, id) {
        Ok(soul) -> dict.insert(souls, id, tick_soul(soul, dt))
        Error(_) -> souls
      }
    })

  PoolState(..state, souls: new_souls)
}

fn tick_soul(soul: SoulState, dt: Float) -> SoulState {
  case soul.alive {
    False -> soul
    True -> {
      // Apply momentum
      let new_pad = pad.add(soul.pad, pad.scale(soul.mood_momentum, dt))

      // Decay momentum
      let decay_factor = 1.0 -. soul.decay_rate *. dt
      let new_momentum = pad.scale(soul.mood_momentum, decay_factor)

      // Clamp PAD to valid range
      let clamped_pad = clamp_pad(new_pad)

      // Update glyph based on PAD (simple encoding)
      let new_glyph = update_glyph(soul.glyph, clamped_pad)

      SoulState(
        ..soul,
        pad: clamped_pad,
        glyph: new_glyph,
        mood_momentum: new_momentum,
        tick: soul.tick + 1,
      )
    }
  }
}

fn do_apply_delta(
  state: PoolState,
  id: VivaId,
  dp: Float,
  da: Float,
  dd: Float,
) -> PoolState {
  case dict.get(state.souls, id) {
    Ok(soul) -> {
      let delta = pad.new(dp, da, dd)
      let new_momentum = pad.add(soul.mood_momentum, delta)
      let updated_soul = SoulState(..soul, mood_momentum: new_momentum)
      PoolState(..state, souls: dict.insert(state.souls, id, updated_soul))
    }
    Error(_) -> state
  }
}

fn do_apply_delta_all(
  state: PoolState,
  dp: Float,
  da: Float,
  dd: Float,
) -> PoolState {
  let delta = pad.new(dp, da, dd)
  let new_souls =
    dict.map_values(state.souls, fn(_, soul) {
      let new_momentum = pad.add(soul.mood_momentum, delta)
      SoulState(..soul, mood_momentum: new_momentum)
    })
  PoolState(..state, souls: new_souls)
}

// =============================================================================
// HELPERS
// =============================================================================

fn clamp_pad(p: Pad) -> Pad {
  pad.new(
    clamp(p.pleasure, -1.0, 1.0),
    clamp(p.arousal, -1.0, 1.0),
    clamp(p.dominance, -1.0, 1.0),
  )
}

fn clamp(value: Float, min: Float, max: Float) -> Float {
  case value <. min {
    True -> min
    False ->
      case value >. max {
        True -> max
        False -> value
      }
  }
}

fn update_glyph(g: Glyph, p: Pad) -> Glyph {
  // Simple: encode PAD into first 3 tokens
  let p_token = float.round(p.pleasure *. 127.0) + 128
  let a_token = float.round(p.arousal *. 127.0) + 128
  let d_token = float.round(p.dominance *. 127.0) + 128

  // Keep rest of glyph, update first 3
  let tokens = case g.tokens {
    [_, _, _, ..rest] -> [p_token, a_token, d_token, ..rest]
    _ -> [p_token, a_token, d_token]
  }

  glyph.new(tokens)
}

fn int_to_string(n: Int) -> String {
  case n {
    0 -> "0"
    1 -> "1"
    2 -> "2"
    3 -> "3"
    4 -> "4"
    5 -> "5"
    6 -> "6"
    7 -> "7"
    8 -> "8"
    9 -> "9"
    _ -> {
      let div = n / 10
      let rem = n % 10
      int_to_string(div) <> int_to_string(rem)
    }
  }
}
