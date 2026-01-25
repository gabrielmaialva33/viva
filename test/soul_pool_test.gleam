import gleam/dict
import gleam/list
import gleam/option.{None, Some}
import gleeunit/should
import viva/soul_pool

// =============================================================================
// SPAWN TESTS
// =============================================================================

pub fn spawn_returns_id_test() {
  let assert Ok(pool) = soul_pool.start()
  let id = soul_pool.spawn(pool)
  should.equal(id, 1)

  let id2 = soul_pool.spawn(pool)
  should.equal(id2, 2)

  soul_pool.kill_all(pool)
}

pub fn spawn_many_returns_ids_test() {
  let assert Ok(pool) = soul_pool.start()
  let ids = soul_pool.spawn_many(pool, 5)

  should.equal(list.length(ids), 5)
  should.equal(ids, [1, 2, 3, 4, 5])

  soul_pool.kill_all(pool)
}

pub fn count_tracks_alive_test() {
  let assert Ok(pool) = soul_pool.start()

  should.equal(soul_pool.count(pool), 0)

  let _ids = soul_pool.spawn_many(pool, 10)
  should.equal(soul_pool.count(pool), 10)

  soul_pool.kill(pool, 1)
  should.equal(soul_pool.count(pool), 9)

  soul_pool.kill_all(pool)
}

// =============================================================================
// TICK TESTS
// =============================================================================

pub fn tick_all_updates_all_souls_test() {
  let assert Ok(pool) = soul_pool.start()
  let _ids = soul_pool.spawn_many(pool, 3)

  // Initial tick is 0
  let states = soul_pool.get_all_states(pool)
  dict.each(states, fn(_, s) { should.equal(s.tick, 0) })

  // After tick_all, all should be tick=1
  soul_pool.tick_all(pool, 0.1)
  let states2 = soul_pool.get_all_states(pool)
  dict.each(states2, fn(_, s) { should.equal(s.tick, 1) })

  soul_pool.kill_all(pool)
}

pub fn tick_many_updates_only_specified_test() {
  let assert Ok(pool) = soul_pool.start()
  let _ids = soul_pool.spawn_many(pool, 3)

  // Only tick souls 1 and 2
  soul_pool.tick_many(pool, [1, 2], 0.1)

  let states = soul_pool.get_all_states(pool)

  case dict.get(states, 1) {
    Ok(s) -> should.equal(s.tick, 1)
    Error(_) -> should.fail()
  }
  case dict.get(states, 2) {
    Ok(s) -> should.equal(s.tick, 1)
    Error(_) -> should.fail()
  }
  case dict.get(states, 3) {
    Ok(s) -> should.equal(s.tick, 0)  // Not ticked
    Error(_) -> should.fail()
  }

  soul_pool.kill_all(pool)
}

// =============================================================================
// READ TESTS
// =============================================================================

pub fn get_pad_returns_soul_pad_test() {
  let assert Ok(pool) = soul_pool.start()
  let id = soul_pool.spawn(pool)

  let pad = soul_pool.get_pad(pool, id)
  should.be_true(option.is_some(pad))

  // Non-existent soul
  let pad2 = soul_pool.get_pad(pool, 999)
  should.equal(pad2, None)

  soul_pool.kill_all(pool)
}

pub fn get_all_pads_returns_all_test() {
  let assert Ok(pool) = soul_pool.start()
  let _ids = soul_pool.spawn_many(pool, 5)

  let pads = soul_pool.get_all_pads(pool)
  should.equal(dict.size(pads), 5)

  soul_pool.kill_all(pool)
}

pub fn get_all_snapshots_returns_list_test() {
  let assert Ok(pool) = soul_pool.start()
  let _ids = soul_pool.spawn_many(pool, 3)

  let snapshots = soul_pool.get_all_snapshots(pool)
  should.equal(list.length(snapshots), 3)

  // All should be alive
  list.each(snapshots, fn(s) { should.be_true(s.alive) })

  soul_pool.kill_all(pool)
}

// =============================================================================
// DELTA TESTS
// =============================================================================

pub fn apply_delta_affects_momentum_test() {
  let assert Ok(pool) = soul_pool.start()
  let id = soul_pool.spawn(pool)

  // Apply positive pleasure delta
  soul_pool.apply_delta(pool, id, 0.5, 0.0, 0.0)

  // Tick to apply momentum
  soul_pool.tick_all(pool, 1.0)

  case soul_pool.get_pad(pool, id) {
    Some(p) -> should.be_true(p.pleasure >. 0.0)
    None -> should.fail()
  }

  soul_pool.kill_all(pool)
}

pub fn apply_delta_all_affects_all_test() {
  let assert Ok(pool) = soul_pool.start()
  let _ids = soul_pool.spawn_many(pool, 3)

  // Apply positive arousal delta to all
  soul_pool.apply_delta_all(pool, 0.0, 0.8, 0.0)

  // Tick to apply momentum
  soul_pool.tick_all(pool, 1.0)

  let pads = soul_pool.get_all_pads(pool)
  dict.each(pads, fn(_, p) {
    should.be_true(p.arousal >. 0.0)
  })

  soul_pool.kill_all(pool)
}

// =============================================================================
// KILL TESTS
// =============================================================================

pub fn kill_marks_soul_dead_test() {
  let assert Ok(pool) = soul_pool.start()
  let id = soul_pool.spawn(pool)

  should.equal(soul_pool.count(pool), 1)

  soul_pool.kill(pool, id)

  should.equal(soul_pool.count(pool), 0)  // Dead souls don't count

  // State still exists but marked dead
  case soul_pool.get_state(pool, id) {
    Some(s) -> should.be_false(s.alive)
    None -> should.fail()
  }

  soul_pool.kill_all(pool)
}

pub fn dead_souls_dont_tick_test() {
  let assert Ok(pool) = soul_pool.start()
  let id = soul_pool.spawn(pool)

  soul_pool.kill(pool, id)

  // Try to tick dead soul
  soul_pool.tick_all(pool, 0.1)

  case soul_pool.get_state(pool, id) {
    Some(s) -> should.equal(s.tick, 0)  // Still 0, didn't tick
    None -> should.fail()
  }

  soul_pool.kill_all(pool)
}

// =============================================================================
// LIST ALIVE TEST
// =============================================================================

pub fn list_alive_returns_only_living_test() {
  let assert Ok(pool) = soul_pool.start()
  let _ids = soul_pool.spawn_many(pool, 5)

  // Kill some
  soul_pool.kill(pool, 2)
  soul_pool.kill(pool, 4)

  let alive = soul_pool.list_alive(pool)
  should.equal(list.length(alive), 3)
  should.be_true(list.contains(alive, 1))
  should.be_true(list.contains(alive, 3))
  should.be_true(list.contains(alive, 5))
  should.be_false(list.contains(alive, 2))
  should.be_false(list.contains(alive, 4))

  soul_pool.kill_all(pool)
}
