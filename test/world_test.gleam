//// Physics World Tests
////
//// Tests for world simulation, body management, and queries.

import gleam/dict
import gleam/float
import gleam/list
import gleam/option.{None, Some}
import gleeunit/should
import viva/embodied/physics/body
import viva/embodied/physics/vec4.{Vec4}
import viva/embodied/physics/world
import viva_glyph/glyph

// =============================================================================
// WORLD CREATION TESTS
// =============================================================================

pub fn world_new_test() {
  let w = world.new()

  should.equal(w.tick, 0)
  should.equal(w.next_id, 1)
  should.equal(w.attractor, None)
  should.equal(dict.size(w.bodies), 0)
}

pub fn world_with_config_test() {
  let config =
    world.PhysicsConfig(
      gravity: Vec4(0.0, -0.1, 0.0, 0.0),
      attraction_strength: 0.2,
      damping: 0.9,
      resonance_threshold: 0.8,
      sleep_threshold: 50,
    )

  let w = world.with_config(config)

  should.equal(w.config.attraction_strength, 0.2)
  should.equal(w.config.damping, 0.9)
  should.equal(w.config.sleep_threshold, 50)
}

pub fn world_default_config_test() {
  let config = world.default_config()

  should.equal(config.gravity, vec4.zero)
  should.equal(config.attraction_strength, 0.1)
  should.equal(config.damping, 0.95)
  should.equal(config.resonance_threshold, 0.7)
  should.equal(config.sleep_threshold, 100)
}

// =============================================================================
// BODY MANAGEMENT TESTS
// =============================================================================

pub fn world_add_dynamic_test() {
  let w = world.new()
  let pos = Vec4(0.5, 0.5, 0.5, 0.5)
  let g = glyph.new([128, 128, 128, 128])

  let #(w1, id) = world.add_dynamic(w, pos, g, 1.0)

  should.equal(id, 1)
  should.equal(dict.size(w1.bodies), 1)
  should.equal(w1.next_id, 2)
}

pub fn world_add_static_test() {
  let w = world.new()
  let pos = Vec4(0.0, 0.0, 0.0, 1.0)
  let g = glyph.new([255, 0, 0, 128])

  let #(w1, id) = world.add_static(w, pos, g)

  should.equal(id, 1)

  let assert Some(b) = world.get(w1, id)
  should.equal(b.motion_type, body.Static)
}

pub fn world_add_multiple_bodies_test() {
  let g = glyph.new([128, 128, 128, 128])
  let w = world.new()

  let #(w1, id1) = world.add_dynamic(w, Vec4(0.1, 0.1, 0.0, 0.5), g, 1.0)
  let #(w2, id2) = world.add_dynamic(w1, Vec4(0.5, 0.5, 0.0, 0.5), g, 1.0)
  let #(w3, id3) = world.add_static(w2, Vec4(0.0, 0.0, 0.0, 1.0), g)

  should.equal(id1, 1)
  should.equal(id2, 2)
  should.equal(id3, 3)
  should.equal(dict.size(w3.bodies), 3)
}

pub fn world_remove_test() {
  let w = world.new()
  let g = glyph.new([128, 128, 128, 128])

  let #(w1, id) = world.add_dynamic(w, vec4.zero, g, 1.0)
  should.equal(dict.size(w1.bodies), 1)

  let w2 = world.remove(w1, id)
  should.equal(dict.size(w2.bodies), 0)
}

pub fn world_get_existing_test() {
  let w = world.new()
  let pos = Vec4(0.3, 0.4, 0.5, 0.6)
  let g = glyph.new([100, 150, 200, 128])

  let #(w1, id) = world.add_dynamic(w, pos, g, 2.0)
  let result = world.get(w1, id)

  should.be_some(result)
  let assert Some(b) = result
  should.equal(b.position, pos)
  should.equal(b.mass, 2.0)
}

pub fn world_get_nonexistent_test() {
  let w = world.new()
  let result = world.get(w, 999)

  should.be_none(result)
}

// =============================================================================
// ATTRACTOR TESTS
// =============================================================================

pub fn world_set_attractor_test() {
  let w = world.new()
  let attractor_pos = Vec4(0.5, 0.5, 0.5, 1.0)

  let w1 = world.set_attractor(w, attractor_pos)

  should.equal(w1.attractor, Some(attractor_pos))
}

pub fn world_clear_attractor_test() {
  let w = world.new()
  let w1 = world.set_attractor(w, Vec4(0.5, 0.5, 0.5, 1.0))
  let w2 = world.clear_attractor(w1)

  should.equal(w2.attractor, None)
}

// =============================================================================
// SIMULATION STEP TESTS
// =============================================================================

pub fn world_step_increments_tick_test() {
  let w = world.new()
  should.equal(w.tick, 0)

  let w1 = world.step(w, 0.1)
  should.equal(w1.tick, 1)

  let w2 = world.step(w1, 0.1)
  should.equal(w2.tick, 2)
}

pub fn world_step_single_body_test() {
  let w = world.new()
  let g = glyph.new([128, 128, 128, 128])
  let initial_pos = Vec4(0.5, 0.5, 0.0, 0.5)

  let #(w1, id) = world.add_dynamic(w, initial_pos, g, 1.0)

  // Add velocity to the body
  let assert Some(b) = world.get(w1, id)
  let b_with_velocity = body.MemoryBody(..b, velocity: Vec4(0.1, 0.0, 0.0, 0.0))
  let w2 =
    world.PhysicsWorld(
      ..w1,
      bodies: dict.insert(w1.bodies, id, b_with_velocity),
    )

  // Step
  let w3 = world.step(w2, 1.0)

  let assert Some(b_after) = world.get(w3, id)
  // Position should have changed
  should.be_true(b_after.position.x >. initial_pos.x)
}

pub fn world_step_with_attractor_test() {
  let w = world.new()
  let g = glyph.new([128, 128, 128, 128])

  // Body at (0.8, 0.5, ...)
  let #(w1, id) = world.add_dynamic(w, Vec4(0.8, 0.5, 0.0, 0.5), g, 1.0)

  // Attractor at origin
  let w2 = world.set_attractor(w1, Vec4(0.0, 0.0, 0.0, 0.5))

  // Step multiple times
  let w3 = world.step(w2, 0.1)
  let w4 = world.step(w3, 0.1)
  let w5 = world.step(w4, 0.1)

  let assert Some(b_after) = world.get(w5, id)

  // Body should be moving toward attractor (velocity.x should be negative)
  should.be_true(b_after.velocity.x <. 0.0)
}

pub fn world_step_static_unchanged_test() {
  let w = world.new()
  let g = glyph.new([128, 128, 128, 128])
  let pos = Vec4(0.5, 0.5, 0.5, 1.0)

  let #(w1, id) = world.add_static(w, pos, g)

  // Even with attractor, static body shouldn't move
  let w2 = world.set_attractor(w1, vec4.zero)
  let w3 = world.step(w2, 0.1)
  let w4 = world.step(w3, 0.1)

  let assert Some(b) = world.get(w4, id)
  should.equal(b.position, pos)
}

pub fn world_step_sleeping_unchanged_test() {
  let w = world.new()
  let g = glyph.new([128, 128, 128, 128])

  let #(w1, id) = world.add_dynamic(w, Vec4(0.5, 0.5, 0.0, 0.5), g, 1.0)

  // Put body to sleep
  let assert Some(b) = world.get(w1, id)
  let sleeping = body.put_to_sleep(b)
  let w2 =
    world.PhysicsWorld(..w1, bodies: dict.insert(w1.bodies, id, sleeping))

  // Add attractor
  let w3 = world.set_attractor(w2, vec4.zero)
  let w4 = world.step(w3, 0.1)

  let assert Some(b_after) = world.get(w4, id)
  // Should still be sleeping and at same position
  should.be_true(body.is_sleeping(b_after))
}

pub fn world_step_multiple_bodies_test() {
  let w = world.new()
  let g = glyph.new([128, 128, 128, 128])

  let #(w1, _id1) = world.add_dynamic(w, Vec4(0.2, 0.2, 0.0, 0.5), g, 1.0)
  let #(w2, _id2) = world.add_dynamic(w1, Vec4(0.8, 0.8, 0.0, 0.5), g, 1.0)
  let #(w3, _id3) = world.add_static(w2, vec4.zero, g)

  let w4 = world.set_attractor(w3, Vec4(0.5, 0.5, 0.0, 0.5))

  // Step multiple times
  let w5 = world.step(w4, 0.1)
  let w6 = world.step(w5, 0.1)

  // Should not crash and tick should advance
  should.equal(w6.tick, 2)
  should.equal(dict.size(w6.bodies), 3)
}

// =============================================================================
// WAKE TESTS
// =============================================================================

pub fn world_wake_test() {
  let w = world.new()
  let g = glyph.new([128, 128, 128, 128])

  let #(w1, id) = world.add_dynamic(w, vec4.zero, g, 1.0)

  // Put to sleep
  let assert Some(b) = world.get(w1, id)
  let sleeping = body.put_to_sleep(b)
  let w2 =
    world.PhysicsWorld(..w1, bodies: dict.insert(w1.bodies, id, sleeping))

  // Verify sleeping
  let assert Some(b_sleeping) = world.get(w2, id)
  should.be_true(body.is_sleeping(b_sleeping))

  // Wake
  let w3 = world.wake(w2, id)

  let assert Some(b_awake) = world.get(w3, id)
  should.be_false(body.is_sleeping(b_awake))
}

pub fn world_wake_nonexistent_test() {
  let w = world.new()

  // Should not crash when waking non-existent body
  let w1 = world.wake(w, 999)

  should.equal(dict.size(w1.bodies), 0)
}

// =============================================================================
// QUERY TESTS
// =============================================================================

pub fn world_query_radius_test() {
  let w = world.new()
  let g = glyph.new([128, 128, 128, 128])

  // Bodies at different distances from origin (all w=0 for clean distance calc)
  let #(w1, id1) = world.add_dynamic(w, Vec4(0.1, 0.0, 0.0, 0.0), g, 1.0)
  let #(w2, id2) = world.add_dynamic(w1, Vec4(0.3, 0.0, 0.0, 0.0), g, 1.0)
  let #(w3, _id3) = world.add_dynamic(w2, Vec4(0.9, 0.0, 0.0, 0.0), g, 1.0)

  // Query within radius 0.5
  let results = world.query_radius(w3, vec4.zero, 0.5)

  // Should find id1 (dist 0.1) and id2 (dist 0.3), but not id3 (dist 0.9)
  should.equal(list.length(results), 2)
  should.be_true(list.contains(results, id1))
  should.be_true(list.contains(results, id2))
}

pub fn world_query_radius_empty_test() {
  let w = world.new()
  let g = glyph.new([128, 128, 128, 128])

  let #(w1, _) = world.add_dynamic(w, Vec4(0.9, 0.9, 0.9, 0.5), g, 1.0)

  // Query with very small radius at origin
  let results = world.query_radius(w1, vec4.zero, 0.1)

  should.equal(list.length(results), 0)
}

pub fn world_query_nearest_test() {
  let w = world.new()
  let g = glyph.new([128, 128, 128, 128])

  // All w=0 for clean distance calculation
  let #(w1, id1) = world.add_dynamic(w, Vec4(0.1, 0.0, 0.0, 0.0), g, 1.0)
  let #(w2, id2) = world.add_dynamic(w1, Vec4(0.5, 0.0, 0.0, 0.0), g, 1.0)
  let #(w3, _id3) = world.add_dynamic(w2, Vec4(0.9, 0.0, 0.0, 0.0), g, 1.0)

  // Query 2 nearest to origin
  let results = world.query_nearest(w3, vec4.zero, 2)

  should.equal(list.length(results), 2)

  // First should be closest
  let assert Ok(#(closest_id, closest_dist)) = list.first(results)
  should.equal(closest_id, id1)
  should.be_true(float.loosely_equals(closest_dist, 0.1, 0.0001))

  // Second should be id2 (use drop + first instead of at)
  let assert Ok(#(second_id, _)) = results |> list.drop(1) |> list.first
  should.equal(second_id, id2)
}

pub fn world_query_resonance_test() {
  let w = world.new()

  // Different glyphs
  let g1 = glyph.new([255, 0, 0, 128])
  let g2 = glyph.new([250, 5, 5, 130])
  // Similar to g1
  let g3 = glyph.new([0, 255, 0, 128])
  // Very different

  let #(w1, id1) = world.add_dynamic(w, Vec4(0.1, 0.0, 0.0, 0.5), g1, 1.0)
  let #(w2, _id2) = world.add_dynamic(w1, Vec4(0.5, 0.0, 0.0, 0.5), g2, 1.0)
  let #(w3, _id3) = world.add_dynamic(w2, Vec4(0.9, 0.0, 0.0, 0.5), g3, 1.0)

  // Query similar to g1
  let results = world.query_resonance(w3, g1, 0.9)

  // Should find g1 (exact match) and g2 (very similar), but not g3
  should.be_true(list.length(results) >= 1)

  // First result should be g1 with similarity 1.0
  let assert Ok(#(best_id, best_sim)) = list.first(results)
  should.equal(best_id, id1)
  should.be_true(float.loosely_equals(best_sim, 1.0, 0.0001))
}

pub fn world_query_near_and_resonant_test() {
  let w = world.new()
  let g_query = glyph.new([128, 128, 128, 128])

  // Body close and similar (identical glyph for high similarity)
  let g_similar = glyph.new([128, 128, 128, 128])
  let #(w1, id1) =
    world.add_dynamic(w, Vec4(0.1, 0.0, 0.0, 0.0), g_similar, 1.0)

  // Body close but different
  let g_different = glyph.new([0, 255, 0, 128])
  let #(w2, _id2) =
    world.add_dynamic(w1, Vec4(0.1, 0.1, 0.0, 0.0), g_different, 1.0)

  // Body far but similar
  let #(w3, _id3) =
    world.add_dynamic(w2, Vec4(0.9, 0.9, 0.0, 0.0), g_similar, 1.0)

  // Query: close (radius 0.5) AND similar (threshold 0.9)
  let results = world.query_near_and_resonant(w3, vec4.zero, g_query, 0.5, 0.9)

  // Should only find id1 (close AND similar)
  should.equal(list.length(results), 1)
  let assert Ok(#(found_id, _, _)) = list.first(results)
  should.equal(found_id, id1)
}

// =============================================================================
// STATS TESTS
// =============================================================================

pub fn world_stats_empty_test() {
  let w = world.new()
  let s = world.stats(w)

  should.equal(s.total, 0)
  should.equal(s.awake, 0)
  should.equal(s.sleeping, 0)
  should.equal(s.dynamic, 0)
  should.equal(s.static_count, 0)
  should.equal(s.tick, 0)
}

pub fn world_stats_with_bodies_test() {
  let w = world.new()
  let g = glyph.new([128, 128, 128, 128])

  let #(w1, _) = world.add_dynamic(w, Vec4(0.1, 0.1, 0.0, 0.5), g, 1.0)
  let #(w2, _) = world.add_dynamic(w1, Vec4(0.2, 0.2, 0.0, 0.5), g, 1.0)
  let #(w3, _) = world.add_static(w2, vec4.zero, g)

  // Step once
  let w4 = world.step(w3, 0.1)

  let s = world.stats(w4)

  should.equal(s.total, 3)
  should.equal(s.dynamic, 2)
  should.equal(s.static_count, 1)
  should.equal(s.tick, 1)
}

pub fn world_stats_string_test() {
  let w = world.new()
  let g = glyph.new([128, 128, 128, 128])

  let #(w1, _) = world.add_dynamic(w, vec4.zero, g, 1.0)

  let s = world.stats(w1)
  let str = world.stats_string(s)

  // Should contain relevant info
  should.be_true(str != "")
}

// =============================================================================
// AUTO-SLEEP TESTS
// =============================================================================

pub fn world_auto_sleep_after_threshold_test() {
  // Use low threshold for test
  let config = world.PhysicsConfig(..world.default_config(), sleep_threshold: 3)
  let w = world.with_config(config)
  let g = glyph.new([128, 128, 128, 128])

  // Add stationary body (no velocity)
  let #(w1, id) = world.add_dynamic(w, Vec4(0.5, 0.5, 0.0, 0.5), g, 1.0)

  // Step multiple times (more than threshold)
  let w2 = world.step(w1, 0.1)
  let w3 = world.step(w2, 0.1)
  let w4 = world.step(w3, 0.1)
  let w5 = world.step(w4, 0.1)
  let w6 = world.step(w5, 0.1)

  let assert Some(b) = world.get(w6, id)
  // Should be sleeping after exceeding threshold
  should.be_true(body.is_sleeping(b))
}
