//// Physics Module Tests
////
//// Tests for vec4, body, and world modules

import gleam/float
import gleeunit/should
import viva/physics/body
import viva/physics/vec4.{Vec4}
import viva_glyph/glyph

// =============================================================================
// VEC4 TESTS
// =============================================================================

pub fn vec4_add_test() {
  let a = Vec4(1.0, 2.0, 3.0, 4.0)
  let b = Vec4(0.5, 0.5, 0.5, 0.5)
  let result = vec4.add(a, b)

  should.equal(result.x, 1.5)
  should.equal(result.y, 2.5)
  should.equal(result.z, 3.5)
  should.equal(result.w, 4.5)
}

pub fn vec4_sub_test() {
  let a = Vec4(1.0, 2.0, 3.0, 4.0)
  let b = Vec4(0.5, 0.5, 0.5, 0.5)
  let result = vec4.sub(a, b)

  should.equal(result.x, 0.5)
  should.equal(result.y, 1.5)
  should.equal(result.z, 2.5)
  should.equal(result.w, 3.5)
}

pub fn vec4_scale_test() {
  let v = Vec4(1.0, 2.0, 3.0, 4.0)
  let result = vec4.scale(v, 2.0)

  should.equal(result.x, 2.0)
  should.equal(result.y, 4.0)
  should.equal(result.z, 6.0)
  should.equal(result.w, 8.0)
}

pub fn vec4_dot_test() {
  let a = Vec4(1.0, 0.0, 0.0, 0.0)
  let b = Vec4(1.0, 0.0, 0.0, 0.0)
  vec4.dot(a, b) |> should.equal(1.0)

  let c = Vec4(1.0, 2.0, 3.0, 4.0)
  let d = Vec4(1.0, 1.0, 1.0, 1.0)
  vec4.dot(c, d) |> should.equal(10.0)
}

pub fn vec4_length_test() {
  let v = Vec4(3.0, 0.0, 4.0, 0.0)
  vec4.length(v) |> should.equal(5.0)

  vec4.length(vec4.zero) |> should.equal(0.0)
}

pub fn vec4_length_sq_test() {
  let v = Vec4(2.0, 2.0, 2.0, 2.0)
  vec4.length_sq(v) |> should.equal(16.0)
}

pub fn vec4_distance_test() {
  let a = Vec4(0.0, 0.0, 0.0, 0.0)
  let b = Vec4(3.0, 4.0, 0.0, 0.0)
  vec4.distance(a, b) |> should.equal(5.0)
}

pub fn vec4_normalize_test() {
  let v = Vec4(3.0, 0.0, 4.0, 0.0)
  let normalized = vec4.normalize(v)

  // length should be 1
  let len = vec4.length(normalized)
  should.be_true(float.loosely_equals(len, 1.0, 0.0001))

  // direction preserved
  should.be_true(float.loosely_equals(normalized.x, 0.6, 0.0001))
  should.be_true(float.loosely_equals(normalized.z, 0.8, 0.0001))
}

pub fn vec4_normalize_zero_test() {
  // Normalizing zero vector should return zero
  let result = vec4.normalize(vec4.zero)
  should.equal(result, vec4.zero)
}

pub fn vec4_lerp_test() {
  let a = Vec4(0.0, 0.0, 0.0, 0.0)
  let b = Vec4(10.0, 10.0, 10.0, 10.0)

  let mid = vec4.lerp(a, b, 0.5)
  should.equal(mid.x, 5.0)
  should.equal(mid.y, 5.0)

  let start = vec4.lerp(a, b, 0.0)
  should.equal(start, a)

  let end = vec4.lerp(a, b, 1.0)
  should.equal(end, b)
}

pub fn vec4_clamp_test() {
  let v = Vec4(2.0, -2.0, 0.5, 1.5)
  let min = Vec4(-1.0, -1.0, -1.0, 0.0)
  let max = Vec4(1.0, 1.0, 1.0, 1.0)

  let result = vec4.clamp(v, min, max)
  should.equal(result.x, 1.0)
  // clamped from 2.0
  should.equal(result.y, -1.0)
  // clamped from -2.0
  should.equal(result.z, 0.5)
  // unchanged
  should.equal(result.w, 1.0)
  // clamped from 1.5
}

pub fn vec4_clamp_pad_test() {
  let v = Vec4(2.0, -2.0, 0.5, -0.5)
  let result = vec4.clamp_pad(v)

  should.equal(result.x, 1.0)
  // PAD max
  should.equal(result.y, -1.0)
  // PAD min
  should.equal(result.z, 0.5)
  // unchanged
  should.equal(result.w, 0.0)
  // intensity min is 0
}

pub fn vec4_from_pad_test() {
  let v = vec4.from_pad(0.5, -0.3, 0.8, 0.9)
  should.equal(v.x, 0.5)
  should.equal(v.y, -0.3)
  should.equal(v.z, 0.8)
  should.equal(v.w, 0.9)
}

pub fn vec4_min_max_test() {
  let a = Vec4(1.0, 5.0, 3.0, 7.0)
  let b = Vec4(2.0, 3.0, 4.0, 6.0)

  let min_result = vec4.min(a, b)
  should.equal(min_result, Vec4(1.0, 3.0, 3.0, 6.0))

  let max_result = vec4.max(a, b)
  should.equal(max_result, Vec4(2.0, 5.0, 4.0, 7.0))
}

// =============================================================================
// BODY TESTS
// =============================================================================

pub fn body_new_dynamic_test() {
  let pos = Vec4(0.5, 0.5, 0.5, 0.5)
  let g = glyph.new([128, 128, 128, 128])
  let b = body.new_dynamic(1, pos, g, 1.0, 0)

  should.equal(b.id, 1)
  should.equal(b.position, pos)
  should.equal(b.velocity, vec4.zero)
  should.equal(b.motion_type, body.Dynamic)
  should.equal(b.sleep_state, body.Awake)
  should.equal(b.mass, 1.0)
}

pub fn body_new_static_test() {
  let pos = Vec4(0.0, 0.0, 0.0, 1.0)
  let g = glyph.new([255, 0, 0, 128])
  let b = body.new_static(1, pos, g, 0)

  should.equal(b.motion_type, body.Static)
  should.equal(b.mass, 1000.0)
  // High mass for static
}

pub fn body_is_dynamic_test() {
  let g = glyph.new([128, 128, 128, 128])
  let dynamic = body.new_dynamic(1, vec4.zero, g, 1.0, 0)
  let static_body = body.new_static(2, vec4.zero, g, 0)

  body.is_dynamic(dynamic) |> should.be_true
  body.is_dynamic(static_body) |> should.be_false
}

pub fn body_is_sleeping_test() {
  let g = glyph.new([128, 128, 128, 128])
  let b = body.new_dynamic(1, vec4.zero, g, 1.0, 0)

  body.is_sleeping(b) |> should.be_false

  let sleeping = body.put_to_sleep(b)
  body.is_sleeping(sleeping) |> should.be_true
}

pub fn body_wake_up_test() {
  let g = glyph.new([128, 128, 128, 128])
  let b = body.new_dynamic(1, vec4.zero, g, 1.0, 0)
  let sleeping = body.put_to_sleep(b)
  let awake = body.wake_up(sleeping, 100)

  body.is_sleeping(awake) |> should.be_false
  should.equal(awake.last_accessed, 100)
  should.equal(awake.inactive_ticks, 0)
}

pub fn body_integrate_dynamic_test() {
  let g = glyph.new([128, 128, 128, 128])
  let b =
    body.MemoryBody(
      ..body.new_dynamic(1, vec4.zero, g, 1.0, 0),
      velocity: Vec4(0.1, 0.1, 0.0, 0.0),
    )

  let dt = 1.0
  let integrated = body.integrate(b, dt)

  // Position should change by velocity * dt
  should.be_true(float.loosely_equals(integrated.position.x, 0.1, 0.0001))
  should.be_true(float.loosely_equals(integrated.position.y, 0.1, 0.0001))
}

pub fn body_integrate_static_unchanged_test() {
  let g = glyph.new([128, 128, 128, 128])
  let b =
    body.MemoryBody(
      ..body.new_static(1, vec4.zero, g, 0),
      velocity: Vec4(1.0, 1.0, 1.0, 1.0),
    )

  let integrated = body.integrate(b, 1.0)

  // Static body should not move
  should.equal(integrated.position, vec4.zero)
}

pub fn body_integrate_clamps_position_test() {
  let g = glyph.new([128, 128, 128, 128])
  let pos = Vec4(0.9, 0.0, 0.0, 0.5)
  let b =
    body.MemoryBody(
      ..body.new_dynamic(1, pos, g, 1.0, 0),
      velocity: Vec4(0.5, 0.0, 0.0, 0.0),
      // Would go to 1.4
    )

  let integrated = body.integrate(b, 1.0)

  // Should be clamped to PAD max (1.0)
  should.equal(integrated.position.x, 1.0)
}

pub fn body_apply_force_test() {
  let g = glyph.new([128, 128, 128, 128])
  let b = body.new_dynamic(1, vec4.zero, g, 2.0, 0)
  // mass = 2.0
  let force = Vec4(4.0, 0.0, 0.0, 0.0)
  let dt = 1.0

  let result = body.apply_force(b, force, dt)

  // acceleration = force / mass = 4 / 2 = 2
  // velocity = old_velocity + acceleration * dt = 0 + 2 * 1 = 2
  should.be_true(float.loosely_equals(result.velocity.x, 2.0, 0.0001))
}

pub fn body_apply_force_static_unchanged_test() {
  let g = glyph.new([128, 128, 128, 128])
  let b = body.new_static(1, vec4.zero, g, 0)
  let force = Vec4(100.0, 100.0, 100.0, 100.0)

  let result = body.apply_force(b, force, 1.0)

  // Static body should not change velocity
  should.equal(result.velocity, vec4.zero)
}

pub fn body_apply_damping_test() {
  let g = glyph.new([128, 128, 128, 128])
  let b =
    body.MemoryBody(
      ..body.new_dynamic(1, vec4.zero, g, 1.0, 0),
      velocity: Vec4(10.0, 10.0, 10.0, 10.0),
    )

  let damped = body.apply_damping(b, 0.9)

  should.be_true(float.loosely_equals(damped.velocity.x, 9.0, 0.0001))
  should.be_true(float.loosely_equals(damped.velocity.y, 9.0, 0.0001))
}

pub fn body_tick_inactive_moving_test() {
  let g = glyph.new([128, 128, 128, 128])
  let b =
    body.MemoryBody(
      ..body.new_dynamic(1, vec4.zero, g, 1.0, 0),
      velocity: Vec4(1.0, 0.0, 0.0, 0.0),
      // Moving
      inactive_ticks: 10,
    )

  let result = body.tick_inactive(b)

  // Moving body should reset inactive counter
  should.equal(result.inactive_ticks, 0)
}

pub fn body_tick_inactive_stationary_test() {
  let g = glyph.new([128, 128, 128, 128])
  let b =
    body.MemoryBody(
      ..body.new_dynamic(1, vec4.zero, g, 1.0, 0),
      velocity: vec4.zero,
      // Not moving
      inactive_ticks: 10,
    )

  let result = body.tick_inactive(b)

  // Stationary body should increment counter
  should.equal(result.inactive_ticks, 11)
}

pub fn body_put_to_sleep_zeros_velocity_test() {
  let g = glyph.new([128, 128, 128, 128])
  let b =
    body.MemoryBody(
      ..body.new_dynamic(1, vec4.zero, g, 1.0, 0),
      velocity: Vec4(1.0, 2.0, 3.0, 4.0),
    )

  let sleeping = body.put_to_sleep(b)

  should.equal(sleeping.velocity, vec4.zero)
  should.equal(sleeping.sleep_state, body.Sleeping)
}

// =============================================================================
// INTEGRATION TESTS (vec4 + body)
// =============================================================================

pub fn physics_simulation_step_test() {
  // Simulate a body with gravity
  let g = glyph.new([128, 128, 128, 128])
  let initial_pos = Vec4(0.0, 0.5, 0.0, 0.5)
  let b = body.new_dynamic(1, initial_pos, g, 1.0, 0)

  let gravity = Vec4(0.0, -0.1, 0.0, 0.0)
  let dt = 0.1
  let damping = 0.99

  // One simulation step
  let b1 =
    b
    |> body.apply_force(gravity, dt)
    |> body.apply_damping(damping)
    |> body.integrate(dt)

  // Position should have moved down slightly
  should.be_true(b1.position.y <. initial_pos.y)

  // Velocity should be negative (falling)
  should.be_true(b1.velocity.y <. 0.0)
}

pub fn physics_attraction_test() {
  // Simulate attraction toward a point
  let g = glyph.new([128, 128, 128, 128])
  let pos = Vec4(0.5, 0.0, 0.0, 0.5)
  let b = body.new_dynamic(1, pos, g, 1.0, 0)
  let attractor = Vec4(0.0, 0.0, 0.0, 0.5)

  // Calculate attraction force
  let direction = vec4.sub(attractor, b.position)
  let force = vec4.scale(direction, 0.1)
  // Weak attraction

  let b1 = body.apply_force(b, force, 1.0)

  // Should be moving toward attractor (negative x)
  should.be_true(b1.velocity.x <. 0.0)
}
