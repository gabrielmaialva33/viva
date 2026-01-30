//// Tests for VIVA Billiards 3D with Jolt Physics

import gleeunit/should
import viva/embodied/billiards/table.{
  CueBall, EightBall, Shot, Solid, Stripe,
}
import viva/embodied/billiards/fitness
import viva/embodied/billiards/trainer
import viva/lifecycle/jolt.{Vec3}

// =============================================================================
// TABLE TESTS
// =============================================================================

pub fn table_creation_test() {
  let t = table.new()

  // Should have 16 balls (cue + 15 object balls)
  table.balls_on_table(t)
  |> should.equal(16)
}

pub fn table_has_cue_ball_test() {
  let t = table.new()

  case table.get_cue_ball_position(t) {
    Ok(Vec3(x, y, z)) -> {
      // Cue ball at head spot
      should.be_true(x <. 0.0)  // Left side of table
      should.be_true(y >. 0.0)  // Above surface
      should.be_true(z >. -0.1 && z <. 0.1)  // Near center line
    }
    Error(_) -> should.fail()
  }
}

pub fn table_shoot_moves_cue_ball_test() {
  let t = table.new()

  // Get initial position
  let initial_pos = case table.get_cue_ball_position(t) {
    Ok(pos) -> pos
    Error(_) -> Vec3(0.0, 0.0, 0.0)
  }

  // Shoot towards rack
  let shot = Shot(angle: 0.0, power: 0.8, english: 0.0, elevation: 0.0)
  let t2 = table.shoot(t, shot)

  // Simulate a bit
  let t3 = table.step_n(t2, 10, 1.0 /. 60.0)

  // Cue ball should have moved
  case table.get_cue_ball_position(t3) {
    Ok(Vec3(x, _y, _z)) -> {
      let Vec3(ix, _iy, _iz) = initial_pos
      should.be_true(x >. ix)  // Moved right (towards rack)
    }
    Error(_) -> should.fail()
  }
}

pub fn table_is_settled_initially_test() {
  let t = table.new()

  // New table should be settled (all balls stationary)
  table.is_settled(t)
  |> should.be_true()
}

pub fn table_not_settled_after_shot_test() {
  let t = table.new()
  let shot = Shot(angle: 0.0, power: 1.0, english: 0.0, elevation: 0.0)
  let t2 = table.shoot(t, shot)
  let t3 = table.step(t2, 1.0 /. 60.0)

  // Should not be settled immediately after shot
  table.is_settled(t3)
  |> should.be_false()
}

pub fn table_simulate_until_settled_test() {
  let t = table.new()
  let shot = Shot(angle: 0.0, power: 0.5, english: 0.0, elevation: 0.0)
  let t2 = table.shoot(t, shot)

  // Simulate until settled (or max steps) - give more time for physics
  let t3 = table.simulate_until_settled(t2, 3000)

  // After many steps, should either be settled OR max steps reached
  // This is more a smoke test that the simulation runs without crashing
  table.balls_on_table(t3)
  |> should.equal(16)  // All balls should still be on table (no pockets from weak shot)
}

pub fn table_reset_cue_ball_test() {
  let t = table.new()

  // Move cue ball somewhere
  let shot = Shot(angle: 0.0, power: 0.3, english: 0.0, elevation: 0.0)
  let t2 = table.shoot(t, shot)
  let t3 = table.simulate_until_settled(t2, 500)

  // Reset cue ball
  let t4 = table.reset_cue_ball(t3)

  // Should be back at head spot
  case table.get_cue_ball_position(t4) {
    Ok(Vec3(x, _y, _z)) -> {
      should.be_true(x <. -0.5)  // Near head spot
    }
    Error(_) -> should.fail()
  }
}

// =============================================================================
// FITNESS TESTS
// =============================================================================

pub fn fitness_default_config_test() {
  let config = fitness.default_config()

  should.be_true(config.pocket_weight >. 0.0)
  should.be_true(config.scratch_penalty <. 0.0)
  should.be_true(config.eight_ball_penalty <. 0.0)
}

pub fn fitness_quick_evaluate_test() {
  let t = table.new()
  let shot = Shot(angle: 0.0, power: 0.5, english: 0.0, elevation: 0.0)

  let #(fitness_score, _final_table) = fitness.quick_evaluate(t, shot, 600)

  // Should return some fitness value (could be positive or negative)
  should.be_true(fitness_score >. -100.0)
  should.be_true(fitness_score <. 100.0)
}

// =============================================================================
// TRAINER TESTS
// =============================================================================

pub fn trainer_encode_inputs_test() {
  let t = table.new()
  let inputs = trainer.encode_inputs(t)

  // Should have 6 inputs
  case inputs {
    [a, b, c, d, e, f] -> {
      // All should be normalized to [-1, 1]
      should.be_true(a >=. -1.0 && a <=. 1.0)
      should.be_true(b >=. -1.0 && b <=. 1.0)
      should.be_true(c >=. -1.0 && c <=. 1.0)
      should.be_true(d >=. -1.0 && d <=. 1.0)
      should.be_true(e >=. -1.0 && e <=. 1.0)
      should.be_true(f >=. -1.0 && f <=. 1.0)
    }
    _ -> should.fail()
  }
}

pub fn trainer_decode_outputs_test() {
  let outputs = [0.5, 0.8, 0.25]
  let shot = trainer.decode_outputs(outputs)

  // Angle should be ~PI (0.5 * 2 * PI)
  should.be_true(shot.angle >. 3.0 && shot.angle <. 3.3)

  // Power should be high (0.1 + 0.8 * 0.9 = 0.82)
  should.be_true(shot.power >. 0.8 && shot.power <. 0.9)

  // English should be negative (0.25 * 2 - 1 = -0.5)
  should.be_true(shot.english <. 0.0)
}

pub fn trainer_billiards_neat_config_test() {
  let config = trainer.billiards_neat_config(100)

  should.equal(config.population_size, 100)
  should.equal(config.num_inputs, 6)
  should.equal(config.num_outputs, 3)
}

// =============================================================================
// INTEGRATION TESTS
// =============================================================================

pub fn full_shot_simulation_test() {
  let t = table.new()

  // Break shot - hit the rack hard
  let break_shot = Shot(angle: 0.0, power: 1.0, english: 0.0, elevation: 0.0)
  let t2 = table.shoot(t, break_shot)

  // Simulate until settled
  let t3 = table.simulate_until_settled(t2, 1200)
  let t4 = table.update_pocketed(t3)

  // After break, we might have pocketed some balls
  let pocketed = table.get_pocketed_balls(t4)
  let on_table = table.balls_on_table(t4)

  // Total should be 16
  should.equal(on_table + { case pocketed { [] -> 0 _ -> 1 } }, 16)
}

pub fn ball_type_to_string_test() {
  table.ball_type_to_string(CueBall)
  |> should.equal("Cue")

  table.ball_type_to_string(Solid(3))
  |> should.equal("Solid 3")

  table.ball_type_to_string(Stripe(11))
  |> should.equal("Stripe 11")

  table.ball_type_to_string(EightBall)
  |> should.equal("8-Ball")
}
