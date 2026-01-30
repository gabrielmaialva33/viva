//// Tests for VIVA Sinuca - Brazilian Bar Pool

import gleeunit/should
import gleam/list
import gleam/option
import viva/embodied/billiards/sinuca.{
  Black, Blue, Brown, Green, Pink, Red, Shot, White, Yellow,
}
import viva/embodied/billiards/sinuca_fitness as fitness
import viva/embodied/billiards/sinuca_trainer as trainer
import viva/lifecycle/jolt.{Vec3}

// =============================================================================
// TABLE TESTS
// =============================================================================

pub fn sinuca_table_creation_test() {
  let table = sinuca.new()

  // Should have 8 balls (cue + 7 colored)
  sinuca.balls_on_table(table)
  |> should.equal(8)
}

pub fn sinuca_has_cue_ball_test() {
  let table = sinuca.new()

  case sinuca.get_cue_ball_position(table) {
    option.Some(Vec3(x, y, _z)) -> {
      // Cue ball at break area (left side)
      should.be_true(x <. 0.0)
      should.be_true(y >. 0.0)
    }
    option.None -> should.fail()
  }
}

pub fn sinuca_target_ball_starts_red_test() {
  let table = sinuca.new()

  // Target ball should be Red (lowest value)
  table.target_ball
  |> should.equal(Red)
}

pub fn sinuca_point_values_test() {
  sinuca.point_value(White) |> should.equal(0)
  sinuca.point_value(Red) |> should.equal(1)
  sinuca.point_value(Yellow) |> should.equal(2)
  sinuca.point_value(Green) |> should.equal(3)
  sinuca.point_value(Brown) |> should.equal(4)
  sinuca.point_value(Blue) |> should.equal(5)
  sinuca.point_value(Pink) |> should.equal(6)
  sinuca.point_value(Black) |> should.equal(7)
}

pub fn sinuca_shoot_moves_cue_ball_test() {
  let table = sinuca.new()

  // Get initial position
  let initial_x = case sinuca.get_cue_ball_position(table) {
    option.Some(Vec3(x, _y, _z)) -> x
    option.None -> 0.0
  }

  // Shoot towards rack
  let shot = Shot(angle: 0.0, power: 0.8, english: 0.0, elevation: 0.0)
  let table2 = sinuca.shoot(table, shot)

  // Simulate a bit
  let table3 = sinuca.step_n(table2, 10, 1.0 /. 60.0)

  // Cue ball should have moved right
  case sinuca.get_cue_ball_position(table3) {
    option.Some(Vec3(x, _y, _z)) -> {
      should.be_true(x >. initial_x)
    }
    option.None -> should.fail()
  }
}

pub fn sinuca_is_settled_initially_test() {
  let table = sinuca.new()

  // New table should be settled
  sinuca.is_settled(table)
  |> should.be_true()
}

pub fn sinuca_color_names_test() {
  sinuca.color_name(White) |> should.equal("White")
  sinuca.color_name(Red) |> should.equal("Red")
  sinuca.color_name(Black) |> should.equal("Black")
}

// =============================================================================
// FITNESS TESTS
// =============================================================================

pub fn sinuca_fitness_default_config_test() {
  let config = fitness.default_config()

  should.be_true(config.target_pocketed >. 0.0)
  should.be_true(config.scratch_penalty >. 0.0)  // Now stored as positive
}

pub fn sinuca_fitness_quick_evaluate_test() {
  let table = sinuca.new()
  let shot = Shot(angle: 0.0, power: 0.5, english: 0.0, elevation: 0.0)

  let #(score, _final) = fitness.quick_evaluate(table, shot, 300)

  // Should return some fitness value
  should.be_true(score >. -100.0)
  should.be_true(score <. 100.0)
}

pub fn sinuca_distance_to_target_test() {
  let table = sinuca.new()

  let dist = fitness.distance_to_target(table)

  // Distance should be positive and reasonable
  should.be_true(dist >. 0.0)
  should.be_true(dist <. 5.0)
}

pub fn sinuca_angle_to_target_test() {
  let table = sinuca.new()

  let angle = fitness.angle_to_target(table)

  // Angle should be roughly towards the rack (positive X)
  should.be_true(angle >. -1.0)
  should.be_true(angle <. 1.0)
}

// =============================================================================
// TRAINER TESTS
// =============================================================================

pub fn sinuca_trainer_encode_inputs_test() {
  let table = sinuca.new()

  // Test basic inputs (8 floats)
  let basic_inputs = trainer.encode_inputs(table, False)
  should.equal(list.length(basic_inputs), 8)

  // Test full inputs (42 floats)
  let full_inputs = trainer.encode_inputs(table, True)
  should.equal(list.length(full_inputs), 42)

  // All should be normalized to reasonable range
  list.each(basic_inputs, fn(x) {
    should.be_true(x >=. -2.5 && x <=. 2.5)
  })
}

pub fn sinuca_trainer_decode_outputs_test() {
  // 4 outputs: angle, power, english, elevation
  let outputs = [0.5, 0.8, 0.25, 0.1]
  let shot = trainer.decode_outputs(outputs)

  // Angle should be ~PI (0.5 * 2 * PI)
  should.be_true(shot.angle >. 3.0 && shot.angle <. 3.3)

  // Power should be high (0.15 + 0.8 * 0.85 = 0.83)
  should.be_true(shot.power >. 0.8)

  // English should be negative ((0.25 * 2 - 1) * 0.8 = -0.4)
  should.be_true(shot.english <. 0.0)

  // Elevation should be small (0.1 * 0.3 = 0.03)
  should.be_true(shot.elevation <. 0.1)
}

pub fn sinuca_trainer_neat_config_test() {
  // Basic mode (8 inputs)
  let config_basic = trainer.sinuca_neat_config(100, False)
  should.equal(config_basic.population_size, 100)
  should.equal(config_basic.num_inputs, 8)
  should.equal(config_basic.num_outputs, 4)

  // Full mode (42 inputs)
  let config_full = trainer.sinuca_neat_config(100, True)
  should.equal(config_full.num_inputs, 42)
  should.equal(config_full.num_outputs, 4)
}

// =============================================================================
// INTEGRATION TESTS
// =============================================================================

pub fn sinuca_full_shot_test() {
  let table = sinuca.new()

  // Break shot
  let break_shot = Shot(angle: 0.0, power: 1.0, english: 0.0, elevation: 0.0)
  let table2 = sinuca.shoot(table, break_shot)

  // Simulate
  let table3 = sinuca.simulate_until_settled(table2, 600)
  let table4 = sinuca.update_pocketed(table3)

  // Should still have balls (might have pocketed some)
  let remaining = sinuca.balls_on_table(table4)
  should.be_true(remaining >= 1)
  should.be_true(remaining <= 8)
}

pub fn sinuca_next_target_ball_test() {
  let table = sinuca.new()

  // Initially target should be Red
  let target = sinuca.next_target_ball(table)
  should.equal(target, Red)
}
