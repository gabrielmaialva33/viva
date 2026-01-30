//// Tests for VIVA Sinuca - Brazilian Bar Pool

import gleeunit/should
import gleam/option
import viva/embodied/billiards/sinuca.{
  Black, Blue, Brown, Green, Pink, Red, Shot, White, Yellow,
}
import viva/lifecycle/jolt.{Vec3}

// =============================================================================
// TABLE TESTS
// =============================================================================

pub fn sinuca_table_creation_test() {
  let table = sinuca.new()
  sinuca.balls_on_table(table) |> should.equal(8)
}

pub fn sinuca_has_cue_ball_test() {
  let table = sinuca.new()
  case sinuca.get_cue_ball_position(table) {
    option.Some(Vec3(x, y, _z)) -> {
      should.be_true(x <. 0.0)
      should.be_true(y >. 0.0)
    }
    option.None -> should.fail()
  }
}

pub fn sinuca_target_ball_starts_red_test() {
  let table = sinuca.new()
  table.target_ball |> should.equal(Red)
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

pub fn sinuca_shot_creation_test() {
  let shot = Shot(angle: 1.57, power: 0.5, english: 0.0, elevation: 0.0)
  should.be_true(shot.angle >. 1.5)
  should.be_true(shot.power >. 0.4)
}

pub fn sinuca_color_names_test() {
  sinuca.color_name(White) |> should.equal("White")
  sinuca.color_name(Red) |> should.equal("Red")
  sinuca.color_name(Black) |> should.equal("Black")
}
