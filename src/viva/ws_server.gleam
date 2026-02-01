//// WebSocket Server for VIVA Billiards Visualization
////
//// Connects burn_physics with Bevy 3D visualization.
//// Uses REAL physics with ball-ball collisions!

import gleam/bytes_tree
import gleam/erlang/process
import gleam/float
import gleam/http/request.{type Request}
import gleam/http/response.{type Response}
import gleam/int
import gleam/json
import gleam/list
import gleam/option.{None, Some}
import mist.{type Connection, type ResponseData}
import viva/embodied/billiards/sinuca_trainer.{type TrainingState}
import viva/neural/neat
import viva_telemetry/log

// External math functions
@external(erlang, "math", "cos")
fn cos(x: Float) -> Float

@external(erlang, "math", "sin")
fn sin(x: Float) -> Float

@external(erlang, "math", "sqrt")
fn sqrt(x: Float) -> Float

// =============================================================================
// PHYSICS CONSTANTS
// =============================================================================

const table_half_length: Float = 1.27
const table_half_width: Float = 0.635
const ball_radius: Float = 0.026
const cue_ball_radius: Float = 0.028
// Rolling friction per frame (~60fps): 1 - (mu_roll * g * dt) ≈ 0.998
const rolling_friction: Float = 0.998
// Sliding friction per frame: 1 - (mu_slide * g * dt) ≈ 0.967
const sliding_friction: Float = 0.967
// Ball-ball restitution (Mathavan 2014)
const ball_restitution: Float = 0.93
// Cushion restitution (K-66 rubber)
const cushion_restitution: Float = 0.75
const pocket_radius: Float = 0.05

// Pocket positions (6 pockets)
fn pocket_positions() -> List(#(Float, Float)) {
  [
    #(-1.27, -0.635),   // Bottom left
    #(-1.27, 0.635),    // Top left
    #(0.0, -0.635),     // Bottom center
    #(0.0, 0.635),      // Top center
    #(1.27, -0.635),    // Bottom right
    #(1.27, 0.635),     // Top right
  ]
}

// =============================================================================
// TYPES
// =============================================================================

pub type Ball {
  Ball(
    id: Int,
    x: Float,
    z: Float,
    vx: Float,
    vz: Float,
    radius: Float,
    active: Bool,
  )
}

pub type GameState {
  GameState(
    balls: List(Ball),
    generation: Int,
    fitness: Float,
    score: Int,
    shot_count: Int,
    timestamp: Float,
  )
}

// =============================================================================
// INITIAL STATE
// =============================================================================

pub fn initial_balls() -> List(Ball) {
  // Brazilian Sinuca Par/Impar - 15 balls + cue ball
  // P1 (VIVA): Odd balls (1,3,5,7,9,11,13,15)
  // P2 (VIVA): Even balls (2,4,6,8,10,12,14)
  let d = ball_radius *. 2.1  // Ball diameter with small gap
  let row = d *. 0.866        // Row offset (sqrt(3)/2)
  let x0 = 0.635              // Rack apex X position

  [
    // Cue ball (tacadeira) - id 0
    Ball(0, -0.635, 0.0, 0.0, 0.0, cue_ball_radius, True),

    // Row 1: apex - Ball 1 (Yellow, odd/P1)
    Ball(1, x0, 0.0, 0.0, 0.0, ball_radius, True),

    // Row 2: 2 balls
    Ball(2, x0 +. row, d *. 0.5, 0.0, 0.0, ball_radius, True),   // Blue, even/P2
    Ball(3, x0 +. row, d *. -0.5, 0.0, 0.0, ball_radius, True),  // Red, odd/P1

    // Row 3: 3 balls
    Ball(4, x0 +. row *. 2.0, d, 0.0, 0.0, ball_radius, True),        // Purple, even/P2
    Ball(8, x0 +. row *. 2.0, 0.0, 0.0, 0.0, ball_radius, True),      // Black (8), even/P2 - center
    Ball(5, x0 +. row *. 2.0, d *. -1.0, 0.0, 0.0, ball_radius, True), // Orange, odd/P1

    // Row 4: 4 balls
    Ball(6, x0 +. row *. 3.0, d *. 1.5, 0.0, 0.0, ball_radius, True),  // Green, even/P2
    Ball(7, x0 +. row *. 3.0, d *. 0.5, 0.0, 0.0, ball_radius, True),  // Maroon, odd/P1
    Ball(9, x0 +. row *. 3.0, d *. -0.5, 0.0, 0.0, ball_radius, True), // Yellow stripe, odd/P1
    Ball(10, x0 +. row *. 3.0, d *. -1.5, 0.0, 0.0, ball_radius, True), // Blue stripe, even/P2

    // Row 5: 5 balls (base)
    Ball(11, x0 +. row *. 4.0, d *. 2.0, 0.0, 0.0, ball_radius, True),  // Red stripe, odd/P1
    Ball(12, x0 +. row *. 4.0, d, 0.0, 0.0, ball_radius, True),         // Purple stripe, even/P2
    Ball(15, x0 +. row *. 4.0, 0.0, 0.0, 0.0, ball_radius, True),       // Maroon stripe, odd/P1 - back center
    Ball(14, x0 +. row *. 4.0, d *. -1.0, 0.0, 0.0, ball_radius, True), // Green stripe, even/P2
    Ball(13, x0 +. row *. 4.0, d *. -2.0, 0.0, 0.0, ball_radius, True), // Orange stripe, odd/P1
  ]
}

pub fn initial_game() -> GameState {
  GameState(
    balls: initial_balls(),
    generation: 0,
    fitness: 0.0,
    score: 0,
    shot_count: 0,
    timestamp: 0.0,
  )
}

// =============================================================================
// PHYSICS - REAL COLLISIONS!
// =============================================================================

/// Step physics with REAL ball-ball collisions
fn step_physics(state: GameState, dt: Float) -> GameState {
  // 1. Move balls
  let moved_balls = list.map(state.balls, fn(ball) {
    case ball.active {
      False -> ball
      True -> {
        // Detect sliding vs rolling (simplified: high speed = sliding)
        let speed = sqrt(ball.vx *. ball.vx +. ball.vz *. ball.vz)
        let fric = case speed >. 0.5 {
          True -> sliding_friction   // High speed = sliding
          False -> rolling_friction  // Low speed = rolling
        }
        Ball(
          ..ball,
          x: ball.x +. ball.vx *. dt,
          z: ball.z +. ball.vz *. dt,
          vx: ball.vx *. fric,
          vz: ball.vz *. fric,
        )
      }
    }
  })

  // 2. Ball-ball collisions
  let collided_balls = do_ball_collisions(moved_balls)

  // 3. Cushion collisions
  let cushion_balls = list.map(collided_balls, fn(ball) {
    case ball.active {
      False -> ball
      True -> collide_cushion(ball)
    }
  })

  // 4. Check pockets
  let #(final_balls, pocketed_count) = check_pockets(cushion_balls)

  GameState(
    ..state,
    balls: final_balls,
    score: state.score + pocketed_count,
    timestamp: state.timestamp +. dt,
  )
}

/// Ball-ball collision detection and response
fn do_ball_collisions(balls: List(Ball)) -> List(Ball) {
  do_ball_collisions_loop(balls, [])
}

fn do_ball_collisions_loop(remaining: List(Ball), processed: List(Ball)) -> List(Ball) {
  case remaining {
    [] -> processed
    [ball, ..rest] -> {
      // Check collision with all other balls
      let #(updated_ball, updated_rest) = collide_with_all(ball, rest, [])
      do_ball_collisions_loop(updated_rest, [updated_ball, ..processed])
    }
  }
}

fn collide_with_all(ball: Ball, others: List(Ball), checked: List(Ball)) -> #(Ball, List(Ball)) {
  case others {
    [] -> #(ball, checked)
    [other, ..rest] -> {
      case ball.active && other.active {
        False -> collide_with_all(ball, rest, [other, ..checked])
        True -> {
          let #(new_ball, new_other) = collide_pair(ball, other)
          collide_with_all(new_ball, rest, [new_other, ..checked])
        }
      }
    }
  }
}

/// Elastic collision between two balls
fn collide_pair(a: Ball, b: Ball) -> #(Ball, Ball) {
  let dx = b.x -. a.x
  let dz = b.z -. a.z
  let dist_sq = dx *. dx +. dz *. dz
  let min_dist = a.radius +. b.radius

  case dist_sq <. min_dist *. min_dist {
    False -> #(a, b)  // No collision
    True -> {
      let dist = sqrt(dist_sq)
      let dist_safe = case dist <. 0.001 { True -> 0.001 False -> dist }

      // Normal vector
      let nx = dx /. dist_safe
      let nz = dz /. dist_safe

      // Relative velocity
      let dvx = a.vx -. b.vx
      let dvz = a.vz -. b.vz

      // Relative velocity along normal
      let rel_vel = dvx *. nx +. dvz *. nz

      case rel_vel >. 0.0 {
        False -> #(a, b)  // Moving apart
        True -> {
          // Impulse (equal mass assumed)
          let impulse = rel_vel *. { 1.0 +. ball_restitution } *. 0.5

          // Apply impulse
          let new_a = Ball(
            ..a,
            vx: a.vx -. impulse *. nx,
            vz: a.vz -. impulse *. nz,
          )
          let new_b = Ball(
            ..b,
            vx: b.vx +. impulse *. nx,
            vz: b.vz +. impulse *. nz,
          )

          // Separate balls
          let overlap = min_dist -. dist_safe
          let sep = overlap *. 0.5 +. 0.001

          let sep_a = Ball(..new_a, x: new_a.x -. sep *. nx, z: new_a.z -. sep *. nz)
          let sep_b = Ball(..new_b, x: new_b.x +. sep *. nx, z: new_b.z +. sep *. nz)

          #(sep_a, sep_b)
        }
      }
    }
  }
}

/// Cushion collision
fn collide_cushion(ball: Ball) -> Ball {
  let x = ball.x
  let z = ball.z
  let r = ball.radius
  let vx = ball.vx
  let vz = ball.vz

  // Left cushion
  let #(x1, vx1) = case x <. { float.negate(table_half_length) +. r } {
    True -> #(float.negate(table_half_length) +. r, float.negate(vx) *. cushion_restitution)
    False -> #(x, vx)
  }

  // Right cushion
  let #(x2, vx2) = case x1 >. { table_half_length -. r } {
    True -> #(table_half_length -. r, float.negate(vx1) *. cushion_restitution)
    False -> #(x1, vx1)
  }

  // Bottom cushion
  let #(z1, vz1) = case z <. { float.negate(table_half_width) +. r } {
    True -> #(float.negate(table_half_width) +. r, float.negate(vz) *. cushion_restitution)
    False -> #(z, vz)
  }

  // Top cushion
  let #(z2, vz2) = case z1 >. { table_half_width -. r } {
    True -> #(table_half_width -. r, float.negate(vz1) *. cushion_restitution)
    False -> #(z1, vz1)
  }

  Ball(..ball, x: x2, z: z2, vx: vx2, vz: vz2)
}

/// Check if balls fell into pockets
fn check_pockets(balls: List(Ball)) -> #(List(Ball), Int) {
  let pockets = pocket_positions()
  let #(new_balls, count) = list.fold(balls, #([], 0), fn(acc, ball) {
    let #(acc_balls, acc_count) = acc
    case ball.active && ball.id != 0 {  // Don't pocket cue ball
      False -> #([ball, ..acc_balls], acc_count)
      True -> {
        let in_pocket = list.any(pockets, fn(p) {
          let #(px, pz) = p
          let dx = ball.x -. px
          let dz = ball.z -. pz
          dx *. dx +. dz *. dz <. pocket_radius *. pocket_radius
        })
        case in_pocket {
          True -> {
            log.info("Ball " <> int.to_string(ball.id) <> " pocketed!", [])
            #([Ball(..ball, active: False, vx: 0.0, vz: 0.0), ..acc_balls], acc_count + 1)
          }
          False -> #([ball, ..acc_balls], acc_count)
        }
      }
    }
  })
  #(list.reverse(new_balls), count)
}

/// Apply a shot to the cue ball
fn apply_shot(state: GameState, angle: Float, power: Float) -> GameState {
  let velocity = power *. 3.0  // Scale power
  let vx = velocity *. cos(angle)
  let vz = velocity *. sin(angle)

  log.info("Shot: angle=" <> float.to_string(angle) <> " power=" <> float.to_string(power), [])

  let new_balls = list.map(state.balls, fn(ball) {
    case ball.id {
      0 -> Ball(..ball, vx: vx, vz: vz)
      _ -> ball
    }
  })

  GameState(..state, balls: new_balls, shot_count: state.shot_count + 1)
}

/// Check if all balls are still
fn balls_are_still(balls: List(Ball)) -> Bool {
  list.all(balls, fn(ball) {
    let speed = sqrt(ball.vx *. ball.vx +. ball.vz *. ball.vz)
    speed <. 0.01
  })
}

// =============================================================================
// JSON ENCODING
// =============================================================================

fn ball_to_json(ball: Ball) -> json.Json {
  json.object([
    #("id", json.int(ball.id)),
    #("position", json.array([ball.x, ball.radius, ball.z], json.float)),
    #("velocity", json.array([ball.vx, 0.0, ball.vz], json.float)),
    #("angular_velocity", json.array([0.0, 0.0, 0.0], json.float)),
    #("active", json.bool(ball.active)),
  ])
}

fn physics_update_json(state: GameState) -> String {
  json.object([
    #("type", json.string("physics")),
    #("timestamp", json.float(state.timestamp)),
    #("balls", json.array(state.balls, ball_to_json)),
  ])
  |> json.to_string
}

fn training_update_json(state: GameState) -> String {
  let remaining = list.filter(state.balls, fn(b) { b.active && b.id != 0 })
  json.object([
    #("type", json.string("training")),
    #("generation", json.int(state.generation)),
    #("fitness", json.float(state.fitness)),
    #("score", json.int(state.score)),
    #("balls_remaining", json.int(list.length(remaining))),
    #("best_fitness", json.float(state.fitness)),
    #("games_played", json.int(state.shot_count)),
  ])
  |> json.to_string
}

// =============================================================================
// WEBSOCKET GAME LOOP
// =============================================================================

pub type WsState {
  WsState(game: GameState, frame: Int, waiting_for_shot: Bool)
}

fn run_game_loop(conn: mist.WebsocketConnection) -> Nil {
  let state = WsState(game: initial_game(), frame: 0, waiting_for_shot: True)
  do_game_loop(conn, state)
}

fn do_game_loop(conn: mist.WebsocketConnection, state: WsState) -> Nil {
  // Step physics
  let new_game = step_physics(state.game, 0.016)  // ~60 FPS

  // Check if we should take a new shot
  let #(final_game, new_waiting) = case state.waiting_for_shot && balls_are_still(new_game.balls) {
    True -> {
      // Take a shot!
      let angle = int.to_float({ state.frame / 3 } % 628) /. 100.0  // Vary angle
      let power = 0.5 +. int.to_float(state.frame % 30) /. 100.0
      let shot_game = apply_shot(new_game, angle, power)
      #(shot_game, False)
    }
    False -> {
      // Check if balls stopped after a shot
      let still = balls_are_still(new_game.balls)
      #(new_game, still && state.game.shot_count > 0)
    }
  }

  // Send physics update
  let physics_json = physics_update_json(final_game)
  let _ = mist.send_text_frame(conn, physics_json)

  // Send training update every 30 frames
  case state.frame % 30 {
    0 -> {
      let training_json = training_update_json(final_game)
      let _ = mist.send_text_frame(conn, training_json)
      Nil
    }
    _ -> Nil
  }

  // Sleep for frame time
  process.sleep(16)

  // Continue loop
  let new_state = WsState(game: final_game, frame: state.frame + 1, waiting_for_shot: new_waiting)
  do_game_loop(conn, new_state)
}

pub fn handle_ws_message(
  state: WsState,
  message: mist.WebsocketMessage(String),
  _conn: mist.WebsocketConnection,
) -> mist.Next(WsState, String) {
  case message {
    mist.Text(text) -> {
      log.info("Received: " <> text, [])
      mist.continue(state)
    }
    mist.Binary(_) -> mist.continue(state)
    mist.Custom(_) -> mist.continue(state)
    mist.Closed | mist.Shutdown -> mist.stop()
  }
}

// =============================================================================
// HTTP SERVER
// =============================================================================

pub fn start(port: Int) -> Result(Nil, String) {
  log.info("Starting VIVA Billiards WebSocket server on port " <> int.to_string(port), [])
  log.info("Physics: REAL ball-ball collisions enabled!", [])
  log.info("Training: NEAT self-play available at /train", [])

  let handler = fn(req: Request(Connection)) -> Response(ResponseData) {
    case request.path_segments(req) {
      [] | ["ws"] -> {
        mist.websocket(
          request: req,
          on_init: fn(conn) {
            log.info("WebSocket client connected! Starting game loop...", [])
            let _ = process.spawn(fn() { run_game_loop(conn) })
            let state = WsState(game: initial_game(), frame: 0, waiting_for_shot: True)
            #(state, None)
          },
          on_close: fn(_state) {
            log.info("WebSocket client disconnected", [])
          },
          handler: handle_ws_message,
        )
      }
      ["train"] -> {
        // Start NEAT training WebSocket
        mist.websocket(
          request: req,
          on_init: fn(conn) {
            log.info("Training client connected! Initializing NEAT...", [])
            let config = sinuca_trainer.fast_config()
            let training_state = sinuca_trainer.init_training(config)
            let _ = process.spawn(fn() { run_training_loop(conn, training_state) })
            let state = TrainWsState(training: training_state, generation: 0, running: True)
            #(state, None)
          },
          on_close: fn(_state) {
            log.info("Training client disconnected", [])
          },
          handler: handle_train_ws_message,
        )
      }
      ["health"] -> {
        response.new(200)
        |> response.set_body(mist.Bytes(bytes_tree.from_string("OK")))
      }
      _ -> {
        let body = "VIVA Billiards WebSocket Server\n\nEndpoints:\n- ws://localhost:" <> int.to_string(port) <> "/ws - Game visualization\n- ws://localhost:" <> int.to_string(port) <> "/train - NEAT self-play training\n- /health - Health check\n\nPhysics: REAL collisions!"
        response.new(200)
        |> response.set_body(mist.Bytes(bytes_tree.from_string(body)))
      }
    }
  }

  case mist.new(handler) |> mist.port(port) |> mist.start {
    Ok(_) -> {
      log.info("Server started! Connect Bevy to ws://localhost:" <> int.to_string(port), [])
      process.sleep_forever()
      Ok(Nil)
    }
    Error(_e) -> {
      log.error("Failed to start server", [])
      Error("Server start failed")
    }
  }
}

pub fn main() {
  let _ = start(9000)
  Nil
}

// =============================================================================
// TRAINING WEBSOCKET
// =============================================================================

pub type TrainWsState {
  TrainWsState(
    training: TrainingState,
    generation: Int,
    running: Bool,
  )
}

/// Run NEAT training loop with WebSocket updates
fn run_training_loop(conn: mist.WebsocketConnection, state: TrainingState) -> Nil {
  do_training_loop(conn, state, 0)
}

fn do_training_loop(
  conn: mist.WebsocketConnection,
  state: TrainingState,
  gen: Int,
) -> Nil {
  // Check if we should continue
  case gen >= state.config.max_generations {
    True -> {
      log.info("Training complete! Best fitness: " <> float.to_string(state.best_fitness), [])
      let final_json = training_complete_json(state)
      let _ = mist.send_text_frame(conn, final_json)
      Nil
    }
    False -> {
      // Run one generation
      let new_state = sinuca_trainer.run_generation(state)

      // Send progress update
      let stats = neat.get_stats(new_state.population)
      let progress_json = training_progress_json(new_state, stats)
      let _ = mist.send_text_frame(conn, progress_json)

      // Log to console
      log.info(
        "Gen " <> int.to_string(gen) <>
        " | Best: " <> float.to_string(stats.best_fitness) <>
        " | Species: " <> int.to_string(stats.num_species), []
      )

      // Small delay between generations
      process.sleep(100)

      // Continue
      do_training_loop(conn, new_state, gen + 1)
    }
  }
}

fn training_progress_json(state: TrainingState, stats: neat.PopulationStats) -> String {
  let best_genome_info = case state.best_genome {
    Some(g) -> neat.genome_to_string(g)
    None -> "No best genome yet"
  }

  json.object([
    #("type", json.string("training_progress")),
    #("generation", json.int(state.generation)),
    #("best_fitness", json.float(state.best_fitness)),
    #("avg_fitness", json.float(stats.avg_fitness)),
    #("num_species", json.int(stats.num_species)),
    #("total_games", json.int(state.total_games)),
    #("avg_nodes", json.float(stats.avg_nodes)),
    #("avg_connections", json.float(stats.avg_connections)),
    #("best_genome", json.string(best_genome_info)),
  ])
  |> json.to_string
}

fn training_complete_json(state: TrainingState) -> String {
  json.object([
    #("type", json.string("training_complete")),
    #("final_generation", json.int(state.generation)),
    #("best_fitness", json.float(state.best_fitness)),
    #("total_games", json.int(state.total_games)),
  ])
  |> json.to_string
}

pub fn handle_train_ws_message(
  state: TrainWsState,
  message: mist.WebsocketMessage(String),
  _conn: mist.WebsocketConnection,
) -> mist.Next(TrainWsState, String) {
  case message {
    mist.Text(text) -> {
      log.info("Training received: " <> text, [])
      // Could add commands like "pause", "resume", "stop" here
      mist.continue(state)
    }
    mist.Binary(_) -> mist.continue(state)
    mist.Custom(_) -> mist.continue(state)
    mist.Closed | mist.Shutdown -> mist.stop()
  }
}
