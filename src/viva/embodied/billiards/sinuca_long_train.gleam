//// VIVA Sinuca - Extended Training (1 hour)
////
//// Quality training with full inputs and advanced fitness.

import gleam/int
import gleam/io
import gleam/list
import gleam/float
import viva/embodied/billiards/sinuca
import viva/embodied/billiards/sinuca_trainer as trainer
import viva/embodied/billiards/sinuca_fitness as fitness

pub fn main() {
  io.println("=== VIVA Sinuca - Extended Training ===")
  io.println("Duration: ~1 hour | 500 generations")
  io.println("Population: 100 | 8 inputs | 5 shots/episode")
  io.println("")

  let config = trainer.TrainerConfig(
    population_size: 100,
    max_steps_per_shot: 300,
    shots_per_episode: 5,
    log_interval: 10,
    fitness_config: fitness.default_config(),
  )

  // 500 generations should take ~1 hour
  let #(_pop, best) = trainer.train(500, config)

  io.println("")
  io.println("=== Extended Training Complete ===")
  io.println("Best fitness: " <> float_to_string(best.fitness))
  io.println("Network size:")
  io.println("  Nodes: " <> int.to_string(list.length(best.nodes)))
  io.println("  Connections: " <> int.to_string(list.length(best.connections)))

  // Play multiple demo games
  io.println("")
  io.println("=== Demo Games ===")

  io.println("\n--- Game 1 ---")
  let game1 = trainer.play_game(best, 15, True)
  trainer.display_game_results(game1)

  io.println("\n--- Game 2 ---")
  let game2 = trainer.play_game(best, 15, True)
  trainer.display_game_results(game2)

  io.println("\n--- Game 3 ---")
  let game3 = trainer.play_game(best, 15, True)
  trainer.display_game_results(game3)

  // Summary
  let total_pocketed = count_pocketed(game1) + count_pocketed(game2) + count_pocketed(game3)
  let total_fouls = count_fouls(game1) + count_fouls(game2) + count_fouls(game3)

  io.println("")
  io.println("=== Summary (3 games) ===")
  io.println("Total pocketed: " <> int.to_string(total_pocketed))
  io.println("Total fouls: " <> int.to_string(total_fouls))
}

fn count_pocketed(results: List(#(sinuca.Shot, Int, Bool, Int))) -> Int {
  list.fold(results, 0, fn(acc, r) {
    let #(_shot, pocketed, _foul, _combo) = r
    acc + pocketed
  })
}

fn count_fouls(results: List(#(sinuca.Shot, Int, Bool, Int))) -> Int {
  list.fold(results, 0, fn(acc, r) {
    let #(_shot, _pocketed, foul, _combo) = r
    case foul {
      True -> acc + 1
      False -> acc
    }
  })
}

fn float_to_string(f: Float) -> String {
  let rounded = float.round(f *. 100.0)
  int.to_string(rounded / 100) <> "." <> pad_left(int.to_string(int.absolute_value(rounded % 100)), 2, "0")
}

fn pad_left(s: String, len: Int, pad: String) -> String {
  case string_length(s) >= len {
    True -> s
    False -> pad_left(pad <> s, len, pad)
  }
}

@external(erlang, "string", "length")
fn string_length(s: String) -> Int
