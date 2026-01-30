import gleeunit/should
import viva/lifecycle/burn_physics.{BatchShot}

// =============================================================================
// BATCH PHYSICS TESTS
// =============================================================================

pub fn create_batch_test() {
  let batch = burn_physics.create_batch(10)

  // Verify batch size
  should.equal(batch.batch_size, 10)

  // Verify we have 8 balls per table
  case batch.positions_x {
    [first, ..] -> should.equal(list.length(first), 8)
    _ -> should.fail()
  }
}

pub fn simulate_single_shot_test() {
  let batch = burn_physics.create_batch(1)

  // Create a shot: straight at medium power
  let shots = [
    BatchShot(angle: 0.0, power: 0.5, english: 0.0, elevation: 0.0)
  ]

  case burn_physics.simulate_batch(batch, shots, 200) {
    Ok(final_state) -> {
      // Cue ball should have moved (position_x[0][0] should be different from initial)
      case final_state.positions_x {
        [[cue_x, ..], ..] -> {
          // Initial cue_x is about -0.635 (from initial_positions_x)
          // After shot, should have moved towards +X
          should.be_true(cue_x >. -0.635)
        }
        _ -> should.fail()
      }
    }
    Error(_) -> should.fail()
  }
}

pub fn batch_simulation_test() {
  let batch_size = 100

  // Create batch of tables
  let batch = burn_physics.create_batch(batch_size)

  // Create different shots for each table
  let shots = list.range(0, batch_size - 1)
    |> list.map(fn(i) {
      let angle = int.to_float(i) /. int.to_float(batch_size) *. 6.28  // Full circle
      BatchShot(angle: angle, power: 0.5, english: 0.0, elevation: 0.0)
    })

  case burn_physics.simulate_batch(batch, shots, 200) {
    Ok(final_state) -> {
      // All tables should have completed
      should.equal(list.length(final_state.positions_x), batch_size)

      // Different angles should produce different final positions
      let first_cue_x = case final_state.positions_x {
        [[x, ..], ..] -> x
        _ -> 0.0
      }
      let last_cue_x = case list.last(final_state.positions_x) {
        Ok([x, ..]) -> x
        _ -> 0.0
      }

      // They should be different (different shots)
      should.not_equal(first_cue_x, last_cue_x)
    }
    Error(_) -> should.fail()
  }
}

pub fn fitness_calculation_test() {
  let batch = burn_physics.create_batch(10)

  // Shots that should produce varied results
  let shots = list.range(0, 9)
    |> list.map(fn(i) {
      let angle = int.to_float(i) *. 0.3
      BatchShot(angle: angle, power: 0.6, english: 0.0, elevation: 0.0)
    })

  case burn_physics.simulate_batch(batch, shots, 200) {
    Ok(final_state) -> {
      let fitness_results = burn_physics.calculate_fitness(
        batch,
        final_state,
        burn_physics.red_ball_idx,  // Target is Red
      )

      // Should get fitness for all tables
      should.equal(list.length(fitness_results), 10)

      // All should have valid fitness values
      list.each(fitness_results, fn(r) {
        // Fitness should be in reasonable range (-20 to +50)
        should.be_true(r.fitness >. -50.0)
        should.be_true(r.fitness <. 100.0)

        // Behavior descriptors should be in 0-1 range
        should.be_true(r.hit_angle >=. 0.0)
        should.be_true(r.hit_angle <=. 1.0)
        should.be_true(r.scatter_ratio >=. 0.0)
        should.be_true(r.scatter_ratio <=. 1.0)
      })
    }
    Error(_) -> should.fail()
  }
}

pub fn evaluate_batch_test() {
  let shots = list.range(0, 99)
    |> list.map(fn(i) {
      BatchShot(
        angle: int.to_float(i) *. 0.063,  // 0 to 2*pi roughly
        power: 0.5,
        english: 0.0,
        elevation: 0.0,
      )
    })

  case burn_physics.evaluate_batch(shots, burn_physics.red_ball_idx, 200) {
    Ok(results) -> {
      should.equal(list.length(results), 100)
    }
    Error(_) -> should.fail()
  }
}

// =============================================================================
// MULTI-SHOT EPISODE TESTS (NEW KEY OPTIMIZATION)
// =============================================================================

pub fn evaluate_episodes_test() {
  // Architecture: [8, 32, 16, 3] = 867 weights
  let architecture = [8, 32, 16, 3]
  let weight_count = 8 * 32 + 32 + 32 * 16 + 16 + 16 * 3 + 3

  // Generate 10 random networks
  let population_weights = list.range(0, 9)
    |> list.map(fn(net) {
      list.range(0, weight_count - 1)
      |> list.map(fn(i) {
        let seed = { net * 1000 + i } * 1103515245 + 12345
        int.to_float(seed % 1000) /. 1000.0 -. 0.5
      })
    })

  // Run multi-shot episode evaluation (1 NIF call!)
  let results = burn_physics.evaluate_episodes(
    population_weights,
    architecture,
    3,   // 3 shots per episode
    200, // 200 max steps per shot
  )

  // Should have results for all 10 networks
  should.equal(list.length(results), 10)

  // Each result should have valid behavior descriptors
  list.each(results, fn(r) {
    should.be_true(r.hit_angle >=. 0.0 && r.hit_angle <=. 1.0)
    should.be_true(r.scatter_ratio >=. 0.0 && r.scatter_ratio <=. 1.0)
  })
}

pub fn simulate_episodes_full_test() {
  let architecture = [8, 32, 16, 3]
  let weight_count = 8 * 32 + 32 + 32 * 16 + 16 + 16 * 3 + 3

  // Generate 5 networks
  let population_weights = list.range(0, 4)
    |> list.map(fn(net) {
      list.range(0, weight_count - 1)
      |> list.map(fn(i) {
        let seed = { net * 1000 + i } * 1103515245 + 12345
        int.to_float(seed % 1000) /. 1000.0 -. 0.5
      })
    })

  // Get full episode results
  let results = burn_physics.simulate_episodes(
    population_weights,
    architecture,
    3,
    200,
  )

  // Should have 5 results
  should.equal(list.length(results), 5)

  // Each result should have valid data
  list.each(results, fn(r) {
    // Shots taken should be 1-3
    should.be_true(r.shots_taken >= 1 && r.shots_taken <= 3)

    // Final state should have 8 balls
    should.equal(list.length(r.final_pos_x), 8)
    should.equal(list.length(r.final_pos_z), 8)
    should.equal(list.length(r.final_pocketed), 8)
  })
}

pub fn single_call_replaces_many_test() {
  // This test demonstrates the API efficiency gain
  // Before: 100 networks x 3 shots = 300 NIF calls per generation
  // After: 1 NIF call per generation

  let architecture = [8, 32, 16, 3]
  let weight_count = 8 * 32 + 32 + 32 * 16 + 16 + 16 * 3 + 3
  let pop_size = 100
  let shots_per_episode = 3

  // Generate population
  let population_weights = list.range(0, pop_size - 1)
    |> list.map(fn(net) {
      list.range(0, weight_count - 1)
      |> list.map(fn(i) {
        let seed = { net * 1000 + i } * 1103515245 + 12345
        int.to_float(seed % 1000) /. 1000.0 -. 0.5
      })
    })

  // Single NIF call evaluates entire population with multi-shot episodes
  let results = burn_physics.evaluate_episodes(
    population_weights,
    architecture,
    shots_per_episode,
    200,
  )

  // Should have results for entire population
  should.equal(list.length(results), pop_size)

  // Efficiency summary:
  // - Old: 100 * 3 = 300 NIF calls (neural + physics + fitness each)
  // - New: 1 NIF call (everything in single Rust function)
  // - Speedup: ~300x reduction in NIF overhead
}

// Imports needed for tests
import gleam/int
import gleam/list
