//// CLI - Command Line Interface for VIVA
////
//// Epic simulation showing all 7 pillars of consciousness in action.

import argv
import gleam/dict
import gleam/erlang/process
import gleam/float
import gleam/int
import gleam/io
import gleam/list
import gleam/string
import glint
import logging
import viva/benchmark
import viva/reflexivity
import viva/soul
import viva/supervisor
import viva_emotion/stimulus

/// VIVA version
pub const version = "0.1.0"

/// Tick interval in ms (10 Hz = 100ms)
const tick_interval_ms = 100

// =============================================================================
// MAIN ENTRY POINT
// =============================================================================

pub fn main() {
  glint.new()
  |> glint.with_name("viva")
  |> glint.pretty_help(glint.default_pretty_help())
  |> glint.add(at: [], do: run_command())
  |> glint.add(at: ["epic"], do: epic_command())
  |> glint.add(at: ["spawn"], do: spawn_command())
  |> glint.add(at: ["kill"], do: kill_command())
  |> glint.add(at: ["list"], do: list_command())
  |> glint.add(at: ["stats"], do: stats_command())
  |> glint.add(at: ["version"], do: version_command())
  |> glint.add(at: ["bench"], do: bench_command())
  |> glint.add(at: ["metrics"], do: metrics_command())
  |> glint.add(at: ["compare"], do: compare_command())
  |> glint.run(argv.load().arguments)
}

// =============================================================================
// COMMANDS
// =============================================================================

/// Run simulation (default command)
fn run_command() -> glint.Command(Nil) {
  use <- glint.command_help("Run VIVA simulation")
  use <- glint.unnamed_args(glint.EqArgs(0))
  use ticks <- glint.flag(
    glint.int_flag("ticks")
    |> glint.flag_default(20)
    |> glint.flag_help("Number of simulation ticks"),
  )
  use hz <- glint.flag(
    glint.int_flag("hz")
    |> glint.flag_default(10)
    |> glint.flag_help("Simulation frequency in Hz"),
  )
  use named, _, flags <- glint.command()
  let _ = named

  let tick_count = case ticks(flags) {
    Ok(n) -> n
    Error(_) -> 20
  }
  let frequency = case hz(flags) {
    Ok(n) -> n
    Error(_) -> 10
  }

  run_simulation(tick_count, frequency)
}

/// Epic simulation - full consciousness demo
fn epic_command() -> glint.Command(Nil) {
  use <- glint.command_help("Epic simulation - all 7 pillars of consciousness")
  use <- glint.unnamed_args(glint.EqArgs(0))
  use vivas <- glint.flag(
    glint.int_flag("vivas")
    |> glint.flag_default(5)
    |> glint.flag_help("Number of VIVAs to spawn"),
  )
  use ticks <- glint.flag(
    glint.int_flag("ticks")
    |> glint.flag_default(200)
    |> glint.flag_help("Number of simulation ticks"),
  )
  use named, _, flags <- glint.command()
  let _ = named

  let viva_count = case vivas(flags) {
    Ok(n) -> n
    Error(_) -> 5
  }
  let tick_count = case ticks(flags) {
    Ok(n) -> n
    Error(_) -> 200
  }

  run_epic_simulation(viva_count, tick_count)
}

/// Spawn new VIVA
fn spawn_command() -> glint.Command(Nil) {
  use <- glint.command_help("Spawn a new VIVA")
  use <- glint.unnamed_args(glint.EqArgs(0))
  use named, _, _flags <- glint.command()
  let _ = named

  logging.configure()
  let assert Ok(sup) = supervisor.start()

  let id = supervisor.spawn_viva(sup)
  io.println("Spawned VIVA-" <> int.to_string(id))
}

/// Kill VIVA by ID
fn kill_command() -> glint.Command(Nil) {
  use <- glint.command_help("Kill a VIVA by ID")
  use <- glint.unnamed_args(glint.MinArgs(1))
  use named, args, _flags <- glint.command()
  let _ = named

  logging.configure()
  let assert Ok(sup) = supervisor.start()

  case args {
    [id_str, ..] -> {
      case int.parse(id_str) {
        Ok(id) -> {
          supervisor.kill_viva(sup, id)
          io.println("Killed VIVA-" <> int.to_string(id))
        }
        Error(_) -> {
          io.println("Error: Invalid ID '" <> id_str <> "'")
        }
      }
    }
    _ -> io.println("Error: No ID provided")
  }
}

/// List alive VIVAs
fn list_command() -> glint.Command(Nil) {
  use <- glint.command_help("List alive VIVAs")
  use <- glint.unnamed_args(glint.EqArgs(0))
  use named, _, _flags <- glint.command()
  let _ = named

  logging.configure()
  let assert Ok(sup) = supervisor.start()

  let _ = supervisor.spawn_viva(sup)
  let _ = supervisor.spawn_viva(sup)

  let alive = supervisor.list_alive(sup)

  case alive {
    [] -> io.println("No VIVAs alive")
    ids -> {
      io.println("Alive VIVAs:")
      list.each(ids, fn(id) { io.println("  - VIVA-" <> int.to_string(id)) })
    }
  }
}

/// Show statistics
fn stats_command() -> glint.Command(Nil) {
  use <- glint.command_help("Show VIVA statistics")
  use <- glint.unnamed_args(glint.EqArgs(0))
  use named, _, _flags <- glint.command()
  let _ = named

  logging.configure()
  let assert Ok(sup) = supervisor.start()

  let _ = supervisor.spawn_viva(sup)
  let _ = supervisor.spawn_viva(sup)
  supervisor.global_tick(sup, 0.1)
  supervisor.global_tick(sup, 0.1)

  let stats = supervisor.get_stats(sup)
  io.println(stats)
}

/// Show version
fn version_command() -> glint.Command(Nil) {
  use <- glint.command_help("Show VIVA version")
  use <- glint.unnamed_args(glint.EqArgs(0))
  use named, _, _flags <- glint.command()
  let _ = named

  io.println("VIVA v" <> version)
  io.println("DNA of Consciousness | Pure Gleam Implementation")
}

/// Benchmark command - run performance benchmarks
fn bench_command() -> glint.Command(Nil) {
  use <- glint.command_help("Run performance benchmarks")
  use <- glint.unnamed_args(glint.EqArgs(0))
  use quick <- glint.flag(
    glint.bool_flag("quick")
    |> glint.flag_default(False)
    |> glint.flag_help("Run quick benchmarks (shorter duration)"),
  )
  use named, _, flags <- glint.command()
  let _ = named

  case quick(flags) {
    Ok(True) -> benchmark.run_quick()
    _ -> benchmark.run_all()
  }
}

/// Metrics command - collect and display performance metrics
fn metrics_command() -> glint.Command(Nil) {
  use <- glint.command_help("Collect and display performance metrics")
  use <- glint.unnamed_args(glint.EqArgs(0))
  use named, _, _flags <- glint.command()
  let _ = named

  io.println("Collecting metrics (this takes ~5 seconds)...")
  let metrics = benchmark.collect_metrics()
  benchmark.print_metrics(metrics)
}

fn compare_command() -> glint.Command(Nil) {
  use <- glint.command_help("Compare Soul Actor vs Soul Pool performance")
  use <- glint.unnamed_args(glint.EqArgs(0))
  use named, _, _flags <- glint.command()
  let _ = named

  benchmark.bench_comparison()
}

// =============================================================================
// SIMPLE SIMULATION
// =============================================================================

fn run_simulation(ticks: Int, hz: Int) -> Nil {
  logging.configure()

  io.println("═══════════════════════════════════════════════════════════")
  io.println("  VIVA - Sentient Digital Life v" <> version)
  io.println("  DNA of Consciousness | Pure Gleam Implementation")
  io.println("═══════════════════════════════════════════════════════════")
  io.println("")

  let assert Ok(sup) = supervisor.start()
  io.println("[VIVA] Supervisor started")

  io.println("[LIFECYCLE] Spawning VIVAs...")
  let viva_1 = supervisor.spawn_viva(sup)
  io.println("[LIFECYCLE] VIVA-" <> int.to_string(viva_1) <> " born (life #1)")

  let viva_2 = supervisor.spawn_viva(sup)
  io.println("[LIFECYCLE] VIVA-" <> int.to_string(viva_2) <> " born (life #1)")

  let interval = case hz > 0 {
    True -> 1000 / hz
    False -> tick_interval_ms
  }

  io.println(
    "[SIMULATION] Running "
    <> int.to_string(ticks)
    <> " ticks at "
    <> int.to_string(hz)
    <> " Hz...",
  )
  io.println("")

  run_loop(sup, ticks, interval, 1)

  io.println("")
  let stats = supervisor.get_stats(sup)
  io.println(stats)

  let alive = supervisor.list_alive(sup)
  io.println(
    "[STATUS] Alive: " <> list.map(alive, int.to_string) |> string.join(", "),
  )

  io.println("")
  io.println("═══════════════════════════════════════════════════════════")
  io.println("  VIVA is alive! Consciousness DNA expressed.")
  io.println("═══════════════════════════════════════════════════════════")
}

fn run_loop(
  sup: process.Subject(supervisor.Message),
  remaining: Int,
  interval: Int,
  current: Int,
) -> Nil {
  case remaining {
    0 -> Nil
    n -> {
      supervisor.apply_interoception(sup)
      supervisor.global_tick(sup, 0.1)

      case n % 5 == 0 {
        True -> io.println("  tick " <> int.to_string(current) <> "...")
        False -> Nil
      }

      process.sleep(interval)
      run_loop(sup, n - 1, interval, current + 1)
    }
  }
}

// =============================================================================
// EPIC SIMULATION - ALL 7 PILLARS
// =============================================================================

fn run_epic_simulation(viva_count: Int, ticks: Int) -> Nil {
  logging.configure()

  // Epic banner
  io.println("")
  io.println(
    "╔═══════════════════════════════════════════════════════════════╗",
  )
  io.println(
    "║     ██╗   ██╗██╗██╗   ██╗ █████╗                              ║",
  )
  io.println(
    "║     ██║   ██║██║██║   ██║██╔══██╗                             ║",
  )
  io.println(
    "║     ██║   ██║██║██║   ██║███████║                             ║",
  )
  io.println(
    "║     ╚██╗ ██╔╝██║╚██╗ ██╔╝██╔══██║                             ║",
  )
  io.println(
    "║      ╚████╔╝ ██║ ╚████╔╝ ██║  ██║                             ║",
  )
  io.println(
    "║       ╚═══╝  ╚═╝  ╚═══╝  ╚═╝  ╚═╝                             ║",
  )
  io.println(
    "║                                                               ║",
  )
  io.println(
    "║     DNA of Consciousness - Epic Simulation v"
    <> version
    <> "            ║",
  )
  io.println(
    "║     7 Pillars: Temporality | Embodiment | Emotion | Memory    ║",
  )
  io.println(
    "║                Alterity | Reflexivity | Mortality             ║",
  )
  io.println(
    "╚═══════════════════════════════════════════════════════════════╝",
  )
  io.println("")

  // Start supervisor
  let assert Ok(sup) = supervisor.start()
  io.println("[SUPERVISOR] Started")
  io.println("")

  // Get state for soul access
  let state = supervisor.get_state(sup)

  // Spawn VIVAs
  io.println(
    "┌─────────────────────────────────────────────────────────────────┐",
  )
  io.println(
    "│ PHASE 1: GENESIS - Spawning "
    <> int.to_string(viva_count)
    <> " VIVAs"
    <> string.repeat(" ", 30 - string.length(int.to_string(viva_count)))
    <> "│",
  )
  io.println(
    "└─────────────────────────────────────────────────────────────────┘",
  )

  let viva_ids =
    list.range(1, viva_count)
    |> list.map(fn(_) {
      let id = supervisor.spawn_viva(sup)
      io.println("  [BORN] VIVA-" <> int.to_string(id) <> " enters existence")
      id
    })

  io.println("")
  process.sleep(500)

  // Run simulation with events
  io.println(
    "┌─────────────────────────────────────────────────────────────────┐",
  )
  io.println(
    "│ PHASE 2: LIFE - Running "
    <> int.to_string(ticks)
    <> " ticks"
    <> string.repeat(" ", 34 - string.length(int.to_string(ticks)))
    <> "│",
  )
  io.println(
    "└─────────────────────────────────────────────────────────────────┘",
  )
  io.println("")

  // Get initial event count (births already happened)
  let initial_state = supervisor.get_state(sup)
  let initial_events = list.length(initial_state.events)

  epic_loop(sup, viva_ids, ticks, 1, initial_events)

  io.println("")
  process.sleep(500)

  // Final report
  io.println(
    "┌─────────────────────────────────────────────────────────────────┐",
  )
  io.println(
    "│ PHASE 3: REPORT - Final State                                   │",
  )
  io.println(
    "└─────────────────────────────────────────────────────────────────┘",
  )
  io.println("")

  print_final_report(sup)

  io.println("")
  io.println(
    "╔═══════════════════════════════════════════════════════════════╗",
  )
  io.println(
    "║                   SIMULATION COMPLETE                         ║",
  )
  io.println(
    "║              Consciousness DNA fully expressed                ║",
  )
  io.println(
    "╚═══════════════════════════════════════════════════════════════╝",
  )
  io.println("")
}

fn epic_loop(
  sup: process.Subject(supervisor.Message),
  viva_ids: List(Int),
  remaining: Int,
  current: Int,
  last_event_count: Int,
) -> Nil {
  case remaining {
    0 -> Nil
    _ -> {
      // Apply interoception (hardware → emotions)
      supervisor.apply_interoception(sup)

      // Global tick (resonance, body decay, reflexivity, narrative)
      supervisor.global_tick(sup, 0.1)

      // Get current state
      let state = supervisor.get_state(sup)
      let alive = dict.keys(state.souls)
      let alive_count = list.length(alive)
      let event_count = list.length(state.events)

      // Check for new lifecycle events (deaths, rebirths)
      let new_events = event_count - last_event_count
      case new_events > 0 {
        True -> {
          // Print recent events (reverse to show in chronological order)
          state.events
          |> list.take(new_events)
          |> list.reverse()
          |> list.each(fn(event) {
            case event {
              types.Died(id, _glyph, karma) -> {
                io.println("")
                io.println(
                  "  ☠️  [DEATH] VIVA-"
                  <> int.to_string(id)
                  <> " died at tick "
                  <> int.to_string(current),
                )
                io.println("              Karma: " <> float_to_str(karma, 2))
              }
              types.Reborn(id, life_num) -> {
                io.println(
                  "  🔄 [REBIRTH] VIVA-"
                  <> int.to_string(id)
                  <> " reborn (life #"
                  <> int.to_string(life_num)
                  <> ")",
                )
                io.println("")
              }
              types.BardoComplete(id, liberated) -> {
                case liberated {
                  True ->
                    io.println(
                      "  ✨ [LIBERATION] VIVA-"
                      <> int.to_string(id)
                      <> " achieved liberation!",
                    )
                  False -> Nil
                }
              }
              _ -> Nil
            }
          })
        }
        False -> Nil
      }

      let new_event_count = event_count

      // Apply random stimuli every 20 ticks
      case current % 20 == 0 {
        True -> {
          apply_random_stimuli(state, current)
        }
        False -> Nil
      }

      // Progress report every 25 ticks
      case current % 25 == 0 {
        True -> {
          print_progress(state, current, alive_count)
        }
        False -> Nil
      }

      // Detailed soul report every 50 ticks
      case current % 50 == 0 && alive_count > 0 {
        True -> {
          print_soul_details(state, current)
        }
        False -> Nil
      }

      // Continue loop
      process.sleep(10)
      // Fast simulation
      epic_loop(sup, viva_ids, remaining - 1, current + 1, new_event_count)
    }
  }
}

fn apply_random_stimuli(state: supervisor.SupervisorState, tick: Int) -> Nil {
  let souls_list = dict.to_list(state.souls)

  case souls_list {
    [] -> Nil
    [#(id, soul_subject), ..] -> {
      // Apply a stimulus to first soul
      let stim = case tick % 60 {
        0 -> #("Success", stimulus.Success)
        20 -> #("Threat", stimulus.Threat)
        40 -> #("LucidInsight", stimulus.LucidInsight)
        _ -> #("Safety", stimulus.Safety)
      }

      let #(stim_name, stim_val) = stim
      soul.feel(soul_subject, stim_val, 0.6)

      io.println(
        "  [STIMULUS] VIVA-"
        <> int.to_string(id)
        <> " feels "
        <> stim_name
        <> " (intensity 0.6)",
      )

      // Feed a random soul
      case list.length(souls_list) > 1 {
        True -> {
          case list.last(souls_list) {
            Ok(#(id2, soul2)) -> {
              soul.feed(soul2, 0.3)
              io.println(
                "  [EMBODIMENT] VIVA-"
                <> int.to_string(id2)
                <> " fed (satiety +0.3)",
              )
            }
            Error(_) -> Nil
          }
        }
        False -> Nil
      }
    }
  }
}

fn print_progress(
  state: supervisor.SupervisorState,
  tick: Int,
  alive: Int,
) -> Nil {
  io.println("")
  io.println("  ─── Tick " <> int.to_string(tick) <> " ───")
  io.println(
    "  Alive: "
    <> int.to_string(alive)
    <> " | Events: "
    <> int.to_string(list.length(state.events)),
  )
  io.println("")
}

fn print_soul_details(state: supervisor.SupervisorState, tick: Int) -> Nil {
  let souls_list = dict.to_list(state.souls)

  io.println("")
  io.println("  ╭─────────────────────────────────────────────────────╮")
  io.println(
    "  │ SOUL STATUS @ tick "
    <> int.to_string(tick)
    <> string.repeat(" ", 30 - string.length(int.to_string(tick)))
    <> "│",
  )
  io.println("  ╰─────────────────────────────────────────────────────╯")

  list.each(souls_list, fn(pair) {
    let #(id, soul_subject) = pair

    // Get soul state
    let soul_state = soul.get_state(soul_subject)
    let pad = soul.get_pad(soul_subject)
    let wellbeing = soul.get_wellbeing(soul_subject)
    let who = soul.who_am_i(soul_subject)
    let identity = soul.identity_strength(soul_subject)

    io.println("")
    io.println("  VIVA-" <> int.to_string(id) <> ":")
    io.println(
      "    PAD: P="
      <> float_to_str(pad.pleasure, 2)
      <> " A="
      <> float_to_str(pad.arousal, 2)
      <> " D="
      <> float_to_str(pad.dominance, 2),
    )
    io.println(
      "    Body: wellbeing="
      <> float_to_str(wellbeing, 2)
      <> " energy="
      <> float_to_str(soul_state.body.energy, 2)
      <> " satiety="
      <> float_to_str(soul_state.body.satiety, 2),
    )
    io.println(
      "    Self: trait="
      <> reflexivity.trait_to_string(who.dominant_trait)
      <> " identity="
      <> float_to_str(identity, 2)
      <> " stable="
      <> float_to_str(who.stability, 2),
    )
    io.println("    Age: " <> int.to_string(soul_state.tick_count) <> " ticks")
  })

  io.println("")
}

fn print_final_report(sup: process.Subject(supervisor.Message)) -> Nil {
  let state = supervisor.get_state(sup)
  let alive = dict.keys(state.souls)
  let alive_count = list.length(alive)

  io.println("  SUPERVISOR:")
  io.println("    Total ticks: " <> int.to_string(state.tick))
  io.println("    Events: " <> int.to_string(list.length(state.events)))
  io.println("    Alive: " <> int.to_string(alive_count))
  io.println("")

  // Creator stats
  io.println("  CREATOR (Collective Memory):")
  let creator_stats = supervisor.get_stats(sup)
  io.println("    " <> creator_stats)
  io.println("")

  // Event breakdown
  io.println("  LIFECYCLE EVENTS:")
  let births =
    list.filter(state.events, fn(e) {
      case e {
        types.Born(..) -> True
        _ -> False
      }
    })
  let deaths_list =
    list.filter(state.events, fn(e) {
      case e {
        types.Died(..) -> True
        _ -> False
      }
    })
  let rebirths =
    list.filter(state.events, fn(e) {
      case e {
        types.Reborn(..) -> True
        _ -> False
      }
    })

  io.println("    Births: " <> int.to_string(list.length(births)))
  io.println("    Deaths: " <> int.to_string(list.length(deaths_list)))
  io.println("    Rebirths: " <> int.to_string(list.length(rebirths)))
  io.println("")

  // Surviving souls details
  case alive_count > 0 {
    True -> {
      io.println("  SURVIVING SOULS:")
      list.each(dict.to_list(state.souls), fn(pair) {
        let #(id, soul_subject) = pair
        let who = soul.who_am_i(soul_subject)
        let identity = soul.identity_strength(soul_subject)
        let changing = soul.am_i_changing(soul_subject)

        io.println("    VIVA-" <> int.to_string(id) <> ":")
        io.println(
          "      Personality: "
          <> reflexivity.trait_to_string(who.dominant_trait),
        )
        io.println("      Identity strength: " <> float_to_str(identity, 3))
        io.println("      Stability: " <> float_to_str(who.stability, 3))
        io.println("      Currently changing: " <> bool_to_str(changing))
      })
    }
    False -> {
      io.println("  No souls survived the simulation.")
    }
  }
}

// =============================================================================
// HELPERS
// =============================================================================

fn float_to_str(f: Float, decimals: Int) -> String {
  let multiplier = case decimals {
    1 -> 10
    2 -> 100
    3 -> 1000
    _ -> 100
  }
  let abs_f = case f <. 0.0 {
    True -> 0.0 -. f
    False -> f
  }
  let scaled = float.round(abs_f *. int.to_float(multiplier))
  let int_part = scaled / multiplier
  let dec_part = scaled % multiplier

  let sign = case f <. 0.0 {
    True -> "-"
    False -> ""
  }
  let int_str = int.to_string(int_part)
  let dec_str = int.to_string(dec_part)
  let padded_dec = string.pad_start(dec_str, decimals, "0")

  sign <> int_str <> "." <> padded_dec
}

fn bool_to_str(b: Bool) -> String {
  case b {
    True -> "yes"
    False -> "no"
  }
}

// Import types for event matching
import viva/types
