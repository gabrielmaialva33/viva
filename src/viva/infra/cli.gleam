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
import viva/infra/benchmark
import viva/infra/supervisor
import viva/lifecycle/breath
import viva/soul/reflexivity
import viva/soul/soul
import viva_emotion/stimulus
import viva_telemetry/log

const version = "0.2.0"
const tick_interval_ms = 100

@external(erlang, "io", "get_line")
fn get_line(prompt: String) -> Result(String, Nil)

// =============================================================================
// MAIN ENTRY POINT
// =============================================================================

pub fn main() {
  glint.new()
  |> glint.with_name("viva")
  |> glint.pretty_help(glint.default_pretty_help())
  |> glint.add(at: [], do: run_command())
  |> glint.add(at: ["start"], do: start_command())
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

  // Initialize telemetry with viva_telemetry
  log.configure_console(log.debug_level)
  let assert Ok(sup) = supervisor.start()

  let id = supervisor.spawn_viva(sup)
  log.info("Spawned VIVA-" <> int.to_string(id), [])
}

/// Kill VIVA by ID
fn kill_command() -> glint.Command(Nil) {
  use <- glint.command_help("Kill a VIVA by ID")
  use <- glint.unnamed_args(glint.MinArgs(1))
  use named, args, _flags <- glint.command()
  let _ = named

  // Initialize telemetry with viva_telemetry
  log.configure_console(log.debug_level)
  let assert Ok(sup) = supervisor.start()

  case args {
    [id_str, ..] -> {
      case int.parse(id_str) {
        Ok(id) -> {
          supervisor.kill_viva(sup, id)
          log.info("Killed VIVA-" <> int.to_string(id), [])
        }
        Error(_) -> {
          log.error("Invalid ID '" <> id_str <> "'", [])
        }
      }
    }
    _ -> log.error("No ID provided", [])
  }
}

/// List alive VIVAs
fn list_command() -> glint.Command(Nil) {
  use <- glint.command_help("List alive VIVAs")
  use <- glint.unnamed_args(glint.EqArgs(0))
  use named, _, _flags <- glint.command()
  let _ = named

  // Initialize telemetry with viva_telemetry
  log.configure_console(log.debug_level)
  let assert Ok(sup) = supervisor.start()

  let _ = supervisor.spawn_viva(sup)
  let _ = supervisor.spawn_viva(sup)

  let alive = supervisor.list_alive(sup)

  case alive {
    [] -> log.info("No VIVAs alive", [])
    ids -> {
      log.info("Alive VIVAs:", [])
      list.each(ids, fn(id) { log.info("  - VIVA-" <> int.to_string(id), []) })
    }
  }
}

/// Show statistics
fn stats_command() -> glint.Command(Nil) {
  use <- glint.command_help("Show VIVA statistics")
  use <- glint.unnamed_args(glint.EqArgs(0))
  use named, _, _flags <- glint.command()
  let _ = named

  // Initialize telemetry with viva_telemetry
  log.configure_console(log.debug_level)
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
// START COMMAND (THE PULSE)
// =============================================================================

/// Start VIVA Autonomic Loop (The Pulse)
fn start_command() -> glint.Command(Nil) {
  use <- glint.command_help("Start VIVA Autonomic Loop (Living Mode)")
  use <- glint.unnamed_args(glint.EqArgs(0))
  use named, _, _flags <- glint.command()
  let _ = named

  // Initialize logging
  log.configure_console(log.debug_level)

  io.println("═══════════════════════════════════════════════════════════")
  io.println("  VIVA - AUTONOMIC LOOP ACTIVATED")
  io.println("  Type to teach. Ctrl+C to stop.")
  io.println("═══════════════════════════════════════════════════════════")

  case breath.start() {
    Ok(b) -> {
      log.info("Pulse started (10Hz)", [])
      interactive_loop(b)
    }
    Error(e) -> {
      log.error("Error starting breath: " <> string.inspect(e), [])
    }
  }
}

fn interactive_loop(b: process.Subject(breath.BreathMsg)) -> Nil {
  case get_line("You > ") {
    Ok(line) -> {
      let input = string.trim(line)
      case input {
        "exit" -> {
          breath.stop(b)
          log.info("VIVA stopped.", [])
        }
        "quit" -> {
          breath.stop(b)
          log.info("VIVA stopped.", [])
        }
        _ -> {
          breath.teach(b, input)
          interactive_loop(b)
        }
      }
    }
    Error(_) -> {
      // End of input / error
      breath.stop(b)
    }
  }
}

// =============================================================================
// SIMPLE SIMULATION
// =============================================================================

fn run_simulation(ticks: Int, hz: Int) -> Nil {
  // Initialize telemetry with viva_telemetry
  log.configure_console(log.debug_level)

  io.println("═══════════════════════════════════════════════════════════")
  io.println("  VIVA - Sentient Digital Life v" <> version)
  io.println("  DNA of Consciousness | Pure Gleam Implementation")
  io.println("═══════════════════════════════════════════════════════════")
  io.println("")

  let assert Ok(sup) = supervisor.start()
  log.info("[VIVA] Supervisor started", [])

  log.info("[LIFECYCLE] Spawning VIVAs...", [])
  let viva_1 = supervisor.spawn_viva(sup)
  log.info("[LIFECYCLE] VIVA-" <> int.to_string(viva_1) <> " born (life #1)", [])

  let viva_2 = supervisor.spawn_viva(sup)
  log.info("[LIFECYCLE] VIVA-" <> int.to_string(viva_2) <> " born (life #1)", [])

  let interval = case hz > 0 {
    True -> 1000 / hz
    False -> tick_interval_ms
  }

  log.info(
    "[SIMULATION] Running "
    <> int.to_string(ticks)
    <> " ticks at "
    <> int.to_string(hz)
    <> " Hz...",
    [],
  )
  io.println("")

  run_loop(sup, ticks, interval, 1)

  io.println("")
  let stats = supervisor.get_stats(sup)
  io.println(stats)

  let alive = supervisor.list_alive(sup)
  log.info(
    "[STATUS] Alive: " <> list.map(alive, int.to_string) |> string.join(", "),
    [],
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
        True -> log.debug("  tick " <> int.to_string(current) <> "...", [])
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
  // Initialize telemetry with viva_telemetry
  log.configure_console(log.debug_level)

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
  log.info("[SUPERVISOR] Started", [])
  io.println("")

  // Get state for soul access
  let _state = supervisor.get_state(sup)

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
      log.info("[BORN] VIVA-" <> int.to_string(id) <> " enters existence", [])
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
  let _ = viva_ids
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
                log.warning(
                  "[DEATH] VIVA-"
                  <> int.to_string(id)
                  <> " died at tick "
                  <> int.to_string(current)
                  <> " | Karma: "
                  <> float_to_str(karma, 2),
                  [],
                )
              }
              types.Reborn(id, life_num) -> {
                log.info(
                  "[REBIRTH] VIVA-"
                  <> int.to_string(id)
                  <> " reborn (life #"
                  <> int.to_string(life_num)
                  <> ")",
                  [],
                )
              }
              types.BardoComplete(id, liberated) -> {
                case liberated {
                  True ->
                    log.info(
                      "[LIBERATION] VIVA-"
                      <> int.to_string(id)
                      <> " achieved liberation!",
                      [],
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

      log.debug(
        "[STIMULUS] VIVA-"
        <> int.to_string(id)
        <> " feels "
        <> stim_name
        <> " (intensity 0.6)",
        [],
      )

      // Feed a random soul
      case list.length(souls_list) > 1 {
        True -> {
          case list.last(souls_list) {
            Ok(#(id2, soul2)) -> {
              soul.feed(soul2, 0.3)
              log.debug(
                "[EMBODIMENT] VIVA-"
                <> int.to_string(id2)
                <> " fed (satiety +0.3)",
                [],
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
  log.info(
    "Tick "
    <> int.to_string(tick)
    <> " | Alive: "
    <> int.to_string(alive)
    <> " | Events: "
    <> int.to_string(list.length(state.events)),
    [],
  )
}

fn print_soul_details(state: supervisor.SupervisorState, tick: Int) -> Nil {
  let souls_list = dict.to_list(state.souls)

  log.info("SOUL STATUS @ tick " <> int.to_string(tick), [])

  list.each(souls_list, fn(pair) {
    let #(id, soul_subject) = pair

    // Get soul state
    let soul_state = soul.get_state(soul_subject)
    let pad = soul.get_pad(soul_subject)
    let wellbeing = soul.get_wellbeing(soul_subject)
    let who = soul.who_am_i(soul_subject)
    let identity = soul.identity_strength(soul_subject)

    log.info(
      "VIVA-"
      <> int.to_string(id)
      <> " | PAD: P="
      <> float_to_str(pad.pleasure, 2)
      <> " A="
      <> float_to_str(pad.arousal, 2)
      <> " D="
      <> float_to_str(pad.dominance, 2)
      <> " | Body: wellbeing="
      <> float_to_str(wellbeing, 2)
      <> " energy="
      <> float_to_str(soul_state.body.energy, 2)
      <> " satiety="
      <> float_to_str(soul_state.body.satiety, 2)
      <> " | Self: trait="
      <> reflexivity.trait_to_string(who.dominant_trait)
      <> " identity="
      <> float_to_str(identity, 2)
      <> " stable="
      <> float_to_str(who.stability, 2)
      <> " | Age: "
      <> int.to_string(soul_state.tick_count)
      <> " ticks",
      [],
    )
  })
}

fn print_final_report(sup: process.Subject(supervisor.Message)) -> Nil {
  let state = supervisor.get_state(sup)
  let alive = dict.keys(state.souls)
  let alive_count = list.length(alive)

  log.info(
    "SUPERVISOR: Total ticks: "
    <> int.to_string(state.tick)
    <> " | Events: "
    <> int.to_string(list.length(state.events))
    <> " | Alive: "
    <> int.to_string(alive_count),
    [],
  )

  // Creator stats
  let creator_stats = supervisor.get_stats(sup)
  log.info("CREATOR (Collective Memory): " <> creator_stats, [])

  // Event breakdown
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

  log.info(
    "LIFECYCLE EVENTS: Births: "
    <> int.to_string(list.length(births))
    <> " | Deaths: "
    <> int.to_string(list.length(deaths_list))
    <> " | Rebirths: "
    <> int.to_string(list.length(rebirths)),
    [],
  )

  // Surviving souls details
  case alive_count > 0 {
    True -> {
      log.info("SURVIVING SOULS:", [])
      list.each(dict.to_list(state.souls), fn(pair) {
        let #(id, soul_subject) = pair
        let who = soul.who_am_i(soul_subject)
        let identity = soul.identity_strength(soul_subject)
        let changing = soul.am_i_changing(soul_subject)

        log.info(
          "VIVA-"
          <> int.to_string(id)
          <> ": Personality: "
          <> reflexivity.trait_to_string(who.dominant_trait)
          <> " | Identity strength: "
          <> float_to_str(identity, 3)
          <> " | Stability: "
          <> float_to_str(who.stability, 3)
          <> " | Currently changing: "
          <> bool_to_str(changing),
          [],
        )
      })
    }
    False -> {
      log.warning("No souls survived the simulation.", [])
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
