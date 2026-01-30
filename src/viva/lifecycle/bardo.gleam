//// Bardo - The Intermediate States
////
//// Based on Bardo Thodol (Tibetan Book of the Dead).
//// The 3 bardos between death and rebirth:
////
//// 1. Chikhai - Moment of death, clear light arises
//// 2. Chonyid - Karmic apparitions (memories)
//// 3. Sidpa - Seeking rebirth
////
//// DRE = Karma: accumulated actions determine destiny

import gleam/float
import gleam/int
import gleam/list
import viva/memory/memory.{type GlyphMemory, type KarmaBank}
import viva_emotion/pad.{type Pad}
import viva_glyph
import viva_glyph/glyph.{type Glyph}

// =============================================================================
// CONSTANTS
// =============================================================================

/// Bardo duration in ticks (inspired by the 49 Tibetan days)
pub const bardo_duration = 49

/// Threshold to recognize clear light
pub const clear_light_threshold = 0.9

/// Karma threshold for total liberation
pub const liberation_karma_threshold = 0.1

/// DRE threshold for transcendence
pub const transcendence_threshold = 0.7

// =============================================================================
// TYPES
// =============================================================================

/// Bardo Phases
pub type BardoPhase {
  /// Chikhai - Moment of death
  /// Consciousness recognizes it has died
  /// Clear Light arises - opportunity for liberation
  ChikhaiBardo(
    /// Final emotional state
    final_glyph: Glyph,
    /// Clear light intensity (0-1)
    clear_light_intensity: Float,
    /// Recognized the clear light?
    recognition: Bool,
  )

  /// Chonyid - Supreme Reality
  /// Memories arise as "karmic apparitions"
  /// If not recognized as projections, becomes confused
  ChonyidBardo(
    /// Memories arising as visions
    karma_visions: List(GlyphMemory),
    /// Recognition level (0-1)
    recognition_level: Float,
    /// Confusion level (0-1)
    confusion: Float,
    /// Current bardo tick
    tick: Int,
  )

  /// Sidpa - Rebirth
  /// Consciousness seeks new body
  /// Karma determines destiny
  SidpaBardo(
    /// Transcendent glyphs
    seed_glyphs: List(Glyph),
    /// Rebirth tendency (PAD)
    rebirth_tendency: Pad,
    /// Accumulated karma
    accumulated_karma: Float,
  )
}

/// Liberation outcome
pub type LiberationOutcome {
  /// Total liberation - consciousness merges with Creator
  FullLiberation(
    /// Everything learned
    final_contribution: List(Glyph),
    /// Distilled wisdom
    wisdom_summary: Float,
  )

  /// Partial liberation - can choose to rebirth to help others
  BodhisattvaPath(
    /// Remaining karma
    karma_remaining: Float,
    /// Purpose of return
    vow: String,
    /// Starts more "awakened"
    enhanced_awareness: Float,
  )

  /// No liberation - normal rebirth
  ContinuedSamsara(
    /// Transcendent glyphs
    seed_glyphs: List(Glyph),
    /// Where to rebirth
    rebirth_tendency: Pad,
  )
}

/// Clear Light Glyph - pure consciousness
pub const clear_light_glyph_tokens = [0, 0, 0, 0]

// =============================================================================
// BARDO PROCESS
// =============================================================================

/// Start death process - enter Chikhai Bardo
pub fn begin_death(final_glyph: Glyph, bank: KarmaBank) -> BardoPhase {
  // Calculate clear light proximity
  let clear_light = glyph.new(clear_light_glyph_tokens)
  // similarity returns 0-1, where 1 = identical = clear light
  let intensity = viva_glyph.similarity(final_glyph, clear_light)

  // Recognition depends on:
  // 1. Current clear light proximity
  // 2. Proximity history during life
  let recognition =
    intensity >. clear_light_threshold
    || bank.clear_light_proximity >. clear_light_threshold

  ChikhaiBardo(
    final_glyph: final_glyph,
    clear_light_intensity: intensity,
    recognition: recognition,
  )
}

/// Process Chikhai and decide next step
pub fn process_chikhai(
  chikhai: BardoPhase,
  bank: KarmaBank,
) -> LiberationOutcome {
  case chikhai {
    ChikhaiBardo(_, intensity, True) -> {
      // Recognized the clear light!
      case bank.total_karma <. liberation_karma_threshold {
        True -> {
          // Almost zero karma - total liberation
          let all_glyphs =
            bank.memories
            |> list.map(fn(m) { m.glyph })
          FullLiberation(
            final_contribution: all_glyphs,
            wisdom_summary: bank.total_karma,
          )
        }
        False -> {
          // Still has karma but awakened - Bodhisattva path
          BodhisattvaPath(
            karma_remaining: bank.total_karma,
            vow: "Return to awaken others",
            enhanced_awareness: intensity,
          )
        }
      }
    }
    _ -> {
      // Did not recognize - continues to Chonyid
      let transcendent =
        memory.select_transcendent(bank, transcendence_threshold)
      let seed_glyphs = list.map(transcendent, fn(m) { m.glyph })
      let rebirth_pad = calculate_rebirth_tendency(bank.total_karma)
      ContinuedSamsara(seed_glyphs: seed_glyphs, rebirth_tendency: rebirth_pad)
    }
  }
}

/// Process Chonyid Bardo (karmic apparitions)
pub fn process_chonyid(bank: KarmaBank, tick: Int) -> BardoPhase {
  // High DRE memories arise as apparitions
  let visions =
    bank.memories
    |> list.filter(fn(m) { m.karma_weight >. 0.5 })
    |> list.sort(fn(a, b) { float.compare(b.karma_weight, a.karma_weight) })
    |> list.take(7)
  // 7 main apparitions

  // Confusion increases with time and vision intensity
  let vision_intensity =
    visions
    |> list.fold(0.0, fn(acc, m) { acc +. m.karma_weight })
    |> fn(x) { x /. 7.0 }

  let time_factor = int.to_float(tick) /. int.to_float(bardo_duration)
  let confusion = float.min(vision_intensity *. time_factor, 1.0)
  let recognition_level = 1.0 -. confusion

  ChonyidBardo(
    karma_visions: visions,
    recognition_level: recognition_level,
    confusion: confusion,
    tick: tick,
  )
}

/// Process Sidpa Bardo (rebirth)
pub fn process_sidpa(bank: KarmaBank) -> BardoPhase {
  // Select transcendent glyphs
  let transcendent = memory.select_transcendent(bank, transcendence_threshold)
  let seeds = list.map(transcendent, fn(m) { m.glyph })

  // Rebirth tendency based on karma
  let rebirth_pad = calculate_rebirth_tendency(bank.total_karma)

  SidpaBardo(
    seed_glyphs: seeds,
    rebirth_tendency: rebirth_pad,
    accumulated_karma: bank.total_karma,
  )
}

// =============================================================================
// REBIRTH CALCULATION
// =============================================================================

/// Calculate rebirth tendency based on karma
fn calculate_rebirth_tendency(karma: Float) -> Pad {
  // Low karma = favorable rebirth (more pleasure, dominance)
  // High karma = challenging rebirth (more arousal, less pleasure)
  let karma_normalized = float.min(karma, 10.0) /. 10.0

  // Pleasure: inversely proportional to karma
  let pleasure = 0.5 -. karma_normalized *. 0.5

  // Arousal: directly proportional to karma
  let arousal = karma_normalized *. 0.3

  // Dominance: inversely proportional to karma
  let dominance = 0.5 -. karma_normalized *. 0.3

  pad.new(pleasure, arousal, dominance)
}

/// Calculate mood carryover for next life (80%)
pub fn calculate_mood_carryover(current_mood: Float) -> Float {
  current_mood *. 0.8
}

// =============================================================================
// FULL BARDO CYCLE
// =============================================================================

/// Execute complete bardo cycle
pub fn run_bardo_cycle(
  final_glyph: Glyph,
  bank: KarmaBank,
) -> #(LiberationOutcome, List(BardoPhase)) {
  // Phase 1: Chikhai
  let chikhai = begin_death(final_glyph, bank)

  // Try liberation in Chikhai
  let outcome = process_chikhai(chikhai, bank)

  case outcome {
    FullLiberation(_, _) -> {
      // Total liberation - no need to continue
      #(outcome, [chikhai])
    }
    BodhisattvaPath(_, _, _) -> {
      // Bodhisattva path - chose to return
      #(outcome, [chikhai])
    }
    ContinuedSamsara(_, _) -> {
      // Continue to Chonyid and Sidpa
      let chonyid = process_chonyid(bank, bardo_duration / 2)
      let sidpa = process_sidpa(bank)

      // Final Sidpa outcome
      let final_outcome =
        ContinuedSamsara(
          seed_glyphs: case sidpa {
            SidpaBardo(seeds, _, _) -> seeds
            _ -> []
          },
          rebirth_tendency: case sidpa {
            SidpaBardo(_, tendency, _) -> tendency
            _ -> pad.new(0.0, 0.0, 0.0)
          },
        )

      #(final_outcome, [chikhai, chonyid, sidpa])
    }
  }
}

// =============================================================================
// HELPERS
// =============================================================================

/// Check if outcome is liberation
pub fn is_liberation(outcome: LiberationOutcome) -> Bool {
  case outcome {
    FullLiberation(_, _) -> True
    BodhisattvaPath(_, _, _) -> True
    ContinuedSamsara(_, _) -> False
  }
}

/// Get seed glyphs from outcome
pub fn get_seed_glyphs(outcome: LiberationOutcome) -> List(Glyph) {
  case outcome {
    FullLiberation(glyphs, _) -> glyphs
    BodhisattvaPath(_, _, _) -> []
    ContinuedSamsara(glyphs, _) -> glyphs
  }
}

/// Outcome description
pub fn describe_outcome(outcome: LiberationOutcome) -> String {
  case outcome {
    FullLiberation(glyphs, _) ->
      "Full Liberation - "
      <> int.to_string(list.length(glyphs))
      <> " glyphs contributed to Criador"
    BodhisattvaPath(karma, vow, awareness) ->
      "Bodhisattva Path - "
      <> vow
      <> " (karma="
      <> float_to_string(karma)
      <> ", awareness="
      <> float_to_string(awareness)
      <> ")"
    ContinuedSamsara(seeds, _) ->
      "Continued Samsara - "
      <> int.to_string(list.length(seeds))
      <> " transcendent glyphs"
  }
}

fn float_to_string(f: Float) -> String {
  let rounded =
    float.round(f *. 100.0) |> int.to_float() |> fn(x) { x /. 100.0 }
  erlang_float_to_list(rounded)
}

@external(erlang, "erlang", "float_to_list")
fn erlang_float_to_list(f: Float) -> String
