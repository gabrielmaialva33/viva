//// Soul - The heart of VIVA
////
//// Actor that maintains emotional state (PAD) and evolves via O-U dynamics.
//// Now integrated with Glyph, KarmaBank, and full lifecycle.

import gleam/erlang/process.{type Subject}
import gleam/int
import gleam/option.{type Option, None, Some}
import gleam/otp/actor
import viva/embodiment.{type Body, type BodyStimulus}
import viva/memory.{type GlyphMemory, type KarmaBank}
import viva/narrative.{type NarrativeMemory}
import viva/reflexivity.{type SelfModel}
import viva/types.{type VivaConfig, type VivaId, type VivaSnapshot}
import viva_emotion
import viva_emotion/emotion
import viva_emotion/pad.{type Pad}
import viva_emotion/stimulus.{type Stimulus}
import viva_glyph/glyph.{type Glyph}

// =============================================================================
// TYPES
// =============================================================================

/// Soul internal state
pub type SoulState {
  SoulState(
    /// Unique ID of this VIVA
    id: VivaId,
    /// Emotional state (PAD + config + tick)
    emotional: viva_emotion.EmotionalState,
    /// Current Glyph (compact state representation)
    current_glyph: Glyph,
    /// Karma bank (memories)
    karma_bank: KarmaBank,
    /// Physical body (needs, wellbeing)
    body: Body,
    /// Processed tick counter
    tick_count: Int,
    /// Whether alive
    alive: Bool,
    /// Name/identity
    name: String,
    /// Life number (reincarnation)
    life_number: Int,
    /// Current context (for memory)
    context_glyph: Glyph,
    /// Self-model (reflexivity)
    self_model: SelfModel,
    /// Narrative memory (causal links)
    narrative: NarrativeMemory,
  )
}

/// Messages the Soul can receive
pub type Message {

  // === Commands (fire-and-forget) ===
  /// Feel a stimulus with intensity
  Feel(stimulus: Stimulus, intensity: Float)

  /// Emotional evolution tick (dt in seconds)
  Tick(dt: Float)

  /// Apply interoception delta
  ApplyDelta(pleasure: Float, arousal: Float, dominance: Float)

  /// Update context
  SetContext(context: Glyph)

  /// Die (initiates death process)
  Die

  /// Rebirth with new config
  Rebirth(config: VivaConfig)

  /// Apply stimulus to body (feed, rest, energize)
  ApplyBodyStimulus(stimulus: BodyStimulus)

  // === Queries (request-reply) ===
  /// Get current PAD
  GetPad(reply: Subject(Pad))

  /// Get complete state
  GetState(reply: Subject(SoulState))

  /// Get snapshot (for resonance)
  GetSnapshot(reply: Subject(VivaSnapshot))

  /// Check if alive
  IsAlive(reply: Subject(Bool))

  /// Get emotional classification
  GetEmotion(reply: Subject(String))

  /// Get current glyph
  GetGlyph(reply: Subject(Glyph))

  /// Get karma bank (for bardo)
  GetKarmaBank(reply: Subject(KarmaBank))

  /// Get body state
  GetBody(reply: Subject(Body))

  /// Get self-model
  GetSelfModel(reply: Subject(SelfModel))

  /// Introspect (observe self vs self-model)
  Introspect(reply: Subject(reflexivity.Introspection))

  /// Who am I?
  WhoAmI(reply: Subject(reflexivity.SelfDescription))

  /// Get memories similar to current state
  Recall(limit: Int, reply: Subject(List(GlyphMemory)))
}

// =============================================================================
// ACTOR API
// =============================================================================

/// Start Soul with default configuration
pub fn start(id: VivaId) -> Result(Subject(Message), actor.StartError) {
  start_with_config(id, types.default_config("VIVA-" <> int.to_string(id)))
}

/// Start Soul with custom config
pub fn start_with_config(
  id: VivaId,
  config: VivaConfig,
) -> Result(Subject(Message), actor.StartError) {
  let state = init(id, config)
  let builder =
    actor.new(state)
    |> actor.on_message(handle_message)

  case actor.start(builder) {
    Ok(started) -> Ok(started.data)
    Error(e) -> Error(e)
  }
}

/// Send stimulus to Soul
pub fn feel(soul: Subject(Message), stimulus: Stimulus, intensity: Float) -> Nil {
  process.send(soul, Feel(stimulus, intensity))
}

/// Send evolution tick
pub fn tick(soul: Subject(Message), dt: Float) -> Nil {
  process.send(soul, Tick(dt))
}

/// Apply interoception delta
pub fn apply_delta(
  soul: Subject(Message),
  pleasure: Float,
  arousal: Float,
  dominance: Float,
) -> Nil {
  process.send(soul, ApplyDelta(pleasure, arousal, dominance))
}

/// Set current context
pub fn set_context(soul: Subject(Message), context: Glyph) -> Nil {
  process.send(soul, SetContext(context))
}

/// Get current PAD (synchronous, 1s timeout)
pub fn get_pad(soul: Subject(Message)) -> Pad {
  process.call(soul, 1000, fn(reply) { GetPad(reply) })
}

/// Get complete state (synchronous)
pub fn get_state(soul: Subject(Message)) -> SoulState {
  process.call(soul, 1000, fn(reply) { GetState(reply) })
}

/// Get snapshot for resonance
pub fn get_snapshot(soul: Subject(Message)) -> VivaSnapshot {
  process.call(soul, 1000, fn(reply) { GetSnapshot(reply) })
}

/// Check if alive
pub fn is_alive(soul: Subject(Message)) -> Bool {
  process.call(soul, 1000, fn(reply) { IsAlive(reply) })
}

/// Get classified emotion
pub fn get_emotion(soul: Subject(Message)) -> String {
  process.call(soul, 1000, fn(reply) { GetEmotion(reply) })
}

/// Get current glyph
pub fn get_glyph(soul: Subject(Message)) -> Glyph {
  process.call(soul, 1000, fn(reply) { GetGlyph(reply) })
}

/// Get karma bank
pub fn get_karma_bank(soul: Subject(Message)) -> KarmaBank {
  process.call(soul, 1000, fn(reply) { GetKarmaBank(reply) })
}

/// Search similar memories
pub fn recall(soul: Subject(Message), limit: Int) -> List(GlyphMemory) {
  process.call(soul, 1000, fn(reply) { Recall(limit, reply) })
}

/// Kill the Soul
pub fn kill(soul: Subject(Message)) -> Nil {
  process.send(soul, Die)
}

/// Rebirth Soul with new config
pub fn rebirth(soul: Subject(Message), config: VivaConfig) -> Nil {
  process.send(soul, Rebirth(config))
}

/// Apply body stimulus (feed, rest, energize)
pub fn body_stimulus(soul: Subject(Message), stimulus: BodyStimulus) -> Nil {
  process.send(soul, ApplyBodyStimulus(stimulus))
}

/// Feed the body
pub fn feed(soul: Subject(Message), amount: Float) -> Nil {
  process.send(soul, ApplyBodyStimulus(embodiment.Feed(amount)))
}

/// Rest the body
pub fn rest(soul: Subject(Message), amount: Float) -> Nil {
  process.send(soul, ApplyBodyStimulus(embodiment.Rest(amount)))
}

/// Energize the body
pub fn energize(soul: Subject(Message), amount: Float) -> Nil {
  process.send(soul, ApplyBodyStimulus(embodiment.Energize(amount)))
}

/// Get body state
pub fn get_body(soul: Subject(Message)) -> Body {
  process.call(soul, 1000, fn(reply) { GetBody(reply) })
}

/// Get body wellbeing (0.0-1.0)
pub fn get_wellbeing(soul: Subject(Message)) -> Float {
  let body = process.call(soul, 1000, fn(reply) { GetBody(reply) })
  embodiment.wellbeing(body)
}

/// Is body suffering?
pub fn is_suffering(soul: Subject(Message)) -> Bool {
  let body = process.call(soul, 1000, fn(reply) { GetBody(reply) })
  embodiment.is_suffering(body)
}

/// Get self-model
pub fn get_self_model(soul: Subject(Message)) -> SelfModel {
  process.call(soul, 1000, fn(reply) { GetSelfModel(reply) })
}

/// Introspect: observe self vs self-model
pub fn introspect(soul: Subject(Message)) -> reflexivity.Introspection {
  process.call(soul, 1000, fn(reply) { Introspect(reply) })
}

/// Who am I? Get self-description
pub fn who_am_i(soul: Subject(Message)) -> reflexivity.SelfDescription {
  process.call(soul, 1000, fn(reply) { WhoAmI(reply) })
}

/// Get identity strength (0.0 = undefined, 1.0 = strong sense of self)
pub fn identity_strength(soul: Subject(Message)) -> Float {
  let self_model = process.call(soul, 1000, fn(reply) { GetSelfModel(reply) })
  reflexivity.identity_strength(self_model)
}

/// Am I changing?
pub fn am_i_changing(soul: Subject(Message)) -> Bool {
  let self_model = process.call(soul, 1000, fn(reply) { GetSelfModel(reply) })
  let state = process.call(soul, 1000, fn(reply) { GetState(reply) })
  reflexivity.am_i_changing(self_model, state.tick_count)
}

// =============================================================================
// INTERNAL
// =============================================================================

fn init(id: VivaId, config: VivaConfig) -> SoulState {
  // Create initial emotional state
  let emotional = viva_emotion.new()

  // Initial glyph neutral or from first inherited
  let initial_glyph = case config.inherited_glyphs {
    [first, ..] -> first
    [] -> glyph.neutral()
  }

  // Initial KarmaBank
  let karma_bank = case config.life_number {
    1 -> memory.new(1)
    n -> memory.from_transcendent(n, [], config.initial_mood)
  }

  // Initialize self-model from initial state
  let initial_pad = viva_emotion.get_pad(emotional)
  let self_model = reflexivity.from_initial(initial_pad, initial_glyph)

  SoulState(
    id: id,
    emotional: emotional,
    current_glyph: initial_glyph,
    karma_bank: karma_bank,
    body: embodiment.new(),
    tick_count: 0,
    alive: True,
    name: config.name,
    life_number: config.life_number,
    context_glyph: glyph.neutral(),
    self_model: self_model,
    narrative: narrative.new(),
  )
}

fn handle_message(
  state: SoulState,
  message: Message,
) -> actor.Next(SoulState, Message) {
  case message {
    // === Commands ===
    Feel(stimulus, intensity) -> {
      // Apply stimulus
      let emotional = viva_emotion.feel(state.emotional, stimulus, intensity)

      // Update glyph
      let new_glyph = pad_to_glyph(viva_emotion.get_pad(emotional))

      // Store experience if intensity is significant
      let karma_bank = case intensity >. 0.3 {
        True -> {
          let trajectory = compute_trajectory(state.current_glyph, new_glyph)
          memory.store(
            state.karma_bank,
            new_glyph,
            state.context_glyph,
            trajectory,
            intensity,
          )
        }
        False -> state.karma_bank
      }

      // Narrative: record causal link if glyph changed significantly
      let new_narrative = case
        intensity >. 0.3 && !glyph.equals(state.current_glyph, new_glyph)
      {
        True -> {
          // Record: previous state CAUSED new state (via stimulus)
          narrative.record_caused(
            state.narrative,
            state.current_glyph,
            new_glyph,
            state.tick_count,
          )
        }
        False -> state.narrative
      }

      actor.continue(
        SoulState(
          ..state,
          emotional: emotional,
          current_glyph: new_glyph,
          karma_bank: karma_bank,
          narrative: new_narrative,
        ),
      )
    }

    Tick(dt) -> {
      let #(emotional, _jumped) = viva_emotion.tick(state.emotional, dt)

      // Tick the body (decay needs, accumulate stress)
      let new_body = embodiment.tick(state.body)

      // Apply body's influence on emotions
      let body_delta = embodiment.to_pad_delta(new_body)
      let current_pad = viva_emotion.get_pad(emotional)
      let body_influenced_pad =
        pad.new(
          current_pad.pleasure +. body_delta.x,
          current_pad.arousal +. body_delta.y,
          current_pad.dominance +. body_delta.z,
        )
      let emotional =
        viva_emotion.EmotionalState(..emotional, pad: body_influenced_pad)

      // Update glyph
      let new_glyph = pad_to_glyph(viva_emotion.get_pad(emotional))

      // Update clear light proximity
      let karma_bank = memory.update_clear_light(state.karma_bank, new_glyph)

      // Karma bank tick
      let karma_bank = memory.tick(karma_bank)

      // Self-observation (reflexivity)
      let current_pad = viva_emotion.get_pad(emotional)
      let new_self_model =
        reflexivity.observe(
          state.self_model,
          current_pad,
          new_glyph,
          state.tick_count + 1,
        )

      // Narrative: record temporal link if glyph changed
      let new_narrative = case glyph.equals(state.current_glyph, new_glyph) {
        True -> narrative.tick(state.narrative)
        // Same state, just decay
        False -> {
          // Record: previous glyph preceded current glyph
          state.narrative
          |> narrative.record_preceded(
            state.current_glyph,
            new_glyph,
            state.tick_count + 1,
          )
          |> narrative.tick()
        }
      }

      let new_state =
        SoulState(
          ..state,
          emotional: emotional,
          current_glyph: new_glyph,
          karma_bank: karma_bank,
          body: new_body,
          self_model: new_self_model,
          narrative: new_narrative,
          tick_count: state.tick_count + 1,
        )
      actor.continue(new_state)
    }

    ApplyDelta(pleasure, arousal, dominance) -> {
      // Apply interoception delta to PAD
      let current_pad = viva_emotion.get_pad(state.emotional)
      let new_pad =
        pad.new(
          current_pad.pleasure +. pleasure,
          current_pad.arousal +. arousal,
          current_pad.dominance +. dominance,
        )

      // Recreate emotional state with new PAD (spread update)
      let emotional =
        viva_emotion.EmotionalState(..state.emotional, pad: new_pad)
      let new_glyph = pad_to_glyph(new_pad)

      actor.continue(
        SoulState(..state, emotional: emotional, current_glyph: new_glyph),
      )
    }

    SetContext(context) -> {
      actor.continue(SoulState(..state, context_glyph: context))
    }

    Die -> {
      // Apply death trauma
      let emotional =
        viva_emotion.feel(state.emotional, stimulus.NearDeath, 1.0)
      let final_glyph = pad_to_glyph(viva_emotion.get_pad(emotional))

      // Mark as dead - supervisor will process bardo
      actor.continue(
        SoulState(
          ..state,
          emotional: emotional,
          current_glyph: final_glyph,
          alive: False,
        ),
      )
    }

    Rebirth(config) -> {
      // Reinitialize with new config
      let new_state = init(state.id, config)
      actor.continue(new_state)
    }

    ApplyBodyStimulus(stimulus) -> {
      // Apply stimulus to body (feed, rest, energize, etc)
      let new_body = embodiment.apply_stimulus(state.body, stimulus)
      actor.continue(SoulState(..state, body: new_body))
    }

    // === Queries ===
    GetPad(reply) -> {
      process.send(reply, viva_emotion.get_pad(state.emotional))
      actor.continue(state)
    }

    GetState(reply) -> {
      process.send(reply, state)
      actor.continue(state)
    }

    GetSnapshot(reply) -> {
      let snapshot =
        types.VivaSnapshot(
          id: state.id,
          pad: viva_emotion.get_pad(state.emotional),
          glyph: state.current_glyph,
          alive: state.alive,
          tick: state.tick_count,
          life_number: state.life_number,
        )
      process.send(reply, snapshot)
      actor.continue(state)
    }

    IsAlive(reply) -> {
      process.send(reply, state.alive)
      actor.continue(state)
    }

    GetEmotion(reply) -> {
      let classified = viva_emotion.classify(state.emotional)
      let emotion_name = emotion.to_string(classified.emotion)
      process.send(reply, emotion_name)
      actor.continue(state)
    }

    GetGlyph(reply) -> {
      process.send(reply, state.current_glyph)
      actor.continue(state)
    }

    GetKarmaBank(reply) -> {
      process.send(reply, state.karma_bank)
      actor.continue(state)
    }

    GetBody(reply) -> {
      process.send(reply, state.body)
      actor.continue(state)
    }

    GetSelfModel(reply) -> {
      process.send(reply, state.self_model)
      actor.continue(state)
    }

    Introspect(reply) -> {
      let current_pad = viva_emotion.get_pad(state.emotional)
      let result =
        reflexivity.introspect(
          state.self_model,
          current_pad,
          state.current_glyph,
          state.tick_count,
        )
      process.send(reply, result)
      actor.continue(state)
    }

    WhoAmI(reply) -> {
      let description = reflexivity.who_am_i(state.self_model)
      process.send(reply, description)
      actor.continue(state)
    }

    Recall(limit, reply) -> {
      let query =
        memory.GlyphQuery(
          glyph: state.current_glyph,
          context_glyph: state.context_glyph,
        )
      let memories = memory.recall(state.karma_bank, query, limit)
      process.send(reply, memories)
      actor.continue(state)
    }
  }
}

// =============================================================================
// HELPERS
// =============================================================================

/// Convert PAD to Glyph
fn pad_to_glyph(p: Pad) -> Glyph {
  // Map PAD [-1, 1] to tokens [0, 255]
  let t1 = float_to_token(p.pleasure)
  let t2 = float_to_token(p.arousal)
  let t3 = float_to_token(p.dominance)
  // 4th token: magnitude (overall intensity)
  let magnitude = abs(p.pleasure) +. abs(p.arousal) +. abs(p.dominance)
  let t4 = float_to_token(magnitude /. 3.0)

  glyph.new([t1, t2, t3, t4])
}

fn float_to_token(f: Float) -> Int {
  // [-1, 1] -> [0, 255]
  let normalized = { f +. 1.0 } /. 2.0
  let clamped = clamp(normalized, 0.0, 1.0)
  float_to_int(clamped *. 255.0)
}

fn clamp(value: Float, min: Float, max: Float) -> Float {
  case value {
    v if v <. min -> min
    v if v >. max -> max
    v -> v
  }
}

fn abs(f: Float) -> Float {
  case f <. 0.0 {
    True -> 0.0 -. f
    False -> f
  }
}

/// Compute trajectory between two glyphs
fn compute_trajectory(from: Glyph, to: Glyph) -> Option(Glyph) {
  // Trajectory = difference between glyphs
  let from_tokens = from.tokens
  let to_tokens = to.tokens

  case from_tokens, to_tokens {
    [f1, f2, f3, f4], [t1, t2, t3, t4] -> {
      // Delta normalized to [0, 255]
      let d1 = clamp_int(128 + { t1 - f1 }, 0, 255)
      let d2 = clamp_int(128 + { t2 - f2 }, 0, 255)
      let d3 = clamp_int(128 + { t3 - f3 }, 0, 255)
      let d4 = clamp_int(128 + { t4 - f4 }, 0, 255)
      Some(glyph.new([d1, d2, d3, d4]))
    }
    _, _ -> None
  }
}

fn clamp_int(value: Int, min: Int, max: Int) -> Int {
  case value {
    v if v < min -> min
    v if v > max -> max
    v -> v
  }
}

@external(erlang, "erlang", "trunc")
fn float_to_int(f: Float) -> Int
