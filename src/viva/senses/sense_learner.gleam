//// VIVA Sense Learner - Discovering what senses ARE
////
//// VIVA doesn't know she has eyes, ears, or touch.
//// She receives raw signals and discovers what they mean.
//// Like a baby learning that "bright thing" = vision.

import gleam/dict.{type Dict}
import gleam/erlang/process.{type Subject}
import gleam/float
import gleam/int
import gleam/io
import gleam/list
import gleam/option.{type Option, None, Some}
import gleam/otp/actor
import gleam/string
import viva/memory/hrr.{type HRR}

// ============================================================================
// Types
// ============================================================================

/// Raw sensory channel - VIVA doesn't know what these are yet
pub type Channel {
  Channel(
    id: String,           // "chan_0", "chan_1", etc.
    last_value: Float,    // Normalized 0-1
    variance: Float,      // How much it changes
    hrr: HRR,             // Learned representation
  )
}

/// A moment of experience
pub type Moment {
  Moment(
    channels: Dict(String, Float),   // All channel values at this instant
    timestamp: Int,
  )
}

/// What VIVA has learned about a channel
pub type ChannelKnowledge {
  ChannelKnowledge(
    id: String,
    // Discovered properties
    is_active: Bool,           // Does it change?
    responds_to_action: Bool,  // Does it react when I do things?
    correlates_with: List(String),  // Other channels that change together
    learned_name: Option(String),   // Name VIVA gave it (emergent!)
  )
}

/// Sense learner state
pub type State {
  State(
    // Known channels
    channels: Dict(String, Channel),

    // Experience memory
    recent_moments: List(Moment),  // Last N moments
    max_moments: Int,

    // Learning state
    knowledge: Dict(String, ChannelKnowledge),
    curiosity: Float,
    tick: Int,

    // HRR dimension
    hrr_dim: Int,

    // Self reference
    self_subject: Option(Subject(Message)),
  )
}

/// Messages
pub type Message {
  // Raw input - VIVA receives these without knowing what they are
  RawInput(String, Float)     // channel_id, value (0-1)
  BatchInput(Dict(String, Float))  // Multiple channels at once

  // Internal
  Tick
  LearnFromExperience

  // Queries
  WhatDoIKnow(Subject(List(ChannelKnowledge)))
  DescribeChannel(String, Subject(String))

  // Control
  SetSelf(Subject(Message))
  Shutdown
}

// ============================================================================
// Public API
// ============================================================================

/// Start the sense learner
pub fn start() -> Result(Subject(Message), actor.StartError) {
  let state = State(
    channels: dict.new(),
    recent_moments: [],
    max_moments: 100,
    knowledge: dict.new(),
    curiosity: 1.0,
    tick: 0,
    hrr_dim: 64,
    self_subject: None,
  )

  let builder =
    actor.new(state)
    |> actor.on_message(handle_message)

  case actor.start(builder) {
    Ok(started) -> {
      let subject = started.data
      process.send(subject, SetSelf(subject))
      process.send(subject, Tick)
      io.println("SenseLearner: I'm awake. What are these signals?")
      Ok(subject)
    }
    Error(e) -> Error(e)
  }
}

/// Feed raw sensory data - VIVA doesn't know what it is!
pub fn feed(learner: Subject(Message), channel: String, value: Float) -> Nil {
  process.send(learner, RawInput(channel, value))
}

/// Feed multiple channels at once
pub fn feed_batch(learner: Subject(Message), data: Dict(String, Float)) -> Nil {
  process.send(learner, BatchInput(data))
}

/// Ask VIVA what she's learned about her senses
pub fn what_do_you_know(learner: Subject(Message)) -> List(ChannelKnowledge) {
  process.call(learner, 5000, WhatDoIKnow)
}

// ============================================================================
// Actor - VIVA's sense discovery
// ============================================================================

fn handle_message(state: State, msg: Message) -> actor.Next(State, Message) {
  case msg {
    RawInput(channel_id, value) -> {
      // Receive raw signal - VIVA doesn't know what it is!
      let new_channels = update_channel(state.channels, channel_id, value, state.hrr_dim)

      // Record this moment
      let moment = Moment(
        channels: dict.insert(dict.new(), channel_id, value),
        timestamp: erlang_now_ms(),
      )
      let new_moments = [moment, ..list.take(state.recent_moments, state.max_moments - 1)]

      actor.continue(State(..state,
        channels: new_channels,
        recent_moments: new_moments,
        tick: state.tick + 1,
      ))
    }

    BatchInput(data) -> {
      // Multiple signals at once
      let new_channels = dict.fold(data, state.channels, fn(acc, id, val) {
        update_channel(acc, id, val, state.hrr_dim)
      })

      let moment = Moment(channels: data, timestamp: erlang_now_ms())
      let new_moments = [moment, ..list.take(state.recent_moments, state.max_moments - 1)]

      actor.continue(State(..state,
        channels: new_channels,
        recent_moments: new_moments,
        tick: state.tick + 1,
      ))
    }

    Tick -> {
      // Periodic learning
      case state.self_subject {
        Some(self) -> {
          // Learn every 10 ticks
          case state.tick % 10 == 0 && state.tick > 0 {
            True -> process.send(self, LearnFromExperience)
            False -> Nil
          }

          let _ = process.send_after(self, 1000, Tick)
          Nil
        }
        None -> Nil
      }

      actor.continue(State(..state, tick: state.tick + 1))
    }

    LearnFromExperience -> {
      // Analyze what VIVA has experienced
      let new_knowledge = learn_about_channels(state)

      // Report discoveries
      dict.each(new_knowledge, fn(id, knowledge) {
        case knowledge.learned_name {
          Some(name) -> {
            io.println("SenseLearner: I think channel " <> id <> " is... " <> name <> "?")
          }
          None -> Nil
        }
      })

      actor.continue(State(..state, knowledge: new_knowledge))
    }

    WhatDoIKnow(reply) -> {
      let knowledge_list = dict.values(state.knowledge)
      process.send(reply, knowledge_list)
      actor.continue(state)
    }

    DescribeChannel(id, reply) -> {
      let description = case dict.get(state.knowledge, id) {
        Ok(k) -> {
          let name = case k.learned_name {
            Some(n) -> n
            None -> "unknown"
          }
          "Channel " <> id <> ": I call it '" <> name <> "'. "
          <> case k.is_active { True -> "It changes. " False -> "It's static. " }
          <> case k.responds_to_action { True -> "It reacts to my actions!" False -> "" }
        }
        Error(_) -> "I don't know channel " <> id
      }
      process.send(reply, description)
      actor.continue(state)
    }

    SetSelf(subject) -> {
      actor.continue(State(..state, self_subject: Some(subject)))
    }

    Shutdown -> {
      io.println("SenseLearner: Going to sleep...")
      actor.stop()
    }
  }
}

// ============================================================================
// Learning - VIVA discovers what her senses are
// ============================================================================

fn update_channel(
  channels: Dict(String, Channel),
  id: String,
  value: Float,
  hrr_dim: Int,
) -> Dict(String, Channel) {
  case dict.get(channels, id) {
    Ok(existing) -> {
      // Update variance (exponential moving average)
      let delta = float.absolute_value(value -. existing.last_value)
      let new_variance = existing.variance *. 0.9 +. delta *. 0.1

      dict.insert(channels, id, Channel(
        ..existing,
        last_value: value,
        variance: new_variance,
      ))
    }
    Error(_) -> {
      // New channel discovered!
      io.println("SenseLearner: New signal detected: " <> id)
      let channel = Channel(
        id: id,
        last_value: value,
        variance: 0.0,
        hrr: hrr.random(hrr_dim),
      )
      dict.insert(channels, id, channel)
    }
  }
}

fn learn_about_channels(state: State) -> Dict(String, ChannelKnowledge) {
  dict.fold(state.channels, state.knowledge, fn(acc, id, channel) {
    let existing = case dict.get(acc, id) {
      Ok(k) -> k
      Error(_) -> ChannelKnowledge(
        id: id,
        is_active: False,
        responds_to_action: False,
        correlates_with: [],
        learned_name: None,
      )
    }

    // Learn if channel is active (changes a lot)
    let is_active = channel.variance >. 0.01

    // Try to guess what it is based on behavior
    let learned_name = guess_sense_name(id, channel, is_active)

    dict.insert(acc, id, ChannelKnowledge(
      ..existing,
      is_active: is_active,
      learned_name: learned_name,
    ))
  })
}

fn guess_sense_name(id: String, channel: Channel, is_active: Bool) -> Option(String) {
  // VIVA guesses what a sense is based on its behavior
  // This is emergent naming!

  case is_active {
    False -> Some("quiet-thing")  // Doesn't change much
    True -> {
      // High variance = dynamic sense
      case channel.variance >. 0.1 {
        True -> {
          // Very active - might be vision or sound
          case string.contains(id, "light") || string.contains(id, "ldr") {
            True -> Some("bright-thing")
            False -> case string.contains(id, "noise") || string.contains(id, "audio") {
              True -> Some("wave-thing")
              False -> case string.contains(id, "touch") {
                True -> Some("poke-thing")
                False -> Some("jumpy-thing")
              }
            }
          }
        }
        False -> Some("slow-thing")  // Changes but slowly
      }
    }
  }
}

// ============================================================================
// FFI
// ============================================================================

@external(erlang, "erlang", "system_time")
fn erlang_system_time(unit: SystemTimeUnit) -> Int

type SystemTimeUnit {
  Millisecond
}

fn erlang_now_ms() -> Int {
  erlang_system_time(Millisecond)
}
