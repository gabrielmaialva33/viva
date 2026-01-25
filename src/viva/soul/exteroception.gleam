//// Exteroception - The Sense of "Reading" External Minds
////
//// This module handles the interface with the Neural Glands (LLMs).
//// It treats the LLM output as a sensory input (Exteroception).

import gleam/erlang/atom.{type Atom}
import gleam/erlang/process.{type Subject}
import gleam/otp/actor

// =============================================================================
// EXTERNAL BRIDGE (Elixir/Rust)
// =============================================================================

/// The raw resource pointer to the LLM (Rustler Resource)
pub type LlmResource

@external(erlang, "Elixir.Viva.Llm", "load_model")
pub fn load_model(path: String, gpu_layers: Int) -> Result(LlmResource, Atom)

@external(erlang, "Elixir.Viva.Llm", "predict")
pub fn predict(
  resource: LlmResource,
  prompt: String,
) -> Result(#(String, List(Float)), Atom)

@external(erlang, "Elixir.Viva.Llm", "native_check")
pub fn native_check() -> String

// =============================================================================
// TYPES
// =============================================================================

pub type Message {
  Digest(prompt: String)
}

pub type State {
  State(model: LlmResource)
}

pub type StartError {
  ModelLoadError(Atom)
  ActorError(actor.StartError)
}

// =============================================================================
// ACTOR
// =============================================================================

/// Start exteroception actor
/// First loads model, then starts actor
pub fn start(model_path: String) -> Result(Subject(Message), StartError) {
  case load_model(model_path, 99) {
    Ok(resource) -> {
      let state = State(model: resource)
      let builder =
        actor.new(state)
        |> actor.on_message(handle_message)

      case actor.start(builder) {
        Ok(started) -> Ok(started.data)
        Error(e) -> Error(ActorError(e))
      }
    }
    Error(err) -> Error(ModelLoadError(err))
  }
}

fn handle_message(state: State, msg: Message) -> actor.Next(State, Message) {
  case msg {
    Digest(prompt) -> {
      let _ = predict(state.model, prompt)
      actor.continue(state)
    }
  }
}
