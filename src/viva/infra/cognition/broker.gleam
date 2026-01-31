//// Cognitive Broker - Event-driven communication for cognitive systems
////
//// Decouples narrative, reflexivity, and inner_life through explicit events.
//// Prevents circular dependencies between cognitive modules.
////
//// Pattern: Publish-Subscribe for cognitive events
//// "I felt something" → Broker → Narrative + Reflexivity

import gleam/list
import gleam/option.{type Option}
import viva_emotion/pad.{type Pad}
import viva_glyph/glyph.{type Glyph}

// =============================================================================
// COGNITIVE EVENTS
// =============================================================================

/// Events that flow through the cognitive system
pub type CognitiveEvent {
  /// Emotional state changed
  EmotionalShift(from: Pad, to: Pad, magnitude: Float, tick: Int)
  /// Self-observation occurred
  Introspection(drift: Float, within_range: Bool, tick: Int)
  /// Insight was generated
  InsightGenerated(
    dimension: CognitiveDimension,
    direction: ChangeDirection,
    magnitude: Float,
    tick: Int,
  )
  /// Narrative link formed
  NarrativeLinkFormed(
    cause: Glyph,
    effect: Glyph,
    relation: LinkRelation,
    tick: Int,
  )
  /// Identity crisis detected
  CrisisDetected(severity: Float, duration: Int, tick: Int)
  /// Crisis resolved
  CrisisResolved(new_baseline: Glyph, tick: Int)
  /// Meta-cognition level changed
  MetaLevelChanged(from: Int, to: Int, tick: Int)
  /// Thought generated
  ThoughtGenerated(content: String, weight: Float, tick: Int)
}

/// Cognitive dimensions (simplified from reflexivity)
pub type CognitiveDimension {
  Pleasure
  Arousal
  Dominance
}

/// Change direction
pub type ChangeDirection {
  Increasing
  Decreasing
}

/// Link relation types (simplified from narrative)
pub type LinkRelation {
  Caused
  Preceded
  Associated
  Contrasted
}

// =============================================================================
// EVENT BROKER
// =============================================================================

/// Cognitive broker: routes events to handlers
pub type CognitiveBroker {
  CognitiveBroker(
    /// Event history (recent events)
    history: List(CognitiveEvent),
    /// Max history size
    max_history: Int,
    /// Current tick
    tick: Int,
    /// Event handlers
    handlers: List(EventHandler),
  )
}

/// Event handler function type
pub type EventHandler {
  EventHandler(
    /// Handler name
    name: String,
    /// Event filter (which events to handle)
    filter: EventFilter,
    /// Whether handler is active
    active: Bool,
  )
}

/// Event filter
pub type EventFilter {
  /// Handle all events
  AllEvents
  /// Handle specific event types
  EmotionalEvents
  /// Handle introspection events
  IntrospectionEvents
  /// Handle crisis events
  CrisisEvents
  /// Handle narrative events
  NarrativeEvents
}

/// Event processing result
pub type ProcessResult {
  ProcessResult(
    /// Updated broker
    broker: CognitiveBroker,
    /// Responses generated
    responses: List(EventResponse),
  )
}

/// Response to an event
pub type EventResponse {
  EventResponse(handler: String, message: String)
}

// =============================================================================
// CONSTRUCTORS
// =============================================================================

/// Create new cognitive broker
pub fn new() -> CognitiveBroker {
  CognitiveBroker(
    history: [],
    max_history: 100,
    tick: 0,
    handlers: default_handlers(),
  )
}

/// Create with custom max history
pub fn new_with_history(max: Int) -> CognitiveBroker {
  CognitiveBroker(
    history: [],
    max_history: max,
    tick: 0,
    handlers: default_handlers(),
  )
}

/// Default handlers
fn default_handlers() -> List(EventHandler) {
  [
    EventHandler(name: "narrative", filter: EmotionalEvents, active: True),
    EventHandler(name: "reflexivity", filter: IntrospectionEvents, active: True),
    EventHandler(name: "crisis_monitor", filter: CrisisEvents, active: True),
  ]
}

// =============================================================================
// EVENT PUBLISHING
// =============================================================================

/// Publish an event to the broker
pub fn publish(
  broker: CognitiveBroker,
  event: CognitiveEvent,
) -> CognitiveBroker {
  let new_history = [event, ..broker.history]
  let trimmed = list.take(new_history, broker.max_history)

  CognitiveBroker(..broker, history: trimmed, tick: broker.tick + 1)
}

/// Publish emotional shift event
pub fn emit_emotional_shift(
  broker: CognitiveBroker,
  from: Pad,
  to: Pad,
  magnitude: Float,
) -> CognitiveBroker {
  let event =
    EmotionalShift(from: from, to: to, magnitude: magnitude, tick: broker.tick)
  publish(broker, event)
}

/// Publish introspection event
pub fn emit_introspection(
  broker: CognitiveBroker,
  drift: Float,
  within_range: Bool,
) -> CognitiveBroker {
  let event =
    Introspection(drift: drift, within_range: within_range, tick: broker.tick)
  publish(broker, event)
}

/// Publish insight event
pub fn emit_insight(
  broker: CognitiveBroker,
  dimension: CognitiveDimension,
  direction: ChangeDirection,
  magnitude: Float,
) -> CognitiveBroker {
  let event =
    InsightGenerated(
      dimension: dimension,
      direction: direction,
      magnitude: magnitude,
      tick: broker.tick,
    )
  publish(broker, event)
}

/// Publish narrative link event
pub fn emit_narrative_link(
  broker: CognitiveBroker,
  cause: Glyph,
  effect: Glyph,
  relation: LinkRelation,
) -> CognitiveBroker {
  let event =
    NarrativeLinkFormed(
      cause: cause,
      effect: effect,
      relation: relation,
      tick: broker.tick,
    )
  publish(broker, event)
}

/// Publish crisis detected event
pub fn emit_crisis(
  broker: CognitiveBroker,
  severity: Float,
  duration: Int,
) -> CognitiveBroker {
  let event =
    CrisisDetected(severity: severity, duration: duration, tick: broker.tick)
  publish(broker, event)
}

/// Publish crisis resolved event
pub fn emit_crisis_resolved(
  broker: CognitiveBroker,
  new_baseline: Glyph,
) -> CognitiveBroker {
  let event = CrisisResolved(new_baseline: new_baseline, tick: broker.tick)
  publish(broker, event)
}

/// Publish thought event
pub fn emit_thought(
  broker: CognitiveBroker,
  content: String,
  weight: Float,
) -> CognitiveBroker {
  let event =
    ThoughtGenerated(content: content, weight: weight, tick: broker.tick)
  publish(broker, event)
}

// =============================================================================
// EVENT QUERIES
// =============================================================================

/// Get recent events
pub fn recent_events(
  broker: CognitiveBroker,
  limit: Int,
) -> List(CognitiveEvent) {
  list.take(broker.history, limit)
}

/// Get events by filter
pub fn events_by_filter(
  broker: CognitiveBroker,
  filter: EventFilter,
) -> List(CognitiveEvent) {
  broker.history
  |> list.filter(fn(event) { matches_filter(event, filter) })
}

/// Check if event matches filter
fn matches_filter(event: CognitiveEvent, filter: EventFilter) -> Bool {
  case filter {
    AllEvents -> True
    EmotionalEvents ->
      case event {
        EmotionalShift(..) -> True
        _ -> False
      }
    IntrospectionEvents ->
      case event {
        Introspection(..) -> True
        InsightGenerated(..) -> True
        _ -> False
      }
    CrisisEvents ->
      case event {
        CrisisDetected(..) -> True
        CrisisResolved(..) -> True
        _ -> False
      }
    NarrativeEvents ->
      case event {
        NarrativeLinkFormed(..) -> True
        ThoughtGenerated(..) -> True
        _ -> False
      }
  }
}

/// Get last event of type
pub fn last_emotional_shift(broker: CognitiveBroker) -> Option(CognitiveEvent) {
  broker.history
  |> list.find(fn(e) {
    case e {
      EmotionalShift(..) -> True
      _ -> False
    }
  })
  |> option.from_result()
}

/// Get last crisis event
pub fn last_crisis(broker: CognitiveBroker) -> Option(CognitiveEvent) {
  broker.history
  |> list.find(fn(e) {
    case e {
      CrisisDetected(..) -> True
      _ -> False
    }
  })
  |> option.from_result()
}

/// Count events by type
pub fn count_emotional_events(broker: CognitiveBroker) -> Int {
  broker.history
  |> list.filter(fn(e) {
    case e {
      EmotionalShift(..) -> True
      _ -> False
    }
  })
  |> list.length()
}

/// Count insights
pub fn count_insights(broker: CognitiveBroker) -> Int {
  broker.history
  |> list.filter(fn(e) {
    case e {
      InsightGenerated(..) -> True
      _ -> False
    }
  })
  |> list.length()
}

// =============================================================================
// HANDLER MANAGEMENT
// =============================================================================

/// Add a handler
pub fn add_handler(
  broker: CognitiveBroker,
  handler: EventHandler,
) -> CognitiveBroker {
  CognitiveBroker(..broker, handlers: [handler, ..broker.handlers])
}

/// Remove handler by name
pub fn remove_handler(broker: CognitiveBroker, name: String) -> CognitiveBroker {
  let filtered = list.filter(broker.handlers, fn(h) { h.name != name })
  CognitiveBroker(..broker, handlers: filtered)
}

/// Enable handler
pub fn enable_handler(broker: CognitiveBroker, name: String) -> CognitiveBroker {
  let updated =
    list.map(broker.handlers, fn(h) {
      case h.name == name {
        True -> EventHandler(..h, active: True)
        False -> h
      }
    })
  CognitiveBroker(..broker, handlers: updated)
}

/// Disable handler
pub fn disable_handler(broker: CognitiveBroker, name: String) -> CognitiveBroker {
  let updated =
    list.map(broker.handlers, fn(h) {
      case h.name == name {
        True -> EventHandler(..h, active: False)
        False -> h
      }
    })
  CognitiveBroker(..broker, handlers: updated)
}

/// Get active handlers
pub fn active_handlers(broker: CognitiveBroker) -> List(EventHandler) {
  list.filter(broker.handlers, fn(h) { h.active })
}

// =============================================================================
// STATS
// =============================================================================

/// Get broker stats
pub fn stats(broker: CognitiveBroker) -> BrokerStats {
  BrokerStats(
    total_events: list.length(broker.history),
    emotional_events: count_emotional_events(broker),
    insights: count_insights(broker),
    active_handlers: list.length(active_handlers(broker)),
    current_tick: broker.tick,
  )
}

/// Broker statistics
pub type BrokerStats {
  BrokerStats(
    total_events: Int,
    emotional_events: Int,
    insights: Int,
    active_handlers: Int,
    current_tick: Int,
  )
}

// =============================================================================
// STRING CONVERSIONS
// =============================================================================

/// Event to string
pub fn event_to_string(event: CognitiveEvent) -> String {
  case event {
    EmotionalShift(_, _, magnitude, tick) ->
      "EmotionalShift(magnitude="
      <> float_to_string(magnitude)
      <> ", tick="
      <> int_to_string(tick)
      <> ")"
    Introspection(drift, within_range, tick) ->
      "Introspection(drift="
      <> float_to_string(drift)
      <> ", within="
      <> bool_to_string(within_range)
      <> ", tick="
      <> int_to_string(tick)
      <> ")"
    InsightGenerated(dim, dir, magnitude, _tick) ->
      "Insight("
      <> dimension_to_string(dim)
      <> " "
      <> direction_to_string(dir)
      <> ", magnitude="
      <> float_to_string(magnitude)
      <> ")"
    NarrativeLinkFormed(_, _, relation, tick) ->
      "NarrativeLink("
      <> relation_to_string(relation)
      <> ", tick="
      <> int_to_string(tick)
      <> ")"
    CrisisDetected(severity, duration, _) ->
      "Crisis(severity="
      <> float_to_string(severity)
      <> ", duration="
      <> int_to_string(duration)
      <> ")"
    CrisisResolved(_, tick) ->
      "CrisisResolved(tick=" <> int_to_string(tick) <> ")"
    MetaLevelChanged(from, to, _) ->
      "MetaLevel(" <> int_to_string(from) <> " -> " <> int_to_string(to) <> ")"
    ThoughtGenerated(content, weight, _) ->
      "Thought(\"" <> content <> "\", weight=" <> float_to_string(weight) <> ")"
  }
}

/// Dimension to string
pub fn dimension_to_string(dim: CognitiveDimension) -> String {
  case dim {
    Pleasure -> "pleasure"
    Arousal -> "arousal"
    Dominance -> "dominance"
  }
}

/// Direction to string
pub fn direction_to_string(dir: ChangeDirection) -> String {
  case dir {
    Increasing -> "increasing"
    Decreasing -> "decreasing"
  }
}

/// Relation to string
pub fn relation_to_string(rel: LinkRelation) -> String {
  case rel {
    Caused -> "caused"
    Preceded -> "preceded"
    Associated -> "associated"
    Contrasted -> "contrasted"
  }
}

// Helpers
fn float_to_string(f: Float) -> String {
  // Simple conversion
  case f <. 0.0 {
    True -> "-" <> float_to_string(0.0 -. f)
    False -> {
      let int_part = float_truncate(f)
      let frac = f -. int_to_float(int_part)
      let frac_scaled = float_truncate(frac *. 100.0)
      int_to_string(int_part) <> "." <> int_to_string(frac_scaled)
    }
  }
}

fn bool_to_string(b: Bool) -> String {
  case b {
    True -> "true"
    False -> "false"
  }
}

@external(erlang, "erlang", "integer_to_binary")
fn int_to_string(n: Int) -> String

@external(erlang, "erlang", "trunc")
fn float_truncate(f: Float) -> Int

@external(erlang, "erlang", "float")
fn int_to_float(n: Int) -> Float
