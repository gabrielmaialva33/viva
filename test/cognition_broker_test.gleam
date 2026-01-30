//// Cognitive Broker Tests
////
//// Tests for event-driven cognitive communication.

import gleam/list
import gleam/option.{None, Some}
import gleeunit/should
import viva/infra/cognition/broker.{
  AllEvents, Arousal, Associated, Caused, CrisisEvents, Decreasing,
  EmotionalEvents, Increasing, IntrospectionEvents, NarrativeEvents, Pleasure,
}
import viva_emotion/pad
import viva_glyph/glyph

// =============================================================================
// CREATION
// =============================================================================

pub fn new_creates_empty_broker_test() {
  let b = broker.new()
  let stats = broker.stats(b)

  stats.total_events |> should.equal(0)
  stats.current_tick |> should.equal(0)
}

pub fn new_with_history_sets_limit_test() {
  let b = broker.new_with_history(50)

  // Publish more than 50 events
  let b =
    list.range(1, 60)
    |> list.fold(b, fn(acc, _) { broker.emit_introspection(acc, 0.1, True) })

  let stats = broker.stats(b)
  // Should be capped at 50
  stats.total_events |> should.equal(50)
}

// =============================================================================
// PUBLISHING EVENTS
// =============================================================================

pub fn emit_emotional_shift_adds_event_test() {
  let b = broker.new()
  let from = pad.neutral()
  let to = pad.new(0.5, 0.3, 0.2)

  let b = broker.emit_emotional_shift(b, from, to, 0.5)

  broker.count_emotional_events(b) |> should.equal(1)
}

pub fn emit_introspection_adds_event_test() {
  let b = broker.new()

  let b = broker.emit_introspection(b, 0.3, False)

  let stats = broker.stats(b)
  stats.total_events |> should.equal(1)
}

pub fn emit_insight_adds_event_test() {
  let b = broker.new()

  let b = broker.emit_insight(b, Pleasure, Increasing, 0.7)

  broker.count_insights(b) |> should.equal(1)
}

pub fn emit_narrative_link_adds_event_test() {
  let b = broker.new()
  let cause = glyph.new([100, 100, 100, 100])
  let effect = glyph.new([200, 200, 200, 200])

  let b = broker.emit_narrative_link(b, cause, effect, Caused)

  let events = broker.events_by_filter(b, NarrativeEvents)
  list.length(events) |> should.equal(1)
}

pub fn emit_crisis_adds_event_test() {
  let b = broker.new()

  let b = broker.emit_crisis(b, 0.7, 15)

  case broker.last_crisis(b) {
    Some(_) -> should.be_true(True)
    None -> should.fail()
  }
}

pub fn emit_crisis_resolved_adds_event_test() {
  let b = broker.new()
  let new_baseline = glyph.neutral()

  let b = broker.emit_crisis_resolved(b, new_baseline)

  let events = broker.events_by_filter(b, CrisisEvents)
  list.length(events) |> should.equal(1)
}

pub fn emit_thought_adds_event_test() {
  let b = broker.new()

  let b = broker.emit_thought(b, "I feel happy", 0.8)

  let events = broker.events_by_filter(b, NarrativeEvents)
  list.length(events) |> should.equal(1)
}

// =============================================================================
// EVENT QUERIES
// =============================================================================

pub fn recent_events_returns_latest_test() {
  let b = broker.new()

  let b = broker.emit_introspection(b, 0.1, True)
  let b = broker.emit_introspection(b, 0.2, True)
  let b = broker.emit_introspection(b, 0.3, False)

  let recent = broker.recent_events(b, 2)
  list.length(recent) |> should.equal(2)
}

pub fn events_by_filter_emotional_test() {
  let b = broker.new()
  let from = pad.neutral()
  let to = pad.new(0.5, 0.3, 0.2)

  let b = broker.emit_emotional_shift(b, from, to, 0.5)
  let b = broker.emit_introspection(b, 0.3, True)
  let b = broker.emit_emotional_shift(b, to, from, 0.5)

  let emotional = broker.events_by_filter(b, EmotionalEvents)
  list.length(emotional) |> should.equal(2)
}

pub fn events_by_filter_introspection_test() {
  let b = broker.new()

  let b = broker.emit_introspection(b, 0.3, True)
  let b = broker.emit_insight(b, Arousal, Decreasing, 0.5)
  let b = broker.emit_emotional_shift(b, pad.neutral(), pad.neutral(), 0.0)

  let intro = broker.events_by_filter(b, IntrospectionEvents)
  list.length(intro) |> should.equal(2)
}

pub fn events_by_filter_all_test() {
  let b = broker.new()

  let b = broker.emit_introspection(b, 0.3, True)
  let b = broker.emit_insight(b, Arousal, Decreasing, 0.5)
  let b = broker.emit_thought(b, "test", 0.5)

  let all = broker.events_by_filter(b, AllEvents)
  list.length(all) |> should.equal(3)
}

pub fn last_emotional_shift_finds_event_test() {
  let b = broker.new()
  let from = pad.neutral()
  let to = pad.new(0.5, 0.3, 0.2)

  let b = broker.emit_emotional_shift(b, from, to, 0.5)

  case broker.last_emotional_shift(b) {
    Some(_) -> should.be_true(True)
    None -> should.fail()
  }
}

pub fn last_emotional_shift_returns_none_when_empty_test() {
  let b = broker.new()

  case broker.last_emotional_shift(b) {
    None -> should.be_true(True)
    Some(_) -> should.fail()
  }
}

// =============================================================================
// HANDLER MANAGEMENT
// =============================================================================

pub fn add_handler_increases_count_test() {
  let b = broker.new()
  let handler =
    broker.EventHandler(name: "custom", filter: AllEvents, active: True)

  let b = broker.add_handler(b, handler)

  let active = broker.active_handlers(b)
  // 3 default + 1 custom
  { list.length(active) >= 4 } |> should.be_true()
}

pub fn remove_handler_decreases_count_test() {
  let b = broker.new()

  let initial_count = list.length(broker.active_handlers(b))
  let b = broker.remove_handler(b, "narrative")
  let final_count = list.length(broker.active_handlers(b))

  { final_count < initial_count } |> should.be_true()
}

pub fn disable_handler_deactivates_test() {
  let b = broker.new()

  let b = broker.disable_handler(b, "narrative")

  let active = broker.active_handlers(b)
  let narrative_active = list.any(active, fn(h) { h.name == "narrative" })

  narrative_active |> should.be_false()
}

pub fn enable_handler_reactivates_test() {
  let b = broker.new()

  let b = broker.disable_handler(b, "narrative")
  let b = broker.enable_handler(b, "narrative")

  let active = broker.active_handlers(b)
  let narrative_active = list.any(active, fn(h) { h.name == "narrative" })

  narrative_active |> should.be_true()
}

// =============================================================================
// STATS
// =============================================================================

pub fn stats_counts_correctly_test() {
  let b = broker.new()
  let from = pad.neutral()
  let to = pad.new(0.5, 0.3, 0.2)

  let b = broker.emit_emotional_shift(b, from, to, 0.5)
  let b = broker.emit_insight(b, Pleasure, Increasing, 0.7)
  let b = broker.emit_insight(b, Arousal, Decreasing, 0.5)

  let stats = broker.stats(b)

  stats.total_events |> should.equal(3)
  stats.emotional_events |> should.equal(1)
  stats.insights |> should.equal(2)
}

pub fn tick_increments_on_publish_test() {
  let b = broker.new()

  let b = broker.emit_introspection(b, 0.1, True)
  let b = broker.emit_introspection(b, 0.2, True)

  let stats = broker.stats(b)
  stats.current_tick |> should.equal(2)
}

// =============================================================================
// STRING CONVERSIONS
// =============================================================================

pub fn dimension_to_string_test() {
  broker.dimension_to_string(Pleasure) |> should.equal("pleasure")
  broker.dimension_to_string(Arousal) |> should.equal("arousal")
}

pub fn direction_to_string_test() {
  broker.direction_to_string(Increasing) |> should.equal("increasing")
  broker.direction_to_string(Decreasing) |> should.equal("decreasing")
}

pub fn relation_to_string_test() {
  broker.relation_to_string(Caused) |> should.equal("caused")
  broker.relation_to_string(Associated) |> should.equal("associated")
}
