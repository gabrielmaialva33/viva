//// Realtime Vision Tests
////
//// Tests for the dual-path visual processing system.
//// Note: Integration tests with NV-CLIP/OCR servers require
//// the services to be running (localhost:8050, localhost:8020).

import gleam/list
import gleam/option.{None}
import gleeunit/should
import viva/memory/hrr
import viva/senses/realtime_vision.{
  DramaticChange, MajorChange, MinorChange, NoChange,
}

// =============================================================================
// CONFIGURATION TESTS
// =============================================================================

pub fn default_config_has_valid_thresholds_test() {
  let config = realtime_vision.default_config()

  // Thresholds should be in ascending order
  { config.minor_threshold <. config.major_threshold }
  |> should.be_true()

  { config.major_threshold <. config.dramatic_threshold }
  |> should.be_true()

  // HRR dim should match NV-CLIP output
  config.hrr_dim
  |> should.equal(1024)
}

pub fn new_state_is_empty_test() {
  let state = realtime_vision.new()

  realtime_vision.frame_count(state)
  |> should.equal(0)

  realtime_vision.has_pending_caption(state)
  |> should.be_false()

  realtime_vision.last_timestamp(state)
  |> should.equal(None)
}

// =============================================================================
// CHANGE DETECTION TESTS
// =============================================================================

pub fn identical_embeddings_no_change_test() {
  // Create two identical embeddings
  let emb = hrr.random(1024)

  let change = realtime_vision.detect_change(emb, emb)

  change
  |> should.equal(NoChange)
}

pub fn detect_change_magnitude_identical_test() {
  let emb = hrr.random(1024)

  let magnitude = realtime_vision.detect_change_magnitude(emb, emb)

  // Identical embeddings should have ~0 distance
  { magnitude <. 0.01 }
  |> should.be_true()
}

pub fn different_embeddings_detect_change_test() {
  // Create two different random embeddings
  let emb1 = hrr.random(1024)
  let emb2 = hrr.random(1024)

  let magnitude = realtime_vision.detect_change_magnitude(emb1, emb2)

  // Random embeddings should have some distance
  { magnitude >. 0.0 }
  |> should.be_true()
}

pub fn orthogonal_embeddings_major_change_test() {
  // Create approximately orthogonal embeddings
  let dim = 1024

  // 1/sqrt(512) ~= 0.0442
  let scale = 0.0442

  // First: positive in first half, zero in second
  let emb1_data =
    list.range(0, dim - 1)
    |> list.map(fn(i) {
      case i < dim / 2 {
        True -> scale
        False -> 0.0
      }
    })

  // Second: zero in first half, positive in second
  let emb2_data =
    list.range(0, dim - 1)
    |> list.map(fn(i) {
      case i >= dim / 2 {
        True -> scale
        False -> 0.0
      }
    })

  let emb1 = hrr.from_list(emb1_data)
  let emb2 = hrr.from_list(emb2_data)

  let magnitude = realtime_vision.detect_change_magnitude(emb1, emb2)

  // Orthogonal = cosine similarity ~0, distance ~1.0
  { magnitude >. 0.9 }
  |> should.be_true()
}

pub fn classify_change_thresholds_test() {
  let config = realtime_vision.default_config()
  let avg_change = 0.1

  // Test each threshold boundary
  realtime_vision.classify_change(0.01, config, avg_change)
  |> should.equal(NoChange)

  realtime_vision.classify_change(0.1, config, avg_change)
  |> should.equal(MinorChange)

  realtime_vision.classify_change(0.2, config, avg_change)
  |> should.equal(MajorChange)

  realtime_vision.classify_change(0.5, config, avg_change)
  |> should.equal(DramaticChange)
}

// =============================================================================
// STIMULUS CONVERSION TESTS
// =============================================================================

pub fn no_change_no_stimulus_test() {
  let emb = hrr.random(1024)

  let stimulus = realtime_vision.embedding_to_stimulus(emb, NoChange)

  stimulus
  |> should.equal(None)
}

pub fn dramatic_change_produces_stimulus_test() {
  let emb = hrr.random(1024)

  let stimulus = realtime_vision.embedding_to_stimulus(emb, DramaticChange)

  // Should produce some stimulus
  option.is_some(stimulus)
  |> should.be_true()
}

pub fn change_to_intensity_scaling_test() {
  // Intensity should increase with change severity
  let no_intensity = realtime_vision.change_to_intensity(NoChange)
  let minor_intensity = realtime_vision.change_to_intensity(MinorChange)
  let major_intensity = realtime_vision.change_to_intensity(MajorChange)
  let dramatic_intensity = realtime_vision.change_to_intensity(DramaticChange)

  { no_intensity <. minor_intensity }
  |> should.be_true()

  { minor_intensity <. major_intensity }
  |> should.be_true()

  { major_intensity <. dramatic_intensity }
  |> should.be_true()
}

// =============================================================================
// STATE MANAGEMENT TESTS
// =============================================================================

pub fn receive_caption_updates_state_test() {
  let state = realtime_vision.new()

  // Simulate receiving a caption
  let updated = realtime_vision.receive_caption(state, "A person at a desk")

  // Pending caption should be cleared
  realtime_vision.has_pending_caption(updated)
  |> should.be_false()
}

pub fn buffer_utilization_test() {
  let state = realtime_vision.new()

  // Empty buffer should have 0 utilization
  realtime_vision.buffer_utilization(state)
  |> should.equal(0.0)
}

pub fn average_change_initial_test() {
  let state = realtime_vision.new()

  // Initial average should be the default (0.1)
  let avg = realtime_vision.average_change(state)

  { avg >. 0.0 }
  |> should.be_true()

  { avg <. 0.5 }
  |> should.be_true()
}

// =============================================================================
// HRR INTEGRATION TESTS
// =============================================================================

pub fn hrr_similarity_bounds_test() {
  // Similarity should be in [-1, 1]
  let emb1 = hrr.random(1024)
  let emb2 = hrr.random(1024)

  let sim = hrr.similarity(emb1, emb2)

  { sim >=. -1.0 }
  |> should.be_true()

  { sim <=. 1.0 }
  |> should.be_true()
}

pub fn probe_similarity_empty_state_test() {
  let state = realtime_vision.new()
  let probe = hrr.random(1024)

  // Empty state should return 0 similarity
  realtime_vision.probe_similarity(state, probe)
  |> should.equal(0.0)
}

pub fn search_recent_empty_test() {
  let state = realtime_vision.new()
  let probe = hrr.random(1024)

  let results = realtime_vision.search_recent(state, probe, 0.5)

  list.length(results)
  |> should.equal(0)
}

// =============================================================================
// CAPTION REQUEST TESTS
// =============================================================================

pub fn request_caption_creates_valid_request_test() {
  let request = realtime_vision.request_caption("/tmp/test.png")

  request.completed
  |> should.be_false()

  request.result
  |> should.equal(None)

  // ID should contain source info
  request.image_source
  |> should.equal("/tmp/test.png")
}

// =============================================================================
// DESCRIBE STATE TESTS
// =============================================================================

pub fn describe_state_initial_test() {
  let state = realtime_vision.new()

  let description = realtime_vision.describe_state(state)

  // Should contain "Vision" and frame count
  { description != "" }
  |> should.be_true()
}

// =============================================================================
// EDGE CASES
// =============================================================================

pub fn zero_dimension_hrr_test() {
  // Empty HRR should not crash
  let empty = hrr.from_list([])

  hrr.similarity(empty, empty)
  |> should.equal(0.0)
}

pub fn large_change_magnitude_test() {
  // Create opposite embeddings (all positive vs all negative)
  let pos_data = list.repeat(0.03125, 1024)
  // 1/sqrt(1024) normalized
  let neg_data = list.repeat(-0.03125, 1024)

  let pos_emb = hrr.from_list(pos_data)
  let neg_emb = hrr.from_list(neg_data)

  let magnitude = realtime_vision.detect_change_magnitude(pos_emb, neg_emb)

  // Opposite embeddings: cosine = -1, distance = 2.0
  { magnitude >. 1.5 }
  |> should.be_true()
}
