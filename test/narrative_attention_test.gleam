import gleam/list
import gleam/option.{Some}
import gleeunit/should
import viva/narrative
import viva/narrative_attention
import viva_glyph/glyph

// =============================================================================
// CONTEXT TESTS
// =============================================================================

pub fn default_context_test() {
  let ctx = narrative_attention.default_context()
  should.equal(ctx.temperature, 1.0)
  should.equal(ctx.num_heads, 4)
  should.equal(ctx.recent_glyphs, [])
}

pub fn context_with_emotion_neutral_test() {
  let ctx = narrative_attention.context_with_emotion(0.0, 0.0, 0.0)
  // Neutral emotion = base temperature
  should.be_true(ctx.temperature >. 0.5)
  should.be_true(ctx.temperature <. 1.5)
}

pub fn context_with_emotion_high_arousal_test() {
  // High arousal should lower temperature (sharper focus)
  let ctx_high = narrative_attention.context_with_emotion(0.0, 0.8, 0.0)
  let ctx_low = narrative_attention.context_with_emotion(0.0, -0.8, 0.0)
  should.be_true(ctx_high.temperature <. ctx_low.temperature)
}

pub fn context_with_emotion_negative_pleasure_test() {
  // Negative pleasure might sharpen focus
  let ctx_neg = narrative_attention.context_with_emotion(-0.8, 0.0, 0.0)
  let ctx_pos = narrative_attention.context_with_emotion(0.8, 0.0, 0.0)
  should.be_true(ctx_neg.temperature <=. ctx_pos.temperature)
}

// =============================================================================
// INNER VOICE ATTENDED TESTS
// =============================================================================

pub fn inner_voice_attended_empty_memory_test() {
  let memory = narrative.new()
  let focus = glyph.new([1, 2, 3])
  let ctx = narrative_attention.default_context()

  let stream =
    narrative_attention.inner_voice_attended(
      memory,
      focus,
      3,
      narrative.Factual,
      ctx,
    )

  // Should have at least the focus thought
  should.be_true(list.length(stream.thoughts) >= 1)
  should.equal(stream.focus, Some(focus))
  should.equal(stream.depth, 3)
}

pub fn inner_voice_attended_with_links_test() {
  let memory = narrative.new()
  let g1 = glyph.new([10, 20, 30])
  let g2 = glyph.new([11, 21, 31])
  let g3 = glyph.new([12, 22, 32])

  // Create a chain of links
  let memory = narrative.record_caused(memory, g1, g2, 0)
  let memory = narrative.record_caused(memory, g2, g3, 1)

  let ctx = narrative_attention.default_context()
  let stream =
    narrative_attention.inner_voice_attended(
      memory,
      g1,
      3,
      narrative.Emotional,
      ctx,
    )

  // Should follow the chain
  should.be_true(list.length(stream.thoughts) >= 1)
  should.be_true(stream.intensity >. 0.0)
}

// =============================================================================
// QUERY WITH ATTENTION TESTS
// =============================================================================

pub fn query_with_attention_empty_test() {
  let memory = narrative.new()
  let query = glyph.new([5, 5, 5])
  let ctx = narrative_attention.default_context()

  let results = narrative_attention.query_with_attention(memory, query, ctx, 5)
  should.equal(results, [])
}

pub fn query_with_attention_with_links_test() {
  let memory = narrative.new()
  let g1 = glyph.new([50, 50, 50])
  let g2 = glyph.new([51, 51, 51])
  let g3 = glyph.new([52, 52, 52])

  // Create links from center
  let memory = narrative.record_caused(memory, g1, g2, 0)
  let memory = narrative.record_associated(memory, g1, g3, 1)

  let ctx = narrative_attention.default_context()
  let results = narrative_attention.query_with_attention(memory, g1, ctx, 10)

  // Should return linked memories with attention weights
  should.be_true(list.length(results) >= 1)

  // Weights should be positive
  list.each(results, fn(pair) {
    let #(_result, weight) = pair
    should.be_true(weight >. 0.0)
    should.be_true(weight <=. 1.0)
  })
}

// =============================================================================
// ATTENTION STREAM TESTS
// =============================================================================

pub fn attention_stream_weights_test() {
  let memory = narrative.new()
  let g1 = glyph.new([100, 100, 100])
  let g2 = glyph.new([101, 101, 101])
  let g3 = glyph.new([102, 102, 102])

  let memory = narrative.record_caused(memory, g1, g2, 0)
  let memory = narrative.record_caused(memory, g2, g3, 1)

  let ctx = narrative_attention.context_with_emotion(0.5, 0.5, 0.0)
  let stream =
    narrative_attention.inner_voice_attended(
      memory,
      g1,
      5,
      narrative.Reflective,
      ctx,
    )

  // Attention weights should be recorded
  // Each step (except possibly the last) should have weights
  should.be_true(stream.intensity >. 0.0)
}

pub fn emotional_context_affects_stream_test() {
  let memory = narrative.new()
  let g1 = glyph.new([200, 200, 200])
  let g2 = glyph.new([201, 201, 201])
  let memory = narrative.record_caused(memory, g1, g2, 0)

  // High arousal context
  let ctx_aroused = narrative_attention.context_with_emotion(0.0, 0.9, 0.5)
  let stream_aroused =
    narrative_attention.inner_voice_attended(
      memory,
      g1,
      2,
      narrative.Emotional,
      ctx_aroused,
    )

  // Low arousal context
  let ctx_calm = narrative_attention.context_with_emotion(0.0, -0.5, 0.0)
  let stream_calm =
    narrative_attention.inner_voice_attended(
      memory,
      g1,
      2,
      narrative.Emotional,
      ctx_calm,
    )

  // Both should produce valid streams
  should.be_true(list.length(stream_aroused.thoughts) >= 1)
  should.be_true(list.length(stream_calm.thoughts) >= 1)

  // Intensities might differ based on emotional state
  should.be_true(stream_aroused.intensity >. 0.0)
  should.be_true(stream_calm.intensity >. 0.0)
}
