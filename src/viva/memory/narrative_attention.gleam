//// NarrativeAttention - Attention-enhanced narrative generation
////
//// Integrates Multi-Head Attention with the Narrative System (Qwen3-235B Priority #2)
//// Uses attention mechanisms to:
//// - Weight memories by relevance to current context
//// - Generate more coherent thought streams
//// - Focus on emotionally salient narrative links
////
//// The attention mechanism allows VIVA to "attend to" relevant memories
//// when constructing her inner voice, rather than just following the strongest link.

import gleam/float
import gleam/list
import gleam/option.{type Option, Some}
import viva/memory/narrative.{
  type NarrativeLink, type NarrativeMemory, type NarrativeResult, type Thought,
  type VoiceStyle, FromAssociation, FromCausal,
}
import viva_tensor/tensor.{type Tensor}
import viva_glyph/glyph.{type Glyph}

// =============================================================================
// TYPES
// =============================================================================

/// Attention-enhanced thought stream
pub type AttentionStream {
  AttentionStream(
    /// Thoughts generated with attention weighting
    thoughts: List(Thought),
    /// Attention weights for each step
    attention_weights: List(Tensor),
    /// Focus glyph
    focus: Option(Glyph),
    /// Stream depth
    depth: Int,
    /// Total intensity
    intensity: Float,
  )
}

/// Context for attention computation
pub type NarrativeContext {
  NarrativeContext(
    /// Current emotional state (PAD vector as tensor)
    emotional_state: Tensor,
    /// Recent glyphs (query context)
    recent_glyphs: List(Glyph),
    /// Attention temperature (lower = sharper focus)
    temperature: Float,
    /// Number of attention heads
    num_heads: Int,
  )
}

// =============================================================================
// CONSTRUCTORS
// =============================================================================

/// Create default narrative context
pub fn default_context() -> NarrativeContext {
  NarrativeContext(
    emotional_state: tensor.zeros([3]),
    // PAD neutral
    recent_glyphs: [],
    temperature: 1.0,
    num_heads: 4,
  )
}

/// Create context with emotional state
pub fn context_with_emotion(p: Float, a: Float, d: Float) -> NarrativeContext {
  NarrativeContext(
    emotional_state: tensor.from_list([p, a, d]),
    recent_glyphs: [],
    temperature: emotion_to_temperature(p, a, d),
    num_heads: 4,
  )
}

/// Emotional state affects attention temperature
/// High arousal = sharper focus (lower temperature)
/// Low arousal = diffuse attention (higher temperature)
fn emotion_to_temperature(p: Float, a: Float, _d: Float) -> Float {
  // Base temperature
  let base = 1.0

  // High arousal sharpens focus
  let arousal_effect = 0.0 -. a *. 0.3

  // Negative pleasure might sharpen focus on threats
  let pleasure_effect = case p <. 0.0 {
    True -> 0.0 -. 0.2
    False -> 0.0
  }

  float.max(0.1, base +. arousal_effect +. pleasure_effect)
}

// =============================================================================
// ATTENTION-ENHANCED INNER VOICE
// =============================================================================

/// Generate inner voice using attention to weight memories
/// Instead of just following the strongest link, uses attention to consider
/// multiple narrative paths and select based on context relevance.
pub fn inner_voice_attended(
  memory: NarrativeMemory,
  focus: Glyph,
  depth: Int,
  style: VoiceStyle,
  ctx: NarrativeContext,
) -> AttentionStream {
  let #(thoughts, weights) =
    build_attended_chain(memory, focus, depth, style, ctx, [], [])

  let intensity =
    thoughts
    |> list.map(fn(t) { t.weight })
    |> average_or_default(0.5)

  AttentionStream(
    thoughts: thoughts,
    attention_weights: weights,
    focus: Some(focus),
    depth: depth,
    intensity: intensity,
  )
}

/// Build thought chain using attention mechanism
fn build_attended_chain(
  memory: NarrativeMemory,
  current: Glyph,
  depth: Int,
  style: VoiceStyle,
  ctx: NarrativeContext,
  thought_acc: List(Thought),
  weight_acc: List(Tensor),
) -> #(List(Thought), List(Tensor)) {
  case depth <= 0 {
    True -> #(list.reverse(thought_acc), list.reverse(weight_acc))
    False -> {
      // Get candidate links from current glyph
      let candidates = narrative.what_resulted(memory, current, 10)

      case candidates {
        [] -> {
          // No more links, generate final thought
          let thought = narrative.reflect_on(memory, current, style)
          #(list.reverse([thought, ..thought_acc]), list.reverse(weight_acc))
        }
        _ -> {
          // Use attention to weight candidates
          let #(next_link, attn_weights) =
            attend_to_candidates(candidates, current, ctx)

          // Generate thought about transition
          let thought = attended_thought(current, next_link, style, ctx)

          // Update context with current glyph
          let new_ctx =
            NarrativeContext(..ctx, recent_glyphs: [
              current,
              ..list.take(ctx.recent_glyphs, 4)
            ])

          // Continue chain
          build_attended_chain(
            memory,
            next_link.effect,
            depth - 1,
            style,
            new_ctx,
            [thought, ..thought_acc],
            [attn_weights, ..weight_acc],
          )
        }
      }
    }
  }
}

/// Use attention to select next narrative link
fn attend_to_candidates(
  candidates: List(NarrativeResult),
  current: Glyph,
  ctx: NarrativeContext,
) -> #(NarrativeLink, Tensor) {
  // Build query from current context
  let query = build_query_vector(current, ctx)

  // Build key/value from candidates
  let keys = list.map(candidates, fn(c) { link_to_vector(c.link) })
  let _values = list.map(candidates, fn(c) { c.link.strength })

  // Compute attention scores
  let scores = compute_attention_scores(query, keys, ctx.temperature)

  // Apply softmax to get weights
  let weights = softmax_list(scores)
  let weights_tensor = tensor.from_list(weights)

  // Select candidate with highest attention weight
  let selected_idx = argmax_list(weights)
  let selected = case list.drop(candidates, selected_idx) |> list.first() {
    Ok(c) -> c.link
    Error(_) -> {
      // Fallback to first candidate
      case list.first(candidates) {
        Ok(c) -> c.link
        Error(_) -> panic as "No candidates"
      }
    }
  }

  #(selected, weights_tensor)
}

/// Build query vector from current glyph and context
fn build_query_vector(current: Glyph, ctx: NarrativeContext) -> List(Float) {
  // Combine glyph features with emotional state
  let glyph_features = glyph_to_features(current)
  let emotion_features = tensor.to_list(ctx.emotional_state)

  // Concatenate and normalize
  list.append(glyph_features, emotion_features)
}

/// Convert link to key vector for attention
fn link_to_vector(link: NarrativeLink) -> List(Float) {
  let effect_features = glyph_to_features(link.effect)
  let relation_features = relation_to_features(link.relation)
  let strength_feature = [link.strength]
  let recency_feature = [int_to_float(link.occurrences) /. 10.0]

  list.flatten([
    effect_features,
    relation_features,
    strength_feature,
    recency_feature,
  ])
}

/// Convert glyph to feature vector
fn glyph_to_features(g: Glyph) -> List(Float) {
  // Use glyph's semantic components as features
  let hash = glyph_hash(g)
  // Simple feature extraction from hash
  [
    int_to_float(hash % 100) /. 100.0,
    int_to_float({ hash / 100 } % 100) /. 100.0,
    int_to_float({ hash / 10_000 } % 100) /. 100.0,
    int_to_float({ hash / 1_000_000 } % 100) /. 100.0,
  ]
}

/// Hash for glyph using positional encoding (avoids collisions)
fn glyph_hash(g: Glyph) -> Int {
  case g.tokens {
    [t1, t2, t3, t4] -> t1 + t2 * 256 + t3 * 65536 + t4 * 16_777_216
    [t1, t2, t3] -> t1 + t2 * 256 + t3 * 65536
    [t1, t2] -> t1 + t2 * 256
    [t1] -> t1
    _ -> list.fold(g.tokens, 0, fn(acc, t) { acc * 256 + t })
  }
}

/// Convert relation type to features
fn relation_to_features(relation: narrative.Relation) -> List(Float) {
  case relation {
    narrative.Caused -> [1.0, 0.0, 0.0, 0.0]
    narrative.Preceded -> [0.0, 1.0, 0.0, 0.0]
    narrative.Associated -> [0.0, 0.0, 1.0, 0.0]
    narrative.Contrasted -> [0.0, 0.0, 0.0, 1.0]
  }
}

/// Compute scaled dot-product attention scores
fn compute_attention_scores(
  query: List(Float),
  keys: List(List(Float)),
  temperature: Float,
) -> List(Float) {
  // Compute dot product of query with each key
  keys
  |> list.map(fn(key) {
    let dot = dot_product(query, key)
    // Scale by sqrt of key dimension and temperature
    let scale = float_sqrt(int_to_float(list.length(key))) *. temperature
    dot /. scale
  })
}

/// Generate thought with attention context
fn attended_thought(
  from: Glyph,
  link: NarrativeLink,
  style: VoiceStyle,
  ctx: NarrativeContext,
) -> Thought {
  // Weight based on emotional state and link strength
  let emotional_weight = compute_emotional_weight(link, ctx)
  let content = narrative.narrate_link(link, style)

  let source = case link.relation {
    narrative.Caused | narrative.Preceded -> FromCausal
    narrative.Associated | narrative.Contrasted -> FromAssociation
  }

  narrative.Thought(
    content: content,
    weight: emotional_weight,
    source: source,
    glyphs: [from, link.effect],
  )
}

/// Compute emotional weight for a link
fn compute_emotional_weight(link: NarrativeLink, ctx: NarrativeContext) -> Float {
  let base_weight = link.strength
  let emo_data = tensor.to_list(ctx.emotional_state)

  // Arousal amplifies weight
  let arousal = case list.drop(emo_data, 1) |> list.first() {
    Ok(a) -> a
    Error(_) -> 0.0
  }

  // Dominance affects causal link weight
  let dominance = case list.drop(emo_data, 2) |> list.first() {
    Ok(d) -> d
    Error(_) -> 0.0
  }

  let arousal_bonus = float.absolute_value(arousal) *. 0.2
  let dominance_bonus = case link.relation {
    narrative.Caused -> dominance *. 0.1
    _ -> 0.0
  }

  float.min(1.0, base_weight +. arousal_bonus +. dominance_bonus)
}

// =============================================================================
// MULTI-MEMORY ATTENTION
// =============================================================================

/// Query multiple memories using multi-head attention
/// Returns memories sorted by attention-weighted relevance
pub fn query_with_attention(
  memory: NarrativeMemory,
  query_glyph: Glyph,
  ctx: NarrativeContext,
  max_results: Int,
) -> List(#(NarrativeResult, Float)) {
  // Get all related links
  let caused = narrative.what_caused(memory, query_glyph, max_results * 2)
  let resulted = narrative.what_resulted(memory, query_glyph, max_results * 2)
  let all_links = list.append(caused, resulted)

  case all_links {
    [] -> []
    _ -> {
      // Build query
      let query = build_query_vector(query_glyph, ctx)

      // Compute attention for each link
      let keys = list.map(all_links, fn(r) { link_to_vector(r.link) })
      let scores = compute_attention_scores(query, keys, ctx.temperature)
      let weights = softmax_list(scores)

      // Pair results with weights and sort
      list.zip(all_links, weights)
      |> list.sort(fn(a, b) { float.compare(b.1, a.1) })
      |> list.take(max_results)
    }
  }
}

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

fn dot_product(a: List(Float), b: List(Float)) -> Float {
  list.zip(a, b)
  |> list.fold(0.0, fn(acc, pair) { acc +. pair.0 *. pair.1 })
}

fn softmax_list(scores: List(Float)) -> List(Float) {
  let max_score = list.fold(scores, -1_000_000.0, float.max)
  let exp_scores = list.map(scores, fn(s) { float_exp(s -. max_score) })
  let sum = list.fold(exp_scores, 0.0, fn(a, b) { a +. b })
  list.map(exp_scores, fn(e) { e /. sum })
}

fn argmax_list(values: List(Float)) -> Int {
  values
  |> list.index_map(fn(v, i) { #(i, v) })
  |> list.fold(#(0, -1_000_000.0), fn(best, curr) {
    case curr.1 >. best.1 {
      True -> curr
      False -> best
    }
  })
  |> fn(pair) { pair.0 }
}

fn average_or_default(values: List(Float), default: Float) -> Float {
  case values {
    [] -> default
    _ -> {
      let sum = list.fold(values, 0.0, fn(a, b) { a +. b })
      sum /. int_to_float(list.length(values))
    }
  }
}

// =============================================================================
// FFI - Native performance (replaced O(n) implementations)
// =============================================================================

@external(erlang, "math", "exp")
fn float_exp(x: Float) -> Float

@external(erlang, "math", "sqrt")
fn float_sqrt(x: Float) -> Float

@external(erlang, "erlang", "float")
fn int_to_float(i: Int) -> Float
