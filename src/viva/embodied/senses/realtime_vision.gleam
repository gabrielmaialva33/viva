//// Realtime Vision - Dual-path visual processing for VIVA
////
//// Architecture:
////   FAST PATH (<100ms, always):
////     Frame -> NV-CLIP -> Embedding 1024-dim -> HRR bind -> change detection -> Stimulus
////
////   SLOW PATH (async, when change > threshold):
////     Change > threshold -> spawn VLM caption -> update narrative (non-blocking)
////
//// This module integrates with:
////   - NV-CLIP local server (localhost:8050) for embeddings
////   - PaddleOCR local server (localhost:8020) for text extraction
////   - HRR memory system for holographic binding
////   - Soul's stimulus system for emotional impact
////
//// Performance target: <150ms total latency for fast path

import gleam/float
import gleam/int
import gleam/list
import gleam/option.{type Option, None, Some}
import gleam/string
import viva/memory/hrr.{type HRR}
import viva_emotion/stimulus.{type Stimulus}

// =============================================================================
// TYPES
// =============================================================================

/// Visual frame with embedding and change metadata
pub type VisualFrame {
  VisualFrame(
    /// NV-CLIP embedding (1024 dimensions)
    embedding: HRR,
    /// Unix timestamp in milliseconds
    timestamp: Int,
    /// Change magnitude from previous frame (0.0 to 1.0+)
    change_magnitude: Float,
    /// Type of change detected
    change_type: VisualChange,
    /// Optional caption from slow path
    caption: Option(String),
    /// Raw image path or bytes identifier
    source: String,
  )
}

/// State for realtime vision processing
pub type RealtimeVisionState {
  RealtimeVisionState(
    /// Previous frame for change detection
    previous_frame: Option(VisualFrame),
    /// Circular buffer of recent frames (for HRR memory)
    recent_frames: List(VisualFrame),
    /// Maximum frames to keep in buffer
    buffer_size: Int,
    /// Configuration for change detection
    config: VisionConfig,
    /// Total frames processed
    frame_count: Int,
    /// Pending caption request
    pending_caption: Option(CaptionRequest),
    /// Running average of change magnitude (for adaptation)
    avg_change: Float,
  )
}

/// Configuration for vision processing
pub type VisionConfig {
  VisionConfig(
    /// NV-CLIP server URL
    clip_url: String,
    /// PaddleOCR server URL
    ocr_url: String,
    /// Threshold for Minor change
    minor_threshold: Float,
    /// Threshold for Major change
    major_threshold: Float,
    /// Threshold for Dramatic change
    dramatic_threshold: Float,
    /// HRR dimension (must match NV-CLIP output)
    hrr_dim: Int,
    /// Whether to auto-request captions on major changes
    auto_caption: Bool,
    /// Adaptive threshold factor (multiplies avg_change)
    adaptive_factor: Float,
  )
}

/// Type of visual change detected
pub type VisualChange {
  /// No significant change (< 0.05 cosine distance)
  NoChange
  /// Minor change (0.05 - 0.15) - subtle movement
  MinorChange
  /// Major change (0.15 - 0.3) - scene element changed
  MajorChange
  /// Dramatic change (> 0.3) - scene completely different
  DramaticChange
}

/// Async caption request
pub type CaptionRequest {
  CaptionRequest(
    /// Request ID for correlation
    id: String,
    /// Image source for VLM
    image_source: String,
    /// Timestamp of request
    requested_at: Int,
    /// Whether completed
    completed: Bool,
    /// Result caption (when completed)
    result: Option(String),
  )
}

/// Error types for vision processing
pub type VisionError {
  ClipConnectionFailed(reason: String)
  OcrConnectionFailed(reason: String)
  EmbeddingParseFailed(reason: String)
  InvalidImageData(reason: String)
  Timeout(operation: String)
}

// =============================================================================
// CONSTRUCTORS
// =============================================================================

/// Create default vision configuration
pub fn default_config() -> VisionConfig {
  VisionConfig(
    clip_url: "http://localhost:8050",
    ocr_url: "http://localhost:8020",
    minor_threshold: 0.05,
    major_threshold: 0.15,
    dramatic_threshold: 0.3,
    hrr_dim: 1024,
    auto_caption: True,
    adaptive_factor: 1.5,
  )
}

/// Create new realtime vision state
pub fn new() -> RealtimeVisionState {
  new_with_config(default_config())
}

/// Create new state with custom config
pub fn new_with_config(config: VisionConfig) -> RealtimeVisionState {
  RealtimeVisionState(
    previous_frame: None,
    recent_frames: [],
    buffer_size: 30,
    config: config,
    frame_count: 0,
    pending_caption: None,
    avg_change: 0.1,
  )
}

// =============================================================================
// FAST PATH - Main Processing (<100ms target)
// =============================================================================

/// Process a frame through the fast path
/// Returns updated state and optional stimulus for soul
pub fn process_frame(
  state: RealtimeVisionState,
  raw_image: BitArray,
  source: String,
) -> #(RealtimeVisionState, Option(Stimulus)) {
  let timestamp = now_ms()

  // Step 1: Get embedding from NV-CLIP
  case get_clip_embedding(raw_image, state.config.clip_url) {
    Error(_) -> #(state, None)
    Ok(embedding_list) -> {
      // Step 2: Convert to HRR
      let embedding = hrr.from_list(embedding_list)

      // Step 3: Detect change
      let #(change_magnitude, change_type) = case state.previous_frame {
        None -> #(1.0, DramaticChange)
        Some(prev) -> {
          let magnitude = detect_change_magnitude(prev.embedding, embedding)
          let change =
            classify_change(magnitude, state.config, state.avg_change)
          #(magnitude, change)
        }
      }

      // Step 4: Create frame
      let frame =
        VisualFrame(
          embedding: embedding,
          timestamp: timestamp,
          change_magnitude: change_magnitude,
          change_type: change_type,
          caption: None,
          source: source,
        )

      // Step 5: Update state
      let new_avg_change =
        update_running_average(state.avg_change, change_magnitude)
      let new_frames =
        prepend_bounded(frame, state.recent_frames, state.buffer_size)

      // Step 6: Maybe request caption (async, non-blocking)
      let pending_caption = case
        state.config.auto_caption && should_request_caption(change_type)
      {
        True -> Some(create_caption_request(source, timestamp))
        False -> state.pending_caption
      }

      let new_state =
        RealtimeVisionState(
          ..state,
          previous_frame: Some(frame),
          recent_frames: new_frames,
          frame_count: state.frame_count + 1,
          pending_caption: pending_caption,
          avg_change: new_avg_change,
        )

      // Step 7: Convert to stimulus
      let stimulus = embedding_to_stimulus(embedding, change_type)

      #(new_state, stimulus)
    }
  }
}

/// Process frame from file path
pub fn process_frame_path(
  state: RealtimeVisionState,
  image_path: String,
) -> #(RealtimeVisionState, Option(Stimulus)) {
  case read_image_file(image_path) {
    Ok(bytes) -> process_frame(state, bytes, image_path)
    Error(_) -> #(state, None)
  }
}

// =============================================================================
// CHANGE DETECTION
// =============================================================================

/// Calculate change magnitude between two embeddings (cosine distance)
pub fn detect_change_magnitude(old_emb: HRR, new_emb: HRR) -> Float {
  // Cosine similarity returns [-1, 1], we want distance [0, 2]
  let similarity = hrr.similarity(old_emb, new_emb)
  // Convert to distance: 0 = identical, 1 = orthogonal, 2 = opposite
  1.0 -. similarity
}

/// Classify change based on magnitude and thresholds
pub fn classify_change(
  magnitude: Float,
  config: VisionConfig,
  avg_change: Float,
) -> VisualChange {
  // Use adaptive thresholds based on running average
  let adaptive_minor =
    float.max(
      config.minor_threshold,
      avg_change *. config.adaptive_factor *. 0.5,
    )
  let adaptive_major =
    float.max(config.major_threshold, avg_change *. config.adaptive_factor)
  let adaptive_dramatic =
    float.max(
      config.dramatic_threshold,
      avg_change *. config.adaptive_factor *. 2.0,
    )

  case magnitude {
    m if m <. adaptive_minor -> NoChange
    m if m <. adaptive_major -> MinorChange
    m if m <. adaptive_dramatic -> MajorChange
    _ -> DramaticChange
  }
}

/// Detect change between two embeddings
pub fn detect_change(old_emb: HRR, new_emb: HRR) -> VisualChange {
  let magnitude = detect_change_magnitude(old_emb, new_emb)
  classify_change(magnitude, default_config(), 0.1)
}

// =============================================================================
// STIMULUS CONVERSION
// =============================================================================

/// Convert embedding and change to emotional stimulus
pub fn embedding_to_stimulus(
  embedding: HRR,
  change: VisualChange,
) -> Option(Stimulus) {
  // Extract emotional valence from embedding (simplified: use norm as arousal indicator)
  let embedding_norm = hrr_norm(embedding)

  case change {
    NoChange -> None
    MinorChange -> {
      // Minor changes: subtle safety/comfort
      case embedding_norm >. 0.5 {
        True -> Some(stimulus.Safety)
        False -> None
      }
    }
    MajorChange -> {
      // Major changes: could be threat or insight
      case embedding_norm >. 0.7 {
        True -> Some(stimulus.LucidInsight)
        False -> Some(stimulus.Safety)
      }
    }
    DramaticChange -> {
      // Dramatic changes: high arousal, potential threat
      case embedding_norm >. 0.8 {
        True -> Some(stimulus.Threat)
        False -> Some(stimulus.LucidInsight)
      }
    }
  }
}

/// Convert change type to stimulus intensity
pub fn change_to_intensity(change: VisualChange) -> Float {
  case change {
    NoChange -> 0.0
    MinorChange -> 0.2
    MajorChange -> 0.5
    DramaticChange -> 0.8
  }
}

// =============================================================================
// SLOW PATH - Async Caption (Non-blocking)
// =============================================================================

/// Create a caption request
pub fn request_caption(image_source: String) -> CaptionRequest {
  create_caption_request(image_source, now_ms())
}

/// Receive caption result and update state
pub fn receive_caption(
  state: RealtimeVisionState,
  caption: String,
) -> RealtimeVisionState {
  // Update most recent frame with caption
  let updated_frames = case state.recent_frames {
    [] -> []
    [latest, ..rest] -> {
      let updated = VisualFrame(..latest, caption: Some(caption))
      [updated, ..rest]
    }
  }

  // Clear pending request
  RealtimeVisionState(
    ..state,
    recent_frames: updated_frames,
    pending_caption: None,
  )
}

/// Check if there's a pending caption request
pub fn has_pending_caption(state: RealtimeVisionState) -> Bool {
  option.is_some(state.pending_caption)
}

/// Get pending caption request
pub fn get_pending_caption(state: RealtimeVisionState) -> Option(CaptionRequest) {
  state.pending_caption
}

// =============================================================================
// HRR INTEGRATION
// =============================================================================

/// Bind embedding with timestamp for temporal memory
pub fn bind_with_timestamp(
  embedding: HRR,
  timestamp: Int,
) -> Result(HRR, String) {
  // Create temporal phase encoding
  let dim = embedding.dim
  let time_encoding = temporal_phase_encoding(timestamp, dim)

  case hrr.bind(embedding, time_encoding) {
    Ok(bound) -> Ok(bound)
    Error(_) -> Error("HRR bind failed")
  }
}

/// Create temporal phase encoding for HRR binding
fn temporal_phase_encoding(timestamp: Int, dim: Int) -> HRR {
  // Encode time as periodic phases (hour, minute, second cycles)
  let hour_phase =
    int.to_float(timestamp / 3_600_000 % 24) /. 24.0 *. 2.0 *. pi()
  let minute_phase =
    int.to_float(timestamp / 60_000 % 60) /. 60.0 *. 2.0 *. pi()
  let second_phase = int.to_float(timestamp / 1000 % 60) /. 60.0 *. 2.0 *. pi()

  let data =
    list.range(0, dim - 1)
    |> list.map(fn(i) {
      let base_phase = 2.0 *. pi() *. int.to_float(i) /. int.to_float(dim)
      float_sin(hour_phase +. base_phase)
      +. float_cos(minute_phase +. base_phase *. 2.0)
      +. float_sin(second_phase +. base_phase *. 3.0)
    })

  hrr.from_list(normalize_vector(data))
}

/// Get similarity between current state and a probe embedding
pub fn probe_similarity(state: RealtimeVisionState, probe: HRR) -> Float {
  case state.previous_frame {
    None -> 0.0
    Some(frame) -> hrr.similarity(frame.embedding, probe)
  }
}

/// Search recent frames for similar embedding
pub fn search_recent(
  state: RealtimeVisionState,
  probe: HRR,
  threshold: Float,
) -> List(VisualFrame) {
  state.recent_frames
  |> list.filter(fn(frame) {
    hrr.similarity(frame.embedding, probe) >. threshold
  })
  |> list.sort(fn(a, b) {
    float.compare(
      hrr.similarity(b.embedding, probe),
      hrr.similarity(a.embedding, probe),
    )
  })
}

// =============================================================================
// METRICS & INTROSPECTION
// =============================================================================

/// Get current frame count
pub fn frame_count(state: RealtimeVisionState) -> Int {
  state.frame_count
}

/// Get average change magnitude
pub fn average_change(state: RealtimeVisionState) -> Float {
  state.avg_change
}

/// Get last frame timestamp
pub fn last_timestamp(state: RealtimeVisionState) -> Option(Int) {
  state.previous_frame
  |> option.map(fn(f) { f.timestamp })
}

/// Get buffer utilization
pub fn buffer_utilization(state: RealtimeVisionState) -> Float {
  int.to_float(list.length(state.recent_frames))
  /. int.to_float(state.buffer_size)
}

/// Describe current visual state
pub fn describe_state(state: RealtimeVisionState) -> String {
  let change_desc = case state.previous_frame {
    None -> "no previous frame"
    Some(frame) ->
      case frame.change_type {
        NoChange -> "stable"
        MinorChange -> "minor movement"
        MajorChange -> "significant change"
        DramaticChange -> "dramatic shift"
      }
  }

  let caption_desc = case state.pending_caption {
    None -> ""
    Some(_) -> " (caption pending)"
  }

  "Vision: "
  <> change_desc
  <> ", "
  <> int.to_string(state.frame_count)
  <> " frames, avg_change="
  <> float_to_string(state.avg_change, 3)
  <> caption_desc
}

// =============================================================================
// NV-CLIP INTEGRATION (FFI)
// =============================================================================

/// Get embedding from NV-CLIP server
fn get_clip_embedding(
  image_bytes: BitArray,
  url: String,
) -> Result(List(Float), VisionError) {
  case clip_embed_ffi(image_bytes, url) {
    Ok(embedding) -> Ok(embedding)
    Error(reason) -> Error(ClipConnectionFailed(reason))
  }
}

/// Get embedding from image path
pub fn get_embedding_from_path(
  image_path: String,
  config: VisionConfig,
) -> Result(HRR, VisionError) {
  case read_image_file(image_path) {
    Error(reason) -> Error(InvalidImageData(reason))
    Ok(bytes) -> {
      case get_clip_embedding(bytes, config.clip_url) {
        Ok(embedding_list) -> Ok(hrr.from_list(embedding_list))
        Error(e) -> Error(e)
      }
    }
  }
}

// =============================================================================
// OCR INTEGRATION (for text-aware vision)
// =============================================================================

/// Extract text from image using PaddleOCR
pub fn extract_text(
  image_bytes: BitArray,
  config: VisionConfig,
) -> Result(String, VisionError) {
  case ocr_extract_ffi(image_bytes, config.ocr_url) {
    Ok(text) -> Ok(text)
    Error(reason) -> Error(OcrConnectionFailed(reason))
  }
}

/// Extract text from image path
pub fn extract_text_from_path(
  image_path: String,
  config: VisionConfig,
) -> Result(String, VisionError) {
  case read_image_file(image_path) {
    Error(reason) -> Error(InvalidImageData(reason))
    Ok(bytes) -> extract_text(bytes, config)
  }
}

// =============================================================================
// HELPERS
// =============================================================================

/// Should request caption for this change type?
fn should_request_caption(change: VisualChange) -> Bool {
  case change {
    MajorChange -> True
    DramaticChange -> True
    _ -> False
  }
}

/// Create caption request
fn create_caption_request(source: String, timestamp: Int) -> CaptionRequest {
  CaptionRequest(
    id: int.to_string(timestamp) <> "_" <> string.slice(source, 0, 10),
    image_source: source,
    requested_at: timestamp,
    completed: False,
    result: None,
  )
}

/// Update running average with new value
fn update_running_average(avg: Float, new_value: Float) -> Float {
  // Exponential moving average with alpha = 0.1
  avg *. 0.9 +. new_value *. 0.1
}

/// Prepend to list with max size
fn prepend_bounded(item: a, items: List(a), max_size: Int) -> List(a) {
  [item, ..items]
  |> list.take(max_size)
}

/// Calculate HRR norm (magnitude)
fn hrr_norm(h: HRR) -> Float {
  let data = hrr_to_list(h)
  let sum_sq = list.fold(data, 0.0, fn(acc, x) { acc +. x *. x })
  float_sqrt(sum_sq)
}

/// Convert HRR to list (using internal vector)
fn hrr_to_list(h: HRR) -> List(Float) {
  // Access HRR's underlying vector via tensor
  viva_tensor_to_list(h.vector)
}

/// Normalize vector to unit length
fn normalize_vector(vec: List(Float)) -> List(Float) {
  let sum_sq = list.fold(vec, 0.0, fn(acc, x) { acc +. x *. x })
  let norm = float_sqrt(sum_sq)
  case norm >. 0.0001 {
    True -> list.map(vec, fn(x) { x /. norm })
    False -> vec
  }
}

/// Float to string with precision
fn float_to_string(f: Float, decimals: Int) -> String {
  let multiplier = float_pow(10.0, int.to_float(decimals))
  let rounded = float_round(f *. multiplier) /. multiplier
  float.to_string(rounded)
}

// =============================================================================
// FFI DECLARATIONS
// =============================================================================

/// Current time in milliseconds
@external(erlang, "erlang", "system_time")
fn erlang_system_time(unit: SystemTimeUnit) -> Int

type SystemTimeUnit {
  Millisecond
}

fn now_ms() -> Int {
  erlang_system_time(Millisecond)
}

/// Math functions
@external(erlang, "math", "sqrt")
fn float_sqrt(x: Float) -> Float

@external(erlang, "math", "sin")
fn float_sin(x: Float) -> Float

@external(erlang, "math", "cos")
fn float_cos(x: Float) -> Float

@external(erlang, "math", "pi")
fn pi() -> Float

@external(erlang, "math", "pow")
fn float_pow(base: Float, exp: Float) -> Float

@external(erlang, "erlang", "round")
fn float_round(x: Float) -> Float

/// File I/O
@external(erlang, "viva_vision_ffi", "read_file")
fn read_image_file(path: String) -> Result(BitArray, String)

/// NV-CLIP embedding FFI
/// Calls Python/HTTP service at localhost:8050
@external(erlang, "viva_vision_ffi", "clip_embed")
fn clip_embed_ffi(
  image_bytes: BitArray,
  url: String,
) -> Result(List(Float), String)

/// PaddleOCR FFI
/// Calls Python/HTTP service at localhost:8020
@external(erlang, "viva_vision_ffi", "ocr_extract")
fn ocr_extract_ffi(image_bytes: BitArray, url: String) -> Result(String, String)

/// Tensor to list conversion
@external(erlang, "viva_tensor_ffi", "tensor_to_list")
fn viva_tensor_to_list(tensor: a) -> List(Float)
