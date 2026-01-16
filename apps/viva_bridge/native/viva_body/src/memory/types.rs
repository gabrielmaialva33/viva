#![allow(dead_code)]
//! Memory types and structures for VIVA's vector memory system.
//!
//! Shared types between backends (usearch, SQLite).

use serde::{Deserialize, Serialize};

/// Vector dimension for embeddings (e.g., 384 for all-MiniLM-L6-v2)
pub const VECTOR_DIM: usize = 384;

/// Memory type classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemoryType {
    /// Specific events with timestamp
    Episodic,
    /// General knowledge and patterns
    Semantic,
    /// Emotion-tagged memories
    Emotional,
    /// Skills and procedures
    Procedural,
    /// Unclassified
    Generic,
}

impl MemoryType {
    /// Convert to lowercase string representation
    pub fn as_str(&self) -> &'static str {
        match self {
            MemoryType::Episodic => "episodic",
            MemoryType::Semantic => "semantic",
            MemoryType::Emotional => "emotional",
            MemoryType::Procedural => "procedural",
            MemoryType::Generic => "generic",
        }
    }

    /// Parse from string (case-insensitive)
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "episodic" => MemoryType::Episodic,
            "semantic" => MemoryType::Semantic,
            "emotional" => MemoryType::Emotional,
            "procedural" => MemoryType::Procedural,
            _ => MemoryType::Generic,
        }
    }
}

impl Default for MemoryType {
    fn default() -> Self {
        MemoryType::Generic
    }
}

/// PAD emotion model (Pleasure-Arousal-Dominance)
///
/// A three-dimensional model of affect used in psychology and affective computing.
/// Each dimension ranges from -1.0 to 1.0.
///
/// # Examples
///
/// ```rust,ignore
/// // Joy: high pleasure, medium arousal, high dominance
/// let joy = PadEmotion { pleasure: 0.8, arousal: 0.5, dominance: 0.7 };
///
/// // Fear: low pleasure, high arousal, low dominance
/// let fear = PadEmotion { pleasure: -0.6, arousal: 0.9, dominance: -0.5 };
/// ```
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct PadEmotion {
    /// Pleasure/Valence: -1.0 (unhappy) to 1.0 (happy)
    pub pleasure: f32,
    /// Arousal/Activation: -1.0 (calm) to 1.0 (excited)
    pub arousal: f32,
    /// Dominance/Control: -1.0 (submissive) to 1.0 (dominant)
    pub dominance: f32,
}

/// Memory metadata containing all information about a stored memory
///
/// Includes content, classification, importance, emotional context, and access statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryMeta {
    /// Unique identifier for this memory
    pub id: String,
    /// The actual content/text of the memory
    pub content: String,
    /// Classification (episodic, semantic, emotional, procedural, generic)
    pub memory_type: MemoryType,
    /// Importance score [0.0, 1.0] - modulated by Hebbian learning
    pub importance: f32,
    /// Emotional state when memory was created (for mood-congruent retrieval)
    pub emotion: Option<PadEmotion>,
    /// Unix timestamp when memory was created
    pub timestamp: i64,
    /// Number of times this memory has been accessed
    pub access_count: u32,
    /// Unix timestamp of last access (used for STDP)
    pub last_accessed: i64,
}

impl MemoryMeta {
    /// Create new memory with default importance (0.5) and current timestamp
    pub fn new(id: String, content: String) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;

        Self {
            id,
            content,
            memory_type: MemoryType::Generic,
            importance: 0.5,
            emotion: None,
            timestamp: now,
            access_count: 0,
            last_accessed: now,
        }
    }

    /// Set memory type (builder pattern)
    pub fn with_type(mut self, t: MemoryType) -> Self {
        self.memory_type = t;
        self
    }

    /// Set importance score, clamped to [0.0, 1.0] (builder pattern)
    pub fn with_importance(mut self, i: f32) -> Self {
        self.importance = i.clamp(0.0, 1.0);
        self
    }

    /// Attach emotional context (builder pattern)
    pub fn with_emotion(mut self, e: PadEmotion) -> Self {
        self.emotion = Some(e);
        self
    }
}

/// Time range for temporal filtering
#[derive(Debug, Clone, Copy, Default)]
pub struct TimeRange {
    /// Start timestamp (inclusive), None = no lower bound
    pub start: Option<i64>,
    /// End timestamp (inclusive), None = no upper bound
    pub end: Option<i64>,
}

impl TimeRange {
    /// Create unbounded time range (matches all)
    pub fn all() -> Self {
        Self { start: None, end: None }
    }

    /// Create range from start to now
    pub fn since(start: i64) -> Self {
        Self { start: Some(start), end: None }
    }

    /// Create range from epoch to end
    pub fn until(end: i64) -> Self {
        Self { start: None, end: Some(end) }
    }

    /// Create bounded range [start, end]
    pub fn between(start: i64, end: i64) -> Self {
        Self { start: Some(start), end: Some(end) }
    }

    /// Last N seconds from now
    pub fn last_seconds(seconds: i64) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;
        Self::since(now - seconds)
    }

    /// Last N hours
    pub fn last_hours(hours: i64) -> Self {
        Self::last_seconds(hours * 3600)
    }

    /// Last N days
    pub fn last_days(days: i64) -> Self {
        Self::last_seconds(days * 86400)
    }

    /// Check if timestamp is within range
    pub fn contains(&self, timestamp: i64) -> bool {
        let after_start = self.start.map_or(true, |s| timestamp >= s);
        let before_end = self.end.map_or(true, |e| timestamp <= e);
        after_start && before_end
    }
}

/// Search options for memory queries
///
/// Builder pattern for configuring memory search parameters.
///
/// # Example
///
/// ```rust,ignore
/// let options = SearchOptions::new()
///     .limit(10)
///     .of_type(MemoryType::Episodic)
///     .min_importance(0.3)
///     .last_days(7);
/// ```
#[derive(Debug, Clone)]
pub struct SearchOptions {
    /// Maximum number of results to return
    pub limit: usize,
    /// Filter by memory type (None = all types)
    pub memory_type: Option<MemoryType>,
    /// Minimum importance threshold [0.0, 1.0]
    pub min_importance: f32,
    /// Whether to apply Ebbinghaus decay to scores
    pub apply_decay: bool,
    /// Decay half-life in seconds (default: 1 week)
    pub decay_scale: f64,
    /// Temporal filter - only return memories within this time range
    pub time_range: TimeRange,
}

impl SearchOptions {
    /// Create default search options (limit=10, all types, with decay)
    pub fn new() -> Self {
        Self {
            limit: 10,
            memory_type: None,
            min_importance: 0.0,
            apply_decay: true,
            decay_scale: 604_800.0, // 1 week in seconds
            time_range: TimeRange::all(),
        }
    }

    /// Set maximum number of results
    pub fn limit(mut self, l: usize) -> Self {
        self.limit = l;
        self
    }

    /// Filter by memory type
    pub fn of_type(mut self, t: MemoryType) -> Self {
        self.memory_type = Some(t);
        self
    }

    /// Set minimum importance threshold
    pub fn min_importance(mut self, i: f32) -> Self {
        self.min_importance = i;
        self
    }

    /// Disable temporal decay (raw similarity scores)
    pub fn no_decay(mut self) -> Self {
        self.apply_decay = false;
        self
    }

    /// Filter by time range
    pub fn time_range(mut self, range: TimeRange) -> Self {
        self.time_range = range;
        self
    }

    /// Convenience: only memories from last N days
    pub fn last_days(mut self, days: i64) -> Self {
        self.time_range = TimeRange::last_days(days);
        self
    }

    /// Convenience: only memories from last N hours
    pub fn last_hours(mut self, hours: i64) -> Self {
        self.time_range = TimeRange::last_hours(hours);
        self
    }
}

impl Default for SearchOptions {
    fn default() -> Self {
        Self::new()
    }
}

/// Search result with similarity and decay-adjusted scores
///
/// Contains the memory metadata plus scoring information for ranking.
#[derive(Debug, Clone)]
pub struct MemorySearchResult {
    /// Full memory metadata
    pub meta: MemoryMeta,
    /// Raw cosine similarity [0.0, 1.0] (or [-1.0, 1.0] for unnormalized)
    pub similarity: f32,
    /// Decay-adjusted score combining similarity and recency
    pub decayed_score: f32,
}

/// Calculate temporal decay using Ebbinghaus forgetting curve
///
/// Returns a value in [0.0, 1.0] where:
/// - 1.0 = just created
/// - 0.5 ≈ decay_scale seconds ago
/// - → 0.0 as time increases
///
/// # Arguments
///
/// * `timestamp` - Unix timestamp of memory creation
/// * `decay_scale` - Half-life in seconds (default: 604800 = 1 week)
pub fn calculate_decay(timestamp: i64, decay_scale: f64) -> f32 {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64;

    let age = (now - timestamp).max(0) as f64;
    (-age / decay_scale).exp() as f32
}

/// Generate unique memory ID
pub fn generate_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    format!("mem_{:x}", timestamp)
}
