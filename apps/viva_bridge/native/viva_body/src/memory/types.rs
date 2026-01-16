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
    pub fn as_str(&self) -> &'static str {
        match self {
            MemoryType::Episodic => "episodic",
            MemoryType::Semantic => "semantic",
            MemoryType::Emotional => "emotional",
            MemoryType::Procedural => "procedural",
            MemoryType::Generic => "generic",
        }
    }

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
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct PadEmotion {
    pub pleasure: f32,   // -1.0 to 1.0
    pub arousal: f32,    // -1.0 to 1.0
    pub dominance: f32,  // -1.0 to 1.0
}

/// Memory metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryMeta {
    pub id: String,
    pub content: String,
    pub memory_type: MemoryType,
    pub importance: f32,
    pub emotion: Option<PadEmotion>,
    pub timestamp: i64,
    pub access_count: u32,
    pub last_accessed: i64,
}

impl MemoryMeta {
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

    pub fn with_type(mut self, t: MemoryType) -> Self {
        self.memory_type = t;
        self
    }

    pub fn with_importance(mut self, i: f32) -> Self {
        self.importance = i.clamp(0.0, 1.0);
        self
    }

    pub fn with_emotion(mut self, e: PadEmotion) -> Self {
        self.emotion = Some(e);
        self
    }
}

/// Search options
#[derive(Debug, Clone)]
pub struct SearchOptions {
    pub limit: usize,
    pub memory_type: Option<MemoryType>,
    pub min_importance: f32,
    pub apply_decay: bool,
    pub decay_scale: f64,
}

impl SearchOptions {
    pub fn new() -> Self {
        Self {
            limit: 10,
            memory_type: None,
            min_importance: 0.0,
            apply_decay: true,
            decay_scale: 604_800.0, // 1 week in seconds
        }
    }

    pub fn limit(mut self, l: usize) -> Self {
        self.limit = l;
        self
    }

    pub fn of_type(mut self, t: MemoryType) -> Self {
        self.memory_type = Some(t);
        self
    }

    pub fn min_importance(mut self, i: f32) -> Self {
        self.min_importance = i;
        self
    }

    pub fn no_decay(mut self) -> Self {
        self.apply_decay = false;
        self
    }
}

impl Default for SearchOptions {
    fn default() -> Self {
        Self::new()
    }
}

/// Search result with scores
#[derive(Debug, Clone)]
pub struct MemorySearchResult {
    pub meta: MemoryMeta,
    pub similarity: f32,
    pub decayed_score: f32,
}

/// Calculate temporal decay (Ebbinghaus curve)
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
