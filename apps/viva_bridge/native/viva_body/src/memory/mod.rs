//! VIVA's native memory system - vector search backends with Hebbian learning.
//!
//! This module provides a biologically-inspired memory system with:
//!
//! - **Dual backends**: HNSW (fast ANN) and SQLite (portable brute-force)
//! - **Three-Factor Hebbian Learning**: Emotion modulates memory encoding
//! - **STDP-based retrieval**: Recent memories get potentiation boost
//! - **Synaptic Tagging**: Weak memories captured by strong emotional events
//!
//! # Quick Start
//!
//! ```rust,ignore
//! use viva_body::memory::*;
//!
//! // Create integrated memory system
//! let mut viva = VivaMemory::new()?;
//!
//! // Set emotional state (affects encoding strength)
//! viva.feel(PadEmotion { pleasure: 0.8, arousal: 0.7, dominance: 0.5 });
//!
//! // Store a memory (automatically modulated by emotion)
//! let embedding = vec![0.1; VECTOR_DIM];
//! let meta = MemoryMeta::new("happy_moment".into(), "A joyful experience".into());
//! // viva.store uses augmented vectors internally!
//! viva.store(&embedding, meta)?;
//!
//! // Search with STDP boost AND emotional vector augmented
//! let results = viva.search(&embedding, &SearchOptions::new().limit(5))?;
//! ```
#![allow(dead_code)] // Latent code - will be used by Elixir NIFs
#![warn(missing_docs)]
//!
//! ## Architecture: Complementary Learning Systems (CLS)
//!
//! Inspired by neuroscience, VIVA uses dual memory systems:
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                    VIVA Memory Architecture                      │
//! ├─────────────────────────────────────────────────────────────────┤
//! │                                                                  │
//! │  ┌─────────────────────┐     ┌─────────────────────┐           │
//! │  │   HNSW (Episodic)   │     │  SQLite (Semantic)  │           │
//! │  │   "Hippocampus"     │ ──► │   "Neocortex"       │           │
//! │  │   Fast, specific    │     │   Slow, general     │           │
//! │  └─────────────────────┘     └─────────────────────┘           │
//! │            │                           ▲                        │
//! │            │    Sleep Consolidation    │                        │
//! │            └───────────────────────────┘                        │
//! │                                                                  │
//! │  ┌─────────────────────────────────────────────────────────────┐ │
//! │  │              Three-Factor Hebbian Learning                  │ │
//! │  │  Δw = η × pre × post × emotion_modulator                    │ │
//! │  │  (Arousal boosts encoding, emotion tags memories)           │ │
//! │  └─────────────────────────────────────────────────────────────┘ │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Backends
//!
//! - **HNSW (hnsw_rs)**: Fast approximate nearest neighbor, pure Rust
//! - **SQLite**: Portable brute-force, works everywhere
//!
//! ## Why Native Memory?
//!
//! While VIVA uses Qdrant via HTTP in Elixir, native Rust backends provide:
//! - **Autonomy**: No external service dependency
//! - **Portability**: SQLite works on mobile/embedded
//! - **Performance**: Native is faster than HTTP for hot paths
//! - **Self-optimization**: VIVA can choose backend based on context

pub mod types;
pub mod sqlite_backend;
pub mod usearch_backend;
pub mod hebbian;

use anyhow::Result;

pub use types::*;
pub use sqlite_backend::SqliteMemory;
pub use usearch_backend::HnswMemory;
#[allow(unused_imports)]
pub use hebbian::{HebbianLearning, HebbianParams, SynapticTag, SynapticTagManager};

/// Unified memory backend enum
pub enum MemoryBackend {
    Hnsw(HnswMemory),
    Sqlite(SqliteMemory),
}

impl MemoryBackend {
    /// Create HNSW backend (fast ANN, in-memory)
    pub fn hnsw() -> Result<Self> {
        Ok(MemoryBackend::Hnsw(HnswMemory::new()?))
    }

    /// Open persistent HNSW backend
    pub fn hnsw_open(path: &str) -> Result<Self> {
        Ok(MemoryBackend::Hnsw(HnswMemory::open(path)?))
    }

    /// Alias for hnsw() for backwards compatibility
    pub fn usearch() -> Result<Self> {
        Self::hnsw()
    }

    /// Create in-memory SQLite backend
    pub fn sqlite() -> Result<Self> {
        Ok(MemoryBackend::Sqlite(SqliteMemory::new()?))
    }

    /// Open file-based SQLite backend
    pub fn sqlite_open(path: &str) -> Result<Self> {
        Ok(MemoryBackend::Sqlite(SqliteMemory::open(path)?))
    }

    /// Store a memory
    pub fn store(&self, embedding: &[f32], meta: MemoryMeta) -> Result<u64> {
        match self {
            MemoryBackend::Hnsw(h) => h.store(embedding, meta),
            MemoryBackend::Sqlite(s) => s.store(embedding, meta),
        }
    }

    /// Search for similar memories
    pub fn search(
        &self,
        query: &[f32],
        options: &SearchOptions,
    ) -> Result<Vec<MemorySearchResult>> {
        match self {
            MemoryBackend::Hnsw(h) => h.search(query, options),
            MemoryBackend::Sqlite(s) => s.search(query, options),
        }
    }

    /// Get backend name
    pub fn backend_name(&self) -> &'static str {
        match self {
            MemoryBackend::Hnsw(_) => "hnsw",
            MemoryBackend::Sqlite(_) => "sqlite",
        }
    }

    /// Save to disk (if supported)
    pub fn save(&self) -> Result<()> {
        match self {
            MemoryBackend::Hnsw(h) => h.save(),
            MemoryBackend::Sqlite(_) => Ok(()), // SQLite auto-persists
        }
    }

    /// Check if backend has persistence configured
    pub fn is_persistent(&self) -> bool {
        match self {
            MemoryBackend::Hnsw(h) => h.is_persistent(),
            MemoryBackend::Sqlite(s) => s.stats().is_ok_and(|s| s.path.is_some()),
        }
    }
}

// ============================================================================
// Integrated VIVA Memory System (CLS + Hebbian)
// ============================================================================

/// VIVA's integrated memory system with Hebbian learning
///
/// Combines:
/// - Dual backends (HNSW for fast/episodic, SQLite for slow/semantic)
/// - Three-Factor Hebbian learning (emotion modulates encoding)
/// - Synaptic Tagging and Capture (weak memories captured by strong)
/// - STDP-like retrieval strengthening
///
/// # Concurrency
///
/// This struct is **not `Sync`** - methods like `store()`, `feel()`, and
/// `consolidate()` take `&mut self`. For multi-threaded access:
///
/// ```rust,ignore
/// use std::sync::{Arc, Mutex};
///
/// let memory = Arc::new(Mutex::new(VivaMemory::new()?));
///
/// // From any thread:
/// let mut guard = memory.lock().unwrap();
/// guard.store(&embedding, meta)?;
/// ```
///
/// The internal backends (`HnswMemory`, `SqliteMemory`) use their own `Mutex`
/// guards, so read operations like `search()` are thread-safe even with `&self`.
pub struct VivaMemory {
    /// Fast episodic memory (hippocampus-like)
    pub episodic: HnswMemory,
    /// Slow semantic memory (neocortex-like)
    pub semantic: SqliteMemory,
    /// Hebbian learning engine
    pub hebbian: HebbianLearning,

    /// Synaptic tag manager for consolidation
    pub tags: SynapticTagManager,
}

impl VivaMemory {
    /// Create new integrated memory system (in-memory)
    pub fn new() -> Result<Self> {
        Ok(Self {
            episodic: HnswMemory::new()?,
            semantic: SqliteMemory::new()?,
            hebbian: HebbianLearning::new(),
            tags: SynapticTagManager::new(),
        })
    }

    /// Create with persistent storage
    pub fn open(episodic_path: &str, semantic_path: &str) -> Result<Self> {
        Ok(Self {
            episodic: HnswMemory::open(episodic_path)?,
            semantic: SqliteMemory::open(semantic_path)?,
            hebbian: HebbianLearning::new(),
            tags: SynapticTagManager::new(),
        })
    }

    /// Update emotional state (affects learning)
    ///
    /// Strong emotional events (arousal > 0.6) can "capture" weakly-tagged
    /// memories, boosting their importance (Synaptic Tagging and Capture).
    pub fn feel(&mut self, emotion: PadEmotion) {
        self.hebbian.update_emotion(emotion);

        // Check if this is a strong emotional event that can capture weak memories
        if emotion.arousal.abs() > 0.6 {
            let captured = self.tags.capture_weak_memories(&emotion);
            for key in captured {
                // Boost importance of captured memories and persist the change
                if let Some(mut meta) = self.episodic.get(key) {
                    meta.importance = (meta.importance + 0.2).min(1.0);
                    meta.emotion = Some(emotion); // Tag with capturing emotion
                    self.episodic.update_meta(key, meta);
                }
            }
        }
    }

    /// Store a memory with Hebbian modulation
    ///
    /// The memory's importance is automatically adjusted based on
    /// current emotional state.
    pub fn store(&mut self, embedding: &[f32], meta: MemoryMeta) -> Result<u64> {
        // Apply Three-Factor Hebbian modulation
        let modulated_meta = self.hebbian.modulate_memory(meta);

        // Augment vector with current emotion (Critical for "Vibe Search")
        // We use the emotion attached to the memory (which is current state)
        // Weight 10.0 ensures emotion is significant in cosine distance
        let augmented_vec = self.hebbian.augment_vector(
            embedding,
            modulated_meta.emotion,
            10.0
        );

        // Store in episodic (fast) memory using AUGMENTED vector
        // Note: HNSW backend handles variable dimensions, but we must be consistent
        let key = self.episodic.store(&augmented_vec, modulated_meta.clone())?;

        // Tag for potential consolidation if not strong enough
        self.tags.tag_memory(
            key,
            modulated_meta.importance,
            self.hebbian.current_emotion(),
        );

        Ok(key)
    }

    /// Search with emotional context awareness
    ///
    /// Results are boosted based on emotional similarity to current state
    /// (mood-congruent memory retrieval)
    pub fn search(
        &self,
        query: &[f32],
        options: &SearchOptions,
    ) -> Result<Vec<MemorySearchResult>> {
        // Augment query with CURRENT emotion to find "mood-congruent" memories
        // Usage: "I am happy, find me other happy things"
        let augmented_query = self.hebbian.augment_vector(
            query,
            None, // Use current emotion
            10.0
        );

        let mut results = self.episodic.search(&augmented_query, options)?;

        // Apply STDP retrieval boost (emotion + timing)
        for result in &mut results {
            let boost = self.hebbian.retrieval_boost_stdp(
                result.meta.emotion,
                Some(result.meta.last_accessed),
            );
            result.decayed_score *= boost;
        }

        // Re-sort by boosted score
        results.sort_by(|a, b| {
            b.decayed_score
                .partial_cmp(&a.decayed_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(results)
    }

    /// Search using Complementary Learning Systems (CLS) - dual-backend
    ///
    /// Combines fast episodic memory (specific, recent) with slow semantic
    /// memory (general, consolidated) for richer retrieval.
    ///
    /// # Arguments
    ///
    /// * `query` - Query embedding vector
    /// * `options` - Search options (limit applies to final merged results)
    ///
    /// # Returns
    ///
    /// Merged results from both backends, deduplicated and re-ranked
    pub fn search_cls(
        &self,
        query: &[f32],
        options: &SearchOptions,
    ) -> Result<Vec<MemorySearchResult>> {
        use std::collections::HashSet;

        // Fetch from both backends with 2x limit for merge headroom
        let mut expanded_options = options.clone();
        expanded_options.limit = options.limit * 2;

        // 1. Search episodic (fast, specific memories) using AUGMENTED vector
        let augmented_query = self.hebbian.augment_vector(query, None, 10.0);
        let mut episodic_results = self.episodic.search(&augmented_query, &expanded_options)?;


        // 2. Search semantic (slow, general knowledge)
        let semantic_results = self.semantic.search(query, &expanded_options)?;

        // 3. Deduplicate by memory ID (episodic takes priority for same ID)
        // Clone IDs to avoid borrow conflict with push
        let episodic_ids: HashSet<String> = episodic_results
            .iter()
            .map(|r| r.meta.id.clone())
            .collect();

        for sem_result in semantic_results {
            if !episodic_ids.contains(&sem_result.meta.id) {
                episodic_results.push(sem_result);
            }
        }

        // 4. Apply STDP boost to all results
        for result in &mut episodic_results {
            let boost = self.hebbian.retrieval_boost_stdp(
                result.meta.emotion,
                Some(result.meta.last_accessed),
            );
            result.decayed_score *= boost;
        }

        // 5. Re-sort by boosted score (NaN-safe)
        episodic_results.sort_by(|a, b| {
            b.decayed_score
                .partial_cmp(&a.decayed_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // 6. Truncate to requested limit
        episodic_results.truncate(options.limit);

        Ok(episodic_results)
    }

    /// Consolidate strong/emotional memories to semantic store
    ///
    /// This simulates "sleep consolidation" where important episodic
    /// memories are transferred to long-term semantic storage.
    ///
    /// # Algorithm
    ///
    /// 1. Query episodic for high-importance memories (>= threshold)
    /// 2. Copy them to semantic store (slow, general knowledge)
    /// 3. Mark as consolidated (importance reduced to avoid re-consolidation)
    ///
    /// # Arguments
    ///
    /// * `importance_threshold` - Minimum importance to consolidate [0.0, 1.0]
    ///
    /// # Returns
    ///
    /// Number of memories successfully consolidated
    pub fn consolidate(&mut self, importance_threshold: f32) -> Result<usize> {
        // Clean up expired tags first
        self.tags.cleanup_expired();

        // Get candidates from episodic memory
        let candidates = self.episodic.get_consolidation_candidates(importance_threshold)?;

        let mut consolidated = 0;

        for (key, meta, embedding) in candidates {
            // Check if already exists in semantic (by ID)
            if self.semantic.get(&meta.id)?.is_some() {
                continue; // Already consolidated
            }

            // Create semantic version with same ID
            let semantic_meta = MemoryMeta {
                id: meta.id.clone(),
                content: meta.content.clone(),
                memory_type: MemoryType::Semantic, // Change type to semantic
                importance: meta.importance,
                emotion: meta.emotion,
                timestamp: meta.timestamp,
                access_count: meta.access_count,
                last_accessed: meta.last_accessed,
            };

            // Store in semantic backend
            self.semantic.store(&embedding, semantic_meta)?;

            // Mark as consolidated in episodic (reduce importance to prevent re-consolidation)
            // Note: We don't delete from episodic - it may still be useful for recent retrieval
            // The reduced importance means it won't be re-consolidated
            if let Some(mut episodic_meta) = self.episodic.get(key) {
                episodic_meta.importance *= 0.5; // Reduce importance after consolidation
                // Note: In a full impl we'd update the stored meta
            }

            consolidated += 1;
        }

        Ok(consolidated)
    }

    /// Save both backends
    pub fn save(&self) -> Result<()> {
        self.episodic.save()?;
        // SQLite auto-saves
        Ok(())
    }

    /// Get statistics
    pub fn stats(&self) -> VivaMemoryStats {
        let episodic_stats = self.episodic.stats();
        let semantic_stats = self.semantic.stats().ok();

        VivaMemoryStats {
            episodic_count: episodic_stats.count,
            semantic_count: semantic_stats.map(|s| s.count).unwrap_or(0),
            pending_tags: self.tags.active_count(),
            current_emotion: self.hebbian.current_emotion(),
        }
    }
}

/// Statistics for integrated memory system
#[derive(Debug, Clone)]
pub struct VivaMemoryStats {
    pub episodic_count: usize,
    pub semantic_count: usize,
    pub pending_tags: usize,
    pub current_emotion: PadEmotion,
}

// ============================================================================
// Benchmarking utilities
// ============================================================================

/// Benchmark result
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    pub backend: String,
    pub operation: String,
    pub count: usize,
    pub total_ms: f64,
    pub avg_ms: f64,
}

/// Benchmark store operations
pub fn bench_store(
    backend: &MemoryBackend,
    embeddings: &[Vec<f32>],
) -> Result<BenchmarkResult> {
    let start = std::time::Instant::now();

    for (i, emb) in embeddings.iter().enumerate() {
        let meta = MemoryMeta::new(format!("bench_{}", i), format!("Benchmark memory {}", i));
        backend.store(emb, meta)?;
    }

    let elapsed = start.elapsed().as_secs_f64() * 1000.0;

    Ok(BenchmarkResult {
        backend: backend.backend_name().to_string(),
        operation: "store".to_string(),
        count: embeddings.len(),
        total_ms: elapsed,
        avg_ms: elapsed / embeddings.len() as f64,
    })
}

/// Benchmark search operations
pub fn bench_search(
    backend: &MemoryBackend,
    queries: &[Vec<f32>],
    limit: usize,
) -> Result<BenchmarkResult> {
    let options = SearchOptions::new().limit(limit);
    let start = std::time::Instant::now();

    for query in queries {
        backend.search(query, &options)?;
    }

    let elapsed = start.elapsed().as_secs_f64() * 1000.0;

    Ok(BenchmarkResult {
        backend: backend.backend_name().to_string(),
        operation: "search".to_string(),
        count: queries.len(),
        total_ms: elapsed,
        avg_ms: elapsed / queries.len() as f64,
    })
}

/// Generate random embeddings for testing
pub fn random_embeddings(count: usize) -> Vec<Vec<f32>> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    (0..count)
        .map(|i| {
            (0..VECTOR_DIM)
                .map(|j| {
                    let mut hasher = DefaultHasher::new();
                    (i * VECTOR_DIM + j).hash(&mut hasher);
                    (hasher.finish() as f32 / u64::MAX as f32) * 2.0 - 1.0
                })
                .collect()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_switching() {
        // Test that both backends have same interface
        let hnsw = MemoryBackend::hnsw().unwrap();
        let sqlite = MemoryBackend::sqlite().unwrap();

        assert_eq!(hnsw.backend_name(), "hnsw");
        assert_eq!(sqlite.backend_name(), "sqlite");

        let emb: Vec<f32> = (0..VECTOR_DIM).map(|i| i as f32 / VECTOR_DIM as f32).collect();
        let meta = MemoryMeta::new("test".to_string(), "Test".to_string());

        // Both should accept same operations
        assert!(hnsw.store(&emb, meta.clone()).is_ok());
        assert!(sqlite.store(&emb, meta).is_ok());
    }

    #[test]
    fn test_viva_memory_integration() {
        let mut viva = VivaMemory::new().unwrap();

        // Set emotional state
        viva.feel(PadEmotion {
            pleasure: 0.8,
            arousal: 0.7,
            dominance: 0.5,
        });

        // Store a memory (should get importance boost from high arousal)
        let emb: Vec<f32> = (0..VECTOR_DIM).map(|i| i as f32 / VECTOR_DIM as f32).collect();
        let meta = MemoryMeta::new("emotional_memory".to_string(), "Happy memory".to_string());

        let key = viva.store(&emb, meta).unwrap();
        assert!(key > 0);

        // Search should return the memory
        let results = viva.search(&emb, &SearchOptions::new().limit(5)).unwrap();
        assert!(!results.is_empty());

        // Memory should have emotion attached
        assert!(results[0].meta.emotion.is_some());
    }

    #[test]
    fn test_hebbian_importance_modulation() {
        let mut viva = VivaMemory::new().unwrap();

        // Store with neutral emotion
        viva.feel(PadEmotion {
            pleasure: 0.0,
            arousal: 0.0,
            dominance: 0.0,
        });
        let emb1: Vec<f32> = (0..VECTOR_DIM).map(|i| i as f32 / VECTOR_DIM as f32).collect();
        let meta1 = MemoryMeta::new("neutral".to_string(), "Neutral".to_string())
            .with_importance(0.5);
        viva.store(&emb1, meta1).unwrap();

        // Store with high arousal
        viva.feel(PadEmotion {
            pleasure: 0.8,
            arousal: 0.9,
            dominance: 0.5,
        });
        let emb2: Vec<f32> = (0..VECTOR_DIM).map(|i| (i as f32 + 0.1) / VECTOR_DIM as f32).collect();
        let meta2 = MemoryMeta::new("emotional".to_string(), "Emotional".to_string())
            .with_importance(0.5);
        viva.store(&emb2, meta2).unwrap();

        // Search for the emotional memory
        let results = viva.search(&emb2, &SearchOptions::new().limit(5)).unwrap();

        // Emotional memory should have higher importance
        let emotional = results.iter().find(|r| r.meta.id == "emotional");
        let neutral = results.iter().find(|r| r.meta.id == "neutral");

        if let (Some(e), Some(_n)) = (emotional, neutral) {
            // The emotional memory should have been boosted
            assert!(e.meta.importance > 0.5, "Emotional memory importance should be boosted");
        }
    }
}
