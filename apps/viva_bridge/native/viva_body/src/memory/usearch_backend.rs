#![allow(dead_code)]
//! HNSW backend using hnsw_rs for fast approximate nearest neighbor search.
//!
//! Pure Rust implementation - no C++ dependencies.
//! Ideal for high-performance vector search on larger datasets.
//!
//! ## Persistence
//!
//! This backend now supports full persistence:
//! - HNSW index saved via `file_dump()` (binary format)
//! - Metadata saved as JSON (human-readable, debuggable)
//!
//! Files created:
//! - `<name>.hnsw.data` - HNSW graph data
//! - `<name>.hnsw.graph` - HNSW graph structure
//! - `<name>.meta.json` - Metadata (MemoryMeta entries)

use crate::memory::types::*;
use anndists::dist::DistCosine;
use hnsw_rs::api::AnnT; // Trait that provides file_dump()
use hnsw_rs::hnsw::Hnsw;
use hnsw_rs::hnswio::HnswIo;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufReader, BufWriter};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

/// Serializable metadata state for persistence
#[derive(Serialize, Deserialize)]
struct PersistentState {
    metadata: HashMap<u64, MemoryMeta>,
    next_key: u64,
    version: u32,
}

impl PersistentState {
    const CURRENT_VERSION: u32 = 1;

    fn new() -> Self {
        Self {
            metadata: HashMap::new(),
            next_key: 1,
            version: Self::CURRENT_VERSION,
        }
    }
}

/// HNSW-based memory backend for fast ANN search
///
/// ## Architecture Decision: Hippocampus-like Fast Memory
///
/// In the Complementary Learning Systems (CLS) theory, the hippocampus
/// provides fast, episodic memory storage. This HNSW backend serves that
/// role in VIVA - rapid storage and retrieval of specific memories.
///
/// The SQLite backend serves as the "neocortex" - slower but more
/// persistent semantic memory.
pub struct HnswMemory {
    /// HNSW index for vector search
    index: Mutex<Hnsw<'static, f32, DistCosine>>,
    /// Metadata storage (key -> MemoryMeta)
    meta: Mutex<HashMap<u64, MemoryMeta>>,
    /// Next available key
    next_key: Mutex<u64>,
    /// Path for persistence (None = in-memory only)
    storage_path: Option<PathBuf>,
}

impl HnswMemory {
    /// Create new in-memory HNSW index (no persistence)
    ///
    /// Parameters tuned for ~100k vectors:
    /// - max_nb_connection (M): 16 - edges per node
    /// - ef_construction: 200 - search width during construction
    /// - max_elements: 100_000 - pre-allocated capacity
    pub fn new() -> Result<Self, String> {
        Self::with_capacity(100_000)
    }

    /// Create new in-memory HNSW index with custom capacity
    pub fn with_capacity(max_elements: usize) -> Result<Self, String> {
        let index = Hnsw::<f32, DistCosine>::new(
            16,           // max_nb_connection (M)
            max_elements, // max_elements
            16,           // max_layer
            200,          // ef_construction
            DistCosine {},   // distance metric (dot product for normalized vectors = cosine)
        );

        Ok(Self {
            index: Mutex::new(index),
            meta: Mutex::new(HashMap::new()),
            next_key: Mutex::new(1),
            storage_path: None,
        })
    }

    /// Open or create a persistent HNSW index at the given path
    ///
    /// If files exist at path, loads them. Otherwise creates new index.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        let path = path.as_ref().to_path_buf();
        let meta_path = Self::meta_file_path(&path);

        // Check if existing data exists
        if meta_path.exists() {
            Self::load_from_path(&path)
        } else {
            // Create new with persistence path set
            let mut mem = Self::new()?;
            mem.storage_path = Some(path);
            Ok(mem)
        }
    }

    /// Load existing HNSW index from disk
    fn load_from_path(base_path: &Path) -> Result<Self, String> {
        let dir = base_path
            .parent()
            .ok_or_else(|| "Invalid path: no parent directory".to_string())?;
        let name = base_path
            .file_name()
            .and_then(|n| n.to_str())
            .ok_or_else(|| "Invalid path: no filename".to_string())?;

        // Load HNSW index
        // Note: Box::leak is used because HnswIo::load_hnsw returns Hnsw<'a>
        // with lifetime tied to the reloader. Since we only load once per
        // open() call and the index lives for the program's lifetime, this
        // is an acceptable tradeoff (small one-time memory cost).
        let reloader = Box::leak(Box::new(HnswIo::new(dir, name)));
        let index: Hnsw<'static, f32, DistCosine> = reloader
            .load_hnsw()
            .map_err(|e| format!("Failed to load HNSW index: {:?}", e))?;

        // Load metadata
        let meta_path = Self::meta_file_path(base_path);
        let file = File::open(&meta_path)
            .map_err(|e| format!("Failed to open metadata file: {}", e))?;
        let reader = BufReader::new(file);
        let state: PersistentState = serde_json::from_reader(reader)
            .map_err(|e| format!("Failed to parse metadata: {}", e))?;

        // Version check for future migrations
        if state.version > PersistentState::CURRENT_VERSION {
            return Err(format!(
                "Metadata version {} is newer than supported version {}",
                state.version,
                PersistentState::CURRENT_VERSION
            ));
        }

        Ok(Self {
            index: Mutex::new(index),
            meta: Mutex::new(state.metadata),
            next_key: Mutex::new(state.next_key),
            storage_path: Some(base_path.to_path_buf()),
        })
    }

    /// Get path for metadata JSON file
    fn meta_file_path(base_path: &Path) -> PathBuf {
        let mut meta_path = base_path.to_path_buf();
        let name = meta_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("hnsw");
        meta_path.set_file_name(format!("{}.meta.json", name));
        meta_path
    }

    /// Store a memory with its embedding
    pub fn store(&self, embedding: &[f32], meta: MemoryMeta) -> Result<u64, String> {
        if embedding.len() != VECTOR_DIM {
            return Err(format!(
                "Invalid embedding dimension: {} (expected {})",
                embedding.len(),
                VECTOR_DIM
            ));
        }

        // Normalize for cosine similarity via dot product
        let normalized = normalize_vector(embedding);

        // Get next key
        let key = {
            let mut next = self.next_key.lock().map_err(|e| format!("Lock error: {}", e))?;
            let k = *next;
            *next += 1;
            k
        };

        // Insert into HNSW index
        {
            let index = self.index.lock().map_err(|e| format!("Lock error: {}", e))?;
            index.insert((&normalized, key as usize));
        }

        // Store metadata
        {
            let mut meta_map = self.meta.lock().map_err(|e| format!("Lock error: {}", e))?;
            meta_map.insert(key, meta);
        }

        Ok(key)
    }

    /// Search for similar memories using HNSW
    pub fn search(
        &self,
        query: &[f32],
        options: &SearchOptions,
    ) -> Result<Vec<MemorySearchResult>, String> {
        if query.len() != VECTOR_DIM {
            return Err(format!(
                "Invalid query dimension: {} (expected {})",
                query.len(),
                VECTOR_DIM
            ));
        }

        // Normalize query
        let normalized = normalize_vector(query);

        // Search HNSW - get more results than needed for filtering
        let ef_search = (options.limit * 3).max(50);
        let neighbors = {
            let index = self.index.lock().map_err(|e| format!("Lock error: {}", e))?;
            index.search(&normalized, options.limit * 2, ef_search)
        };

        let meta_map = self.meta.lock().map_err(|e| format!("Lock error: {}", e))?;
        let mut results: Vec<MemorySearchResult> = Vec::new();

        for neighbor in neighbors {
            let key = neighbor.d_id as u64;
            let similarity = 1.0 - neighbor.distance; // Convert distance to similarity

            if let Some(meta) = meta_map.get(&key) {
                // Apply filters
                if let Some(ref filter_type) = options.memory_type {
                    if meta.memory_type != *filter_type {
                        continue;
                    }
                }

                if meta.importance < options.min_importance {
                    continue;
                }

                // Calculate decayed score
                let decayed_score = if options.apply_decay {
                    let decay = calculate_decay(meta.timestamp, options.decay_scale);
                    0.7 * similarity + 0.3 * decay
                } else {
                    similarity
                };

                results.push(MemorySearchResult {
                    meta: meta.clone(),
                    similarity,
                    decayed_score,
                });
            }
        }

        // Sort by decayed score
        results.sort_by(|a, b| {
            b.decayed_score
                .partial_cmp(&a.decayed_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(options.limit);

        Ok(results)
    }

    /// Get memory by key (with STDP-inspired access strengthening)
    ///
    /// Each access increments access_count, implementing a basic
    /// "use it or lose it" principle from synaptic plasticity.
    pub fn get(&self, key: u64) -> Option<MemoryMeta> {
        let mut meta_map = self.meta.lock().ok()?;

        if let Some(meta) = meta_map.get_mut(&key) {
            // Update access stats (STDP: retrieval strengthens)
            meta.access_count += 1;
            meta.last_accessed = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs() as i64)
                .unwrap_or(0);

            Some(meta.clone())
        } else {
            None
        }
    }

    /// Delete a memory (only from metadata, HNSW doesn't support deletion well)
    ///
    /// Note: The vector remains in the HNSW index but won't be returned
    /// since metadata is gone. This is a pragmatic workaround for HNSW's
    /// limitation with deletions.
    pub fn forget(&self, key: u64) -> Result<(), String> {
        let mut meta_map = self.meta.lock().map_err(|e| format!("Lock error: {}", e))?;
        meta_map.remove(&key);
        Ok(())
    }

    /// Save index and metadata to disk
    ///
    /// Requires that the index was opened with a path (via `open()`).
    /// For in-memory indexes created with `new()`, this will return an error.
    pub fn save(&self) -> Result<(), String> {
        let base_path = self
            .storage_path
            .as_ref()
            .ok_or_else(|| "No storage path configured. Use HnswMemory::open() for persistence.".to_string())?;

        let dir = base_path
            .parent()
            .ok_or_else(|| "Invalid path: no parent directory".to_string())?;
        let name = base_path
            .file_name()
            .and_then(|n| n.to_str())
            .ok_or_else(|| "Invalid path: no filename".to_string())?;

        // Ensure directory exists
        if !dir.as_os_str().is_empty() {
            fs::create_dir_all(dir)
                .map_err(|e| format!("Failed to create directory: {}", e))?;
        }

        // Save HNSW index
        {
            let index = self.index.lock().map_err(|e| format!("Lock error: {}", e))?;
            index
                .file_dump(dir, name)
                .map_err(|e| format!("Failed to dump HNSW index: {:?}", e))?;
        }

        // Save metadata as JSON
        let meta_path = Self::meta_file_path(base_path);
        let state = {
            let meta_map = self.meta.lock().map_err(|e| format!("Lock error: {}", e))?;
            let next_key = self.next_key.lock().map_err(|e| format!("Lock error: {}", e))?;
            PersistentState {
                metadata: meta_map.clone(),
                next_key: *next_key,
                version: PersistentState::CURRENT_VERSION,
            }
        };

        let file = File::create(&meta_path)
            .map_err(|e| format!("Failed to create metadata file: {}", e))?;
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, &state)
            .map_err(|e| format!("Failed to write metadata: {}", e))?;

        Ok(())
    }

    /// Get statistics
    pub fn stats(&self) -> MemoryStats {
        let meta_map = self.meta.lock().ok();
        let count = meta_map.map(|m| m.len()).unwrap_or(0);
        MemoryStats {
            backend: "hnsw",
            count,
            index_size_estimate: count * (VECTOR_DIM * 4 + 200), // rough estimate
        }
    }

    /// Check if this index has persistence configured
    pub fn is_persistent(&self) -> bool {
        self.storage_path.is_some()
    }

    /// Get the storage path (if configured)
    pub fn storage_path(&self) -> Option<&Path> {
        self.storage_path.as_deref()
    }
}

impl Default for HnswMemory {
    fn default() -> Self {
        Self::new().expect("Failed to create default HnswMemory")
    }
}

/// Statistics for HNSW backend
#[derive(Debug, Clone)]
pub struct MemoryStats {
    pub backend: &'static str,
    pub count: usize,
    pub index_size_estimate: usize,
}

/// Normalize vector to unit length (for cosine similarity via dot product)
fn normalize_vector(v: &[f32]) -> Vec<f32> {
    let magnitude: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if magnitude > 0.0 {
        v.iter().map(|x| x / magnitude).collect()
    } else {
        v.to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    fn dummy_embedding() -> Vec<f32> {
        let mut v: Vec<f32> = (0..VECTOR_DIM).map(|i| (i as f32 / VECTOR_DIM as f32)).collect();
        // Normalize
        let mag: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        for x in &mut v {
            *x /= mag;
        }
        v
    }

    #[test]
    fn test_hnsw_store_and_search() {
        let mem = HnswMemory::new().unwrap();
        let emb = dummy_embedding();
        let meta = MemoryMeta::new("test_hnsw".to_string(), "HNSW test memory".to_string());

        let key = mem.store(&emb, meta).unwrap();
        assert!(key > 0);

        let results = mem.search(&emb, &SearchOptions::new().limit(5)).unwrap();
        assert!(!results.is_empty());
        assert!(results[0].similarity > 0.9);
    }

    #[test]
    fn test_normalize() {
        let v = vec![3.0, 4.0];
        let n = normalize_vector(&v);
        let mag: f32 = n.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((mag - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_persistence_roundtrip() {
        let temp_dir = env::temp_dir();
        let test_path = temp_dir.join("viva_hnsw_test");

        // Clean up any previous test data
        let _ = fs::remove_file(test_path.with_extension("hnsw.data"));
        let _ = fs::remove_file(test_path.with_extension("hnsw.graph"));
        let _ = fs::remove_file(HnswMemory::meta_file_path(&test_path));

        // Create and populate index
        {
            let mem = HnswMemory::open(&test_path).unwrap();
            let emb = dummy_embedding();
            let meta = MemoryMeta::new("persist_test".to_string(), "Test persistence".to_string())
                .with_importance(0.9);

            mem.store(&emb, meta).unwrap();
            mem.save().unwrap();
        }

        // Reload and verify
        {
            let mem = HnswMemory::open(&test_path).unwrap();
            let stats = mem.stats();
            assert_eq!(stats.count, 1);

            let emb = dummy_embedding();
            let results = mem.search(&emb, &SearchOptions::new().limit(5)).unwrap();
            assert!(!results.is_empty());
            assert_eq!(results[0].meta.id, "persist_test");
            assert!((results[0].meta.importance - 0.9).abs() < 0.01);
        }

        // Clean up
        let _ = fs::remove_file(test_path.with_extension("hnsw.data"));
        let _ = fs::remove_file(test_path.with_extension("hnsw.graph"));
        let _ = fs::remove_file(HnswMemory::meta_file_path(&test_path));
    }

    #[test]
    fn test_in_memory_save_fails_gracefully() {
        let mem = HnswMemory::new().unwrap();
        assert!(!mem.is_persistent());
        assert!(mem.save().is_err());
    }
}
