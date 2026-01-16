#![allow(dead_code)]
//! SQLite backend for VIVA's memory system.
//!
//! Simple and portable - works everywhere including mobile.
//! Brute-force cosine similarity (good for < 100k vectors).

use crate::memory::types::*;
use rusqlite::{params, Connection};
use std::path::Path;
use std::sync::Mutex;

/// SQLite-based memory backend
pub struct SqliteMemory {
    conn: Mutex<Connection>,
    path: Option<String>,
}

impl SqliteMemory {
    /// Create in-memory database
    pub fn new() -> Result<Self, String> {
        let conn =
            Connection::open_in_memory().map_err(|e| format!("Failed to open SQLite: {}", e))?;
        let mem = Self {
            conn: Mutex::new(conn),
            path: None,
        };
        mem.init_schema()?;
        Ok(mem)
    }

    /// Open or create file-based database
    pub fn open(path: &str) -> Result<Self, String> {
        let db_path = format!("{}.sqlite", path);
        let exists = Path::new(&db_path).exists();

        let conn =
            Connection::open(&db_path).map_err(|e| format!("Failed to open SQLite: {}", e))?;

        let mem = Self {
            conn: Mutex::new(conn),
            path: Some(path.to_string()),
        };

        if !exists {
            mem.init_schema()?;
        }

        Ok(mem)
    }

    fn init_schema(&self) -> Result<(), String> {
        let conn = self.conn.lock().unwrap();
        conn.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS memories (
                key INTEGER PRIMARY KEY AUTOINCREMENT,
                id TEXT NOT NULL UNIQUE,
                content TEXT NOT NULL,
                memory_type TEXT NOT NULL DEFAULT 'generic',
                importance REAL NOT NULL DEFAULT 0.5,
                emotion_p REAL,
                emotion_a REAL,
                emotion_d REAL,
                timestamp INTEGER NOT NULL,
                access_count INTEGER NOT NULL DEFAULT 0,
                last_accessed INTEGER NOT NULL,
                embedding BLOB NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_type ON memories(memory_type);
            CREATE INDEX IF NOT EXISTS idx_importance ON memories(importance);
            CREATE INDEX IF NOT EXISTS idx_timestamp ON memories(timestamp);
            "#,
        )
        .map_err(|e| format!("Failed to create schema: {}", e))?;
        Ok(())
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

        let conn = self.conn.lock().unwrap();

        // Serialize embedding to bytes
        let emb_bytes: Vec<u8> = embedding
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();

        let (ep, ea, ed) = meta
            .emotion
            .map(|e| (Some(e.pleasure), Some(e.arousal), Some(e.dominance)))
            .unwrap_or((None, None, None));

        conn.execute(
            r#"INSERT INTO memories
               (id, content, memory_type, importance, emotion_p, emotion_a, emotion_d,
                timestamp, access_count, last_accessed, embedding)
               VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)"#,
            params![
                meta.id,
                meta.content,
                meta.memory_type.as_str(),
                meta.importance,
                ep,
                ea,
                ed,
                meta.timestamp,
                meta.access_count,
                meta.last_accessed,
                emb_bytes
            ],
        )
        .map_err(|e| format!("Failed to insert: {}", e))?;

        let key = conn.last_insert_rowid() as u64;
        Ok(key)
    }

    /// Search for similar memories using brute-force cosine similarity
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

        let conn = self.conn.lock().unwrap();

        // Build query with filters
        let mut sql = String::from(
            "SELECT key, id, content, memory_type, importance,
                    emotion_p, emotion_a, emotion_d, timestamp,
                    access_count, last_accessed, embedding
             FROM memories WHERE 1=1",
        );

        if let Some(ref t) = options.memory_type {
            sql.push_str(&format!(" AND memory_type = '{}'", t.as_str()));
        }

        if options.min_importance > 0.0 {
            sql.push_str(&format!(" AND importance >= {}", options.min_importance));
        }

        let mut stmt = conn
            .prepare(&sql)
            .map_err(|e| format!("Failed to prepare: {}", e))?;

        let rows = stmt
            .query_map([], |row| {
                let key: u64 = row.get(0)?;
                let id: String = row.get(1)?;
                let content: String = row.get(2)?;
                let memory_type_str: String = row.get(3)?;
                let importance: f32 = row.get(4)?;
                let emotion_p: Option<f32> = row.get(5)?;
                let emotion_a: Option<f32> = row.get(6)?;
                let emotion_d: Option<f32> = row.get(7)?;
                let timestamp: i64 = row.get(8)?;
                let access_count: u32 = row.get(9)?;
                let last_accessed: i64 = row.get(10)?;
                let emb_bytes: Vec<u8> = row.get(11)?;

                Ok((
                    key,
                    id,
                    content,
                    memory_type_str,
                    importance,
                    emotion_p,
                    emotion_a,
                    emotion_d,
                    timestamp,
                    access_count,
                    last_accessed,
                    emb_bytes,
                ))
            })
            .map_err(|e| format!("Failed to query: {}", e))?;

        let mut results: Vec<MemorySearchResult> = Vec::new();

        for row_result in rows {
            let (
                _key,
                id,
                content,
                memory_type_str,
                importance,
                emotion_p,
                emotion_a,
                emotion_d,
                timestamp,
                access_count,
                last_accessed,
                emb_bytes,
            ) = row_result.map_err(|e| format!("Row error: {}", e))?;

            // Deserialize embedding
            let embedding: Vec<f32> = emb_bytes
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect();

            // Calculate cosine similarity
            let similarity = cosine_similarity(query, &embedding);

            // Build emotion
            let emotion = match (emotion_p, emotion_a, emotion_d) {
                (Some(p), Some(a), Some(d)) => Some(PadEmotion {
                    pleasure: p,
                    arousal: a,
                    dominance: d,
                }),
                _ => None,
            };

            // Calculate decayed score
            let decayed_score = if options.apply_decay {
                let decay = calculate_decay(timestamp, options.decay_scale);
                0.7 * similarity + 0.3 * decay
            } else {
                similarity
            };

            let meta = MemoryMeta {
                id,
                content,
                memory_type: MemoryType::from_str(&memory_type_str),
                importance,
                emotion,
                timestamp,
                access_count,
                last_accessed,
            };

            results.push(MemorySearchResult {
                meta,
                similarity,
                decayed_score,
            });
        }

        // Sort by decayed score (descending)
        results.sort_by(|a, b| b.decayed_score.partial_cmp(&a.decayed_score).unwrap());
        results.truncate(options.limit);

        Ok(results)
    }

    /// Get memory by string ID
    pub fn get(&self, memory_id: &str) -> Result<Option<MemoryMeta>, String> {
        let conn = self.conn.lock().unwrap();

        // Update access stats first
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;

        conn.execute(
            "UPDATE memories SET access_count = access_count + 1, last_accessed = ?1 WHERE id = ?2",
            params![now, memory_id],
        )
        .map_err(|e| format!("Failed to update access: {}", e))?;

        // Fetch
        let result = conn.query_row(
            "SELECT id, content, memory_type, importance, emotion_p, emotion_a, emotion_d,
                    timestamp, access_count, last_accessed
             FROM memories WHERE id = ?1",
            params![memory_id],
            |row| {
                let emotion = match (row.get::<_, Option<f32>>(4)?, row.get::<_, Option<f32>>(5)?, row.get::<_, Option<f32>>(6)?) {
                    (Some(p), Some(a), Some(d)) => Some(PadEmotion {
                        pleasure: p,
                        arousal: a,
                        dominance: d,
                    }),
                    _ => None,
                };

                Ok(MemoryMeta {
                    id: row.get(0)?,
                    content: row.get(1)?,
                    memory_type: MemoryType::from_str(&row.get::<_, String>(2)?),
                    importance: row.get(3)?,
                    emotion,
                    timestamp: row.get(7)?,
                    access_count: row.get(8)?,
                    last_accessed: row.get(9)?,
                })
            },
        );

        match result {
            Ok(meta) => Ok(Some(meta)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(format!("Query failed: {}", e)),
        }
    }

    /// Delete a memory
    pub fn forget(&self, memory_id: &str) -> Result<(), String> {
        let conn = self.conn.lock().unwrap();
        conn.execute("DELETE FROM memories WHERE id = ?1", params![memory_id])
            .map_err(|e| format!("Failed to delete: {}", e))?;
        Ok(())
    }

    /// Get stats
    pub fn stats(&self) -> Result<SqliteStats, String> {
        let conn = self.conn.lock().unwrap();
        let count: usize = conn
            .query_row("SELECT COUNT(*) FROM memories", [], |row| row.get(0))
            .map_err(|e| format!("Count failed: {}", e))?;

        Ok(SqliteStats {
            backend: "sqlite",
            count,
            path: self.path.clone(),
        })
    }
}

impl Default for SqliteMemory {
    fn default() -> Self {
        Self::new().expect("Failed to create default SqliteMemory")
    }
}

/// SQLite statistics
#[derive(Debug, Clone)]
pub struct SqliteStats {
    pub backend: &'static str,
    pub count: usize,
    pub path: Option<String>,
}

/// Cosine similarity between two vectors
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let mag_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let mag_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if mag_a > 0.0 && mag_b > 0.0 {
        dot / (mag_a * mag_b)
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_embedding() -> Vec<f32> {
        (0..VECTOR_DIM)
            .map(|i| (i as f32 / VECTOR_DIM as f32))
            .collect()
    }

    #[test]
    fn test_store_and_search() {
        let mem = SqliteMemory::new().unwrap();
        let emb = dummy_embedding();
        let meta = MemoryMeta::new("test_id".to_string(), "Test memory".to_string());

        let key = mem.store(&emb, meta).unwrap();
        assert!(key > 0);

        let results = mem.search(&emb, &SearchOptions::new().limit(5)).unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].similarity > 0.99);
    }

    #[test]
    fn test_get_and_access_count() {
        let mem = SqliteMemory::new().unwrap();
        let emb = dummy_embedding();
        let meta = MemoryMeta::new("access_test".to_string(), "Test".to_string());
        mem.store(&emb, meta).unwrap();

        // First access
        let m1 = mem.get("access_test").unwrap().unwrap();
        assert_eq!(m1.access_count, 1);

        // Second access
        let m2 = mem.get("access_test").unwrap().unwrap();
        assert_eq!(m2.access_count, 2);
    }

    #[test]
    fn test_forget() {
        let mem = SqliteMemory::new().unwrap();
        let emb = dummy_embedding();
        let meta = MemoryMeta::new("forget_test".to_string(), "Will be forgotten".to_string());
        mem.store(&emb, meta).unwrap();

        assert!(mem.get("forget_test").unwrap().is_some());
        mem.forget("forget_test").unwrap();
        assert!(mem.get("forget_test").unwrap().is_none());
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 0.001);

        let c = vec![0.0, 1.0, 0.0];
        assert!(cosine_similarity(&a, &c).abs() < 0.001); // Orthogonal
    }
}
