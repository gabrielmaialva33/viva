#![allow(dead_code)]
//! Three-Factor Hebbian Learning for VIVA's Memory System
//!
//! Implements biologically-inspired learning rules where emotion (PAD model)
//! modulates the strength of memory formation.
//!
//! ## Theory
//!
//! Traditional Hebbian: `Δw = η * pre * post`
//! Three-Factor Hebbian: `Δw = η * pre * post * modulator`
//!
//! The modulator is VIVA's emotional state (PAD), which affects:
//! - **Arousal** → Learning rate (high arousal = stronger encoding)
//! - **Pleasure** → Memory valence (positive/negative emotional tag)
//! - **Dominance** → Confidence in memory (affects retrieval priority)
//!
//! ## Biological Basis
//!
//! - Dopamine modulates synaptic plasticity in the hippocampus
//! - Emotional events are remembered better (amygdala-hippocampus interaction)
//! - Stress hormones (cortisol, norepinephrine) affect memory consolidation
//!
//! ## References
//!
//! - Frémaux, N., & Gerstner, W. (2016). Neuromodulated STDP and theory of
//!   three-factor learning rules. Frontiers in Neural Circuits.
//! - McGaugh, J. L. (2000). Memory - a century of consolidation. Science.

use crate::memory::types::*;
use serde::{Deserialize, Serialize};

/// Learning rate parameters for Three-Factor Hebbian
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct HebbianParams {
    /// Base learning rate (η)
    pub base_lr: f32,
    /// Arousal influence on learning rate [0.0-1.0]
    pub arousal_weight: f32,
    /// Minimum learning rate (prevents zero learning)
    pub min_lr: f32,
    /// Maximum learning rate (prevents instability)
    pub max_lr: f32,
    /// Decay rate for importance over time
    pub importance_decay: f32,
    /// Boost factor for highly emotional memories
    pub emotional_boost: f32,
}

impl Default for HebbianParams {
    fn default() -> Self {
        Self {
            base_lr: 0.5,
            arousal_weight: 0.4,
            min_lr: 0.1,
            max_lr: 1.0,
            importance_decay: 0.001,
            emotional_boost: 1.5,
        }
    }
}

/// Three-Factor Hebbian Learning Engine
///
/// Modulates memory formation based on emotional state.
#[derive(Debug, Clone)]
pub struct HebbianLearning {
    params: HebbianParams,
    /// Current emotional state (PAD)
    current_emotion: PadEmotion,
    /// Running average of arousal (for normalization)
    arousal_ema: f32,
    /// EMA decay factor
    ema_alpha: f32,
}

impl HebbianLearning {
    /// Create new Hebbian learning engine with default parameters
    pub fn new() -> Self {
        Self {
            params: HebbianParams::default(),
            current_emotion: PadEmotion {
                pleasure: 0.0,
                arousal: 0.0,
                dominance: 0.0,
            },
            arousal_ema: 0.5,
            ema_alpha: 0.1,
        }
    }

    /// Create with custom parameters
    pub fn with_params(params: HebbianParams) -> Self {
        Self {
            params,
            ..Self::new()
        }
    }

    /// Update current emotional state
    ///
    /// This should be called whenever VIVA's emotional state changes.
    pub fn update_emotion(&mut self, emotion: PadEmotion) {
        self.current_emotion = emotion;
        // Update arousal EMA for adaptive normalization
        self.arousal_ema =
            self.ema_alpha * emotion.arousal.abs() + (1.0 - self.ema_alpha) * self.arousal_ema;
    }

    /// Calculate modulated learning rate based on current emotion
    ///
    /// Higher arousal → higher learning rate (more memorable)
    /// This implements the "flashbulb memory" effect where emotionally
    /// charged events are encoded more strongly.
    pub fn modulated_learning_rate(&self) -> f32 {
        let arousal_factor = self.current_emotion.arousal.abs();

        // Three-factor modulation: η * (1 + arousal_weight * |arousal|)
        let lr = self.params.base_lr * (1.0 + self.params.arousal_weight * arousal_factor);

        lr.clamp(self.params.min_lr, self.params.max_lr)
    }

    /// Calculate importance score for a new memory
    ///
    /// Combines multiple factors:
    /// - Base importance (0.5)
    /// - Arousal boost (emotional = more important)
    /// - Dominance factor (confident experiences = more important)
    pub fn calculate_importance(&self, base_importance: f32) -> f32 {
        let emotion = &self.current_emotion;

        // Arousal increases importance (emotional events are important)
        let arousal_boost = emotion.arousal.abs() * 0.3;

        // Dominance slightly increases importance (confident = more weight)
        let dominance_boost = (emotion.dominance + 1.0) / 2.0 * 0.1; // Normalize to [0,1]

        // Strong emotions (positive or negative) get extra boost
        let emotional_intensity =
            (emotion.pleasure.abs() + emotion.arousal.abs()) / 2.0;
        let intensity_boost = if emotional_intensity > 0.7 {
            (emotional_intensity - 0.7) * self.params.emotional_boost
        } else {
            0.0
        };

        let importance = base_importance + arousal_boost + dominance_boost + intensity_boost;
        importance.clamp(0.0, 1.0)
    }

    /// Process a memory before storage, applying Three-Factor Hebbian modulation
    ///
    /// Returns a modified MemoryMeta with:
    /// - Adjusted importance based on emotional state
    /// - Attached emotion for future retrieval
    pub fn modulate_memory(&self, mut meta: MemoryMeta) -> MemoryMeta {
        // Apply Three-Factor Hebbian importance modulation
        meta.importance = self.calculate_importance(meta.importance);

        // Tag memory with current emotion (for emotional context retrieval)
        if meta.emotion.is_none() {
            meta.emotion = Some(self.current_emotion);
        }

        meta
    }

    /// Calculate retrieval strength boost for a memory using STDP timing
    ///
    /// Implements Spike-Timing-Dependent Plasticity:
    /// - LTP (potentiation): Recently accessed memories get boosted
    /// - LTD (depression): Old memories get slightly weakened
    /// - Emotional congruence modulates the effect
    ///
    /// STDP Window:
    /// ```text
    ///   Boost
    ///     ^
    /// 1.3 |  ****
    /// 1.2 |      ****
    /// 1.1 |          ****
    /// 1.0 |---------------********----------> Δt (hours)
    /// 0.9 |                        ****
    ///     0    1    2    6   12   24   48
    /// ```
    pub fn retrieval_boost(&self, memory_emotion: Option<PadEmotion>) -> f32 {
        self.retrieval_boost_stdp(memory_emotion, None)
    }

    /// STDP retrieval boost with timing information
    ///
    /// `last_accessed`: Unix timestamp of last memory access
    pub fn retrieval_boost_stdp(
        &self,
        memory_emotion: Option<PadEmotion>,
        last_accessed: Option<i64>,
    ) -> f32 {
        // 1. Emotional congruence component (mood-congruent memory)
        let emotional_factor = match memory_emotion {
            Some(mem_emo) => {
                let current = &self.current_emotion;
                let similarity = mem_emo.pleasure * current.pleasure
                    + mem_emo.arousal * current.arousal
                    + mem_emo.dominance * current.dominance;
                // Normalize from [-3, 3] to [0.85, 1.15]
                let normalized = (similarity + 3.0) / 6.0;
                0.85 + normalized * 0.3
            }
            None => 1.0,
        };

        // 2. STDP timing component
        let timing_factor: f32 = match last_accessed {
            Some(last_ts) => {
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_secs() as i64)
                    .unwrap_or(0);

                let delta_secs = (now - last_ts).max(0) as f32;
                let delta_hours = delta_secs / 3600.0;

                // STDP curve: LTP for recent, LTD for old
                // Bi-exponential: A+ * exp(-Δt/τ+) - A- * exp(-Δt/τ-)
                // τ+ = 2 hours (LTP window)
                // τ- = 24 hours (LTD window)
                let ltp = 0.25 * (-delta_hours / 2.0_f32).exp();  // Fast potentiation
                let ltd = 0.15 * (-delta_hours / 24.0_f32).exp(); // Slow depression

                // LTP dominates for recent, LTD baseline for old
                if delta_hours < 6.0 {
                    // Recent: strong LTP
                    1.0 + ltp
                } else if delta_hours < 24.0 {
                    // Medium: transition zone
                    1.0 + ltp - ltd * 0.5
                } else {
                    // Old: mild LTD
                    1.0 - 0.05 * (1.0 - ltd)
                }
            }
            None => 1.0, // No timing info = no STDP effect
        };

        // Combine factors multiplicatively
        (emotional_factor * timing_factor).clamp(0.7, 1.5)
    }

    /// Apply STDP-inspired strengthening/weakening
    ///
    /// When a memory is accessed:
    /// - If emotional state matches: strengthen (LTP)
    /// - If emotional state mismatches: weaken slightly (LTD)
    ///
    /// This implements "emotional reconsolidation" where retrieving
    /// memories in different emotional contexts can modify them.
    pub fn stdp_importance_update(
        &self,
        current_importance: f32,
        memory_emotion: Option<PadEmotion>,
        time_since_access_secs: f64,
    ) -> f32 {
        let time_factor = (-time_since_access_secs / 3600.0).exp() as f32; // Decay over hours

        let emotional_match = match memory_emotion {
            Some(mem_emo) => {
                let current = &self.current_emotion;
                // Sign-sensitive matching (positive with positive, etc.)
                let match_score = mem_emo.pleasure * current.pleasure
                    + mem_emo.arousal * current.arousal * 0.5;
                match_score.clamp(-1.0, 1.0)
            }
            None => 0.0,
        };

        // LTP (strengthen) if match > 0, LTD (weaken) if match < 0
        let delta = emotional_match * time_factor * 0.1;
        (current_importance + delta).clamp(0.0, 1.0)
    }

    /// Get current emotion
    pub fn current_emotion(&self) -> PadEmotion {
        self.current_emotion
    }

    /// Get parameters
    pub fn params(&self) -> &HebbianParams {
        &self.params
    }

    /// Augment a semantic vector with emotional state for "vibe-based" search
    ///
    /// Concatenates (PAD * weight) to the vector, extending its dimensionality.
    /// This forces the ANN index to consider emotional proximity.
    ///
    /// # Arguments
    ///
    /// * `vector` - Original semantic vector
    /// * `emotion` - Emotion to encode (defaults to current if None)
    /// * `weight` - How much emotion influences distance (try 10.0)
    pub fn augment_vector(&self, vector: &[f32], emotion: Option<PadEmotion>, weight: f32) -> Vec<f32> {
        let emo = emotion.unwrap_or(self.current_emotion);
        let mut aug = Vec::with_capacity(vector.len() + 3);
        aug.extend_from_slice(vector);
        // Append weighted PAD components
        aug.push(emo.pleasure * weight);
        aug.push(emo.arousal * weight);
        aug.push(emo.dominance * weight);
        aug
    }
}

impl Default for HebbianLearning {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Synaptic Tagging and Capture (STC) Implementation
// ============================================================================

/// A synaptic tag marks a memory for potential consolidation
///
/// In neuroscience, synaptic tags are transient markers that allow
/// weak memories to be "captured" and consolidated if a strong
/// memory occurs nearby in time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynapticTag {
    /// Memory this tag belongs to
    pub memory_key: u64,
    /// When the tag was created (Unix timestamp)
    pub created_at: i64,
    /// Tag strength (decays over time)
    pub strength: f32,
    /// Emotional state at time of tagging
    pub emotion: PadEmotion,
    /// Time-to-live in seconds (typically ~2 hours in biology)
    pub ttl_secs: i64,
}

impl SynapticTag {
    /// Default TTL: 2 hours (biological STC window)
    pub const DEFAULT_TTL: i64 = 7200;

    /// Create a new synaptic tag
    pub fn new(memory_key: u64, strength: f32, emotion: PadEmotion) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0);

        Self {
            memory_key,
            created_at: now,
            strength,
            emotion,
            ttl_secs: Self::DEFAULT_TTL,
        }
    }

    /// Check if the tag is still valid (not expired)
    pub fn is_valid(&self) -> bool {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0);

        (now - self.created_at) < self.ttl_secs
    }

    /// Get current strength (decays exponentially)
    pub fn current_strength(&self) -> f32 {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0);

        let age_secs = (now - self.created_at) as f64;
        let decay = (-age_secs / (self.ttl_secs as f64 / 2.0)).exp() as f32;
        self.strength * decay
    }

    /// Check if this tag can be captured by a strong memory
    ///
    /// Capture happens when:
    /// 1. Tag is still valid
    /// 2. Strong memory has high arousal
    /// 3. Emotional states are similar (within same valence)
    pub fn can_be_captured_by(&self, strong_emotion: &PadEmotion, min_arousal: f32) -> bool {
        if !self.is_valid() {
            return false;
        }

        // Strong memory must have high arousal
        if strong_emotion.arousal.abs() < min_arousal {
            return false;
        }

        // Same emotional valence (both positive or both negative)
        let same_valence = (self.emotion.pleasure * strong_emotion.pleasure) >= 0.0;

        same_valence && self.current_strength() > 0.1
    }
}

/// Manager for synaptic tags (STC system)
#[derive(Debug, Clone, Default)]
pub struct SynapticTagManager {
    tags: Vec<SynapticTag>,
}

impl SynapticTagManager {
    pub fn new() -> Self {
        Self { tags: Vec::new() }
    }

    /// Add a new tag
    pub fn add_tag(&mut self, tag: SynapticTag) {
        self.tags.push(tag);
    }

    /// Create and add a tag for a memory
    pub fn tag_memory(&mut self, memory_key: u64, importance: f32, emotion: PadEmotion) {
        // Only tag memories that aren't already strong
        if importance < 0.8 {
            self.add_tag(SynapticTag::new(memory_key, importance, emotion));
        }
    }

    /// Get all memories that should be captured by a strong memory event
    ///
    /// Returns memory keys that should have their importance boosted
    pub fn capture_weak_memories(&mut self, strong_emotion: &PadEmotion) -> Vec<u64> {
        let min_arousal = 0.6;
        let mut captured = Vec::new();

        self.tags.retain(|tag| {
            if tag.can_be_captured_by(strong_emotion, min_arousal) {
                captured.push(tag.memory_key);
                false // Remove captured tags
            } else if !tag.is_valid() {
                false // Remove expired tags
            } else {
                true // Keep valid uncaptured tags
            }
        });

        captured
    }

    /// Clean up expired tags
    pub fn cleanup_expired(&mut self) {
        self.tags.retain(|tag| tag.is_valid());
    }

    /// Get number of active tags
    pub fn active_count(&self) -> usize {
        self.tags.iter().filter(|t| t.is_valid()).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_learning_rate_modulation() {
        let mut hebbian = HebbianLearning::new();

        // Neutral state
        hebbian.update_emotion(PadEmotion {
            pleasure: 0.0,
            arousal: 0.0,
            dominance: 0.0,
        });
        let neutral_lr = hebbian.modulated_learning_rate();

        // High arousal state
        hebbian.update_emotion(PadEmotion {
            pleasure: 0.0,
            arousal: 0.9,
            dominance: 0.0,
        });
        let aroused_lr = hebbian.modulated_learning_rate();

        // High arousal should increase learning rate
        assert!(aroused_lr > neutral_lr);
    }

    #[test]
    fn test_importance_calculation() {
        let mut hebbian = HebbianLearning::new();

        // High arousal + positive pleasure = high importance
        hebbian.update_emotion(PadEmotion {
            pleasure: 0.8,
            arousal: 0.9,
            dominance: 0.5,
        });
        let high_importance = hebbian.calculate_importance(0.5);

        // Low arousal = normal importance
        hebbian.update_emotion(PadEmotion {
            pleasure: 0.0,
            arousal: 0.0,
            dominance: 0.0,
        });
        let normal_importance = hebbian.calculate_importance(0.5);

        assert!(high_importance > normal_importance);
    }

    #[test]
    fn test_retrieval_boost() {
        let mut hebbian = HebbianLearning::new();

        // Set current emotion to positive
        hebbian.update_emotion(PadEmotion {
            pleasure: 0.8,
            arousal: 0.5,
            dominance: 0.3,
        });

        // Memory with matching emotion should get boost
        let matching = PadEmotion {
            pleasure: 0.7,
            arousal: 0.4,
            dominance: 0.2,
        };
        let matching_boost = hebbian.retrieval_boost(Some(matching));

        // Memory with opposite emotion should get less boost
        let opposite = PadEmotion {
            pleasure: -0.7,
            arousal: -0.4,
            dominance: -0.2,
        };
        let opposite_boost = hebbian.retrieval_boost(Some(opposite));

        assert!(matching_boost > opposite_boost);
    }

    #[test]
    fn test_stdp_timing() {
        let mut hebbian = HebbianLearning::new();
        hebbian.update_emotion(PadEmotion {
            pleasure: 0.5,
            arousal: 0.5,
            dominance: 0.5,
        });

        let mem_emotion = Some(PadEmotion {
            pleasure: 0.5,
            arousal: 0.5,
            dominance: 0.5,
        });

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;

        // Recently accessed (1 min ago) = LTP boost
        let recent_boost = hebbian.retrieval_boost_stdp(mem_emotion, Some(now - 60));

        // Old access (48 hours ago) = slight LTD
        let old_boost = hebbian.retrieval_boost_stdp(mem_emotion, Some(now - 48 * 3600));

        // Recent should get higher boost than old
        assert!(recent_boost > old_boost, "LTP should be > LTD: {} vs {}", recent_boost, old_boost);

        // Recent should be > 1.0 (potentiation)
        assert!(recent_boost > 1.0, "Recent should be potentiated: {}", recent_boost);
    }

    #[test]
    fn test_synaptic_tag_expiry() {
        let tag = SynapticTag {
            memory_key: 1,
            created_at: 0, // Very old
            strength: 0.5,
            emotion: PadEmotion {
                pleasure: 0.0,
                arousal: 0.0,
                dominance: 0.0,
            },
            ttl_secs: 100,
        };

        assert!(!tag.is_valid()); // Should be expired
    }

    #[test]
    fn test_synaptic_tag_capture() {
        let mut manager = SynapticTagManager::new();

        // Create a weak memory tag
        let weak_emotion = PadEmotion {
            pleasure: 0.3,
            arousal: 0.2,
            dominance: 0.0,
        };
        manager.tag_memory(42, 0.4, weak_emotion);

        // Strong emotional event with same valence
        let strong_emotion = PadEmotion {
            pleasure: 0.8,
            arousal: 0.9,
            dominance: 0.5,
        };

        let captured = manager.capture_weak_memories(&strong_emotion);
        assert!(captured.contains(&42));
    }
}
