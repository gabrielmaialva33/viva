//! Qualia - Mapping physical events to emotional experience
//!
//! Transforms collisions and physics states into PAD (Pleasure-Arousal-Dominance)
//! emotional deltas. This is VIVA's "felt sense" of her physical body.
//!
//! # Liveness Analysis (PCSX2 Pattern)
//! Skip calculations that won't be observed:
//! - `soul_listening`: Skip if Soul isn't reading qualia
//! - `awake_bodies`: Skip collisions involving sleeping bodies
//! - Early-out on empty/below-threshold events

use super::world::CollisionEvent;
use bevy_ecs::system::Resource;
use glam::Vec3;
use std::collections::HashSet;

/// PAD (Pleasure-Arousal-Dominance) emotional delta
#[derive(Clone, Debug, Default)]
pub struct PadDelta {
    pub pleasure: f64,
    pub arousal: f64,
    pub dominance: f64,
}

impl PadDelta {
    pub fn new(pleasure: f64, arousal: f64, dominance: f64) -> Self {
        Self {
            pleasure,
            arousal,
            dominance,
        }
    }

    /// Clamp all values to [-1, 1]
    pub fn clamped(&self) -> Self {
        Self {
            pleasure: self.pleasure.clamp(-1.0, 1.0),
            arousal: self.arousal.clamp(-1.0, 1.0),
            dominance: self.dominance.clamp(-1.0, 1.0),
        }
    }

    /// Combine multiple PAD deltas
    pub fn combine(&self, other: &PadDelta) -> Self {
        Self {
            pleasure: self.pleasure + other.pleasure,
            arousal: self.arousal + other.arousal,
            dominance: self.dominance + other.dominance,
        }
    }
}

/// Configuration for qualia mapping
#[derive(Clone, Debug)]
pub struct QualiaConfig {
    /// Impulse threshold for "significant" collision (Ns)
    pub collision_threshold: f32,
    /// Maximum impulse for scaling (Ns)
    pub max_collision_impulse: f32,
    /// Arousal multiplier for collisions
    pub collision_arousal_factor: f64,
    /// Pleasure reduction for hard impacts
    pub collision_pain_factor: f64,
    /// Dominance reduction for being hit
    pub collision_dominance_factor: f64,
}

impl Default for QualiaConfig {
    fn default() -> Self {
        Self {
            collision_threshold: 0.1,    // Ignore very soft contacts
            max_collision_impulse: 100.0, // Cap for scaling
            collision_arousal_factor: 0.5,
            collision_pain_factor: 0.3,
            collision_dominance_factor: 0.2,
        }
    }
}

/// Liveness statistics for diagnostics
#[derive(Clone, Debug, Default)]
pub struct LivenessStats {
    /// Events processed this frame
    pub events_processed: u32,
    /// Events skipped (liveness analysis)
    pub events_skipped: u32,
    /// Skipped: Soul not listening
    pub skipped_soul_not_listening: u32,
    /// Skipped: Body sleeping
    pub skipped_body_sleeping: u32,
    /// Skipped: Below threshold
    pub skipped_below_threshold: u32,
    /// Skipped: Not self collision
    pub skipped_not_self: u32,
}

impl LivenessStats {
    /// Get skip ratio (0.0 = processed all, 1.0 = skipped all)
    pub fn skip_ratio(&self) -> f32 {
        let total = self.events_processed + self.events_skipped;
        if total == 0 {
            0.0
        } else {
            self.events_skipped as f32 / total as f32
        }
    }

    /// Reset counters
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

/// Qualia processor - converts physics events to emotions
///
/// # Liveness Analysis (PCSX2 Pattern)
/// Uses early-out optimizations to skip unnecessary calculations:
/// - Skip all processing if Soul isn't listening
/// - Skip collisions involving sleeping bodies
/// - Skip collisions below impulse threshold
/// - Skip collisions not involving "self" body
#[derive(Resource)]
pub struct QualiaProcessor {
    config: QualiaConfig,
    /// Body ID of "self" (VIVA's main body)
    self_body: Option<u32>,

    // =========================================================================
    // Liveness Analysis State (PCSX2 pattern)
    // =========================================================================
    /// Is Soul currently listening for qualia updates?
    soul_listening: bool,
    /// Set of awake body IDs (skip sleeping bodies)
    awake_bodies: HashSet<u32>,
    /// Statistics for diagnostics
    stats: LivenessStats,
}

impl QualiaProcessor {
    pub fn new() -> Self {
        Self {
            config: QualiaConfig::default(),
            self_body: None,
            soul_listening: true, // Default to listening
            awake_bodies: HashSet::new(),
            stats: LivenessStats::default(),
        }
    }

    pub fn with_config(config: QualiaConfig) -> Self {
        Self {
            config,
            self_body: None,
            soul_listening: true,
            awake_bodies: HashSet::new(),
            stats: LivenessStats::default(),
        }
    }

    /// Set which body ID represents "self"
    pub fn set_self_body(&mut self, body_id: u32) {
        self.self_body = Some(body_id);
    }

    // =========================================================================
    // Liveness Control API
    // =========================================================================

    /// Set whether Soul is listening for qualia updates
    ///
    /// When false, all qualia processing is skipped (major performance win).
    /// Call this based on Soul's readiness to receive updates.
    #[inline]
    pub fn set_soul_listening(&mut self, listening: bool) {
        self.soul_listening = listening;
    }

    /// Check if Soul is listening
    #[inline]
    pub fn is_soul_listening(&self) -> bool {
        self.soul_listening
    }

    /// Mark a body as awake (will process collisions)
    pub fn wake_body(&mut self, body_id: u32) {
        self.awake_bodies.insert(body_id);
    }

    /// Mark a body as sleeping (skip collisions)
    pub fn sleep_body(&mut self, body_id: u32) {
        self.awake_bodies.remove(&body_id);
    }

    /// Check if a body is awake
    #[inline]
    pub fn is_awake(&self, body_id: u32) -> bool {
        // If no bodies registered, assume all awake
        self.awake_bodies.is_empty() || self.awake_bodies.contains(&body_id)
    }

    /// Clear all awake bodies
    pub fn clear_awake_bodies(&mut self) {
        self.awake_bodies.clear();
    }

    /// Get liveness statistics
    pub fn stats(&self) -> &LivenessStats {
        &self.stats
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats.reset();
    }

    // =========================================================================
    // Core Processing with Liveness Analysis
    // =========================================================================

    /// Process collision events and return emotional delta
    ///
    /// # Liveness Analysis
    /// - Returns early if Soul isn't listening
    /// - Skips sleeping bodies
    /// - Skips below-threshold impulses
    /// - Skips non-self collisions
    pub fn process_collisions(&mut self, events: &[CollisionEvent]) -> PadDelta {
        // LIVENESS CHECK 1: Soul not listening → skip all
        if !self.soul_listening {
            self.stats.skipped_soul_not_listening += events.len() as u32;
            self.stats.events_skipped += events.len() as u32;
            return PadDelta::default();
        }

        // LIVENESS CHECK 2: No events → skip
        if events.is_empty() {
            return PadDelta::default();
        }

        let mut total = PadDelta::default();

        for event in events {
            // LIVENESS CHECK 3: Both bodies sleeping → skip
            if !self.is_awake(event.body1) && !self.is_awake(event.body2) {
                self.stats.skipped_body_sleeping += 1;
                self.stats.events_skipped += 1;
                continue;
            }

            // LIVENESS CHECK 4: Not a self collision → skip
            let is_self_collision = match self.self_body {
                Some(id) => event.body1 == id || event.body2 == id,
                None => true, // If no self defined, process all
            };

            if !is_self_collision {
                self.stats.skipped_not_self += 1;
                self.stats.events_skipped += 1;
                continue;
            }

            // LIVENESS CHECK 5: Below threshold → skip
            if event.impulse < self.config.collision_threshold {
                self.stats.skipped_below_threshold += 1;
                self.stats.events_skipped += 1;
                continue;
            }

            // PROCESS: All checks passed
            let delta = self.collision_to_pad(event);
            total = total.combine(&delta);
            self.stats.events_processed += 1;
        }

        total.clamped()
    }

    /// Process collisions without modifying stats (const version for tests)
    pub fn process_collisions_const(&self, events: &[CollisionEvent]) -> PadDelta {
        if !self.soul_listening || events.is_empty() {
            return PadDelta::default();
        }

        events
            .iter()
            .filter(|e| self.is_awake(e.body1) || self.is_awake(e.body2))
            .filter(|e| match self.self_body {
                Some(id) => e.body1 == id || e.body2 == id,
                None => true,
            })
            .filter(|e| e.impulse >= self.config.collision_threshold)
            .map(|e| self.collision_to_pad(e))
            .fold(PadDelta::default(), |a, b| a.combine(&b))
            .clamped()
    }

    /// Convert single collision to PAD delta
    fn collision_to_pad(&self, event: &CollisionEvent) -> PadDelta {
        // Normalize impulse to [0, 1]
        let intensity = (event.impulse / self.config.max_collision_impulse)
            .min(1.0)
            .max(0.0) as f64;

        // Collision causes:
        // - Arousal increase (startle response)
        // - Pleasure decrease (pain/discomfort) proportional to intensity
        // - Dominance decrease (being acted upon)

        PadDelta {
            pleasure: -intensity * self.config.collision_pain_factor,
            arousal: intensity * self.config.collision_arousal_factor,
            dominance: -intensity * self.config.collision_dominance_factor,
        }
    }

    /// Process velocity for "feeling of motion"
    pub fn velocity_to_pad(&self, velocity: Vec3) -> PadDelta {
        let speed = velocity.length();

        // Fast motion:
        // - Slight arousal increase (excitement)
        // - Slight dominance increase (agency, moving freely)
        // - Neutral pleasure (unless very fast = discomfort)

        let speed_factor = (speed / 10.0).min(1.0) as f64;

        PadDelta {
            pleasure: if speed_factor > 0.8 { -0.1 } else { 0.0 },
            arousal: speed_factor * 0.2,
            dominance: speed_factor * 0.1,
        }
    }

    /// Process height for "fear of falling"
    pub fn height_to_pad(&self, height: f32, falling: bool) -> PadDelta {
        if !falling || height < 2.0 {
            return PadDelta::default();
        }

        // Falling from height:
        // - Strong arousal (fear response)
        // - Pleasure decrease (fear)
        // - Dominance decrease (loss of control)

        let danger = ((height - 2.0) / 10.0).min(1.0) as f64;

        PadDelta {
            pleasure: -danger * 0.4,
            arousal: danger * 0.6,
            dominance: -danger * 0.3,
        }
    }

    /// Process constraint violation for "frustration"
    pub fn constraint_violation_to_pad(&self, violations: u32) -> PadDelta {
        if violations == 0 {
            return PadDelta::default();
        }

        // Constraint violations (can't move where intended):
        // - Frustration (pleasure decrease)
        // - Arousal increase (agitation)
        // - Dominance decrease (blocked)

        let frustration = (violations as f64 / 5.0).min(1.0);

        PadDelta {
            pleasure: -frustration * 0.2,
            arousal: frustration * 0.3,
            dominance: -frustration * 0.4,
        }
    }
}

impl Default for QualiaProcessor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_collision_to_pad() {
        let mut processor = QualiaProcessor::new();

        let event = CollisionEvent {
            body1: 1,
            body2: 2,
            impulse: 50.0, // Medium impact
            contact_point: Vec3::ZERO,
        };

        let pad = processor.collision_to_pad(&event);

        // Should have negative pleasure (pain)
        assert!(pad.pleasure < 0.0);
        // Should have positive arousal (startle)
        assert!(pad.arousal > 0.0);
        // Should have negative dominance (being hit)
        assert!(pad.dominance < 0.0);
    }

    #[test]
    fn test_ignore_soft_collision() {
        let mut processor = QualiaProcessor::new();

        let events = vec![CollisionEvent {
            body1: 1,
            body2: 2,
            impulse: 0.05, // Below threshold
            contact_point: Vec3::ZERO,
        }];

        let pad = processor.process_collisions(&events);

        // Should be zero
        assert_eq!(pad.pleasure, 0.0);
        assert_eq!(pad.arousal, 0.0);
        assert_eq!(pad.dominance, 0.0);

        // Stats should show skipped
        assert_eq!(processor.stats().skipped_below_threshold, 1);
    }

    #[test]
    fn test_velocity_arousal() {
        let processor = QualiaProcessor::new();

        // Slow movement
        let slow = processor.velocity_to_pad(Vec3::new(1.0, 0.0, 0.0));
        // Fast movement
        let fast = processor.velocity_to_pad(Vec3::new(8.0, 0.0, 0.0));

        // Fast should have higher arousal
        assert!(fast.arousal > slow.arousal);
    }

    #[test]
    fn test_falling_fear() {
        let processor = QualiaProcessor::new();

        // Not falling - no fear
        let not_falling = processor.height_to_pad(10.0, false);
        assert_eq!(not_falling.arousal, 0.0);

        // Falling from height - fear!
        let falling = processor.height_to_pad(10.0, true);
        assert!(falling.arousal > 0.0);
        assert!(falling.pleasure < 0.0);
    }

    #[test]
    fn test_frustration() {
        let processor = QualiaProcessor::new();

        let none = processor.constraint_violation_to_pad(0);
        let some = processor.constraint_violation_to_pad(3);

        // Violations should cause frustration (negative pleasure)
        assert_eq!(none.pleasure, 0.0);
        assert!(some.pleasure < 0.0);
    }

    // =========================================================================
    // Liveness Analysis Tests
    // =========================================================================

    #[test]
    fn test_liveness_soul_not_listening() {
        let mut processor = QualiaProcessor::new();

        // Soul not listening - should skip all
        processor.set_soul_listening(false);

        let events = vec![CollisionEvent {
            body1: 1,
            body2: 2,
            impulse: 50.0,
            contact_point: Vec3::ZERO,
        }];

        let pad = processor.process_collisions(&events);

        // Should return zero (skipped)
        assert_eq!(pad.pleasure, 0.0);
        assert_eq!(pad.arousal, 0.0);
        assert_eq!(pad.dominance, 0.0);

        // Stats should reflect skip
        assert_eq!(processor.stats().skipped_soul_not_listening, 1);
        assert_eq!(processor.stats().events_processed, 0);
    }

    #[test]
    fn test_liveness_sleeping_bodies() {
        let mut processor = QualiaProcessor::new();

        // Mark body 1 as awake, body 2 as not registered (sleeping by default when checking)
        processor.wake_body(1);

        let events = vec![
            // Collision with awake body - should process
            CollisionEvent {
                body1: 1,
                body2: 3,
                impulse: 50.0,
                contact_point: Vec3::ZERO,
            },
            // Collision between non-awake bodies - should skip
            CollisionEvent {
                body1: 4,
                body2: 5,
                impulse: 50.0,
                contact_point: Vec3::ZERO,
            },
        ];

        let pad = processor.process_collisions(&events);

        // Should have processed the first collision
        assert!(pad.arousal > 0.0);
        assert_eq!(processor.stats().events_processed, 1);
        assert_eq!(processor.stats().skipped_body_sleeping, 1);
    }

    #[test]
    fn test_liveness_self_collision_filter() {
        let mut processor = QualiaProcessor::new();
        processor.set_self_body(1); // Only process collisions involving body 1

        let events = vec![
            // Self collision - should process
            CollisionEvent {
                body1: 1,
                body2: 2,
                impulse: 50.0,
                contact_point: Vec3::ZERO,
            },
            // Non-self collision - should skip
            CollisionEvent {
                body1: 3,
                body2: 4,
                impulse: 50.0,
                contact_point: Vec3::ZERO,
            },
        ];

        let pad = processor.process_collisions(&events);

        assert!(pad.arousal > 0.0);
        assert_eq!(processor.stats().events_processed, 1);
        assert_eq!(processor.stats().skipped_not_self, 1);
    }

    #[test]
    fn test_liveness_stats_skip_ratio() {
        let mut processor = QualiaProcessor::new();

        // Process 2 events: 1 above threshold, 1 below
        let events = vec![
            CollisionEvent {
                body1: 1,
                body2: 2,
                impulse: 50.0, // Above threshold
                contact_point: Vec3::ZERO,
            },
            CollisionEvent {
                body1: 1,
                body2: 3,
                impulse: 0.01, // Below threshold
                contact_point: Vec3::ZERO,
            },
        ];

        processor.process_collisions(&events);

        // 1 processed, 1 skipped = 50% skip ratio
        assert_eq!(processor.stats().events_processed, 1);
        assert_eq!(processor.stats().events_skipped, 1);
        assert!((processor.stats().skip_ratio() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_liveness_reset_stats() {
        let mut processor = QualiaProcessor::new();

        let events = vec![CollisionEvent {
            body1: 1,
            body2: 2,
            impulse: 50.0,
            contact_point: Vec3::ZERO,
        }];

        processor.process_collisions(&events);
        assert_eq!(processor.stats().events_processed, 1);

        processor.reset_stats();
        assert_eq!(processor.stats().events_processed, 0);
    }
}
