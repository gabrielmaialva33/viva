//! SoulSync - Delta synchronization between Soul (10Hz) and Body (60Hz)
//!
//! Based on PCSX2's EE↔IOP synchronization pattern.
//! Ensures Soul and Body don't desynchronize during parallel execution.
//!
//! # PCSX2 Pattern
//! ```cpp
//! EEsCycle += cpuRegs.cycle - EEoCycle;
//! if (EEsCycle > 0) iopEventAction = true;
//! ```
//!
//! # VIVA Translation
//! - EE (Emotion Engine) = Soul (Elixir GenServers, 10Hz)
//! - IOP (I/O Processor) = Body (Rust ECS, 60Hz)
//! - Body runs 6x faster, must sync at boundaries

use bevy_ecs::system::Resource;

/// Default Soul tick rate (10 Hz)
pub const SOUL_TICKS_PER_SECOND: u64 = 10;

/// Default Body tick rate (60 Hz)
pub const BODY_TICKS_PER_SECOND: u64 = 60;

/// Ratio of Body ticks per Soul tick (60/10 = 6)
pub const BODY_PER_SOUL: u64 = BODY_TICKS_PER_SECOND / SOUL_TICKS_PER_SECOND;

/// Soul↔Body synchronization manager
///
/// Tracks relative progress of Soul and Body to ensure they stay in sync.
/// Body runs faster but must wait for Soul at sync points.
#[derive(Resource, Clone, Debug)]
pub struct SoulBodySync {
    /// Current Body tick (60Hz)
    body_tick: u64,
    /// Current Soul tick (10Hz)
    soul_tick: u64,
    /// Tick ratio (Body ticks per Soul tick)
    ratio: u64,
    /// Accumulated drift (Body ticks ahead of expected)
    drift: i64,
    /// Maximum allowed drift before forcing sync
    max_drift: i64,
    /// Whether Soul is currently waiting for data
    soul_waiting: bool,
    /// Last Body tick when we synced with Soul
    last_sync_body_tick: u64,
}

impl Default for SoulBodySync {
    fn default() -> Self {
        Self::new()
    }
}

impl SoulBodySync {
    /// Create with default 60Hz Body, 10Hz Soul
    pub fn new() -> Self {
        Self::with_ratio(BODY_PER_SOUL)
    }

    /// Create with custom ratio
    pub fn with_ratio(ratio: u64) -> Self {
        Self {
            body_tick: 0,
            soul_tick: 0,
            ratio,
            drift: 0,
            max_drift: (ratio * 2) as i64, // 2 Soul ticks of slack
            soul_waiting: false,
            last_sync_body_tick: 0,
        }
    }

    // =========================================================================
    // Tick Management
    // =========================================================================

    /// Advance Body by one tick
    #[inline]
    pub fn advance_body(&mut self) {
        self.body_tick += 1;
        self.update_drift();
    }

    /// Advance Body by N ticks
    #[inline]
    pub fn advance_body_n(&mut self, n: u64) {
        self.body_tick += n;
        self.update_drift();
    }

    /// Record that Soul has processed one tick
    #[inline]
    pub fn advance_soul(&mut self) {
        self.soul_tick += 1;
        self.soul_waiting = false;
    }

    /// Get current Body tick
    #[inline]
    pub fn body_tick(&self) -> u64 {
        self.body_tick
    }

    /// Get current Soul tick
    #[inline]
    pub fn soul_tick(&self) -> u64 {
        self.soul_tick
    }

    // =========================================================================
    // Synchronization Logic (PCSX2 Pattern)
    // =========================================================================

    /// Update drift calculation
    ///
    /// Drift = actual Body ticks - expected Body ticks for current Soul tick
    fn update_drift(&mut self) {
        let expected_body = self.soul_tick * self.ratio;
        self.drift = self.body_tick as i64 - expected_body as i64;
    }

    /// Should Body pause to let Soul catch up?
    ///
    /// Returns true if Body is too far ahead (drift > max_drift)
    #[inline]
    pub fn should_body_wait(&self) -> bool {
        self.drift > self.max_drift
    }

    /// Should Soul receive an update from Body?
    ///
    /// Returns true when Body has completed enough ticks for the next Soul sync.
    /// Uses PCSX2's checkpoint pattern: check at boundaries, not every tick.
    #[inline]
    pub fn should_sync_soul(&self) -> bool {
        // Body has done enough ticks since last sync
        self.body_tick >= self.last_sync_body_tick + self.ratio
    }

    /// Mark that sync with Soul has happened
    #[inline]
    pub fn acknowledge_sync(&mut self) {
        self.last_sync_body_tick = self.body_tick;
    }

    /// Get current drift (positive = Body ahead, negative = Soul ahead)
    #[inline]
    pub fn drift(&self) -> i64 {
        self.drift
    }

    /// Mark that Soul is waiting for Body data
    pub fn soul_request_update(&mut self) {
        self.soul_waiting = true;
    }

    /// Check if Soul is waiting
    #[inline]
    pub fn is_soul_waiting(&self) -> bool {
        self.soul_waiting
    }

    // =========================================================================
    // Timing Utilities
    // =========================================================================

    /// How many Body ticks until next Soul sync?
    #[inline]
    pub fn ticks_until_sync(&self) -> u64 {
        let next_sync = self.last_sync_body_tick + self.ratio;
        next_sync.saturating_sub(self.body_tick)
    }

    /// Convert Soul tick to equivalent Body tick
    #[inline]
    pub fn soul_to_body_tick(&self, soul_tick: u64) -> u64 {
        soul_tick * self.ratio
    }

    /// Convert Body tick to equivalent Soul tick (rounded down)
    #[inline]
    pub fn body_to_soul_tick(&self, body_tick: u64) -> u64 {
        body_tick / self.ratio
    }

    /// Get elapsed time in seconds (using Body tick rate)
    pub fn elapsed_seconds(&self) -> f64 {
        self.body_tick as f64 / BODY_TICKS_PER_SECOND as f64
    }

    // =========================================================================
    // Snapshot/Restore
    // =========================================================================

    /// Save sync state
    pub fn save(&self) -> SyncSnapshot {
        SyncSnapshot {
            body_tick: self.body_tick,
            soul_tick: self.soul_tick,
            drift: self.drift,
            last_sync_body_tick: self.last_sync_body_tick,
        }
    }

    /// Restore sync state
    pub fn restore(&mut self, snapshot: &SyncSnapshot) {
        self.body_tick = snapshot.body_tick;
        self.soul_tick = snapshot.soul_tick;
        self.drift = snapshot.drift;
        self.last_sync_body_tick = snapshot.last_sync_body_tick;
        self.soul_waiting = false;
    }
}

/// Snapshot of sync state
#[derive(Clone, Debug, Default)]
pub struct SyncSnapshot {
    pub body_tick: u64,
    pub soul_tick: u64,
    pub drift: i64,
    pub last_sync_body_tick: u64,
}

// ============================================================================
// Event Queue for Soul↔Body Communication
// ============================================================================

/// Types of events that can be sent from Body to Soul
#[derive(Clone, Debug)]
pub enum BodyEvent {
    /// State changed (stress, emotions, etc.)
    StateChanged {
        tick: u64,
        stress: f32,
        pleasure: f64,
        arousal: f64,
        dominance: f64,
    },
    /// Collision occurred (for qualia)
    Collision {
        tick: u64,
        impulse: f32,
        body_id: u32,
    },
    /// Prediction completed
    PredictionComplete {
        tick: u64,
        best_action_id: u32,
        confidence: f32,
    },
    /// Heartbeat (periodic liveness signal)
    Heartbeat { tick: u64 },
}

/// Types of commands from Soul to Body
#[derive(Clone, Debug)]
pub enum SoulCommand {
    /// Apply emotional stimulus
    ApplyStimulus {
        pleasure: f64,
        arousal: f64,
        dominance: f64,
    },
    /// Request prediction for given actions
    RequestPrediction { action_ids: Vec<u32> },
    /// Change tick rate
    SetTickRate { ticks_per_second: u64 },
    /// Pause/resume
    SetPaused { paused: bool },
    /// Shutdown
    Shutdown,
}

/// Bounded event queue (avoids unbounded memory growth)
pub struct EventQueue<T> {
    events: Vec<T>,
    max_size: usize,
}

impl<T> EventQueue<T> {
    pub fn new(max_size: usize) -> Self {
        Self {
            events: Vec::with_capacity(max_size),
            max_size,
        }
    }

    /// Push event (drops oldest if full)
    pub fn push(&mut self, event: T) {
        if self.events.len() >= self.max_size {
            self.events.remove(0); // Drop oldest
        }
        self.events.push(event);
    }

    /// Drain all events
    pub fn drain(&mut self) -> Vec<T> {
        std::mem::take(&mut self.events)
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Get count
    pub fn len(&self) -> usize {
        self.events.len()
    }
}

impl<T> Default for EventQueue<T> {
    fn default() -> Self {
        Self::new(64)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sync_creation() {
        let sync = SoulBodySync::new();
        assert_eq!(sync.body_tick(), 0);
        assert_eq!(sync.soul_tick(), 0);
        assert_eq!(sync.drift(), 0);
    }

    #[test]
    fn test_advance_body() {
        let mut sync = SoulBodySync::new();

        // Advance 6 Body ticks (= 1 Soul tick worth)
        sync.advance_body_n(6);
        assert_eq!(sync.body_tick(), 6);
        assert_eq!(sync.drift(), 6); // Body is 6 ticks ahead of Soul

        // Now Soul catches up
        sync.advance_soul();
        sync.update_drift();
        assert_eq!(sync.drift(), 0); // Back in sync
    }

    #[test]
    fn test_should_sync_soul() {
        let mut sync = SoulBodySync::new();

        // Initially at tick 0, not yet time to sync (need 6 ticks first)
        assert!(!sync.should_sync_soul()); // 0 >= 0 + 6? No

        // After 5 Body ticks, not yet time to sync
        sync.advance_body_n(5);
        assert!(!sync.should_sync_soul()); // 5 >= 0 + 6? No

        // After 6 Body ticks, time to sync
        sync.advance_body();
        assert!(sync.should_sync_soul()); // 6 >= 0 + 6? Yes

        // Acknowledge and check next boundary
        sync.acknowledge_sync();
        assert!(!sync.should_sync_soul()); // 6 >= 6 + 6? No

        sync.advance_body_n(6);
        assert!(sync.should_sync_soul()); // 12 >= 6 + 6? Yes
    }

    #[test]
    fn test_should_body_wait() {
        let mut sync = SoulBodySync::new();

        // Body runs way ahead
        sync.advance_body_n(20); // 20 ticks ahead
        assert!(sync.should_body_wait()); // drift 20 > max_drift 12

        // Soul catches up partially
        sync.advance_soul(); // Soul at tick 1
        sync.advance_soul(); // Soul at tick 2
        sync.update_drift();
        assert_eq!(sync.drift(), 8); // 20 - 12 = 8
        assert!(!sync.should_body_wait()); // 8 <= 12
    }

    #[test]
    fn test_conversion() {
        let sync = SoulBodySync::new();

        assert_eq!(sync.soul_to_body_tick(1), 6);
        assert_eq!(sync.soul_to_body_tick(10), 60);

        assert_eq!(sync.body_to_soul_tick(6), 1);
        assert_eq!(sync.body_to_soul_tick(11), 1); // Rounds down
        assert_eq!(sync.body_to_soul_tick(12), 2);
    }

    #[test]
    fn test_snapshot_restore() {
        let mut sync = SoulBodySync::new();
        sync.advance_body_n(100);
        sync.advance_soul();
        sync.advance_soul();
        sync.acknowledge_sync();

        let snapshot = sync.save();

        // Modify state
        sync.advance_body_n(50);
        sync.advance_soul();

        // Restore
        sync.restore(&snapshot);
        assert_eq!(sync.body_tick(), 100);
        assert_eq!(sync.soul_tick(), 2);
    }

    #[test]
    fn test_event_queue() {
        let mut queue: EventQueue<i32> = EventQueue::new(3);

        queue.push(1);
        queue.push(2);
        queue.push(3);
        assert_eq!(queue.len(), 3);

        // Push beyond capacity drops oldest
        queue.push(4);
        assert_eq!(queue.len(), 3);

        let events = queue.drain();
        assert_eq!(events, vec![2, 3, 4]); // 1 was dropped
    }

    #[test]
    fn test_deterministic_sync() {
        // Two sync managers with identical operations must produce identical state
        let mut sync1 = SoulBodySync::new();
        let mut sync2 = SoulBodySync::new();

        for _ in 0..100 {
            sync1.advance_body();
            sync2.advance_body();

            if sync1.should_sync_soul() {
                sync1.advance_soul();
                sync1.acknowledge_sync();
            }
            if sync2.should_sync_soul() {
                sync2.advance_soul();
                sync2.acknowledge_sync();
            }
        }

        assert_eq!(sync1.body_tick(), sync2.body_tick());
        assert_eq!(sync1.soul_tick(), sync2.soul_tick());
        assert_eq!(sync1.drift(), sync2.drift());
    }

    #[test]
    fn test_ticks_until_sync() {
        let mut sync = SoulBodySync::new();
        sync.acknowledge_sync();

        assert_eq!(sync.ticks_until_sync(), 6);

        sync.advance_body_n(4);
        assert_eq!(sync.ticks_until_sync(), 2);

        sync.advance_body_n(2);
        assert_eq!(sync.ticks_until_sync(), 0);
    }
}
