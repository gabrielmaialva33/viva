//! SimulationClock - Deterministic tick-based timing (PCSX2 pattern)
//!
//! Replaces float delta_time with logical tick counter for deterministic replay.
//! Based on emulator timing architecture: every operation advances by fixed ticks.

use bevy_ecs::system::Resource;

/// Fixed ticks per second (60 Hz physics)
pub const TICKS_PER_SECOND: u64 = 60;

/// Nanoseconds per tick (for wall-clock conversion when needed)
pub const NANOS_PER_TICK: u64 = 1_000_000_000 / TICKS_PER_SECOND;

/// SimulationClock - Deterministic timing for physics and prediction
///
/// # Emulator Pattern (PCSX2)
/// Instead of: `physics.step(delta_time)` with float
/// We use: `clock.advance(1); physics.step_tick(&clock)`
///
/// This guarantees:
/// - Same input sequence → identical output (byte-for-byte)
/// - Replay possible across platforms
/// - No floating-point timing drift
#[derive(Resource, Clone, Debug)]
pub struct SimulationClock {
    /// Current tick (absolute counter, never resets)
    tick: u64,

    /// Ticks per second (default: 60)
    ticks_per_second: u64,

    /// Next tick when events should be checked (checkpoint pattern)
    /// Instead of checking every tick, we batch until this threshold.
    next_event_tick: u64,

    /// Event check interval (ticks between mandatory event checks)
    event_interval: u64,

    /// Sub-tick accumulator for variable-rate sources (0..TICKS_PER_SECOND-1)
    /// Used when integrating with external time sources
    sub_tick_accumulator: u64,
}

impl Default for SimulationClock {
    fn default() -> Self {
        Self::new()
    }
}

impl SimulationClock {
    /// Create new clock at tick 0
    pub fn new() -> Self {
        Self::with_rate(TICKS_PER_SECOND)
    }

    /// Create clock with custom tick rate
    pub fn with_rate(ticks_per_second: u64) -> Self {
        Self {
            tick: 0,
            ticks_per_second,
            next_event_tick: 6, // First event check after 6 ticks (100ms at 60Hz)
            event_interval: 6,  // 10Hz event checks (Soul sync rate)
            sub_tick_accumulator: 0,
        }
    }

    // =========================================================================
    // Core API
    // =========================================================================

    /// Get current tick
    #[inline]
    pub fn tick(&self) -> u64 {
        self.tick
    }

    /// Advance by one tick
    #[inline]
    pub fn advance(&mut self) {
        self.tick += 1;
    }

    /// Advance by N ticks
    #[inline]
    pub fn advance_n(&mut self, n: u64) {
        self.tick += n;
    }

    /// Get fixed delta time for physics (always consistent)
    #[inline]
    pub fn delta_time(&self) -> f32 {
        1.0 / self.ticks_per_second as f32
    }

    /// Get ticks per second
    #[inline]
    pub fn ticks_per_second(&self) -> u64 {
        self.ticks_per_second
    }

    // =========================================================================
    // Event Checkpoint Pattern (PCSX2)
    // =========================================================================

    /// Should we check events this tick?
    /// Returns true when tick >= next_event_tick
    #[inline]
    pub fn should_check_events(&self) -> bool {
        self.tick >= self.next_event_tick
    }

    /// Acknowledge event check and schedule next
    #[inline]
    pub fn acknowledge_event_check(&mut self) {
        self.next_event_tick = self.tick + self.event_interval;
    }

    /// Set custom event interval
    pub fn set_event_interval(&mut self, interval: u64) {
        self.event_interval = interval;
    }

    /// Get remaining ticks until next event check
    #[inline]
    pub fn ticks_until_event(&self) -> u64 {
        self.next_event_tick.saturating_sub(self.tick)
    }

    // =========================================================================
    // Wall-Clock Integration (for external time sources)
    // =========================================================================

    /// Convert wall-clock nanoseconds to ticks
    /// Returns (whole_ticks, remaining_nanos)
    pub fn nanos_to_ticks(nanos: u64) -> (u64, u64) {
        let whole_ticks = nanos / NANOS_PER_TICK;
        let remaining = nanos % NANOS_PER_TICK;
        (whole_ticks, remaining)
    }

    /// Accumulate external time and return ticks to advance
    /// This handles variable-rate input (e.g., from Bevy Time)
    pub fn accumulate_nanos(&mut self, nanos: u64) -> u64 {
        let total_nanos = self.sub_tick_accumulator + nanos;
        let (ticks, remaining) = Self::nanos_to_ticks(total_nanos);
        self.sub_tick_accumulator = remaining;
        ticks
    }

    // =========================================================================
    // Snapshot/Restore (for prediction rollback)
    // =========================================================================

    /// Save clock state for rollback
    pub fn save(&self) -> ClockSnapshot {
        ClockSnapshot {
            tick: self.tick,
            next_event_tick: self.next_event_tick,
            sub_tick_accumulator: self.sub_tick_accumulator,
        }
    }

    /// Restore clock state from snapshot
    pub fn restore(&mut self, snapshot: &ClockSnapshot) {
        self.tick = snapshot.tick;
        self.next_event_tick = snapshot.next_event_tick;
        self.sub_tick_accumulator = snapshot.sub_tick_accumulator;
    }

    // =========================================================================
    // Conversion utilities
    // =========================================================================

    /// Convert ticks to seconds (for display/logging)
    #[inline]
    pub fn ticks_to_seconds(&self, ticks: u64) -> f64 {
        ticks as f64 / self.ticks_per_second as f64
    }

    /// Convert seconds to ticks
    #[inline]
    pub fn seconds_to_ticks(&self, seconds: f64) -> u64 {
        (seconds * self.ticks_per_second as f64) as u64
    }

    /// Get elapsed time in seconds since tick 0
    #[inline]
    pub fn elapsed_seconds(&self) -> f64 {
        self.ticks_to_seconds(self.tick)
    }
}

/// Snapshot of clock state for rollback
#[derive(Clone, Debug, Default)]
pub struct ClockSnapshot {
    pub tick: u64,
    pub next_event_tick: u64,
    pub sub_tick_accumulator: u64,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clock_creation() {
        let clock = SimulationClock::new();
        assert_eq!(clock.tick(), 0);
        assert_eq!(clock.ticks_per_second(), 60);
    }

    #[test]
    fn test_advance() {
        let mut clock = SimulationClock::new();
        clock.advance();
        assert_eq!(clock.tick(), 1);

        clock.advance_n(10);
        assert_eq!(clock.tick(), 11);
    }

    #[test]
    fn test_delta_time_consistency() {
        let clock = SimulationClock::new();
        let dt1 = clock.delta_time();
        let dt2 = clock.delta_time();
        assert_eq!(dt1, dt2); // Always same value
        assert!((dt1 - 1.0 / 60.0).abs() < 1e-6);
    }

    #[test]
    fn test_event_checkpoint() {
        let mut clock = SimulationClock::new();
        clock.set_event_interval(6); // Check every 6 ticks

        // Initially, next_event_tick is 6
        assert!(!clock.should_check_events()); // tick 0 < 6

        // Advance to tick 5
        clock.advance_n(5);
        assert!(!clock.should_check_events()); // tick 5 < 6

        // Advance to tick 6
        clock.advance();
        assert!(clock.should_check_events()); // tick 6 >= 6

        // Acknowledge and check next
        clock.acknowledge_event_check();
        assert_eq!(clock.ticks_until_event(), 6); // Next at tick 12
    }

    #[test]
    fn test_nanos_conversion() {
        // 1 second ≈ 60 ticks at 60 Hz
        // Note: Integer division means NANOS_PER_TICK = 16_666_666 (not exact)
        // So 60 * 16_666_666 = 999_999_960, leaving 40 nanos remainder
        let (ticks, remaining) = SimulationClock::nanos_to_ticks(1_000_000_000);
        assert_eq!(ticks, 60);
        assert_eq!(remaining, 40); // Integer division remainder

        // Half a tick
        let (ticks, remaining) = SimulationClock::nanos_to_ticks(NANOS_PER_TICK / 2);
        assert_eq!(ticks, 0);
        assert_eq!(remaining, NANOS_PER_TICK / 2);
    }

    #[test]
    fn test_accumulate_nanos() {
        let mut clock = SimulationClock::new();

        // First call: 0.5 tick worth of nanos
        let ticks1 = clock.accumulate_nanos(NANOS_PER_TICK / 2);
        assert_eq!(ticks1, 0); // Not enough for a full tick

        // Second call: another 0.5 tick → now we have 1 full tick
        let ticks2 = clock.accumulate_nanos(NANOS_PER_TICK / 2);
        assert_eq!(ticks2, 1); // Now we have 1 tick

        // Third call: 1.5 ticks worth
        let ticks3 = clock.accumulate_nanos(NANOS_PER_TICK + NANOS_PER_TICK / 2);
        assert_eq!(ticks3, 1); // 1 full tick, 0.5 remaining
    }

    #[test]
    fn test_snapshot_restore() {
        let mut clock = SimulationClock::new();
        clock.advance_n(100);
        clock.acknowledge_event_check();

        // Save
        let snapshot = clock.save();

        // Modify
        clock.advance_n(50);
        assert_eq!(clock.tick(), 150);

        // Restore
        clock.restore(&snapshot);
        assert_eq!(clock.tick(), 100);
    }

    #[test]
    fn test_determinism() {
        // Two clocks with same operations must produce identical state
        let mut clock1 = SimulationClock::new();
        let mut clock2 = SimulationClock::new();

        for _ in 0..100 {
            clock1.advance();
            clock2.advance();

            if clock1.should_check_events() {
                clock1.acknowledge_event_check();
            }
            if clock2.should_check_events() {
                clock2.acknowledge_event_check();
            }
        }

        assert_eq!(clock1.tick(), clock2.tick());
        assert_eq!(clock1.next_event_tick, clock2.next_event_tick);
    }
}
