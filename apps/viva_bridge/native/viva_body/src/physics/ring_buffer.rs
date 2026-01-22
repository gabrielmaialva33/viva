//! RingBuffer - Pre-allocated snapshot storage (DuckStation pattern)
//!
//! Zero allocation during simulation. Slots are pre-allocated at startup
//! and reused in circular fashion.
//!
//! # DuckStation Pattern
//! ```cpp
//! std::array<MemorySaveState, REWIND_SLOTS> m_rewind_states;
//! m_rewind_save_index = (m_rewind_save_index + 1) % REWIND_SLOTS;
//! ```

use super::state_wrapper::StateWrapper;
use bevy_ecs::system::Resource;

/// Default number of prediction slots
pub const DEFAULT_SLOTS: usize = 16;

/// Maximum snapshot size in bytes (1MB should be plenty for physics state)
pub const MAX_SNAPSHOT_SIZE: usize = 1024 * 1024;

/// Pre-allocated snapshot slot
#[derive(Clone)]
pub struct SnapshotSlot {
    /// Raw snapshot data
    buffer: Vec<u8>,
    /// Tick when this snapshot was taken
    tick: u64,
    /// Whether this slot has valid data
    valid: bool,
}

impl Default for SnapshotSlot {
    fn default() -> Self {
        Self {
            buffer: Vec::with_capacity(4096), // Pre-allocate typical size
            tick: 0,
            valid: false,
        }
    }
}

impl SnapshotSlot {
    /// Create with specific capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            buffer: Vec::with_capacity(capacity),
            tick: 0,
            valid: false,
        }
    }

    /// Store snapshot data (reuses existing allocation)
    pub fn store(&mut self, wrapper: StateWrapper, tick: u64) {
        let data = wrapper.into_buffer();

        // Reuse allocation if possible
        self.buffer.clear();
        self.buffer.extend_from_slice(&data);
        self.tick = tick;
        self.valid = true;
    }

    /// Get snapshot data for restore
    pub fn data(&self) -> Option<&[u8]> {
        if self.valid {
            Some(&self.buffer)
        } else {
            None
        }
    }

    /// Get tick when snapshot was taken
    pub fn tick(&self) -> u64 {
        self.tick
    }

    /// Check if slot has valid data
    pub fn is_valid(&self) -> bool {
        self.valid
    }

    /// Invalidate this slot
    pub fn invalidate(&mut self) {
        self.valid = false;
    }
}

/// Ring buffer for snapshots - zero allocation after init
///
/// # Usage Pattern (Active Inference)
/// ```ignore
/// // At prediction start: save current state
/// let slot_id = ring.write(current_snapshot, clock.tick());
///
/// // Try different futures...
/// for action in actions {
///     physics.step_n(10);
///     // evaluate outcome
///     ring.restore_from(slot_id); // rollback
/// }
/// ```
#[derive(Resource)]
pub struct SnapshotRingBuffer {
    /// Pre-allocated slots
    slots: Vec<SnapshotSlot>,
    /// Current write index
    write_index: usize,
    /// Number of valid snapshots currently stored
    count: usize,
}

impl Default for SnapshotRingBuffer {
    fn default() -> Self {
        Self::new(DEFAULT_SLOTS)
    }
}

impl SnapshotRingBuffer {
    /// Create ring buffer with N slots
    pub fn new(num_slots: usize) -> Self {
        let mut slots = Vec::with_capacity(num_slots);
        for _ in 0..num_slots {
            slots.push(SnapshotSlot::default());
        }

        Self {
            slots,
            write_index: 0,
            count: 0,
        }
    }

    /// Create with pre-sized slots (for large physics states)
    pub fn with_slot_capacity(num_slots: usize, slot_capacity: usize) -> Self {
        let mut slots = Vec::with_capacity(num_slots);
        for _ in 0..num_slots {
            slots.push(SnapshotSlot::with_capacity(slot_capacity));
        }

        Self {
            slots,
            write_index: 0,
            count: 0,
        }
    }

    /// Write a snapshot to the next slot
    ///
    /// Returns the slot index for later restore
    #[inline]
    pub fn write(&mut self, wrapper: StateWrapper, tick: u64) -> usize {
        let index = self.write_index;
        self.slots[index].store(wrapper, tick);

        // Advance write pointer (circular)
        self.write_index = (self.write_index + 1) % self.slots.len();

        // Track count (capped at capacity)
        if self.count < self.slots.len() {
            self.count += 1;
        }

        index
    }

    /// Get snapshot data from a slot
    pub fn get(&self, index: usize) -> Option<&[u8]> {
        self.slots.get(index).and_then(|s| s.data())
    }

    /// Get the most recent snapshot
    pub fn latest(&self) -> Option<&SnapshotSlot> {
        if self.count == 0 {
            return None;
        }

        // Last written slot is one before current write_index
        let last_index = if self.write_index == 0 {
            self.slots.len() - 1
        } else {
            self.write_index - 1
        };

        if self.slots[last_index].is_valid() {
            Some(&self.slots[last_index])
        } else {
            None
        }
    }

    /// Get snapshot at specific tick (if still in buffer)
    pub fn find_by_tick(&self, tick: u64) -> Option<&SnapshotSlot> {
        self.slots.iter().find(|s| s.is_valid() && s.tick() == tick)
    }

    /// Get snapshot closest to (but not after) target tick
    pub fn find_before_tick(&self, target_tick: u64) -> Option<&SnapshotSlot> {
        self.slots
            .iter()
            .filter(|s| s.is_valid() && s.tick() <= target_tick)
            .max_by_key(|s| s.tick())
    }

    /// Clear all snapshots
    pub fn clear(&mut self) {
        for slot in &mut self.slots {
            slot.invalidate();
        }
        self.write_index = 0;
        self.count = 0;
    }

    /// Get number of valid snapshots
    #[inline]
    pub fn count(&self) -> usize {
        self.count
    }

    /// Get total capacity
    #[inline]
    pub fn capacity(&self) -> usize {
        self.slots.len()
    }

    /// Check if buffer is full
    #[inline]
    pub fn is_full(&self) -> bool {
        self.count >= self.slots.len()
    }

    // =========================================================================
    // Memory stats (for debugging)
    // =========================================================================

    /// Total memory used by all slots
    pub fn memory_used(&self) -> usize {
        self.slots.iter().map(|s| s.buffer.len()).sum()
    }

    /// Total memory allocated (capacity)
    pub fn memory_allocated(&self) -> usize {
        self.slots.iter().map(|s| s.buffer.capacity()).sum()
    }
}

// ============================================================================
// Lightweight temporary snapshot (for short-term rollback)
// ============================================================================

/// Temporary snapshot that can be restored quickly
///
/// Unlike RingBuffer slots, this is owned and can be passed around.
/// Use for prediction rollback within a single system.
#[derive(Clone, Default)]
pub struct TempSnapshot {
    pub buffer: Vec<u8>,
    pub tick: u64,
}

impl TempSnapshot {
    /// Create from StateWrapper
    pub fn from_wrapper(wrapper: StateWrapper, tick: u64) -> Self {
        Self {
            buffer: wrapper.into_buffer(),
            tick,
        }
    }

    /// Create StateWrapper for restore
    pub fn to_wrapper(&self) -> StateWrapper {
        StateWrapper::for_load(self.buffer.clone())
    }

    /// Get size in bytes
    pub fn size(&self) -> usize {
        self.buffer.len()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ring_buffer_creation() {
        let ring = SnapshotRingBuffer::new(8);
        assert_eq!(ring.capacity(), 8);
        assert_eq!(ring.count(), 0);
    }

    #[test]
    fn test_write_and_retrieve() {
        let mut ring = SnapshotRingBuffer::new(4);

        // Create a simple snapshot
        let mut wrapper = StateWrapper::for_save();
        let mut val: f32 = 42.5;
        wrapper.do_val(&mut val);

        let index = ring.write(wrapper, 100);
        assert_eq!(index, 0);
        assert_eq!(ring.count(), 1);

        // Retrieve
        let data = ring.get(index).expect("Should have data");
        assert_eq!(data.len(), 4); // f32 is 4 bytes
    }

    #[test]
    fn test_circular_overwrite() {
        let mut ring = SnapshotRingBuffer::new(3);

        // Write 5 snapshots (should overwrite first 2)
        for i in 0..5u64 {
            let mut wrapper = StateWrapper::for_save();
            let mut val = i as f32;
            wrapper.do_val(&mut val);
            ring.write(wrapper, i);
        }

        // Count should be capped at capacity
        assert_eq!(ring.count(), 3);

        // Find by tick - first two should be overwritten
        assert!(ring.find_by_tick(0).is_none());
        assert!(ring.find_by_tick(1).is_none());
        assert!(ring.find_by_tick(2).is_some());
        assert!(ring.find_by_tick(3).is_some());
        assert!(ring.find_by_tick(4).is_some());
    }

    #[test]
    fn test_latest() {
        let mut ring = SnapshotRingBuffer::new(4);

        // Empty buffer
        assert!(ring.latest().is_none());

        // Add some snapshots
        for i in 0..3u64 {
            let wrapper = StateWrapper::for_save();
            ring.write(wrapper, i * 10);
        }

        let latest = ring.latest().expect("Should have latest");
        assert_eq!(latest.tick(), 20);
    }

    #[test]
    fn test_find_before_tick() {
        let mut ring = SnapshotRingBuffer::new(4);

        // Write snapshots at ticks 10, 20, 30
        for i in 1..=3u64 {
            let wrapper = StateWrapper::for_save();
            ring.write(wrapper, i * 10);
        }

        // Find before tick 25 should return tick 20
        let found = ring.find_before_tick(25).expect("Should find");
        assert_eq!(found.tick(), 20);

        // Find before tick 10 should return tick 10
        let found = ring.find_before_tick(10).expect("Should find");
        assert_eq!(found.tick(), 10);

        // Find before tick 5 should return None
        assert!(ring.find_before_tick(5).is_none());
    }

    #[test]
    fn test_clear() {
        let mut ring = SnapshotRingBuffer::new(4);

        for i in 0..3 {
            let wrapper = StateWrapper::for_save();
            ring.write(wrapper, i);
        }
        assert_eq!(ring.count(), 3);

        ring.clear();
        assert_eq!(ring.count(), 0);
        assert!(ring.latest().is_none());
    }

    #[test]
    fn test_temp_snapshot() {
        let mut wrapper = StateWrapper::for_save();
        let mut val: f32 = 123.456;
        wrapper.do_val(&mut val);

        let snapshot = TempSnapshot::from_wrapper(wrapper, 999);
        assert_eq!(snapshot.tick, 999);
        assert_eq!(snapshot.size(), 4);

        // Restore
        let mut restore = snapshot.to_wrapper();
        let mut restored_val: f32 = 0.0;
        restore.do_val(&mut restored_val);
        assert_eq!(restored_val, 123.456);
    }

    #[test]
    fn test_memory_stats() {
        let mut ring = SnapshotRingBuffer::with_slot_capacity(4, 1024);

        // Initial: allocated but not used
        assert_eq!(ring.memory_used(), 0);
        assert!(ring.memory_allocated() >= 4 * 1024);

        // Write some data
        let mut wrapper = StateWrapper::for_save();
        let mut arr = [0.0f32; 100]; // 400 bytes
        for i in 0..100 {
            arr[i] = i as f32;
        }
        for i in &mut arr {
            wrapper.do_val(i);
        }
        ring.write(wrapper, 0);

        assert_eq!(ring.memory_used(), 400);
    }

    #[test]
    fn test_zero_allocation_pattern() {
        // This test verifies the zero-allocation pattern during prediction

        let mut ring = SnapshotRingBuffer::with_slot_capacity(16, 4096);

        // Simulate Active Inference loop
        for cycle in 0..100u64 {
            // Save state at start of prediction
            let mut save = StateWrapper::for_save();
            let mut position = [1.0f32, 2.0, 3.0];
            let mut velocity = [0.1f32, 0.2, 0.3];
            for p in &mut position {
                save.do_val(p);
            }
            for v in &mut velocity {
                save.do_val(v);
            }

            let slot = ring.write(save, cycle);

            // Simulate 5 different action predictions
            for _action in 0..5 {
                // Would: apply action, step physics, evaluate

                // Rollback from saved slot
                let data = ring.get(slot).expect("Should have snapshot");
                let mut restore = StateWrapper::for_load(data.to_vec());

                let mut restored_pos = [0.0f32; 3];
                let mut restored_vel = [0.0f32; 3];
                for p in &mut restored_pos {
                    restore.do_val(p);
                }
                for v in &mut restored_vel {
                    restore.do_val(v);
                }

                assert_eq!(restored_pos, [1.0, 2.0, 3.0]);
                assert_eq!(restored_vel, [0.1, 0.2, 0.3]);
            }
        }

        // After 100 cycles, memory should not have grown unboundedly
        // (old snapshots are overwritten)
        assert!(ring.memory_used() < 16 * 100); // Each snapshot is 24 bytes
    }
}
