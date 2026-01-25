//! Determinism Tests - Verify byte-identical output for same input
//!
//! These tests validate the emulator patterns:
//! - SimulationClock: Tick-based timing is deterministic
//! - StateWrapper: Save/restore produces identical bytes
//! - RingBuffer: Circular storage is deterministic
//! - SoulSync: Synchronization is deterministic

#[cfg(test)]
mod tests {
    use crate::physics::{
        SimulationClock, SnapshotRingBuffer, SoulBodySync, StateWrapper,
    };
    use glam::{Quat, Vec3};

    /// Test: Two clocks with identical operations produce identical state
    #[test]
    fn test_clock_determinism() {
        let mut clock1 = SimulationClock::new();
        let mut clock2 = SimulationClock::new();

        // Run same operations on both
        for _ in 0..1000 {
            clock1.advance();
            clock2.advance();

            if clock1.should_check_events() {
                clock1.acknowledge_event_check();
            }
            if clock2.should_check_events() {
                clock2.acknowledge_event_check();
            }
        }

        // Must be byte-identical
        assert_eq!(clock1.tick(), clock2.tick());
        assert_eq!(clock1.delta_time(), clock2.delta_time());

        // Snapshots must be identical
        let snap1 = clock1.save();
        let snap2 = clock2.save();
        assert_eq!(snap1.tick, snap2.tick);
        assert_eq!(snap1.next_event_tick, snap2.next_event_tick);
    }

    /// Test: StateWrapper produces identical bytes for same data
    #[test]
    fn test_state_wrapper_determinism() {
        let position = Vec3::new(1.23456, 7.89012, 3.45678);
        let velocity = Vec3::new(0.1, 0.2, 0.3);
        let rotation = Quat::from_euler(glam::EulerRot::XYZ, 0.5, 1.0, 0.3);

        // Save twice
        let mut save1 = StateWrapper::for_save();
        let mut pos1 = position;
        let mut vel1 = velocity;
        let mut rot1 = rotation;
        save1.do_vec3(&mut pos1);
        save1.do_vec3(&mut vel1);
        save1.do_quat(&mut rot1);

        let mut save2 = StateWrapper::for_save();
        let mut pos2 = position;
        let mut vel2 = velocity;
        let mut rot2 = rotation;
        save2.do_vec3(&mut pos2);
        save2.do_vec3(&mut vel2);
        save2.do_quat(&mut rot2);

        // Buffers must be byte-identical
        let buf1 = save1.into_buffer();
        let buf2 = save2.into_buffer();
        assert_eq!(buf1, buf2, "StateWrapper must be deterministic");
    }

    /// Test: Save and restore produces identical state
    #[test]
    fn test_state_roundtrip_determinism() {
        let original_pos = Vec3::new(1.23456, 7.89012, 3.45678);
        let original_vel = Vec3::new(0.1, 0.2, 0.3);

        // Save
        let mut save = StateWrapper::for_save();
        let mut pos = original_pos;
        let mut vel = original_vel;
        save.do_vec3(&mut pos);
        save.do_vec3(&mut vel);
        let buffer = save.into_buffer();

        // Restore
        let mut load = StateWrapper::for_load(buffer);
        let mut restored_pos = Vec3::ZERO;
        let mut restored_vel = Vec3::ZERO;
        load.do_vec3(&mut restored_pos);
        load.do_vec3(&mut restored_vel);

        // Must be bit-identical (no floating point drift)
        assert_eq!(
            original_pos.to_array(),
            restored_pos.to_array(),
            "Position must be bit-identical after roundtrip"
        );
        assert_eq!(
            original_vel.to_array(),
            restored_vel.to_array(),
            "Velocity must be bit-identical after roundtrip"
        );
    }

    /// Test: Ring buffer operations are deterministic
    #[test]
    fn test_ring_buffer_determinism() {
        let mut ring1 = SnapshotRingBuffer::new(4);
        let mut ring2 = SnapshotRingBuffer::new(4);

        // Same sequence of writes
        for i in 0..10u64 {
            let mut save1 = StateWrapper::for_save();
            let mut save2 = StateWrapper::for_save();
            let mut val = i as f32;
            save1.do_val(&mut val);
            save2.do_val(&mut val);

            let idx1 = ring1.write(save1, i);
            let idx2 = ring2.write(save2, i);

            assert_eq!(idx1, idx2, "Write indices must match");
        }

        // Same data at same indices
        assert_eq!(ring1.count(), ring2.count());

        for i in 0..4 {
            let data1 = ring1.get(i);
            let data2 = ring2.get(i);
            assert_eq!(data1, data2, "Data at index {} must match", i);
        }
    }

    /// Test: Soul↔Body sync is deterministic
    #[test]
    fn test_soul_sync_determinism() {
        let mut sync1 = SoulBodySync::new();
        let mut sync2 = SoulBodySync::new();

        // Same sequence of operations
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

        // Must be identical
        assert_eq!(sync1.body_tick(), sync2.body_tick());
        assert_eq!(sync1.soul_tick(), sync2.soul_tick());
        assert_eq!(sync1.drift(), sync2.drift());

        // Snapshots must be identical
        let snap1 = sync1.save();
        let snap2 = sync2.save();
        assert_eq!(snap1.body_tick, snap2.body_tick);
        assert_eq!(snap1.soul_tick, snap2.soul_tick);
        assert_eq!(snap1.drift, snap2.drift);
    }

    /// Test: Multiple rollbacks produce identical final state
    #[test]
    fn test_prediction_rollback_determinism() {
        // Simulate Active Inference prediction cycle

        let initial_pos = Vec3::new(0.0, 10.0, 0.0);
        let initial_vel = Vec3::new(0.0, -1.0, 0.0);

        // Save initial state
        let mut save = StateWrapper::for_save();
        let mut pos = initial_pos;
        let mut vel = initial_vel;
        save.do_vec3(&mut pos);
        save.do_vec3(&mut vel);
        let snapshot = save.into_buffer();

        // Simulate 5 different futures, rollback each time
        for action_id in 0..5 {
            // Load state
            let mut load = StateWrapper::for_load(snapshot.clone());
            let mut sim_pos = Vec3::ZERO;
            let mut sim_vel = Vec3::ZERO;
            load.do_vec3(&mut sim_pos);
            load.do_vec3(&mut sim_vel);

            // State should be identical to initial
            assert_eq!(
                sim_pos.to_array(),
                initial_pos.to_array(),
                "Position after rollback {} must be identical",
                action_id
            );
            assert_eq!(
                sim_vel.to_array(),
                initial_vel.to_array(),
                "Velocity after rollback {} must be identical",
                action_id
            );

            // Simulate some physics (doesn't matter, we'll rollback)
            sim_pos += sim_vel * 0.016 * action_id as f32;
        }
    }

    /// Test: Complete simulation cycle is deterministic
    #[test]
    fn test_full_simulation_determinism() {
        // Run two identical simulations and compare

        fn run_simulation(seed: u64) -> (u64, Vec<u8>) {
            let mut clock = SimulationClock::new();
            let mut sync = SoulBodySync::new();
            let mut ring = SnapshotRingBuffer::new(8);

            // Simulate 100 ticks
            for _tick in 0..100 {
                clock.advance();
                sync.advance_body();

                // Periodic snapshot
                if clock.tick() % 10 == 0 {
                    let mut save = StateWrapper::for_save();
                    let mut tick = clock.tick();
                    save.do_val(&mut tick);
                    ring.write(save, clock.tick());
                }

                // Soul sync
                if sync.should_sync_soul() {
                    sync.advance_soul();
                    sync.acknowledge_sync();
                }
            }

            // Return final state
            let mut final_save = StateWrapper::for_save();
            let mut final_tick = clock.tick();
            let mut final_body = sync.body_tick();
            let mut final_soul = sync.soul_tick();
            final_save.do_val(&mut final_tick);
            final_save.do_val(&mut final_body);
            final_save.do_val(&mut final_soul);

            (clock.tick(), final_save.into_buffer())
        }

        // Run twice with same "seed" (operations)
        let (tick1, state1) = run_simulation(42);
        let (tick2, state2) = run_simulation(42);

        assert_eq!(tick1, tick2, "Final tick must be identical");
        assert_eq!(state1, state2, "Final state must be byte-identical");
    }

    /// Test: Floating point operations maintain determinism
    #[test]
    fn test_floating_point_determinism() {
        // This test verifies that our fixed delta_time approach
        // produces consistent results

        let clock = SimulationClock::new();
        let dt = clock.delta_time();

        // Multiple calls must return same value
        assert_eq!(dt, clock.delta_time());
        assert_eq!(dt, clock.delta_time());
        assert_eq!(dt, clock.delta_time());

        // Value must be exactly 1/60
        let expected = 1.0f32 / 60.0;
        assert_eq!(dt, expected);

        // Accumulated dt must be predictable
        let accumulated = dt * 60.0;
        // Note: Due to floating point, this won't be exactly 1.0
        // but it MUST be the same every time
        let accumulated2 = dt * 60.0;
        assert_eq!(accumulated, accumulated2);
    }
}
