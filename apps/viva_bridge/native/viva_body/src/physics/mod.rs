//! # Jolt Physics Integration for VIVA Body
//!
//! AAA-grade deterministic physics engine for Active Inference.
//! Features:
//! - Snapshot/Rollback for predicting futures
//! - Multithreaded simulation
//! - Deterministic results across platforms
//! - Qualia mapping (physics → emotions)
//! - Tick-based timing (PCSX2 emulator pattern)
//!
//! Architecture:
//! ```text
//! SimulationClock (deterministic ticks)
//!     │
//!     ▼
//! PhysicsWorld (Jolt wrapper)
//!     │
//!     ├──▶ PredictionEngine (Active Inference)
//!     │
//!     └──▶ QualiaProcessor (physics → PAD emotions)
//!             │
//!             ▼
//!         PhysicsPlugin (Bevy integration)
//! ```

mod clock;
mod determinism_tests;
mod plugin;
mod prediction;
mod qualia;
mod ring_buffer;
mod soul_sync;
mod state_wrapper;
mod world;

pub use clock::{ClockSnapshot, SimulationClock, NANOS_PER_TICK, TICKS_PER_SECOND};
pub use plugin::PhysicsPlugin;
pub use prediction::{FuturePrediction, PredictionConfig, PredictionEngine};
pub use qualia::{PadDelta, QualiaConfig, QualiaProcessor};
pub use ring_buffer::{SnapshotRingBuffer, SnapshotSlot, TempSnapshot, DEFAULT_SLOTS};
pub use soul_sync::{
    BodyEvent, EventQueue, SoulBodySync, SoulCommand, SyncSnapshot, BODY_PER_SOUL,
    BODY_TICKS_PER_SECOND, SOUL_TICKS_PER_SECOND,
};
pub use state_wrapper::{FreezeThaw, StateMode, StateWrapper};
pub use world::{CollisionEvent, PhysicsSnapshot, PhysicsWorld};
