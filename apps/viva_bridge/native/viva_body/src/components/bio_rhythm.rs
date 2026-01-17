use crate::prelude::*;
use serde::Serialize;

#[derive(Component, Default, Clone, Debug, Serialize)]
pub struct BioRhythm {
    pub stress_level: f32,         // 0.0-1.0 (Composite Hardware Stress)
    pub system_entropy: f32,       // Shannon entropy of system states
    pub heartrate_bpm: f32,        // 60-120bpm (Derived from Energy/Tick)
    pub fatigue: f32,              // Accumulated metabolic waste
    pub needs_rest: bool,          // True if fatigue > threshold
    pub metabolism_rate: f32,      // Current processing speed (0.0-1.0)
}
