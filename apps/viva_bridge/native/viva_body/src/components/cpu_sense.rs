use crate::prelude::*;

#[derive(Component, Default, Clone, Debug)]
pub struct CpuSense {
    pub usage_percent: f32,        // 0-100 (Global User+System)
    pub freq_mhz: Vec<f32>,        // Per-core frequencies
    pub temp_celsius: Option<f32>, // CPU package temp (if available)
    pub core_count: usize,
    pub energy_uj: Option<u64>,    // RAPL microjoules (Accumulated)
    pub power_watts: Option<f32>,  // Derived instantaneous power
}
