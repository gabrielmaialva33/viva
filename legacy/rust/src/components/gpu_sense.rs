use crate::prelude::*;

#[derive(Component, Default, Clone, Debug)]
pub struct GpuSense {
    pub usage_percent: Option<f32>,
    pub vram_used_mb: Option<u64>,
    pub vram_total_mb: Option<u64>,
    pub temp_celsius: Option<f32>,
    pub power_watts: Option<f32>,
    pub name: Option<String>,
}
