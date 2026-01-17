use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuReading {
    pub usage: f32,
    pub temp: Option<f32>,
    pub energy_uj: Option<u64>,
    pub freq_mhz: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuReading {
    pub usage: Option<f32>,
    pub vram_used: Option<u64>,  // MB
    pub vram_total: Option<u64>, // MB
    pub temp: Option<f32>,
    pub power_watts: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MemoryReading {
    pub used_percent: f32,
    pub available_gb: f32,
    pub total_gb: f32,
    pub swap_percent: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ThermalReading {
    pub highest_temp: f32,
    pub pressure: f32, // 0.0-1.0 Heat Pressure
}

pub trait SensorReader: Send + Sync {
    fn read_cpu(&mut self) -> CpuReading;
    fn read_gpu(&mut self) -> Option<GpuReading>;
    fn read_memory(&mut self) -> MemoryReading;
    fn read_thermal(&mut self) -> ThermalReading;

    fn platform_name(&self) -> &'static str;
}
