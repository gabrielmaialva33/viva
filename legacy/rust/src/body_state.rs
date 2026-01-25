use crate::dynamics::{DynAffect, OUParams};
use rustler::NifMap;
use serde::{Deserialize, Serialize};

/// Complete hardware metrics
#[derive(Debug, Clone, Serialize, Deserialize, NifMap)]
pub struct HardwareState {
    pub cpu_usage: f32,
    pub cpu_temp: Option<f32>,
    pub cpu_count: usize,
    pub memory_used_percent: f32,
    pub memory_available_gb: f32,
    pub memory_total_gb: f32,
    pub swap_used_percent: f32,
    pub gpu_usage: Option<f32>,
    pub gpu_vram_used_percent: Option<f32>,
    pub gpu_temp: Option<f32>,
    pub gpu_name: Option<String>,
    pub disk_usage_percent: f32,
    pub disk_read_bytes: u64,
    pub disk_write_bytes: u64,
    pub net_rx_bytes: u64,
    pub net_tx_bytes: u64,
    pub uptime_seconds: u64,
    pub process_count: usize,
    pub load_avg_1m: f64,
    pub load_avg_5m: f64,
    pub load_avg_15m: f64,
    pub cpu_freq_mhz: Option<f32>,
    pub l3_cache_kb: Option<u32>,
    pub context_switches: u64,
    pub interrupts: u64,
    pub system_entropy: f32,
    pub os_jitter: f32,
    pub fan_rpm: Option<u32>,
    pub target_fan_rpm: Option<u32>,
}

impl HardwareState {
    pub fn empty() -> Self {
        Self {
            cpu_usage: 0.0,
            cpu_temp: None,
            cpu_count: 0,
            memory_used_percent: 0.0,
            memory_available_gb: 0.0,
            memory_total_gb: 0.0,
            swap_used_percent: 0.0,
            gpu_usage: None,
            gpu_vram_used_percent: None,
            gpu_temp: None,
            gpu_name: None,
            disk_usage_percent: 0.0,
            disk_read_bytes: 0,
            disk_write_bytes: 0,
            net_rx_bytes: 0,
            net_tx_bytes: 0,
            uptime_seconds: 0,
            process_count: 0,
            load_avg_1m: 0.0,
            load_avg_5m: 0.0,
            load_avg_15m: 0.0,
            cpu_freq_mhz: None,
            l3_cache_kb: None,
            context_switches: 0,
            interrupts: 0,
            system_entropy: 0.0,
            os_jitter: 0.0,
            fan_rpm: None,
            target_fan_rpm: None,
        }
    }
}

/// Complete body state - everything VIVA feels
#[derive(Debug, Clone, Serialize, Deserialize, NifMap)]
pub struct BodyState {
    pub pleasure: f64,
    pub arousal: f64,
    pub dominance: f64,
    pub hardware: HardwareState,
    pub stress_level: f64,
    pub in_bifurcation: bool,
    pub tick: u64,
    pub timestamp_ms: u64,
}

impl BodyState {
    pub fn default() -> Self {
        Self {
            pleasure: 0.0,
            arousal: 0.0,
            dominance: 0.0,
            hardware: HardwareState::empty(),
            stress_level: 0.0,
            in_bifurcation: false,
            tick: 0,
            timestamp_ms: 0,
        }
    }
}

// ... (Rest of BodyConfig and BodyEngine can remain or be stubbed if dependencies fail)
// I will keep BodyConfig struct as it was, but stub BodyEngine methods that used crate::collect_hardware_state
// to avoid circular dependency if I can't resolve it.
// Actually I'll just include the structs.

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BodyConfig {
    pub dt: f64,
    pub cusp_enabled: bool,
    pub cusp_sensitivity: f64,
    pub ou_params: [OUParams; 3],
    pub seed: u64,
}

impl Default for BodyConfig {
    fn default() -> Self {
        Self {
            dt: 0.5,
            cusp_enabled: true,
            cusp_sensitivity: 0.5,
            ou_params: [
                OUParams {
                    theta: 0.3,
                    mu: 0.0,
                    sigma: 0.15,
                },
                OUParams {
                    theta: 0.5,
                    mu: 0.0,
                    sigma: 0.25,
                },
                OUParams {
                    theta: 0.2,
                    mu: 0.0,
                    sigma: 0.10,
                },
            ],
            seed: 0,
        }
    }
}

pub struct BodyEngine {
    // Stubbed BodyEngine since we use ECS now.
    // Keeping this struct to satisfy existing legacy code if any.
}
