//! VIVA Body - Hardware Sensing (Interoception)
//!
//! This module provides hardware interoception for VIVA:
//! - CPU usage → "heartbeat"
//! - CPU temperature → "fever"
//! - RAM pressure → "cognitive load"
//! - Available memory → "mental capacity"
//!
//! These metrics map to emotional qualia, influencing VIVA's decisions.

use rustler::{Encoder, Env, NifResult, Term};
use sysinfo::System;
use std::sync::LazyLock;
use std::sync::Mutex;

// Global System instance for hardware sensing
static SYSTEM: LazyLock<Mutex<System>> = LazyLock::new(|| {
    let mut sys = System::new_all();
    sys.refresh_all();
    Mutex::new(sys)
});

/// Hardware state returned to Elixir
#[derive(Debug)]
pub struct HardwareState {
    pub cpu_usage: f32,
    pub memory_used_percent: f32,
    pub memory_available_gb: f32,
    pub uptime_seconds: u64,
}

impl Encoder for HardwareState {
    fn encode<'a>(&self, env: Env<'a>) -> Term<'a> {
        let map = rustler::types::map::map_new(env);
        let map = map.map_put(
            rustler::types::atom::Atom::from_str(env, "cpu_usage").unwrap(),
            self.cpu_usage,
        ).unwrap();
        let map = map.map_put(
            rustler::types::atom::Atom::from_str(env, "memory_used_percent").unwrap(),
            self.memory_used_percent,
        ).unwrap();
        let map = map.map_put(
            rustler::types::atom::Atom::from_str(env, "memory_available_gb").unwrap(),
            self.memory_available_gb,
        ).unwrap();
        let map = map.map_put(
            rustler::types::atom::Atom::from_str(env, "uptime_seconds").unwrap(),
            self.uptime_seconds,
        ).unwrap();
        map
    }
}

/// Check if VIVA's body is alive
#[rustler::nif]
fn alive() -> &'static str {
    "VIVA body is alive"
}

/// Get current hardware state (interoception)
#[rustler::nif]
fn feel_hardware() -> NifResult<HardwareState> {
    let mut sys = SYSTEM.lock().unwrap();
    sys.refresh_cpu_all();
    sys.refresh_memory();
    
    let cpu_usage = sys.global_cpu_usage();
    let total_memory = sys.total_memory() as f64;
    let used_memory = sys.used_memory() as f64;
    let available_memory = sys.available_memory() as f64;
    
    let memory_used_percent = if total_memory > 0.0 {
        ((used_memory / total_memory) * 100.0) as f32
    } else {
        0.0
    };
    
    let memory_available_gb = (available_memory / 1_073_741_824.0) as f32; // bytes to GB
    
    Ok(HardwareState {
        cpu_usage,
        memory_used_percent,
        memory_available_gb,
        uptime_seconds: System::uptime(),
    })
}

/// Map hardware metrics to emotional qualia
/// Returns a map with PAD-compatible deltas
#[rustler::nif]
fn hardware_to_qualia() -> NifResult<(f64, f64, f64)> {
    let mut sys = SYSTEM.lock().unwrap();
    sys.refresh_cpu_all();
    sys.refresh_memory();
    
    let cpu_usage = sys.global_cpu_usage() as f64;
    let total_memory = sys.total_memory() as f64;
    let used_memory = sys.used_memory() as f64;
    let memory_pressure = if total_memory > 0.0 {
        used_memory / total_memory
    } else {
        0.0
    };
    
    // Map hardware → PAD deltas
    // High CPU = stress (↓pleasure, ↑arousal, ↓dominance)
    // High memory = cognitive load (↓pleasure, ↑arousal)
    
    let cpu_stress = (cpu_usage / 100.0).min(1.0);
    let mem_stress = memory_pressure.min(1.0);
    
    // Combined stress factor
    let stress = (cpu_stress * 0.6 + mem_stress * 0.4).min(1.0);
    
    // PAD deltas (small, to be added to current state)
    let pleasure_delta = -0.05 * stress;      // Stress reduces pleasure
    let arousal_delta = 0.1 * stress;         // Stress increases arousal
    let dominance_delta = -0.03 * stress;     // Stress reduces sense of control
    
    Ok((pleasure_delta, arousal_delta, dominance_delta))
}

rustler::init!("Elixir.VivaBridge.Body");
