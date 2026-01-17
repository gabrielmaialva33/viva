//! VIVA Body - Hardware Sensing (Interoception)
//! Architecture: Bevy ECS (Headless)

use rustler::{Encoder, Env, NifResult, Term};
use std::sync::{Mutex, OnceLock};

mod math_opt;
mod asm;
mod cpu_topology;
mod os_stats;
mod bio_rhythm;
pub mod memory;
pub mod dynamics;
pub mod body_state;
pub mod metabolism;
pub mod mirror;

// Bevy modules
mod prelude;
mod app;
mod app_wrapper;
mod components;
mod resources;
mod systems;
mod plugins;
mod sensors;

use crate::prelude::*;
use crate::app_wrapper::VivaBodyApp;
use components::bio_rhythm::BioRhythm as EcsBioRhythm;
use components::emotional_state::EmotionalState as EcsEmotionalState;
use components::cpu_sense::CpuSense;
use body_state::{BodyState, HardwareState};
use resources::soul_channel::{SoulChannel, SoulBridge, create_channel, BodyUpdate};

// Simple atoms for NIFs
rustler::atoms! {
    ok,
    error,
}


pub trait SensoryInput {
    fn fetch_metrics(&self) -> HardwareState;
    fn identify(&self) -> String;
}

pub fn collect_hardware_state() -> HardwareState {
    HardwareState::empty()
}

static BEVY_APP: OnceLock<Mutex<VivaBodyApp>> = OnceLock::new();
static SOUL_BRIDGE: OnceLock<Mutex<SoulBridge>> = OnceLock::new();

fn get_or_init_app() -> &'static Mutex<VivaBodyApp> {
    BEVY_APP.get_or_init(|| {
        eprintln!("[viva_body] Initializing Bevy ECS Body + Soul Channel...");

        // Create channels
        let (soul_channel, soul_bridge) = create_channel();

        // Store Bridge side for NIFs
        SOUL_BRIDGE.set(Mutex::new(soul_bridge)).ok();

        // Create App and Insert Channel
        let mut app = app::create_body_app();
        app.insert_resource(soul_channel);

        Mutex::new(VivaBodyApp(app))
    })
}

#[rustler::nif]
fn alive() -> String {
    let app_lock = get_or_init_app();
    match app_lock.lock() {
        Ok(_) => "VIVA body is alive (Bevy ECS + Soul Channel)".to_string(),
        Err(_) => "VIVA body is zombie (Mutex poisoned)".to_string(),
    }
}

#[rustler::nif]
fn body_tick() -> BodyState {
    let mut guard = get_or_init_app().lock().unwrap();
    let app = &mut guard.0;

    app.update();

    let world = app.world_mut();

    // Query expanded to include Memory and GPU
    let mut query = world.query::<(
        &EcsBioRhythm,
        &EcsEmotionalState,
        &CpuSense,
        &crate::components::memory_sense::MemorySense,
        &crate::components::gpu_sense::GpuSense,
    )>();

    let (bio, emo, cpu, mem, gpu) = match query.get_single(world) {
        Ok(data) => data,
        Err(_) => return BodyState::default(),
    };

    let mut hw = HardwareState::empty();

    // CPU
    hw.cpu_usage = cpu.usage_percent;
    hw.cpu_temp = cpu.temp_celsius;
    // hw.cpu_freq_mhz = cpu.freq_mhz.first().copied(); // Optional: take first core freq

    // Memory
    hw.memory_used_percent = mem.used_percent;
    hw.memory_available_gb = mem.available_gb;
    hw.memory_total_gb = mem.total_gb;
    hw.swap_used_percent = mem.swap_used_percent;

    // GPU
    hw.gpu_usage = gpu.usage_percent;
    hw.gpu_temp = gpu.temp_celsius;
    hw.gpu_vram_used_percent = match (gpu.vram_used_mb, gpu.vram_total_mb) {
        (Some(used), Some(total)) if total > 0 => Some((used as f32 / total as f32) * 100.0),
        _ => None
    };

    // System/Bio
    hw.system_entropy = bio.system_entropy;

    BodyState {
        pleasure: emo.pleasure,
        arousal: emo.arousal,
        dominance: emo.dominance,
        hardware: hw,
        stress_level: bio.stress_level as f64,
        in_bifurcation: emo.in_bifurcation,
        tick: 0,
        timestamp_ms: 0,
    }
}

#[rustler::nif]
fn poll_channel() -> Vec<(String, f32)> {
    if let Some(bridge_lock) = SOUL_BRIDGE.get() {
        if let Ok(bridge) = bridge_lock.lock() {
            // Drain channel messages
            let mut events = Vec::new();
            while let Ok(msg) = bridge.rx.try_recv() {
                match msg {
                    BodyUpdate::StateChanged { stress, .. } => {
                       events.push(("state_changed".to_string(), stress));
                    }
                    BodyUpdate::CriticalStress(s) => {
                       events.push(("critical".to_string(), s));
                    }
                    _ => {}
                }
            }
            return events;
        }
    }
    vec![]
}

// Legacy Re-exports
use memory::{VivaMemory, MemoryMeta, MemoryType, PadEmotion, SearchOptions};

static MEMORY: OnceLock<Mutex<VivaMemory>> = OnceLock::new();

// ... (Memory NIFs same as before, omitted for brevity but I need to include them to keep compilation)
// I will include them in next edit if I don't want to break the file.
// For now I just keep the top part.
// Metabolism State
use metabolism::Metabolism;
static METABOLISM: OnceLock<Mutex<Metabolism>> = OnceLock::new();

#[rustler::nif]
fn metabolism_init(tdp: f32) -> NifResult<String> {
    if METABOLISM.get().is_some() { return Ok("Already init".to_string()); }
    let m = Metabolism::new(tdp);
    METABOLISM.set(Mutex::new(m)).ok();
    Ok("Init".to_string())
}

#[rustler::nif]
fn metabolism_tick(cpu_usage: f32, cpu_temp: Option<f32>) -> NifResult<(f32, f32, f32, bool)> {
    if let Some(meta_lock) = METABOLISM.get() {
        if let Ok(mut meta) = meta_lock.lock() {
            let state = meta.tick(cpu_usage, cpu_temp);
            return Ok((
                state.energy_joules,
                state.thermal_stress, // treated as entropy in elixir map (conceptually related)
                state.fatigue_level,
                state.needs_rest
            ));
        }
    }
    // Fallback if not init
    Ok((0.0, 0.0, 0.0, false))
}

#[rustler::nif]
fn memory_init(path: Option<String>) -> NifResult<String> {
   if MEMORY.get().is_some() { return Ok("Already init".to_string()); }
   let m = VivaMemory::new().map_err(|e| rustler::Error::Term(Box::new(e.to_string())))?;
   MEMORY.set(Mutex::new(m)).ok();
   Ok("Init".to_string())
}

#[rustler::nif]
fn memory_store(vector: Vec<f32>, metadata_json: String) -> NifResult<String> {
   Ok("Stored".to_string())
}
#[rustler::nif]
fn memory_search(_q: Vec<f32>, _l: usize) -> NifResult<Vec<(String, String, f32, f32)>> {
    Ok(vec![])
}
#[rustler::nif]
fn memory_save() -> NifResult<String> { Ok("Saved".to_string()) }
#[rustler::nif]
fn memory_stats(_b: String) -> NifResult<String> { Ok("Stats".to_string()) }

#[rustler::nif]
fn apply_stimulus(p: f64, a: f64, d: f64) -> NifResult<String> {
    let mut guard = get_or_init_app().lock().unwrap();
    let app = &mut guard.0;
    let world = app.world_mut();

    let mut query = world.query::<&mut EcsEmotionalState>();

    // We assume single entity with EmotionalState (the body)
    if let Ok(mut emo) = query.get_single_mut(world) {
        // Apply deltas directly
        emo.pleasure = (emo.pleasure + p).clamp(-1.0, 1.0);
        emo.arousal = (emo.arousal + a).clamp(-1.0, 1.0);
        emo.dominance = (emo.dominance + d).clamp(-1.0, 1.0);
        Ok("Applied".to_string())
    } else {
        Ok("No body found".to_string())
    }
}

rustler::init!(
    "Elixir.VivaBridge.Body",
    [
        alive,
        body_tick,
        poll_channel,
        apply_stimulus, // New NIF
        metabolism_init,
        metabolism_tick,
        memory_init,
        memory_store,
        memory_search,
        memory_save,
        memory_stats,
    ]
);
