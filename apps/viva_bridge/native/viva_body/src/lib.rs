//! VIVA Body - Hardware Sensing (Interoception)
//!
//! Cross-platform interoception: VIVA feels its hardware as body.
//!
//! ## Sensory Mapping
//!
//! | Hardware | Sensation | PAD Impact |
//! |----------|-----------|------------|
//! | High CPU | Cardiac stress | ↓P, ↑A, ↓D |
//! | High CPU temp | Fever | ↓P, ↑A |
//! | High RAM | Cognitive load | ↓P, ↑A |
//! | High GPU VRAM | Limited imagination | ↓P, ↓D |
//! | High Disk I/O | Slow digestion | ↓A |
//!
//! ## Theoretical Foundation
//!
//! - Interoception (Craig, 2002)
//! - Embodied Cognition (Varela et al., 1991)
//! - PAD Model (Mehrabian, 1996)
//! - Logistic Threshold Model (NOT Weber-Fechner - see sigmoid docs)
//! - Allostasis (Sterling, 2012)

use nvml_wrapper::Nvml;
use rustler::{Encoder, Env, NifResult, Resource, ResourceArc, Term};
use std::sync::LazyLock;
use std::sync::Mutex;
use std::time::{Duration, Instant};
use sysinfo::{Components, Disks, Networks, System};

mod math_opt; // Low-level mathematical optimizations
mod serial_sensor; // Serial/IoT/Arduino support
mod asm; // Inline Assembly (RDTSC/CPUID)
mod cpu_topology; // Cache/Vendor info
mod os_stats; // Kernel metrics (Context Switches)
mod bio_rhythm; // Temporal analysis (Entropy/Jitter)
mod memory; // Vector memory backends (usearch, SQLite)
pub mod dynamics; // Stochastic dynamics (O-U, Cusp catastrophe)
mod body_state; // Unified body state

// ============================================================================
// Pre-defined Atoms (rustler::atoms! macro for performance + panic safety)
// ============================================================================

rustler::atoms! {
    nil,
    // CPU
    cpu_usage,
    cpu_temp,
    cpu_count,
    // Memory
    memory_used_percent,
    memory_available_gb,
    memory_total_gb,
    swap_used_percent,
    // GPU
    gpu_usage,
    gpu_vram_used_percent,
    gpu_temp,
    gpu_name,
    // Disk
    disk_usage_percent,
    disk_read_bytes,
    disk_write_bytes,
    // Network
    net_rx_bytes,
    net_tx_bytes,
    // System
    uptime_seconds,
    process_count,
    load_avg_1m,
    load_avg_5m,
    load_avg_15m,
    // Low Level
    cpu_freq_mhz,
    l3_cache_kb,
    context_switches,
    interrupts,
    // Bio Rhythm
    system_entropy,
    os_jitter,
    // Dynamics (PAD)
    pleasure,
    arousal,
    dominance,
    equilibria,
    is_bifurcation,
}

// Cache duration for hardware metrics (reduces lock contention)
const CACHE_DURATION_MS: u64 = 500;

// ============================================================================
// Panic-Safe Lock Helper
// ============================================================================

/// Acquires a lock safely, recovering from poison if necessary.
///
/// Mutex poisoning occurs when a thread panics while holding the lock.
/// Instead of propagating the panic (which would crash the BEAM scheduler),
/// we recover the inner MutexGuard - data may be inconsistent, but the
/// system keeps running.
///
/// For NIFs, returning potentially stale data is better than crashing.
fn safe_lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|poisoned| {
        eprintln!("[viva_body] WARN: Mutex was poisoned, recovering...");
        poisoned.into_inner()
    })
}

// ============================================================================
// Global State (Thread-Safe)
// ============================================================================

static SYSTEM: LazyLock<Mutex<System>> = LazyLock::new(|| {
    let mut sys = System::new_all();
    sys.refresh_all();
    Mutex::new(sys)
});

static COMPONENTS: LazyLock<Mutex<Components>> =
    LazyLock::new(|| Mutex::new(Components::new_with_refreshed_list()));

static DISKS: LazyLock<Mutex<Disks>> =
    LazyLock::new(|| Mutex::new(Disks::new_with_refreshed_list()));

static NETWORKS: LazyLock<Mutex<Networks>> =
    LazyLock::new(|| Mutex::new(Networks::new_with_refreshed_list()));

static BIO_RHYTHM: LazyLock<Mutex<bio_rhythm::BioRhythm>> =
    LazyLock::new(|| Mutex::new(bio_rhythm::BioRhythm::new()));

// NVIDIA GPU (initialized once, None if unavailable)
static NVML: LazyLock<Option<Nvml>> = LazyLock::new(|| match Nvml::init() {
    Ok(nvml) => {
        eprintln!("[viva_body] NVML initialized - NVIDIA GPU detected");
        Some(nvml)
    }
    Err(e) => {
        eprintln!(
            "[viva_body] NVML unavailable: {:?} - GPU sensing disabled",
            e
        );
        None
    }
});

// Cache for hardware state (reduces lock contention under high load)
static HARDWARE_CACHE: LazyLock<Mutex<(Option<HardwareState>, Instant)>> =
    LazyLock::new(|| Mutex::new((None, Instant::now())));

// ============================================================================
// Hardware State Struct
// ============================================================================

// ============================================================================
// Universal Sensory Abstraction
// ============================================================================

/// Trait for any device that can feel/sense (Local PC, IoT, Server)
pub trait SensoryInput {
    /// Fetches current metrics normalized to HardwareState
    fn fetch_metrics(&self) -> HardwareState;

    /// Identifies the device (e.g., "LocalHost", "Arduino-X")
    fn identify(&self) -> String;
}

/// Driver: System Monitoring (sysinfo + nvml)
pub struct LocalSystem;

impl SensoryInput for LocalSystem {
    fn identify(&self) -> String {
        "Host System".to_string()
    }

    fn fetch_metrics(&self) -> HardwareState {
        collect_hardware_state()
    }
}

// ============================================================================
// Hardware State Struct (The "Body Schema")
// ============================================================================

use serde::{Deserialize, Serialize};

/// Estado completo do hardware - o "corpo" de VIVA
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareState {
    // CPU
    pub cpu_usage: f32,
    pub cpu_temp: Option<f32>, // °C - None se indisponível
    pub cpu_count: usize,

    // Memory
    pub memory_used_percent: f32,
    pub memory_available_gb: f32,
    pub memory_total_gb: f32,
    pub swap_used_percent: f32,

    // GPU (opcional)
    pub gpu_usage: Option<f32>, // % - None se sem GPU/driver
    pub gpu_vram_used_percent: Option<f32>,
    pub gpu_temp: Option<f32>,
    pub gpu_name: Option<String>,

    // Disk
    pub disk_usage_percent: f32,
    pub disk_read_bytes: u64,
    pub disk_write_bytes: u64,

    // Network
    pub net_rx_bytes: u64,
    pub net_tx_bytes: u64,

    // System
    pub uptime_seconds: u64,
    pub process_count: usize,
    pub load_avg_1m: f64,
    pub load_avg_5m: f64,
    pub load_avg_15m: f64,

    // Low-Level (New)
    pub cpu_freq_mhz: Option<f32>,
    pub l3_cache_kb: Option<u32>,
    pub context_switches: u64,
    pub interrupts: u64,

    // Bio-Rhythm (New)
    pub system_entropy: f32, // 0.0 (Order) - 1.0 (Chaos)
    pub os_jitter: f32,      // 0.0 (Stable) - 1.0 (Tremor)
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
        }
    }
}

impl Encoder for HardwareState {
    fn encode<'a>(&self, env: Env<'a>) -> Term<'a> {
        use rustler::types::map::map_new;

        // Helper: put value with pre-defined atom key (no unwrap, uses expect for map_put)
        fn put<'a, T: Encoder>(
            map: Term<'a>,
            key: rustler::Atom,
            val: T,
            env: Env<'a>,
        ) -> Term<'a> {
            map.map_put(key.encode(env), val.encode(env))
                .expect("map_put should not fail for valid terms")
        }

        // Helper: put optional f32 (nil if None)
        fn put_opt_f32<'a>(
            map: Term<'a>,
            key: rustler::Atom,
            val: Option<f32>,
            env: Env<'a>,
        ) -> Term<'a> {
            match val {
                Some(v) => put(map, key, v, env),
                None => put(map, key, nil(), env),
            }
        }

        // Helper: put optional String (nil if None)
        fn put_opt_str<'a>(
            map: Term<'a>,
            key: rustler::Atom,
            val: &Option<String>,
            env: Env<'a>,
        ) -> Term<'a> {
            match val {
                Some(v) => put(map, key, v.as_str(), env),
                None => put(map, key, nil(), env),
            }
        }

        let map = map_new(env);

        // CPU (using pre-defined atoms from rustler::atoms! macro)
        let map = put(map, cpu_usage(), self.cpu_usage, env);
        let map = put_opt_f32(map, cpu_temp(), self.cpu_temp, env);
        let map = put(map, cpu_count(), self.cpu_count, env);

        // Memory
        let map = put(map, memory_used_percent(), self.memory_used_percent, env);
        let map = put(map, memory_available_gb(), self.memory_available_gb, env);
        let map = put(map, memory_total_gb(), self.memory_total_gb, env);
        let map = put(map, swap_used_percent(), self.swap_used_percent, env);

        // GPU
        let map = put_opt_f32(map, gpu_usage(), self.gpu_usage, env);
        let map = put_opt_f32(
            map,
            gpu_vram_used_percent(),
            self.gpu_vram_used_percent,
            env,
        );
        let map = put_opt_f32(map, gpu_temp(), self.gpu_temp, env);
        let map = put_opt_str(map, gpu_name(), &self.gpu_name, env);

        // Disk
        let map = put(map, disk_usage_percent(), self.disk_usage_percent, env);
        let map = put(map, disk_read_bytes(), self.disk_read_bytes, env);
        let map = put(map, disk_write_bytes(), self.disk_write_bytes, env);

        // Network
        let map = put(map, net_rx_bytes(), self.net_rx_bytes, env);
        let map = put(map, net_tx_bytes(), self.net_tx_bytes, env);

        // System
        let map = put(map, uptime_seconds(), self.uptime_seconds, env);
        let map = put(map, process_count(), self.process_count, env);
        // Helper to put u32 as Optional (nil if None)
        fn put_opt_u32<'a>(
             map: Term<'a>,
             key: rustler::Atom,
             val: Option<u32>,
             env: Env<'a>,
         ) -> Term<'a> {
             match val {
                 Some(v) => put(map, key, v, env),
                 None => put(map, key, nil(), env),
             }
         }

        let map = put(map, load_avg_1m(), self.load_avg_1m, env);
        let map = put(map, load_avg_5m(), self.load_avg_5m, env);
        let map = put(map, load_avg_15m(), self.load_avg_15m, env);

        // Low Level
        let map = put_opt_f32(map, cpu_freq_mhz(), self.cpu_freq_mhz, env);
        let map = put_opt_u32(map, l3_cache_kb(), self.l3_cache_kb, env);
        let map = put(map, context_switches(), self.context_switches, env);
        let map = put(map, interrupts(), self.interrupts, env);

        // Bio Rhythm
        let map = put(map, system_entropy(), self.system_entropy, env);
        let map = put(map, os_jitter(), self.os_jitter, env);

        map
    }
}

// ============================================================================
// NIF Functions
// ============================================================================

/// Checks if VIVA's body is alive
#[rustler::nif]
fn alive() -> &'static str {
    "VIVA body is alive"
}

/// Collects hardware state (internal function, with 500ms cache)
fn collect_hardware_state() -> HardwareState {
    // Check cache first (using safe_lock to avoid panic on poison)
    {
        let cache = safe_lock(&HARDWARE_CACHE);
        if let (Some(ref cached_state), last_update) = &*cache {
            if last_update.elapsed() < Duration::from_millis(CACHE_DURATION_MS) {
                return cached_state.clone();
            }
        }
    }

    // Cache miss or expired - refresh all data sources
    // Using safe_lock to recover from poison instead of crashing
    let mut sys = safe_lock(&SYSTEM);
    sys.refresh_all();

    let mut components = safe_lock(&COMPONENTS);
    components.refresh();

    let mut disks = safe_lock(&DISKS);
    disks.refresh();

    let mut networks = safe_lock(&NETWORKS);
    networks.refresh();

    // CPU
    let cpu_usage = sys.global_cpu_usage();
    let cpu_count = sys.cpus().len();
    let cpu_temp = get_cpu_temp(&components);

    // Memory
    let total_memory = sys.total_memory() as f64;
    let used_memory = sys.used_memory() as f64;
    let available_memory = sys.available_memory() as f64;
    let total_swap = sys.total_swap() as f64;
    let used_swap = sys.used_swap() as f64;

    let memory_used_percent = if total_memory > 0.0 {
        ((used_memory / total_memory) * 100.0) as f32
    } else {
        0.0
    };

    let swap_used_percent = if total_swap > 0.0 {
        ((used_swap / total_swap) * 100.0) as f32
    } else {
        0.0
    };

    // GPU (tenta NVML se disponível)
    let (gpu_usage, gpu_vram, gpu_temp, gpu_name) = get_gpu_info();

    // Disk
    let (disk_usage, disk_read, disk_write) = get_disk_info(&disks);

    // Network
    let (net_rx, net_tx) = get_network_info(&networks);

    // Load average (Unix-like)
    let load = System::load_average();

    // Low-Level Metrics
    let cache_info = cpu_topology::detect_cache_topology();
    let os_stats = os_stats::read_os_stats(&sys);

    // Update Bio-Rhythm
    // We update the buffer with CURRENT values, then calculate based on history.
    let mut rhythm = safe_lock(&BIO_RHYTHM);
    // Explicit deref or let generic flow, but compiler complained about type inference.
    // Actually safe_lock returns MutexGuard<'_, BioRhythm>.
    // The previous error was likely due to BIO_RHYTHM not being found or resolved.
    // But let's be safe.
    rhythm.update(cpu_usage, os_stats.context_switches);
    let system_entropy = rhythm.cpu_entropy();
    let os_jitter = rhythm.context_switch_jitter();

    let state = HardwareState {
        cpu_usage,
        cpu_temp,
        cpu_count,
        memory_used_percent,
        memory_available_gb: (available_memory / 1_073_741_824.0) as f32,
        memory_total_gb: (total_memory / 1_073_741_824.0) as f32,
        swap_used_percent,
        gpu_usage,
        gpu_vram_used_percent: gpu_vram,
        gpu_temp,
        gpu_name,
        disk_usage_percent: disk_usage,
        disk_read_bytes: disk_read,
        disk_write_bytes: disk_write,
        net_rx_bytes: net_rx,
        net_tx_bytes: net_tx,
        uptime_seconds: System::uptime(),
        process_count: sys.processes().len(),
        load_avg_1m: load.one,
        load_avg_5m: load.five,
        load_avg_15m: load.fifteen,
        // New fields
        cpu_freq_mhz: os_stats.cpu_freq_mhz,
        l3_cache_kb: if cache_info.l3_kb > 0 { Some(cache_info.l3_kb) } else { None },
        context_switches: os_stats.context_switches,
        interrupts: os_stats.interrupts,
        system_entropy,
        os_jitter,
    };

    // Update cache (using safe_lock)
    {
        let mut cache = safe_lock(&HARDWARE_CACHE);
        *cache = (Some(state.clone()), Instant::now());
    }

    state
}

/// Feels current hardware state (complete interoception)
#[rustler::nif]
fn feel_hardware() -> NifResult<HardwareState> {
    Ok(collect_hardware_state())
}

/// Returns current CPU cycle count (RDTSC)
/// Direct hardware timestamp, useful for micro-benchmarking sensation latency.
#[rustler::nif]
fn get_cycles() -> u64 {
    unsafe { asm::rdtsc() }
}

/// Converts hardware metrics to qualia (PAD deltas)
///
/// Returns (pleasure_delta, arousal_delta, dominance_delta)
///
/// Base teórica:
/// - Interocepção (Craig, 2002)
/// - PAD (Mehrabian, 1996)
/// - Sigmoid thresholds (Logistic Threshold Model)
/// - Allostasis (Sterling, 2012)
#[rustler::nif]
fn hardware_to_qualia() -> NifResult<(f64, f64, f64)> {
    let hw = collect_hardware_state();

    // ========================================================================
    // SIGMOID THRESHOLDS (Logistic Threshold Model)
    // ========================================================================
    // Biological systems do NOT respond linearly.
    // Sigmoid simulates threshold: "feel nothing until 80%, then FEEL A LOT"
    //
    // k = steepness (10.0 = abrupt transition)
    // x0 = threshold (0.8 = 80% starts to "hurt")

    // CPU: threshold 80%, k=12 (abrupto)
    let cpu_raw = (hw.cpu_usage as f64 / 100.0).clamp(0.0, 1.0);
    let cpu_stress = normalized_sigmoid(cpu_raw, 12.0, 0.80);

    // Memory: threshold 75%, k=10 (moderado)
    let mem_raw = (hw.memory_used_percent as f64 / 100.0).clamp(0.0, 1.0);
    let mem_stress = normalized_sigmoid(mem_raw, 10.0, 0.75);

    // Swap: qualquer uso de swap é ruim - threshold 20%, k=15
    let swap_raw = (hw.swap_used_percent as f64 / 100.0).clamp(0.0, 1.0);
    let swap_stress = normalized_sigmoid(swap_raw, 15.0, 0.20);

    // Temperatura: threshold 70°C, k=8 (gradual)
    let temp_stress = match hw.cpu_temp {
        Some(t) => {
            let temp_raw = ((t as f64 - 40.0) / 50.0).clamp(0.0, 1.0);
            normalized_sigmoid(temp_raw, 8.0, 0.60) // 70°C = 0.6 no range 40-90
        }
        None => 0.0,
    };

    // GPU VRAM: threshold 85%, k=10
    let gpu_stress = match hw.gpu_vram_used_percent {
        Some(v) => {
            let gpu_raw = (v as f64 / 100.0).clamp(0.0, 1.0);
            normalized_sigmoid(gpu_raw, 10.0, 0.85)
        }
        None => 0.0,
    };

    // Load average: threshold = 0.8 per core, k=10
    let cores = hw.cpu_count.max(1) as f64;
    let load_raw = (hw.load_avg_1m / cores).clamp(0.0, 1.5) / 1.5; // Normalize para [0,1]
    let load_stress = normalized_sigmoid(load_raw, 10.0, 0.53); // 0.8/1.5 ≈ 0.53

    // Disk: threshold 90%, k=12
    let disk_raw = (hw.disk_usage_percent as f64 / 100.0).clamp(0.0, 1.0);
    let disk_stress = normalized_sigmoid(disk_raw, 12.0, 0.90);

    // ========================================================================
    // ALLOSTASIS (Sterling, 2012) - Anticipatory Regulation
    // ========================================================================
    // VIVA doesn't just react to current stress, but ANTICIPATES based on trend
    // load_avg_1m vs load_avg_5m = short-term trend

    let allostasis = allostasis_delta(hw.load_avg_1m, hw.load_avg_5m);

    // ========================================================================
    // Mapeamento Hardware → PAD (baseado em literatura de interocepção)
    // ========================================================================
    //
    // Pesos calibrados empiricamente:
    // - CPU/Load: 30% do impacto (stress cardíaco)
    // - Memory: 25% (carga cognitiva)
    // - Temperature: 20% (febre/desconforto)
    // - GPU: 15% (capacidade imaginativa)
    // - Disk/Swap: 10% (digestão/armazenamento)

    let composite_stress = cpu_stress * 0.15
        + load_stress * 0.15
        + mem_stress * 0.20
        + swap_stress * 0.05
        + temp_stress * 0.20
        + gpu_stress * 0.15
        + disk_stress * 0.10;

    // Allostasis ajusta composite: antecipação de stress
    // +10% se load subindo, -10% se load caindo
    let allostatic_adjustment = 1.0 + (allostasis * 0.10);
    let adjusted_stress = (composite_stress * allostatic_adjustment).clamp(0.0, 1.0);

    // ========================================================================
    // PAD Deltas (Yerkes-Dodson + Interocepção)
    // ========================================================================

    // Pleasure: Stress → desconforto (negativo)
    // Formula: δP = -k_p × σ
    let mut pleasure_delta = -0.12 * adjusted_stress;

    // Arousal: Stress → activation (positive, up to a point)
    // Formula: δA = k_a × (2σ - σ²) - Yerkes-Dodson inverted U
    // Peak at σ=1, then decreases (exhaustion)
    let mut arousal_delta = 0.15 * (2.0 * adjusted_stress - adjusted_stress.powi(2));

    // Dominance: Stress → perda de controle (negativo)
    // Mais impactado por GPU (capacidade) e Load (overwhelm)
    let dominance_stress = load_stress * 0.4 + gpu_stress * 0.3 + mem_stress * 0.3;
    let mut dominance_delta = -0.09 * dominance_stress;

    // ========================================================================
    // BIO-RHYTHM IMPACT (New - The "Supreme" Layer)
    // ========================================================================
    //
    // 1. System Entropy (Chaos vs Order):
    //    High entropy (chaotic CPU usage) -> Cognitive dissonance (Low Dominance)
    //    Low entropy (steady CPU usage) -> Flow state (High Dominance)
    dominance_delta -= hw.system_entropy as f64 * 0.15; // Chaos erodes control

    // 2. OS Jitter (Tremor):
    //    High jitter -> Anxiety/Fear (High Arousal, Low Pleasure)
    let jitter_impact = hw.os_jitter as f64;
    arousal_delta += jitter_impact * 0.20; // Fear activates
    pleasure_delta -= jitter_impact * 0.15; // Uncertainty is unpleasant

    Ok((pleasure_delta, arousal_delta, dominance_delta))
}

// ============================================================================
// Mathematical Functions (Biologically-Inspired)
// ============================================================================

/// Applies sigmoid with normalization to maintain range [0, 1]
/// Compensates for the fact that sigmoid(0) != 0 and sigmoid(1) != 1
///
/// Uses optimized SIMD/AVX2 implementation via `math_opt` module.
#[inline]
fn normalized_sigmoid(x: f64, k: f64, x0: f64) -> f64 {
    let x_f32 = x as f32;
    let k_f32 = k as f32;
    let x0_f32 = x0 as f32;

    let min_val = math_opt::sigmoid_optimized(0.0, k_f32, x0_f32) as f64;
    let max_val = math_opt::sigmoid_optimized(1.0, k_f32, x0_f32) as f64;
    let raw = math_opt::sigmoid_optimized(x_f32, k_f32, x0_f32) as f64;

    (raw - min_val) / (max_val - min_val)
}

/// Allostasis: Anticipatory stress response
///
/// Theoretical basis: Sterling (2012) - "Allostasis: A model of predictive regulation"
/// Body anticipates demands based on trend, not just current state
///
/// Parameters:
/// - current: current value (e.g., load_avg_1m)
/// - baseline: comparison baseline (e.g., load_avg_5m)
///
/// Returns anticipation delta [-1, 1]:
/// - Positive: load rising → VIVA anticipates stress
/// - Negative: load falling → VIVA relaxes in advance
#[inline]
fn allostasis_delta(current: f64, baseline: f64) -> f64 {
    // Delta normalizado: (current - baseline) / baseline
    // Clamped para evitar explosão em valores baixos
    if baseline < 0.1 {
        return 0.0;
    }
    ((current - baseline) / baseline).clamp(-1.0, 1.0)
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Gets CPU temperature (cross-platform)
fn get_cpu_temp(components: &Components) -> Option<f32> {
    // Procura por componentes de CPU em ordem de preferência
    let cpu_labels = [
        "coretemp", // Intel Linux
        "k10temp",  // AMD Linux
        "cpu",      // Genérico
        "CPU",      // Windows
        "Core",     // Intel per-core
        "Tctl",     // AMD Ryzen
        "Package",  // Intel package
    ];

    for component in components.iter() {
        let label = component.label().to_lowercase();
        for cpu_label in &cpu_labels {
            if label.contains(&cpu_label.to_lowercase()) {
                let temp = component.temperature();
                if temp > 0.0 && temp < 150.0 {
                    // Sanity check
                    return Some(temp);
                }
            }
        }
    }

    // Fallback: primeira temperatura válida
    for component in components.iter() {
        let temp = component.temperature();
        if temp > 0.0 && temp < 150.0 {
            return Some(temp);
        }
    }

    None
}

/// Reads GPU info from cache file (~/.cache/viva/gpu.csv)
/// Format: utilization,mem_used,mem_total,temp,name
/// Updated by external daemon to avoid fork() issues in BEAM
fn read_gpu_cache() -> Option<(Option<f32>, Option<f32>, Option<f32>, Option<String>)> {
    let home = std::env::var("HOME").ok()?;
    let cache_path = format!("{}/.cache/viva/gpu.csv", home);

    let content = std::fs::read_to_string(&cache_path).ok()?;
    let line = content.lines().next()?;
    let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();

    if parts.len() >= 5 {
        let usage = parts[0].parse::<f32>().ok();
        let mem_used = parts[1].parse::<f64>().unwrap_or(0.0);
        let mem_total = parts[2].parse::<f64>().unwrap_or(1.0);
        let vram_percent = if mem_total > 0.0 {
            Some((mem_used / mem_total * 100.0) as f32)
        } else {
            None
        };
        let temp = parts[3].parse::<f32>().ok();
        let name = Some(parts[4].to_string());

        return Some((usage, vram_percent, temp, name));
    }

    None
}

/// Helper: Executes nvidia-smi via CLI (Fallback when NVML unavailable)
fn get_gpu_info_cli() -> (Option<f32>, Option<f32>, Option<f32>, Option<String>) {
    // First, try reading from cache file (updated by external daemon)
    // This avoids fork() issues inside BEAM VM
    if let Some(result) = read_gpu_cache() {
        return result;
    }

    // Fallback: Direct command (works on native Linux/Windows, not WSL)
    let output = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,name",
            "--format=csv,noheader,nounits",
        ])
        .output();

    match output {
        Ok(o) if o.status.success() => {
             // eprintln!("[viva_body] DEBUG: nvidia-smi CLI success"); // Verbose
            let stdout = String::from_utf8_lossy(&o.stdout);
            let line = stdout.lines().next().unwrap_or("");
            let parts: Vec<&str> = line.split(',').collect();

            if parts.len() >= 5 {
                // Parse com tratamento de erro (unwrap_or default)
                let usage = parts[0].trim().parse::<f32>().ok();

                let mem_used = parts[1].trim().parse::<f64>().unwrap_or(0.0);
                let mem_total = parts[2].trim().parse::<f64>().unwrap_or(1.0); // evita div/0

                let vram_percent = if mem_total > 0.0 {
                    Some((mem_used / mem_total * 100.0) as f32)
                } else {
                    None
                };

                let temp = parts[3].trim().parse::<f32>().ok();
                let name = Some(parts[4].trim().to_string());

                return (usage, vram_percent, temp, name);
            }
        }
        _ => {
            if let Err(e) = &output {
               eprintln!("[viva_body] WARN: nvidia-smi CLI failed to execute: {:?}", e);
            } else if let Ok(o) = &output {
               if !o.status.success() {
                    let stderr = String::from_utf8_lossy(&o.stderr);
                    eprintln!("[viva_body] WARN: nvidia-smi CLI returned error: {}", stderr);
               }
            }
        } // Falha silenciosa no fallback
    }

    (None, None, None, None)
}

/// Gets GPU info via NVML or Fallback CLI
/// Returns (usage%, vram%, temp, name)
fn get_gpu_info() -> (Option<f32>, Option<f32>, Option<f32>, Option<String>) {
    use nvml_wrapper::enum_wrappers::device::TemperatureSensor;

    // 1. Tenta NVML (Performance preferida)
    if let Some(nvml) = NVML.as_ref() {
        if let Ok(device) = nvml.device_by_index(0) {
            // Utilização (GPU compute %)
            let usage = device.utilization_rates().ok().map(|u| u.gpu as f32);

            // VRAM
            let vram = device
                .memory_info()
                .ok()
                .map(|m| (m.used as f64 / m.total as f64 * 100.0) as f32);

            // Temperatura
            let temp = device
                .temperature(TemperatureSensor::Gpu)
                .ok()
                .map(|t| t as f32);

            // Nome
            let name = device.name().ok();

            return (usage, vram, temp, name);
        }
    }

    // 2. Fallback: CLI (WSL/Drivers sem lib linkada)
    get_gpu_info_cli()
}

/// Gets disk info
/// Returns (usage%, read_bytes, write_bytes)
fn get_disk_info(disks: &Disks) -> (f32, u64, u64) {
    let mut total_space = 0u64;
    let mut total_used = 0u64;

    for disk in disks.iter() {
        total_space += disk.total_space();
        total_used += disk.total_space() - disk.available_space();
    }

    let usage = if total_space > 0 {
        (total_used as f64 / total_space as f64 * 100.0) as f32
    } else {
        0.0
    };

    // sysinfo doesn't provide I/O rates directly, returning 0
    // Could use /proc/diskstats on Linux
    (usage, 0, 0)
}

/// Gets network info
/// Returns (rx_bytes, tx_bytes)
fn get_network_info(networks: &Networks) -> (u64, u64) {
    let mut rx = 0u64;
    let mut tx = 0u64;

    for (_, data) in networks.iter() {
        rx += data.total_received();
        tx += data.total_transmitted();
    }

    (rx, tx)
}

// ============================================================================
// Memory System NIFs
// ============================================================================

#[allow(unused_imports)]
use memory::{MemoryBackend, MemoryMeta, MemorySearchResult, MemoryType, SearchOptions};

/// Wrapper for MemoryBackend as Rustler Resource
pub struct MemoryResource {
    backend: Mutex<MemoryBackend>,
}

impl Resource for MemoryResource {}

/// Create HNSW-based memory backend (fast ANN search)
// TODO: Enable when VivaBridge.Body declares memory functions
// #[rustler::nif]
#[allow(dead_code)]
fn memory_new_hnsw() -> NifResult<ResourceArc<MemoryResource>> {
    let backend = MemoryBackend::usearch()
        .map_err(|e| rustler::Error::Term(Box::new(format!("Failed to create HNSW backend: {}", e))))?;
    Ok(ResourceArc::new(MemoryResource { backend: Mutex::new(backend) }))
}

/// Create SQLite-based memory backend (portable, brute-force)
// #[rustler::nif]
#[allow(dead_code)]
fn memory_new_sqlite() -> NifResult<ResourceArc<MemoryResource>> {
    let backend = MemoryBackend::sqlite()
        .map_err(|e| rustler::Error::Term(Box::new(format!("Failed to create SQLite backend: {}", e))))?;
    Ok(ResourceArc::new(MemoryResource { backend: Mutex::new(backend) }))
}

/// Store a memory with embedding
/// Returns the internal key (u64)
// #[rustler::nif]
#[allow(dead_code)]
fn memory_store(
    resource: ResourceArc<MemoryResource>,
    embedding: Vec<f32>,
    id: String,
    content: String,
    memory_type: String,
    importance: f32,
) -> NifResult<u64> {
    let backend = resource.backend.lock().unwrap();

    let meta = MemoryMeta::new(id, content)
        .with_type(MemoryType::from_str(&memory_type))
        .with_importance(importance);

    backend.store(&embedding, meta)
        .map_err(|e| rustler::Error::Term(Box::new(e.to_string())))
}

/// Search for similar memories
/// Returns list of (id, content, similarity, decayed_score)
// #[rustler::nif]
#[allow(dead_code)]
fn memory_search(
    resource: ResourceArc<MemoryResource>,
    query: Vec<f32>,
    limit: usize,
    memory_type: Option<String>,
    apply_decay: bool,
) -> NifResult<Vec<(String, String, f32, f32)>> {
    let backend = resource.backend.lock().unwrap();

    let mut options = SearchOptions::new().limit(limit);

    if let Some(ref t) = memory_type {
        options = options.of_type(MemoryType::from_str(t));
    }

    if !apply_decay {
        options = options.no_decay();
    }

    let results = backend.search(&query, &options)
        .map_err(|e| rustler::Error::Term(Box::new(e.to_string())))?;

    Ok(results
        .into_iter()
        .map(|r| (r.meta.id, r.meta.content, r.similarity, r.decayed_score))
        .collect())
}

/// Get backend name
// #[rustler::nif]
#[allow(dead_code)]
fn memory_backend_name(resource: ResourceArc<MemoryResource>) -> String {
    let backend = resource.backend.lock().unwrap();
    backend.backend_name().to_string()
}

// ============================================================================
// Dynamics NIFs (Stochastic Emotional Dynamics)
// ============================================================================

/// Default O-U parameters for PAD model (empirically calibrated)
/// Based on emotional dynamics literature (Kuppens et al., 2010)
fn default_ou_params() -> [dynamics::OUParams; 3] {
    [
        // Pleasure: slower reversion, moderate volatility
        dynamics::OUParams { theta: 0.3, mu: 0.0, sigma: 0.15 },
        // Arousal: faster reversion, higher volatility
        dynamics::OUParams { theta: 0.5, mu: 0.0, sigma: 0.25 },
        // Dominance: slowest reversion, low volatility
        dynamics::OUParams { theta: 0.2, mu: 0.0, sigma: 0.10 },
    ]
}

/// Single O-U step for PAD state
///
/// Input: (p, a, d) current state, dt timestep, (n1, n2, n3) gaussian noises
/// Output: (p', a', d') new state (bounded to [-1, 1])
#[rustler::nif]
fn dynamics_ou_step(
    p: f64, a: f64, d: f64,
    dt: f64,
    noise_p: f64, noise_a: f64, noise_d: f64,
) -> (f64, f64, f64) {
    let params = default_ou_params();
    let mut pad = [p, a, d];
    let noises = [noise_p, noise_a, noise_d];

    dynamics::ou_step_pad_bounded(&mut pad, &params, dt, &noises);

    (pad[0], pad[1], pad[2])
}

/// Cusp catastrophe equilibria finder
///
/// Returns list of equilibrium points for given control parameters (c, y)
/// - c: "splitting factor" (bifurcation control)
/// - y: "normal factor" (asymmetry)
#[rustler::nif]
fn dynamics_cusp_equilibria(c: f64, y: f64) -> Vec<f64> {
    dynamics::cusp_equilibria(c, y)
}

/// Check if (c, y) is in bifurcation region
///
/// Bifurcation region: 27y² < 4c³ (cusp catastrophe surface)
#[rustler::nif]
fn dynamics_cusp_is_bifurcation(c: f64, y: f64) -> bool {
    dynamics::cusp_is_bifurcation(c, y)
}

/// Cusp-modulated mood step
///
/// Applies cusp catastrophe dynamics to pleasure (mood) based on arousal
/// This creates sudden mood transitions at bifurcation boundaries
///
/// Input: mood, arousal, external_bias, dt
/// Output: new_mood (arousal unchanged)
#[rustler::nif]
fn dynamics_cusp_mood_step(
    mood: f64, arousal: f64,
    external_bias: f64, dt: f64,
) -> f64 {
    let mut new_mood = mood;
    dynamics::cusp_mood_step(&mut new_mood, arousal, external_bias, dt);
    new_mood
}

/// Full DynAffect step (O-U + optional Cusp)
///
/// Combines mean-reverting stochastic dynamics with catastrophe theory
/// for realistic emotional transitions including sudden mood shifts.
///
/// Input: (p, a, d), dt, noises, cusp_enabled, cusp_sensitivity, external_bias
/// Output: (p', a', d') bounded to [-1, 1]
#[rustler::nif]
fn dynamics_step(
    p: f64, a: f64, d: f64,
    dt: f64,
    noise_p: f64, noise_a: f64, noise_d: f64,
    cusp_enabled: bool,
    cusp_sensitivity: f64,
    external_bias: f64,
) -> (f64, f64, f64) {
    let dynaffect = dynamics::DynAffect {
        ou_params: default_ou_params(),
        cusp_enabled,
        cusp_sensitivity,
    };

    let mut pad = [p, a, d];
    let noises = [noise_p, noise_a, noise_d];
    dynaffect.step(&mut pad, dt, &noises, external_bias);

    (pad[0], pad[1], pad[2])
}

// ============================================================================
// Body State NIFs (Unified Interoception)
// ============================================================================

use body_state::{BodyConfig, BodyEngine, BodyState};

/// Resource wrapper for BodyEngine (stateful)
pub struct BodyEngineResource {
    engine: Mutex<BodyEngine>,
}

impl Resource for BodyEngineResource {}

/// Create a new body engine with default config
#[rustler::nif]
fn body_engine_new() -> ResourceArc<BodyEngineResource> {
    ResourceArc::new(BodyEngineResource {
        engine: Mutex::new(BodyEngine::default()),
    })
}

/// Create body engine with custom config
#[rustler::nif]
fn body_engine_new_with_config(
    dt: f64,
    cusp_enabled: bool,
    cusp_sensitivity: f64,
    seed: u64,
) -> ResourceArc<BodyEngineResource> {
    let config = BodyConfig {
        dt,
        cusp_enabled,
        cusp_sensitivity,
        seed,
        ..Default::default()
    };
    ResourceArc::new(BodyEngineResource {
        engine: Mutex::new(BodyEngine::new(config)),
    })
}

/// Execute one body tick - the main integration function
///
/// Returns complete BodyState as a map with all metrics
#[rustler::nif]
fn body_engine_tick(resource: ResourceArc<BodyEngineResource>) -> NifResult<BodyState> {
    let mut engine = safe_lock(&resource.engine);
    Ok(engine.tick())
}

/// Get current PAD state without ticking
#[rustler::nif]
fn body_engine_get_pad(resource: ResourceArc<BodyEngineResource>) -> (f64, f64, f64) {
    let engine = safe_lock(&resource.engine);
    engine.pad()
}

/// Set PAD state directly
#[rustler::nif]
fn body_engine_set_pad(resource: ResourceArc<BodyEngineResource>, p: f64, a: f64, d: f64) {
    let mut engine = safe_lock(&resource.engine);
    engine.set_pad(p, a, d);
}

/// Apply external stimulus (e.g., from conversation, events)
#[rustler::nif]
fn body_engine_apply_stimulus(
    resource: ResourceArc<BodyEngineResource>,
    p_delta: f64,
    a_delta: f64,
    d_delta: f64,
) {
    let mut engine = safe_lock(&resource.engine);
    engine.apply_stimulus(p_delta, a_delta, d_delta);
}

// Implement Encoder for BodyState (Elixir map)
impl Encoder for BodyState {
    fn encode<'a>(&self, env: Env<'a>) -> Term<'a> {
        use rustler::types::map::map_new;

        let map = map_new(env);

        // PAD state
        let map = map
            .map_put(pleasure().encode(env), self.pleasure.encode(env))
            .expect("map_put");
        let map = map
            .map_put(arousal().encode(env), self.arousal.encode(env))
            .expect("map_put");
        let map = map
            .map_put(dominance().encode(env), self.dominance.encode(env))
            .expect("map_put");

        // Derived metrics
        let map = map
            .map_put(
                rustler::types::atom::Atom::from_str(env, "stress_level").unwrap().encode(env),
                self.stress_level.encode(env),
            )
            .expect("map_put");
        let map = map
            .map_put(
                rustler::types::atom::Atom::from_str(env, "in_bifurcation").unwrap().encode(env),
                self.in_bifurcation.encode(env),
            )
            .expect("map_put");

        // Metadata
        let map = map
            .map_put(
                rustler::types::atom::Atom::from_str(env, "tick").unwrap().encode(env),
                self.tick.encode(env),
            )
            .expect("map_put");
        let map = map
            .map_put(
                rustler::types::atom::Atom::from_str(env, "timestamp_ms").unwrap().encode(env),
                self.timestamp_ms.encode(env),
            )
            .expect("map_put");

        // Embed hardware state
        let map = map
            .map_put(
                rustler::types::atom::Atom::from_str(env, "hardware").unwrap().encode(env),
                self.hardware.encode(env),
            )
            .expect("map_put");

        map
    }
}

// ============================================================================
// NIF Registration
// ============================================================================

rustler::init!(
    "Elixir.VivaBridge.Body",
    [
        // Core sensing
        alive,
        feel_hardware,
        get_cycles,
        hardware_to_qualia,
        // Dynamics (emotional state evolution)
        dynamics_ou_step,
        dynamics_cusp_equilibria,
        dynamics_cusp_is_bifurcation,
        dynamics_cusp_mood_step,
        dynamics_step,
        // Body State (unified interoception)
        body_engine_new,
        body_engine_new_with_config,
        body_engine_tick,
        body_engine_get_pad,
        body_engine_set_pad,
        body_engine_apply_stimulus,
    ]
);
