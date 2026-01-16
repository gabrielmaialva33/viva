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
use rustler::{Encoder, Env, NifResult, Term};
use std::sync::LazyLock;
use std::sync::Mutex;
use std::time::{Duration, Instant};
use sysinfo::{Components, Disks, Networks, System};

mod math_opt; // Low-level mathematical optimizations
mod serial_sensor; // Serial/IoT/Arduino support
mod asm; // Inline Assembly (RDTSC/CPUID)
mod cpu_topology; // Cache/Vendor info
mod os_stats; // Kernel metrics (Context Switches)

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
    let pleasure_delta = -0.12 * adjusted_stress;

    // Arousal: Stress → activation (positive, up to a point)
    // Formula: δA = k_a × (2σ - σ²) - Yerkes-Dodson inverted U
    // Peak at σ=1, then decreases (exhaustion)
    let arousal_delta = 0.15 * (2.0 * adjusted_stress - adjusted_stress.powi(2));

    // Dominance: Stress → perda de controle (negativo)
    // Mais impactado por GPU (capacidade) e Load (overwhelm)
    let dominance_stress = load_stress * 0.4 + gpu_stress * 0.3 + mem_stress * 0.3;
    let dominance_delta = -0.09 * dominance_stress;

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

/// Helper: Executa nvidia-smi via CLI (Fallback para WSL/Sem NVML)
fn get_gpu_info_cli() -> (Option<f32>, Option<f32>, Option<f32>, Option<String>) {
    // Formato CSV: utilisation.gpu, memory.used, memory.total, temperature.gpu, name
    // NOTE: nvidia-smi might be in /usr/lib/wsl/lib/ on WSL2
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
// NIF Registration
// ============================================================================

rustler::init!("Elixir.VivaBridge.Body");
