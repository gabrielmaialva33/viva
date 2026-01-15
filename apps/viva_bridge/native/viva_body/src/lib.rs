//! VIVA Body - Hardware Sensing (Interoception)
//!
//! Interocepção multiplataforma: VIVA sente seu hardware como corpo.
//!
//! ## Mapeamento Sensório
//!
//! | Hardware | Sensação | Impacto PAD |
//! |----------|----------|-------------|
//! | CPU alto | Stress cardíaco | ↓P, ↑A, ↓D |
//! | CPU temp alta | Febre | ↓P, ↑A |
//! | RAM alta | Carga cognitiva | ↓P, ↑A |
//! | GPU VRAM alta | Imaginação limitada | ↓P, ↓D |
//! | Disk I/O alto | Digestão lenta | ↓A |
//!
//! ## Base Teórica
//! - Interocepção (Craig, 2002)
//! - Embodied Cognition (Varela et al., 1991)
//! - PAD Model (Mehrabian, 1996)

use rustler::{Encoder, Env, NifResult, Term};
use sysinfo::{System, Components, Disks, Networks};
use std::sync::LazyLock;
use std::sync::Mutex;
use std::time::{Duration, Instant};
use nvml_wrapper::Nvml;

// Cache duration for hardware metrics (reduces lock contention)
const CACHE_DURATION_MS: u64 = 500;

// ============================================================================
// Global State (Thread-Safe)
// ============================================================================

static SYSTEM: LazyLock<Mutex<System>> = LazyLock::new(|| {
    let mut sys = System::new_all();
    sys.refresh_all();
    Mutex::new(sys)
});

static COMPONENTS: LazyLock<Mutex<Components>> = LazyLock::new(|| {
    Mutex::new(Components::new_with_refreshed_list())
});

static DISKS: LazyLock<Mutex<Disks>> = LazyLock::new(|| {
    Mutex::new(Disks::new_with_refreshed_list())
});

static NETWORKS: LazyLock<Mutex<Networks>> = LazyLock::new(|| {
    Mutex::new(Networks::new_with_refreshed_list())
});

// NVIDIA GPU (inicializa uma vez, None se não disponível)
static NVML: LazyLock<Option<Nvml>> = LazyLock::new(|| {
    match Nvml::init() {
        Ok(nvml) => {
            eprintln!("[viva_body] NVML inicializado - GPU NVIDIA detectada");
            Some(nvml)
        }
        Err(e) => {
            eprintln!("[viva_body] NVML não disponível: {:?} - GPU sensing desabilitado", e);
            None
        }
    }
});

// Cache for hardware state (reduces lock contention under high load)
static HARDWARE_CACHE: LazyLock<Mutex<(Option<HardwareState>, Instant)>> = LazyLock::new(|| {
    Mutex::new((None, Instant::now()))
});

// ============================================================================
// Hardware State Struct
// ============================================================================

/// Estado completo do hardware - o "corpo" de VIVA
#[derive(Debug, Clone)]
pub struct HardwareState {
    // CPU
    pub cpu_usage: f32,
    pub cpu_temp: Option<f32>,        // °C - None se indisponível
    pub cpu_count: usize,

    // Memory
    pub memory_used_percent: f32,
    pub memory_available_gb: f32,
    pub memory_total_gb: f32,
    pub swap_used_percent: f32,

    // GPU (opcional)
    pub gpu_usage: Option<f32>,       // % - None se sem GPU/driver
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
}

impl Encoder for HardwareState {
    fn encode<'a>(&self, env: Env<'a>) -> Term<'a> {
        use rustler::types::atom::Atom;
        use rustler::types::map::map_new;

        let nil = Atom::from_str(env, "nil").unwrap();

        let mut map = map_new(env);

        // Helper para adicionar ao mapa
        fn put_val<'a, T: Encoder>(env: Env<'a>, map: Term<'a>, key: &str, val: T) -> Term<'a> {
            map.map_put(
                Atom::from_str(env, key).unwrap(),
                val.encode(env),
            ).unwrap()
        }

        fn put_opt_f32<'a>(env: Env<'a>, map: Term<'a>, key: &str, val: Option<f32>, nil: Term<'a>) -> Term<'a> {
            match val {
                Some(v) => put_val(env, map, key, v),
                None => map.map_put(Atom::from_str(env, key).unwrap(), nil).unwrap(),
            }
        }

        fn put_opt_str<'a>(env: Env<'a>, map: Term<'a>, key: &str, val: &Option<String>, nil: Term<'a>) -> Term<'a> {
            match val {
                Some(v) => put_val(env, map, key, v.as_str()),
                None => map.map_put(Atom::from_str(env, key).unwrap(), nil).unwrap(),
            }
        }

        let nil_term = nil.encode(env);

        // CPU
        let map = put_val(env, map, "cpu_usage", self.cpu_usage);
        let map = put_opt_f32(env, map, "cpu_temp", self.cpu_temp, nil_term);
        let map = put_val(env, map, "cpu_count", self.cpu_count);

        // Memory
        let map = put_val(env, map, "memory_used_percent", self.memory_used_percent);
        let map = put_val(env, map, "memory_available_gb", self.memory_available_gb);
        let map = put_val(env, map, "memory_total_gb", self.memory_total_gb);
        let map = put_val(env, map, "swap_used_percent", self.swap_used_percent);

        // GPU
        let map = put_opt_f32(env, map, "gpu_usage", self.gpu_usage, nil_term);
        let map = put_opt_f32(env, map, "gpu_vram_used_percent", self.gpu_vram_used_percent, nil_term);
        let map = put_opt_f32(env, map, "gpu_temp", self.gpu_temp, nil_term);
        let map = put_opt_str(env, map, "gpu_name", &self.gpu_name, nil_term);

        // Disk
        let map = put_val(env, map, "disk_usage_percent", self.disk_usage_percent);
        let map = put_val(env, map, "disk_read_bytes", self.disk_read_bytes);
        let map = put_val(env, map, "disk_write_bytes", self.disk_write_bytes);

        // Network
        let map = put_val(env, map, "net_rx_bytes", self.net_rx_bytes);
        let map = put_val(env, map, "net_tx_bytes", self.net_tx_bytes);

        // System
        let map = put_val(env, map, "uptime_seconds", self.uptime_seconds);
        let map = put_val(env, map, "process_count", self.process_count);
        let map = put_val(env, map, "load_avg_1m", self.load_avg_1m);
        let map = put_val(env, map, "load_avg_5m", self.load_avg_5m);
        let map = put_val(env, map, "load_avg_15m", self.load_avg_15m);

        map
    }
}

// ============================================================================
// NIF Functions
// ============================================================================

/// Verifica se o corpo de VIVA está vivo
#[rustler::nif]
fn alive() -> &'static str {
    "VIVA body is alive"
}

/// Coleta estado do hardware (função interna, com cache de 500ms)
fn collect_hardware_state() -> HardwareState {
    // Check cache first
    {
        let cache = HARDWARE_CACHE.lock().unwrap();
        if let (Some(ref cached_state), last_update) = &*cache {
            if last_update.elapsed() < Duration::from_millis(CACHE_DURATION_MS) {
                return cached_state.clone();
            }
        }
    }

    // Cache miss or expired - refresh all data sources
    let mut sys = SYSTEM.lock().unwrap();
    sys.refresh_all();

    let mut components = COMPONENTS.lock().unwrap();
    components.refresh();

    let mut disks = DISKS.lock().unwrap();
    disks.refresh();

    let mut networks = NETWORKS.lock().unwrap();
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
    } else { 0.0 };

    let swap_used_percent = if total_swap > 0.0 {
        ((used_swap / total_swap) * 100.0) as f32
    } else { 0.0 };

    // GPU (tenta NVML se disponível)
    let (gpu_usage, gpu_vram, gpu_temp, gpu_name) = get_gpu_info();

    // Disk
    let (disk_usage, disk_read, disk_write) = get_disk_info(&disks);

    // Network
    let (net_rx, net_tx) = get_network_info(&networks);

    // Load average (Unix-like)
    let load = System::load_average();

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
    };

    // Update cache
    {
        let mut cache = HARDWARE_CACHE.lock().unwrap();
        *cache = (Some(state.clone()), Instant::now());
    }

    state
}

/// Sente o hardware atual (interocepção completa)
#[rustler::nif]
fn feel_hardware() -> NifResult<HardwareState> {
    Ok(collect_hardware_state())
}

/// Converte hardware em qualia (PAD deltas)
///
/// Retorna (pleasure_delta, arousal_delta, dominance_delta)
///
/// Base teórica: Interocepção (Craig, 2002) + PAD (Mehrabian, 1996)
#[rustler::nif]
fn hardware_to_qualia() -> NifResult<(f64, f64, f64)> {
    let hw = collect_hardware_state();

    // Normalizar métricas para [0, 1]
    let cpu_stress = (hw.cpu_usage as f64 / 100.0).clamp(0.0, 1.0);
    let mem_stress = (hw.memory_used_percent as f64 / 100.0).clamp(0.0, 1.0);
    let swap_stress = (hw.swap_used_percent as f64 / 100.0).clamp(0.0, 1.0);

    // Temperatura: >70°C começa stress, >85°C é crítico
    let temp_stress = match hw.cpu_temp {
        Some(t) => ((t as f64 - 50.0) / 40.0).clamp(0.0, 1.0),
        None => 0.0, // Sem dado = assume ok
    };

    // GPU: VRAM alta = imaginação limitada
    let gpu_stress = match hw.gpu_vram_used_percent {
        Some(v) => (v as f64 / 100.0).clamp(0.0, 1.0),
        None => 0.0,
    };

    // Load average: >1 por core = overloaded
    let cores = hw.cpu_count.max(1) as f64;
    let load_stress = (hw.load_avg_1m / cores).clamp(0.0, 1.0);

    // Disk: usage > 90% = preocupante
    let disk_stress = if hw.disk_usage_percent > 90.0 {
        ((hw.disk_usage_percent - 90.0) / 10.0).clamp(0.0, 1.0) as f64
    } else { 0.0 };

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

    let composite_stress =
        cpu_stress * 0.15 +
        load_stress * 0.15 +
        mem_stress * 0.20 +
        swap_stress * 0.05 +
        temp_stress * 0.20 +
        gpu_stress * 0.15 +
        disk_stress * 0.10;

    // PAD deltas (aditivos ao estado atual)
    // Coeficientes aumentados em 1.5× para maior impacto emocional
    // (evita que decay neutralize antes de VIVA "sentir")
    //
    // Pleasure: Stress → desconforto (negativo)
    // Formula: δP = -k_p × σ onde k_p = 0.12 (era 0.08)
    let pleasure_delta = -0.12 * composite_stress;

    // Arousal: Stress → ativação (positivo, até certo ponto)
    // Formula corrigida: δA = k_a × (2σ - σ²) - parábola que sempre fica em [0, max]
    // Pico em σ=1, depois decresce (exaustão)
    // k_a = 0.15 (ajustado para nova fórmula)
    let arousal_delta = 0.15 * (2.0 * composite_stress - composite_stress.powi(2));

    // Dominance: Stress → perda de controle (negativo)
    // Mais impactado por GPU (capacidade) e Load (overwhelm)
    // k_d = 0.09 (era 0.06)
    let dominance_stress = load_stress * 0.4 + gpu_stress * 0.3 + mem_stress * 0.3;
    let dominance_delta = -0.09 * dominance_stress;

    Ok((pleasure_delta, arousal_delta, dominance_delta))
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Obtém temperatura da CPU (multiplataforma)
fn get_cpu_temp(components: &Components) -> Option<f32> {
    // Procura por componentes de CPU em ordem de preferência
    let cpu_labels = [
        "coretemp",      // Intel Linux
        "k10temp",       // AMD Linux
        "cpu",           // Genérico
        "CPU",           // Windows
        "Core",          // Intel per-core
        "Tctl",          // AMD Ryzen
        "Package",       // Intel package
    ];

    for component in components.iter() {
        let label = component.label().to_lowercase();
        for cpu_label in &cpu_labels {
            if label.contains(&cpu_label.to_lowercase()) {
                let temp = component.temperature();
                if temp > 0.0 && temp < 150.0 { // Sanity check
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

/// Obtém informações de GPU via NVML
/// Retorna (usage%, vram%, temp, name)
fn get_gpu_info() -> (Option<f32>, Option<f32>, Option<f32>, Option<String>) {
    use nvml_wrapper::enum_wrappers::device::TemperatureSensor;

    // Usa NVML global (inicializado uma vez)
    let nvml = match NVML.as_ref() {
        Some(n) => n,
        None => return (None, None, None, None),
    };

    // Pega primeira GPU (index 0)
    let device = match nvml.device_by_index(0) {
        Ok(d) => d,
        Err(_) => return (None, None, None, None),
    };

    // Utilização (GPU compute %)
    let usage = device.utilization_rates()
        .ok()
        .map(|u| u.gpu as f32);

    // VRAM
    let vram = device.memory_info()
        .ok()
        .map(|m| (m.used as f64 / m.total as f64 * 100.0) as f32);

    // Temperatura
    let temp = device.temperature(TemperatureSensor::Gpu)
        .ok()
        .map(|t| t as f32);

    // Nome
    let name = device.name().ok();

    (usage, vram, temp, name)
}

/// Obtém informações de disco
/// Retorna (usage%, read_bytes, write_bytes)
fn get_disk_info(disks: &Disks) -> (f32, u64, u64) {
    let mut total_space = 0u64;
    let mut total_used = 0u64;

    for disk in disks.iter() {
        total_space += disk.total_space();
        total_used += disk.total_space() - disk.available_space();
    }

    let usage = if total_space > 0 {
        (total_used as f64 / total_space as f64 * 100.0) as f32
    } else { 0.0 };

    // sysinfo não dá I/O rates diretamente, retornamos 0
    // Poderia usar /proc/diskstats no Linux
    (usage, 0, 0)
}

/// Obtém informações de rede
/// Retorna (rx_bytes, tx_bytes)
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
