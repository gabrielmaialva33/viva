//! Unified Body State - VIVA's complete interoceptive state
//!
//! This module unifies all body subsystems into a single coherent state:
//! - Hardware sensing (HardwareState)
//! - Emotional dynamics (PAD via O-U + Cusp)
//! - Temporal patterns (entropy, jitter)
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────┐
//! │                      BodyState                          │
//! ├─────────────────────────────────────────────────────────┤
//! │  PAD State (Emotional)                                  │
//! │  ├─ pleasure: f64  [-1, 1]                             │
//! │  ├─ arousal: f64   [-1, 1]                             │
//! │  └─ dominance: f64 [-1, 1]                             │
//! ├─────────────────────────────────────────────────────────┤
//! │  Hardware State (Interoception)                         │
//! │  ├─ cpu_usage, cpu_temp, memory_used_percent, ...      │
//! │  ├─ system_entropy (chaos vs order)                    │
//! │  └─ os_jitter (tremor/instability)                     │
//! ├─────────────────────────────────────────────────────────┤
//! │  Dynamics Config                                        │
//! │  ├─ cusp_enabled: bool                                 │
//! │  ├─ cusp_sensitivity: f64                              │
//! │  └─ dt: f64 (tick duration)                            │
//! └─────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Tick Cycle
//!
//! Each `body_tick()` call:
//! 1. Senses hardware (HardwareState)
//! 2. Computes qualia (PAD deltas from hardware stress)
//! 3. Generates stochastic noise
//! 4. Evolves PAD state via DynAffect (O-U + Cusp)
//! 5. Returns complete BodyState
//!
//! ## References
//!
//! - Craig (2002) - Interoception
//! - Mehrabian (1996) - PAD Model
//! - Kuppens et al. (2010) - Emotional dynamics
//! - Zeeman (1977) - Cusp catastrophe

use serde::{Deserialize, Serialize};

use crate::dynamics::{DynAffect, OUParams};
use crate::HardwareState;

/// Complete body state - everything VIVA feels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BodyState {
    // === Emotional State (PAD) ===
    /// Pleasure dimension: -1 (suffering) to +1 (joy)
    pub pleasure: f64,
    /// Arousal dimension: -1 (calm/sleepy) to +1 (excited/alert)
    pub arousal: f64,
    /// Dominance dimension: -1 (submissive/overwhelmed) to +1 (in control)
    pub dominance: f64,

    // === Hardware State (Interoception) ===
    /// Complete hardware metrics
    pub hardware: HardwareState,

    // === Derived Metrics ===
    /// Composite stress level (0-1), derived from hardware
    pub stress_level: f64,
    /// Whether currently in bifurcation region (mood can flip suddenly)
    pub in_bifurcation: bool,

    // === Metadata ===
    /// Tick counter (monotonic)
    pub tick: u64,
    /// Timestamp (Unix ms)
    pub timestamp_ms: u64,
}

/// Configuration for body dynamics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BodyConfig {
    /// Time step per tick (seconds)
    pub dt: f64,
    /// Enable cusp catastrophe overlay
    pub cusp_enabled: bool,
    /// Cusp sensitivity (0-1)
    pub cusp_sensitivity: f64,
    /// O-U parameters for each PAD dimension
    pub ou_params: [OUParams; 3],
    /// Random seed (0 = use system entropy)
    pub seed: u64,
}

impl Default for BodyConfig {
    fn default() -> Self {
        Self {
            dt: 0.5, // 500ms ticks
            cusp_enabled: true,
            cusp_sensitivity: 0.5,
            ou_params: [
                // Pleasure: slow reversion, moderate volatility
                OUParams {
                    theta: 0.3,
                    mu: 0.0,
                    sigma: 0.15,
                },
                // Arousal: fast reversion, high volatility
                OUParams {
                    theta: 0.5,
                    mu: 0.0,
                    sigma: 0.25,
                },
                // Dominance: slowest reversion, low volatility
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

/// Mutable body state holder for tick evolution
pub struct BodyEngine {
    /// Current PAD state
    pad: [f64; 3],
    /// Dynamics engine
    dynaffect: DynAffect,
    /// Configuration
    config: BodyConfig,
    /// Tick counter
    tick: u64,
    /// Simple LCG RNG state (fast, deterministic if seeded)
    rng_state: u64,
}

impl BodyEngine {
    /// Create new body engine with config
    pub fn new(config: BodyConfig) -> Self {
        let seed = if config.seed == 0 {
            // Use system time as seed
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(12345)
        } else {
            config.seed
        };

        Self {
            pad: [0.0, 0.0, 0.0], // Start neutral
            dynaffect: DynAffect {
                ou_params: config.ou_params,
                cusp_enabled: config.cusp_enabled,
                cusp_sensitivity: config.cusp_sensitivity,
            },
            config,
            tick: 0,
            rng_state: seed,
        }
    }

    /// Create with default config
    pub fn default() -> Self {
        Self::new(BodyConfig::default())
    }

    /// Generate next pseudo-random f64 in N(0,1) using Box-Muller
    fn next_gaussian(&mut self) -> f64 {
        // LCG step (constants from Numerical Recipes)
        self.rng_state = self.rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let u1 = (self.rng_state as f64) / (u64::MAX as f64);

        self.rng_state = self.rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let u2 = (self.rng_state as f64) / (u64::MAX as f64);

        // Box-Muller transform
        let u1_clamped = u1.max(1e-10); // Avoid log(0)
        (-2.0 * u1_clamped.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    /// Execute one tick of the body
    ///
    /// This is the main integration point:
    /// 1. Sense hardware
    /// 2. Compute qualia (stress → PAD deltas)
    /// 3. Apply dynamics (O-U + Cusp)
    /// 4. Return complete state
    pub fn tick(&mut self) -> BodyState {
        // 1. Sense hardware
        let hardware = crate::collect_hardware_state();

        // 2. Compute stress and qualia
        let stress_level = self.compute_stress(&hardware);
        let (p_delta, a_delta, d_delta) = self.compute_qualia(&hardware, stress_level);

        // 3. Generate noise
        let noises = [self.next_gaussian(), self.next_gaussian(), self.next_gaussian()];

        // 4. External bias from hardware qualia
        let external_bias = p_delta; // Hardware stress biases mood

        // 5. Evolve PAD via dynamics
        self.dynaffect.step(&mut self.pad, self.config.dt, &noises, external_bias);

        // 6. Apply direct qualia influence (small, additive)
        // This ensures hardware stress has immediate (not just stochastic) effect
        self.pad[0] = (self.pad[0] + p_delta * 0.1).clamp(-1.0, 1.0);
        self.pad[1] = (self.pad[1] + a_delta * 0.1).clamp(-1.0, 1.0);
        self.pad[2] = (self.pad[2] + d_delta * 0.1).clamp(-1.0, 1.0);

        // 7. Check bifurcation status
        let c = self.pad[1].abs() * 1.5; // Arousal → bifurcation control
        let in_bifurcation = crate::dynamics::cusp_is_bifurcation(c, external_bias);

        // 8. Update tick counter
        self.tick += 1;

        // 9. Build complete state
        BodyState {
            pleasure: self.pad[0],
            arousal: self.pad[1],
            dominance: self.pad[2],
            hardware,
            stress_level,
            in_bifurcation,
            tick: self.tick,
            timestamp_ms: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0),
        }
    }

    /// Compute composite stress from hardware (0-1)
    fn compute_stress(&self, hw: &HardwareState) -> f64 {
        // Weighted stress components (same as hardware_to_qualia)
        let cpu_stress = sigmoid_threshold(hw.cpu_usage as f64 / 100.0, 12.0, 0.80);
        let mem_stress = sigmoid_threshold(hw.memory_used_percent as f64 / 100.0, 10.0, 0.75);
        let swap_stress = sigmoid_threshold(hw.swap_used_percent as f64 / 100.0, 15.0, 0.20);

        let temp_stress = hw.cpu_temp.map_or(0.0, |t| {
            sigmoid_threshold((t as f64 - 40.0) / 50.0, 8.0, 0.60)
        });

        let gpu_stress = hw.gpu_vram_used_percent.map_or(0.0, |v| {
            sigmoid_threshold(v as f64 / 100.0, 10.0, 0.85)
        });

        let cores = hw.cpu_count.max(1) as f64;
        let load_stress = sigmoid_threshold((hw.load_avg_1m / cores).min(1.5) / 1.5, 10.0, 0.53);

        let disk_stress = sigmoid_threshold(hw.disk_usage_percent as f64 / 100.0, 12.0, 0.90);

        // Weighted sum
        (cpu_stress * 0.15
            + load_stress * 0.15
            + mem_stress * 0.20
            + swap_stress * 0.05
            + temp_stress * 0.20
            + gpu_stress * 0.15
            + disk_stress * 0.10)
            .clamp(0.0, 1.0)
    }

    /// Compute PAD deltas from hardware state
    fn compute_qualia(&self, hw: &HardwareState, stress: f64) -> (f64, f64, f64) {
        // Allostasis: anticipatory adjustment based on load trend
        let allostasis = if hw.load_avg_5m > 0.1 {
            ((hw.load_avg_1m - hw.load_avg_5m) / hw.load_avg_5m).clamp(-1.0, 1.0)
        } else {
            0.0
        };

        let adjusted_stress = (stress * (1.0 + allostasis * 0.10)).clamp(0.0, 1.0);

        // Pleasure: stress → discomfort
        let mut p_delta = -0.12 * adjusted_stress;

        // Arousal: stress → activation (Yerkes-Dodson inverted U)
        let mut a_delta = 0.15 * (2.0 * adjusted_stress - adjusted_stress.powi(2));

        // Dominance: specific stressors → loss of control
        let gpu_stress = hw.gpu_vram_used_percent.map_or(0.0, |v| v as f64 / 100.0);
        let load_stress = (hw.load_avg_1m / hw.cpu_count.max(1) as f64).min(1.0);
        let mem_stress = hw.memory_used_percent as f64 / 100.0;
        let dominance_stress = load_stress * 0.4 + gpu_stress * 0.3 + mem_stress * 0.3;
        let mut d_delta = -0.09 * dominance_stress;

        // Bio-rhythm impact
        d_delta -= hw.system_entropy as f64 * 0.15; // Chaos erodes control
        a_delta += hw.os_jitter as f64 * 0.20; // Jitter → anxiety
        p_delta -= hw.os_jitter as f64 * 0.15; // Uncertainty is unpleasant

        (p_delta, a_delta, d_delta)
    }

    /// Get current PAD state
    pub fn pad(&self) -> (f64, f64, f64) {
        (self.pad[0], self.pad[1], self.pad[2])
    }

    /// Set PAD state directly (for initialization or external events)
    pub fn set_pad(&mut self, p: f64, a: f64, d: f64) {
        self.pad[0] = p.clamp(-1.0, 1.0);
        self.pad[1] = a.clamp(-1.0, 1.0);
        self.pad[2] = d.clamp(-1.0, 1.0);
    }

    /// Apply external emotional stimulus
    ///
    /// This allows external events (messages, interactions) to influence PAD
    pub fn apply_stimulus(&mut self, p_delta: f64, a_delta: f64, d_delta: f64) {
        self.pad[0] = (self.pad[0] + p_delta).clamp(-1.0, 1.0);
        self.pad[1] = (self.pad[1] + a_delta).clamp(-1.0, 1.0);
        self.pad[2] = (self.pad[2] + d_delta).clamp(-1.0, 1.0);
    }
}

/// Sigmoid threshold function (same as in lib.rs)
#[inline]
fn sigmoid_threshold(x: f64, k: f64, x0: f64) -> f64 {
    let raw = 1.0 / (1.0 + (-k * (x - x0)).exp());
    let min_val = 1.0 / (1.0 + (-k * (0.0 - x0)).exp());
    let max_val = 1.0 / (1.0 + (-k * (1.0 - x0)).exp());
    ((raw - min_val) / (max_val - min_val)).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_body_engine_creation() {
        let engine = BodyEngine::default();
        let (p, a, d) = engine.pad();
        assert_eq!(p, 0.0);
        assert_eq!(a, 0.0);
        assert_eq!(d, 0.0);
    }

    #[test]
    fn test_gaussian_distribution() {
        let mut engine = BodyEngine::new(BodyConfig {
            seed: 42,
            ..Default::default()
        });

        // Generate many samples
        let samples: Vec<f64> = (0..1000).map(|_| engine.next_gaussian()).collect();

        // Check mean is close to 0
        let mean: f64 = samples.iter().sum::<f64>() / samples.len() as f64;
        assert!(mean.abs() < 0.1, "Mean should be ~0, got {}", mean);

        // Check std is close to 1
        let variance: f64 = samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / samples.len() as f64;
        let std = variance.sqrt();
        assert!((std - 1.0).abs() < 0.15, "Std should be ~1, got {}", std);
    }

    #[test]
    fn test_set_pad() {
        let mut engine = BodyEngine::default();
        engine.set_pad(0.5, -0.3, 0.8);
        let (p, a, d) = engine.pad();
        assert_eq!(p, 0.5);
        assert_eq!(a, -0.3);
        assert_eq!(d, 0.8);
    }

    #[test]
    fn test_set_pad_clamps() {
        let mut engine = BodyEngine::default();
        engine.set_pad(2.0, -3.0, 1.5);
        let (p, a, d) = engine.pad();
        assert_eq!(p, 1.0);
        assert_eq!(a, -1.0);
        assert_eq!(d, 1.0);
    }

    #[test]
    fn test_apply_stimulus() {
        let mut engine = BodyEngine::default();
        engine.apply_stimulus(0.5, 0.3, -0.2);
        let (p, a, d) = engine.pad();
        assert_eq!(p, 0.5);
        assert_eq!(a, 0.3);
        assert_eq!(d, -0.2);
    }

    #[test]
    fn test_sigmoid_threshold() {
        // At threshold, output should be ~0.5
        let at_threshold = sigmoid_threshold(0.8, 12.0, 0.8);
        assert!((at_threshold - 0.5).abs() < 0.1);

        // Below threshold, output should be low
        let below = sigmoid_threshold(0.5, 12.0, 0.8);
        assert!(below < 0.2);

        // Above threshold, output should be high
        let above = sigmoid_threshold(0.95, 12.0, 0.8);
        assert!(above > 0.8);
    }
}
