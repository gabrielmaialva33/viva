//! VIVA Metabolism - Digital Thermodynamics
//!
//! Metabolic model based on real physics:
//! - Energy (Joules): Computational cost
//! - Entropy (Heat): Accumulated disorder
//! - Fatigue: Cognitive exhaustion state
//!
//! Theoretical references:
//! - 2nd Law of Thermodynamics (Entropy)
//! - Free Energy Principle (Friston, 2010)
//! - Dissipative Structures (Prigogine, 1977)

use std::fs;
use std::path::Path;
use std::time::Instant;

/// System metabolic state
#[derive(Debug, Clone)]
pub struct MetabolicState {
    /// Energy consumed in Joules (cost of "thinking")
    pub energy_joules: f32,
    /// Normalized thermal stress: 0.0 (cold) → 1.0 (critical)
    pub thermal_stress: f32,
    /// Electrical voltage (if available)
    pub voltage_tension: f32,
    /// Fatigue level: 0.0 (fresh) → 1.0 (exhausted)
    pub fatigue_level: f32,
    /// System needs rest (trigger for memory consolidation)
    pub needs_rest: bool,
}

impl Default for MetabolicState {
    fn default() -> Self {
        Self {
            energy_joules: 0.0,
            thermal_stress: 0.0,
            voltage_tension: 0.0,
            fatigue_level: 0.0,
            needs_rest: false,
        }
    }
}

/// Main metabolic engine
pub struct Metabolism {
    /// Processor TDP in Watts (fallback when RAPL not available)
    tdp_watts: f32,
    /// Total energy consumed (accumulator)
    energy_consumed_j: f64,
    /// Last sample timestamp
    last_sample: Instant,
    /// Fatigue recovery rate per second
    fatigue_decay: f32,
    /// Current fatigue level
    fatigue: f32,
    /// RAPL path (Linux)
    rapl_path: Option<String>,
    /// Last RAPL energy value (to calculate delta)
    pub last_rapl_uj: Option<u64>,
    /// Max RAPL counter value before wrap (for overflow)
    max_rapl_uj: Option<u64>,
}

impl Metabolism {
    /// # Arguments
    /// * `tdp_watts` - Processor TDP (e.g., 125.0 for i9-13900K)
    pub fn new(tdp_watts: f32) -> Self {
        // Try to detect RAPL on Linux
        let (rapl_path, max_rapl_uj) = Self::detect_rapl();

        Self {
            tdp_watts,
            energy_consumed_j: 0.0,
            last_sample: Instant::now(),
            fatigue_decay: 0.05, // 5% recovery per second when idle
            fatigue: 0.0,
            rapl_path,
            last_rapl_uj: None,
            max_rapl_uj,
        }
    }

    /// Detects RAPL path (Intel Running Average Power Limit) and max_energy_range
    fn detect_rapl() -> (Option<String>, Option<u64>) {
        let rapl_dirs = [
            "/sys/class/powercap/intel-rapl/intel-rapl:0",
            "/sys/class/powercap/intel-rapl:0",
        ];

        for dir in &rapl_dirs {
            let energy_path = format!("{}/energy_uj", dir);
            let max_range_path = format!("{}/max_energy_range_uj", dir);

            if Path::new(&energy_path).exists() {
                // Read max_energy_range_uj to calculate overflow correctly
                let max_range = fs::read_to_string(&max_range_path)
                    .ok()
                    .and_then(|s| s.trim().parse().ok());

                return (Some(energy_path), max_range);
            }
        }
        (None, None)
    }

    /// Reads RAPL energy in microjoules
    fn read_rapl_uj(&self) -> Option<u64> {
        self.rapl_path.as_ref().and_then(|path| {
            fs::read_to_string(path)
                .ok()
                .and_then(|s| s.trim().parse().ok())
        })
    }

    /// Updates metabolic state based on current usage
    ///
    /// # Arguments
    /// * `cpu_usage` - CPU usage in percentage (0-100)
    /// * `cpu_temp` - CPU temperature in Celsius (optional)
    ///
    /// # Returns
    /// Updated metabolic state
    pub fn tick(&mut self, cpu_usage: f32, cpu_temp: Option<f32>) -> MetabolicState {
        let now = Instant::now();
        let dt = now.duration_since(self.last_sample).as_secs_f32();

        // 1. Calculate energy consumed (BEFORE updating last_sample)
        let power_watts = self.estimate_power(cpu_usage, dt);
        let energy_delta_j = power_watts * dt;
        self.energy_consumed_j += energy_delta_j as f64;

        // Update timestamp AFTER energy calculation
        self.last_sample = now;

        // 2. Calculate thermal stress
        // Based on temperature (thermodynamic simplification)
        let base_temp = cpu_temp.unwrap_or(40.0);
        let thermal_stress = (base_temp - 30.0).max(0.0) / 70.0; // Normalized 0-1 (30-100C)

        // 3. Update fatigue
        // Fatigue increases with intense usage, decreases when idle
        if cpu_usage > 50.0 {
            // Intense work: fatigue increases
            let fatigue_rate = (cpu_usage - 50.0) / 50.0 * 0.01; // Max 1% per second at 100%
            self.fatigue = (self.fatigue + fatigue_rate * dt).min(1.0);
        } else {
            // Recovery: fatigue decreases
            self.fatigue = (self.fatigue - self.fatigue_decay * dt).max(0.0);
        }

        // 4. Determine if rest is needed
        let needs_rest = self.fatigue > 0.8;

        MetabolicState {
            energy_joules: energy_delta_j,
            thermal_stress,
            voltage_tension: 0.0, // TODO: Read from sensors if available
            fatigue_level: self.fatigue,
            needs_rest,
        }
    }

    /// Estimates power consumed in Watts
    ///
    /// # Arguments
    /// * `cpu_usage` - CPU usage in percentage (0-100)
    /// * `dt` - Delta time in seconds since last sample
    fn estimate_power(&mut self, cpu_usage: f32, dt: f32) -> f32 {
        // Try RAPL first
        if let Some(current_uj) = self.read_rapl_uj() {
            if let Some(last_uj) = self.last_rapl_uj {
                if dt > 0.0 {
                    // Delta in microjoules -> Watts
                    let delta_uj = if current_uj >= last_uj {
                        current_uj - last_uj
                    } else {
                        // RAPL counter overflow: calculate wrap correctly
                        // delta = (max_range - last) + current
                        match self.max_rapl_uj {
                            Some(max_range) => (max_range - last_uj) + current_uj,
                            // Fallback if max_range not available: use u64::MAX
                            None => (u64::MAX - last_uj) + current_uj,
                        }
                    };
                    self.last_rapl_uj = Some(current_uj);
                    return (delta_uj as f32) / 1_000_000.0 / dt;
                }
            }
            self.last_rapl_uj = Some(current_uj);
        }

        // Fallback: heuristic model based on TDP
        // Power = TDP * (idle_fraction + load_fraction * usage)
        // Assuming idle = 10% of TDP
        let idle_power = self.tdp_watts * 0.10;
        let load_power = self.tdp_watts * 0.90 * (cpu_usage / 100.0);
        idle_power + load_power
    }

    /// Returns total energy consumed since start (in Joules)
    pub fn total_energy(&self) -> f64 {
        self.energy_consumed_j
    }

    /// Resets period counter (tick)
    pub fn reset_period(&mut self) {
        self.fatigue = 0.0;
        // Do not reset total energy (it is historical)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metabolism_basic() {
        let mut meta = Metabolism::new(125.0); // i9-13900K TDP

        // Simulate tick with CPU idle
        let state = meta.tick(5.0, Some(35.0));
        assert!(state.energy_joules > 0.0);
        assert!(state.fatigue_level < 0.1);
        assert!(!state.needs_rest);
    }

    #[test]
    fn test_fatigue_accumulation() {
        let mut meta = Metabolism::new(125.0);

        // Simulate intense work
        for _ in 0..100 {
            meta.tick(95.0, Some(80.0));
            std::thread::sleep(std::time::Duration::from_millis(10));
        }

        // Fatigue should have increased (rate approx 0.009/s -> in 1s ~= 0.009)
        assert!(
            meta.fatigue > 0.005,
            "Fatigue should accumulate (got {})",
            meta.fatigue
        );

        // Test recovery
        let peak_fatigue = meta.fatigue;
        for _ in 0..50 {
            meta.tick(10.0, Some(35.0)); // Idle
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        assert!(meta.fatigue < peak_fatigue, "Fatigue should decay in idle");
    }
}
