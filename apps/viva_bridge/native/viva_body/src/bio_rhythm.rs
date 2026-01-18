use std::collections::VecDeque;

/// Short-term memory buffer size (in 500ms ticks)
/// 20 ticks = 10 seconds of history
const HISTORY_SIZE: usize = 20;

/// Maintains temporal history for biological rhythm analysis
pub struct BioRhythm {
    cpu_history: VecDeque<f32>,
    ctx_switch_history: VecDeque<u64>,
}

impl BioRhythm {
    pub fn new() -> Self {
        Self {
            cpu_history: VecDeque::with_capacity(HISTORY_SIZE),
            ctx_switch_history: VecDeque::with_capacity(HISTORY_SIZE),
        }
    }

    /// Updates the state with new data
    pub fn update(&mut self, cpu_usage: f32, ctx_switches: u64) {
        if self.cpu_history.len() >= HISTORY_SIZE {
            self.cpu_history.pop_front();
        }
        self.cpu_history.push_back(cpu_usage);

        if self.ctx_switch_history.len() >= HISTORY_SIZE {
            self.ctx_switch_history.pop_front();
        }
        self.ctx_switch_history.push_back(ctx_switches);
    }

    /// Calculates Shannon Entropy (Chaos vs Order) of CPU usage
    /// Returns 0.0 (Total Order) to 1.0 (Total Chaos)
    pub fn cpu_entropy(&self) -> f32 {
        if self.cpu_history.len() < 2 {
            return 0.0;
        }

        // 1. Normalize values for probability distribution
        // Create a 10-bin histogram (0-10%, 10-20%...)
        let mut bins = [0.0f32; 10];
        let total = self.cpu_history.len() as f32;

        for &val in &self.cpu_history {
            let idx = (val / 10.0).floor().clamp(0.0, 9.0) as usize;
            bins[idx] += 1.0;
        }

        // 2. Calculate Shannon Entropy: H = -sum(p * log2(p))
        let mut entropy = 0.0;
        for &count in &bins {
            if count > 0.0 {
                let p = count / total;
                entropy -= p * p.log2();
            }
        }

        // Normalize by maximum possible (log2(10 bins) ≈ 3.32)
        (entropy / 3.32).clamp(0.0, 1.0)
    }

    /// Calculates the "Jitter" (Standard Deviation) of Context Switches
    /// Indicates operating system instability
    pub fn context_switch_jitter(&self) -> f32 {
        if self.ctx_switch_history.len() < 2 {
            return 0.0;
        }

        // We need the deltas, not absolute values
        let mut deltas = Vec::new();
        let mut iter = self.ctx_switch_history.iter();
        if let Some(mut last) = iter.next() {
            for val in iter {
                // Simple delta (can be negative if counter resets, but unlikely here)
                let delta = if val >= last { val - last } else { 0 };
                deltas.push(delta as f32);
                last = val;
            }
        }

        if deltas.is_empty() {
            return 0.0;
        }

        // Mean
        let mean: f32 = deltas.iter().sum::<f32>() / deltas.len() as f32;

        // Variance
        let variance: f32 =
            deltas.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / deltas.len() as f32;

        // Normalized standard deviation (Coefficient of Variation)
        // If mean is too low, jitter can explode, so we clamp
        if mean > 1.0 {
            (variance.sqrt() / mean).clamp(0.0, 1.0)
        } else {
            0.0
        }
    }
}
