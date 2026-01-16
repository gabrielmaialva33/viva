use crate::memory::PadEmotion;
use super::sdr::Sdr;


/// The Plastic Mind.
/// A single-layer associative network that maps SDRs (Sensory) to Concepts (Latent Space).
/// Uses Oja's Rule for stability (weights don't explode).
pub struct Cortex {
    /// Synaptic weights: [Input Dimension x Latent Dimension] (e.g., 2048 x 384)
    /// We keep it 1D flattened for cache locality.
    weights: Vec<f32>,
    input_dim: usize,
    output_dim: usize,
    learning_rate: f32,
}

impl Cortex {
    pub fn new(input_dim: usize, output_dim: usize) -> Self {
        // Initialize weights with deterministic noise (Golden Ratio mixed frequencies).
        // Avoids periodicity of simple sin(i).
        let mut weights = Vec::with_capacity(input_dim * output_dim);
        for i in 0..(input_dim * output_dim) {
            let i_f32 = i as f32;
            let val = (i_f32.sin() * 0.5 + (i_f32 * 1.618033).sin() * 0.5) * 0.001;
            weights.push(val);
        }

        Cortex {
            weights,
            input_dim,
            output_dim,
            learning_rate: 0.01,
        }
    }

    /// Process an experience: Input SDR -> Latent Vector
    /// Also learns from it if 'plasticity' is enabled.
    pub fn process(&mut self, input: &Sdr, emotion: PadEmotion) -> Vec<f32> {
        let mut output = vec![0.0; self.output_dim];

        // 1. Forward Pass (Activation)
        for &i in &input.active_bits {
            let row_start = i * self.output_dim;
            for j in 0..self.output_dim {
                output[j] += self.weights[row_start + j];
            }
        }

        // 2. Learning (Hebbian / Oja's Rule)
        let intensity = (emotion.arousal.abs() + emotion.pleasure.abs()) / 2.0;
        let effective_lr = self.learning_rate * (1.0 + intensity * 5.0);

        if effective_lr > 0.0001 {
            self.learn_oja(input, &output, effective_lr);
        }

        // 3. L2 Normalization (Critical for HNSW/Cosine Similarity)
        normalize_l2(&mut output);

        output
    }

    /// Oja's Learning Rule: dw_ij = lr * y_j * (x_i - y_j * w_ij)
    /// Prevents unbounded weight growth.
    fn learn_oja(&mut self, input: &Sdr, output: &[f32], lr: f32) {
        for &i in &input.active_bits {
            let row_start = i * self.output_dim;
            for j in 0..self.output_dim {
                let w = self.weights[row_start + j];
                let y = output[j];
                // x_i is 1.0 (since we only iterate active bits)
                let delta = lr * y * (1.0 - y * w);
                self.weights[row_start + j] += delta;
            }
        }
    }
}

fn normalize_l2(vec: &mut [f32]) {
    let mag: f32 = vec.iter().map(|&x| x * x).sum::<f32>().sqrt();
    if mag > 1e-6 {
        for x in vec.iter_mut() {
            *x /= mag;
        }
    }
}
