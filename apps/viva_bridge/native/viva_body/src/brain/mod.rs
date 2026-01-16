use std::sync::Mutex;
use std::sync::OnceLock;
use anyhow::{Result, anyhow};
use crate::memory::PadEmotion;
use self::sdr::Sdr;
use self::cortex::Cortex;

pub mod sdr;
pub mod cortex;

// Global Brain Singleton
static BRAIN: OnceLock<Mutex<BrainManager>> = OnceLock::new();

pub struct BrainManager {
    cortex: Cortex,
}

impl BrainManager {
    pub fn init() -> Result<()> {
        // Dimensions:
        // Input: 2048 (SDR size from retina)
        // Output: 384 (Latent concept space size)
        let cortex = Cortex::new(2048, 384);
        let manager = BrainManager { cortex };

        BRAIN.set(Mutex::new(manager))
            .map_err(|_| anyhow!("Brain already initialized"))?;
        Ok(())
    }

    pub fn global() -> Option<&'static Mutex<BrainManager>> {
        BRAIN.get()
    }

    pub fn process_experience(text: &str, emotion: PadEmotion) -> Result<Vec<f32>> {
        let brain_lock = Self::global()
            .ok_or_else(|| anyhow!("Brain not initialized"))?;

        let mut brain = brain_lock.lock()
            .map_err(|_| anyhow!("Brain mutex poisoned"))?;

        // 1. Retina: Encode text to SDR
        // Sparsity: 2% of 2048 = ~40 bits
        let sdr = Sdr::encode_text(text, 2048, 40);

        // 2. Cortex: Synaptic activation & Hebbian learning
        let output_vector = brain.cortex.process(&sdr, emotion);

        Ok(output_vector)
    }
}
