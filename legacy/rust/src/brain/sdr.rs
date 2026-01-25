use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// A Sparse Distributed Representation (SDR) vector.
/// Instead of dense floats, we use a bit-array where semantic meaning is distributed.
/// Based on Pentti Kanerva's Sparse Distributed Memory (SDM).
pub struct Sdr {
    /// The sparse vector (indices of active bits)
    pub active_bits: Vec<usize>,
    /// Total dimensionality of the space (e.g., 2048)
    pub dimensions: usize,
}

impl Sdr {
    /// Encodes a string into a Sparse Distributed Representation using Random Projections (hashing).
    /// This creates a deterministic, stable SDR: similar inputs -> overlapping active bits.
    pub fn encode_text(text: &str, dimensions: usize, sparsity: usize) -> Self {
        let mut active_bits = Vec::with_capacity(sparsity);

        // Simulating a "Retina" with 'sparsity' receptors that fire based on content features.
        // We use a sliding window (n-gram) approach implicitly via hashing seeds.

        // To make it semantic-ish without training:
        // We hash trigrams of the text. Similar texts share trigrams -> share active bits.
        // This is effectively SimHash / MinHash.

        // Simple implementation:
        // 1. Hash the whole string with different seeds to pick N active bits?
        //    -> No, that's unstable for slight variations.
        // 2. Hash K-Shingles (substrings) and map them to bits.

        let shingle_size = 3.max(text.len() / 10); // Adaptive shingle size
        let chars: Vec<char> = text.chars().collect();

        if chars.len() <= shingle_size {
             // Text too short, just hash it recursively with salts
             for i in 0..sparsity {
                 let bit = hash_with_salt(text, i) % dimensions;
                 if !active_bits.contains(&bit) {
                     active_bits.push(bit);
                 }
             }
        } else {
             // Hash sliding windows
             for window in chars.windows(shingle_size) {
                 let s: String = window.iter().collect();
                 let bit = (hash_string(&s) % dimensions as u64) as usize;
                 if !active_bits.contains(&bit) {
                     active_bits.push(bit);
                 }
             }

             // RE-THINKING FOR ROBUSTNESS:
             // Let's use a standard Random Projection approach.
             // We want "Cat" and "Cats" to overlap.

             // K-Min-Hash Sketching is better but complex.
             // Let's go with a simpler "Concept Fingerprint":
             // Hash every word, map to N bits.

             for word in text.split_whitespace() {
                 let h = hash_string(word);
                 // Each word activates 'sparsity / 10' bits
                 for i in 0..(sparsity / 5).max(1) {
                     let bit = (h.wrapping_add(i as u64).wrapping_mul(11400714819323198485)) % dimensions as u64;
                     let bit = bit as usize;
                     if !active_bits.contains(&bit) {
                         active_bits.push(bit);
                     }
                 }
             }
        }

        // Ensure we respect the sparsity limit and sort for efficient operations
        active_bits.truncate(sparsity);
        active_bits.sort();
        active_bits.dedup();

        Sdr {
            active_bits,
            dimensions,
        }
    }

    /// Calculate overlap (similarity) between two SDRs
    pub fn overlap(&self, other: &Sdr) -> usize {
        // Intersection of sorted arrays
        let mut count = 0;
        let mut i = 0;
        let mut j = 0;

        while i < self.active_bits.len() && j < other.active_bits.len() {
            if self.active_bits[i] < other.active_bits[j] {
                i += 1;
            } else if self.active_bits[i] > other.active_bits[j] {
                j += 1;
            } else {
                count += 1;
                i += 1;
                j += 1;
            }
        }
        count
    }
}

// Helpers
fn hash_string(s: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    s.hash(&mut hasher);
    hasher.finish()
}

fn hash_with_salt(s: &str, salt: usize) -> usize {
    let mut hasher = DefaultHasher::new();
    s.hash(&mut hasher);
    salt.hash(&mut hasher);
    hasher.finish() as usize
}
