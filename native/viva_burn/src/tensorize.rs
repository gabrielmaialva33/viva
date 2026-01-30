//! Tensorize NEAT Genomes
//!
//! Converts variable-topology NEAT networks into fixed-size tensors
//! for efficient GPU batch processing.
//!
//! Inspired by TensorNEAT (2024) - 500x speedups via tensorization.

/// Calculate total weight count for architecture
pub fn weight_count(architecture: &[usize]) -> usize {
    let mut count = 0;
    for i in 0..architecture.len() - 1 {
        let in_size = architecture[i];
        let out_size = architecture[i + 1];
        count += in_size * out_size + out_size; // weights + biases
    }
    count
}

/// Flatten network weights into a single vector
/// Order: [W1, b1, W2, b2, ...]
/// W is row-major: W[out][in]
pub fn flatten_weights(
    layer_weights: &[Vec<Vec<f32>>],  // [layer][out][in]
    layer_biases: &[Vec<f32>],        // [layer][out]
) -> Vec<f32> {
    let mut flat = Vec::new();

    for (weights, biases) in layer_weights.iter().zip(layer_biases.iter()) {
        // Flatten weight matrix (row-major)
        for row in weights {
            flat.extend(row);
        }
        // Add biases
        flat.extend(biases);
    }

    flat
}

/// Unflatten weights back to layer structure
pub fn unflatten_weights(
    flat: &[f32],
    architecture: &[usize],
) -> (Vec<Vec<Vec<f32>>>, Vec<Vec<f32>>) {
    let mut layer_weights = Vec::new();
    let mut layer_biases = Vec::new();
    let mut offset = 0;

    for i in 0..architecture.len() - 1 {
        let in_size = architecture[i];
        let out_size = architecture[i + 1];

        // Extract weights
        let mut weights = Vec::with_capacity(out_size);
        for _ in 0..out_size {
            let row: Vec<f32> = flat[offset..offset + in_size].to_vec();
            weights.push(row);
            offset += in_size;
        }
        layer_weights.push(weights);

        // Extract biases
        let biases: Vec<f32> = flat[offset..offset + out_size].to_vec();
        layer_biases.push(biases);
        offset += out_size;
    }

    (layer_weights, layer_biases)
}

/// Generate random weights for an architecture
pub fn random_weights(architecture: &[usize], seed: u64) -> Vec<f32> {
    use rand::{rngs::StdRng, Rng, SeedableRng};

    let mut rng = StdRng::seed_from_u64(seed);
    let count = weight_count(architecture);

    // Xavier initialization
    let mut weights = Vec::with_capacity(count);
    let mut offset = 0;

    for i in 0..architecture.len() - 1 {
        let in_size = architecture[i];
        let out_size = architecture[i + 1];

        // Xavier scale
        let scale = (2.0 / (in_size + out_size) as f32).sqrt();

        // Weights
        for _ in 0..in_size * out_size {
            weights.push(rng.gen_range(-scale..scale));
        }

        // Biases (zero init)
        for _ in 0..out_size {
            weights.push(0.0);
        }

        offset += in_size * out_size + out_size;
    }

    weights
}

/// Tensorize a NEAT genome into fixed-size representation
///
/// For variable-topology NEAT, we:
/// 1. Map node innovations to fixed indices
/// 2. Pad missing connections with zeros
/// 3. Create dense weight matrix
pub fn tensorize_neat_genome(
    connections: &[(usize, usize, f32, bool)],  // (in_node, out_node, weight, enabled)
    max_nodes: usize,
) -> Vec<f32> {
    // Create adjacency matrix
    let mut matrix = vec![0.0f32; max_nodes * max_nodes];

    for &(in_node, out_node, weight, enabled) in connections {
        if enabled && in_node < max_nodes && out_node < max_nodes {
            matrix[out_node * max_nodes + in_node] = weight;
        }
    }

    matrix
}

/// Detensorize back to NEAT-style connections
pub fn detensorize_to_connections(
    matrix: &[f32],
    max_nodes: usize,
    threshold: f32,
) -> Vec<(usize, usize, f32, bool)> {
    let mut connections = Vec::new();

    for out_node in 0..max_nodes {
        for in_node in 0..max_nodes {
            let weight = matrix[out_node * max_nodes + in_node];
            if weight.abs() > threshold {
                connections.push((in_node, out_node, weight, true));
            }
        }
    }

    connections
}

/// Batch tensorize multiple genomes
pub fn batch_tensorize(
    genomes: &[Vec<(usize, usize, f32, bool)>],
    max_nodes: usize,
) -> Vec<Vec<f32>> {
    genomes
        .iter()
        .map(|g| tensorize_neat_genome(g, max_nodes))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_weight_count() {
        // 8 -> 32 -> 16 -> 3
        let arch = vec![8, 32, 16, 3];
        let count = weight_count(&arch);
        // 8*32 + 32 + 32*16 + 16 + 16*3 + 3 = 256+32+512+16+48+3 = 867
        assert_eq!(count, 867);
    }

    #[test]
    fn test_flatten_unflatten() {
        let arch = vec![2, 3, 1];
        let weights = random_weights(&arch, 42);

        let (layer_w, layer_b) = unflatten_weights(&weights, &arch);
        let reflattened = flatten_weights(&layer_w, &layer_b);

        assert_eq!(weights.len(), reflattened.len());
        for (a, b) in weights.iter().zip(reflattened.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_tensorize_neat() {
        let connections = vec![
            (0, 2, 0.5, true),
            (1, 2, -0.3, true),
            (0, 1, 0.1, false),  // disabled
        ];

        let matrix = tensorize_neat_genome(&connections, 3);

        assert_eq!(matrix.len(), 9);
        // Check that enabled connections are present
        assert!((matrix[2 * 3 + 0] - 0.5).abs() < 1e-6);  // 0->2
        assert!((matrix[2 * 3 + 1] - (-0.3)).abs() < 1e-6);  // 1->2
        // Disabled connection should be zero
        assert!((matrix[1 * 3 + 0] - 0.0).abs() < 1e-6);  // 0->1 disabled
    }
}
