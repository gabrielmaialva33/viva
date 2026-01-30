//! VIVA Burn - GPU-accelerated Neural Network Operations
//!
//! Batch forward pass for 1000+ neural networks in parallel.
//! Uses burn-rs with CUDA backend for RTX 4090 optimization.
//!
//! Architecture targets:
//! - 16,384 CUDA Cores
//! - 512 Tensor Cores (4th Gen)
//! - 24GB GDDR6X VRAM
//! - i9-13900K (24 cores)

use rayon::prelude::*;
use rustler::NifResult;

mod batch_forward;
mod batch_physics;
mod tensorize;
mod tensor_physics;
mod cma_es;
mod cuda_kernels;

// Re-export for NIF - common functions
use batch_forward::{
    batch_dense_forward, batch_network_forward,
    perturb_weights_auto, nes_gradient_auto, update_weights_auto,
    benchmark_nes,
};

// GPU-specific imports (CUDA only)
#[cfg(feature = "cuda")]
use batch_forward::{
    perturb_weights_gpu, nes_gradient_gpu, update_weights_gpu,
    batch_update_weights_gpu,
};

// =============================================================================
// BACKEND SELECTION
// =============================================================================

// Type alias - the actual backend is selected in batch_forward.rs
#[cfg(feature = "cuda")]
use burn_cuda::Cuda;

#[cfg(feature = "cuda")]
type B = Cuda<f32, i32>;

#[cfg(not(feature = "cuda"))]
use burn_ndarray::NdArray;

#[cfg(not(feature = "cuda"))]
type B = NdArray;

// Explicit NIF registration for CUDA build (39 NIFs)
#[cfg(feature = "cuda")]
rustler::init!(
    "Elixir.Viva.Burn.Native",
    [
        burn_check,
        burn_batch_forward,
        burn_batch_dense,
        // NES auto-select
        burn_perturb_weights,
        burn_nes_gradient,
        burn_update_weights,
        burn_nes_step,
        // NES GPU-explicit
        burn_perturb_weights_gpu,
        burn_nes_gradient_gpu,
        burn_update_weights_gpu,
        burn_batch_update_weights_gpu,
        // Activations
        burn_batch_sigmoid,
        burn_batch_tanh,
        burn_batch_relu,
        // Benchmarks
        burn_benchmark,
        burn_benchmark_nes,
        // Batch Physics (CPU Rayon)
        burn_batch_physics_simulate,
        burn_batch_physics_simulate_with_spin,
        burn_batch_physics_create_tables,
        burn_batch_physics_calculate_fitness,
        burn_batch_physics_benchmark,
        // Multi-shot Episodes
        burn_batch_simulate_episodes,
        burn_batch_evaluate_episodes,
        // Tensor Physics (GPU)
        burn_tensor_physics_simulate,
        burn_tensor_physics_benchmark,
        // Custom CUDA Kernels (NEW - Ultra Performance)
        burn_cuda_kernels_benchmark,
        burn_cuda_fused_forward,
        burn_cuda_physics_simulate,
        burn_cuda_parallel_fitness,
        // CMA-ES
        burn_cma_es_init,
        burn_cma_es_init_auto,
        burn_cma_es_sample,
        burn_cma_es_update,
        burn_cma_es_step,
        burn_cma_es_get_mean,
        burn_cma_es_get_sigma,
        burn_cma_es_get_diagnostics,
        burn_cma_es_benchmark,
    ]
);

// Explicit NIF registration for CPU-only build (27 NIFs)
#[cfg(not(feature = "cuda"))]
rustler::init!(
    "Elixir.Viva.Burn.Native",
    [
        burn_check,
        burn_batch_forward,
        burn_batch_dense,
        burn_perturb_weights,
        burn_nes_gradient,
        burn_update_weights,
        burn_nes_step,
        burn_batch_sigmoid,
        burn_batch_tanh,
        burn_batch_relu,
        burn_benchmark,
        burn_benchmark_nes,
        // Batch Physics
        burn_batch_physics_simulate,
        burn_batch_physics_simulate_with_spin,
        burn_batch_physics_create_tables,
        burn_batch_physics_calculate_fitness,
        burn_batch_physics_benchmark,
        // Multi-shot Episodes
        burn_batch_simulate_episodes,
        burn_batch_evaluate_episodes,
        // CMA-ES
        burn_cma_es_init,
        burn_cma_es_init_auto,
        burn_cma_es_sample,
        burn_cma_es_update,
        burn_cma_es_step,
        burn_cma_es_get_mean,
        burn_cma_es_get_sigma,
        burn_cma_es_get_diagnostics,
        burn_cma_es_benchmark,
    ]
);

// =============================================================================
// NIF FUNCTIONS
// =============================================================================

/// Check GPU/Backend status
#[rustler::nif]
fn burn_check() -> String {
    let threads = rayon::current_num_threads();

    #[cfg(feature = "cuda")]
    {
        use burn_cuda::CudaDevice;

        // Try to initialize CUDA
        match std::panic::catch_unwind(|| {
            let _device = CudaDevice::default();
            "RTX 4090 Ready"
        }) {
            Ok(status) => format!(
                "VIVA_BURN_OK (Backend: CUDA {}, Rayon: {} threads)",
                status, threads
            ),
            Err(_) => format!(
                "VIVA_BURN_WARN (CUDA init failed - check LD_LIBRARY_PATH=/usr/lib/wsl/lib, Rayon: {} threads)",
                threads
            ),
        }
    }

    #[cfg(not(feature = "cuda"))]
    {
        format!(
            "VIVA_BURN_OK (Backend: NdArray CPU, Rayon: {} threads)",
            threads
        )
    }
}

/// Batch forward pass for multiple networks
///
/// weights: List of flattened weight vectors [pop_size][weight_count]
/// inputs: List of input vectors [pop_size][input_size]
/// architecture: [input_size, hidden1, hidden2, ..., output_size]
///
/// Returns: List of output vectors [pop_size][output_size]
#[rustler::nif(schedule = "DirtyCpu")]
fn burn_batch_forward(
    weights_list: Vec<Vec<f32>>,
    inputs_list: Vec<Vec<f32>>,
    architecture: Vec<usize>,
) -> NifResult<Vec<Vec<f32>>> {
    if weights_list.is_empty() || inputs_list.is_empty() {
        return Ok(vec![]);
    }

    let pop_size = weights_list.len();
    if pop_size != inputs_list.len() {
        return Err(rustler::Error::Term(Box::new(
            "weights and inputs must have same length",
        )));
    }

    // Process in parallel using Rayon (works for both CPU and as CUDA prep)
    let results: Vec<Vec<f32>> = weights_list
        .par_iter()
        .zip(inputs_list.par_iter())
        .map(|(weights, inputs)| {
            batch_network_forward::<B>(weights, inputs, &architecture)
        })
        .collect();

    Ok(results)
}

/// Batch forward for fixed-topology dense network
/// More efficient when all networks have same architecture
///
/// weights_batch: [pop_size, total_weights] flattened
/// inputs_batch: [pop_size, input_size]
/// layer_sizes: [input, h1, h2, ..., output]
#[rustler::nif(schedule = "DirtyCpu")]
fn burn_batch_dense(
    weights_batch: Vec<Vec<f32>>,
    inputs_batch: Vec<Vec<f32>>,
    layer_sizes: Vec<usize>,
) -> NifResult<Vec<Vec<f32>>> {
    let results = batch_dense_forward::<B>(&weights_batch, &inputs_batch, &layer_sizes);
    Ok(results)
}

// =============================================================================
// NES OPERATIONS (AUTO-SELECT GPU/CPU)
// =============================================================================

/// Generate random weight perturbations for NES (auto-selects GPU/CPU)
///
/// base_weights: Original weights [weight_count]
/// num_perturbations: How many perturbed copies (typically 8-16)
/// std_dev: Standard deviation of Gaussian noise
/// seed: Random seed
///
/// Returns: List of perturbed weight vectors
///
/// GPU: Uses cuRAND-backed random generation on RTX 4090
/// CPU: Falls back to Rayon parallel with rand crate
#[rustler::nif(schedule = "DirtyCpu")]
fn burn_perturb_weights(
    base_weights: Vec<f32>,
    num_perturbations: usize,
    std_dev: f32,
    seed: u64,
) -> Vec<Vec<f32>> {
    perturb_weights_auto(&base_weights, num_perturbations, std_dev, seed)
}

/// GPU-only perturbation generation (explicit)
#[cfg(feature = "cuda")]
#[rustler::nif(schedule = "DirtyCpu")]
fn burn_perturb_weights_gpu(
    base_weights: Vec<f32>,
    num_perturbations: usize,
    std_dev: f32,
    _seed: u64,  // GPU uses device RNG
) -> Vec<Vec<f32>> {
    perturb_weights_gpu(&base_weights, num_perturbations, std_dev, _seed)
}

/// Compute NES gradient from evaluations (auto-selects GPU/CPU)
///
/// perturbations: List of perturbation vectors [n_pert][weight_count]
/// fitnesses: Fitness of each perturbation [n_pert]
/// std_dev: Standard deviation used for perturbations
///
/// Returns: Gradient vector [weight_count]
///
/// GPU: Uses GEMM with Tensor Core acceleration
///      gradient = advantages^T @ perturbations / (n * sigma^2)
/// CPU: Falls back to Rayon parallel computation
#[rustler::nif(schedule = "DirtyCpu")]
fn burn_nes_gradient(
    perturbations: Vec<Vec<f32>>,
    fitnesses: Vec<f32>,
    std_dev: f32,
) -> Vec<f32> {
    nes_gradient_auto(&perturbations, &fitnesses, std_dev)
}

/// GPU-only NES gradient computation (explicit)
#[cfg(feature = "cuda")]
#[rustler::nif(schedule = "DirtyCpu")]
fn burn_nes_gradient_gpu(
    perturbations: Vec<Vec<f32>>,
    fitnesses: Vec<f32>,
    std_dev: f32,
) -> Vec<f32> {
    nes_gradient_gpu(&perturbations, &fitnesses, std_dev)
}

/// Apply gradient update to weights (auto-selects GPU/CPU)
#[rustler::nif]
fn burn_update_weights(
    weights: Vec<f32>,
    gradient: Vec<f32>,
    learning_rate: f32,
) -> Vec<f32> {
    update_weights_auto(&weights, &gradient, learning_rate)
}

/// GPU-only weight update (explicit)
#[cfg(feature = "cuda")]
#[rustler::nif]
fn burn_update_weights_gpu(
    weights: Vec<f32>,
    gradient: Vec<f32>,
    learning_rate: f32,
) -> Vec<f32> {
    update_weights_gpu(&weights, &gradient, learning_rate)
}

/// Batch weight updates for multiple networks on GPU
#[cfg(feature = "cuda")]
#[rustler::nif(schedule = "DirtyCpu")]
fn burn_batch_update_weights_gpu(
    weights_batch: Vec<Vec<f32>>,
    gradients_batch: Vec<Vec<f32>>,
    learning_rate: f32,
) -> Vec<Vec<f32>> {
    batch_update_weights_gpu(&weights_batch, &gradients_batch, learning_rate)
}

/// Complete NES step (perturb -> evaluate -> gradient -> update) with GPU auto-select
///
/// This is more efficient than calling individual functions as it reduces
/// CPU-GPU transfer overhead.
#[rustler::nif(schedule = "DirtyCpu")]
fn burn_nes_step(
    base_weights: Vec<f32>,
    perturbations: Vec<Vec<f32>>,
    fitnesses: Vec<f32>,
    std_dev: f32,
    learning_rate: f32,
) -> Vec<f32> {
    let gradient = nes_gradient_auto(&perturbations, &fitnesses, std_dev);
    update_weights_auto(&base_weights, &gradient, learning_rate)
}

/// Benchmark NES operations (compares GPU vs CPU)
#[rustler::nif(schedule = "DirtyCpu")]
fn burn_benchmark_nes(
    weight_count: usize,
    num_perturbations: usize,
    iterations: usize,
) -> String {
    benchmark_nes(weight_count, num_perturbations, iterations)
}

/// Batch sigmoid activation
#[rustler::nif]
fn burn_batch_sigmoid(values: Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    values
        .par_iter()
        .map(|v| v.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect())
        .collect()
}

/// Batch tanh activation
#[rustler::nif]
fn burn_batch_tanh(values: Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    values
        .par_iter()
        .map(|v| v.iter().map(|&x| x.tanh()).collect())
        .collect()
}

/// Batch ReLU activation
#[rustler::nif]
fn burn_batch_relu(values: Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    values
        .par_iter()
        .map(|v| v.iter().map(|&x| x.max(0.0)).collect())
        .collect()
}

// =============================================================================
// BENCHMARK
// =============================================================================

/// Benchmark batch forward performance
#[rustler::nif(schedule = "DirtyCpu")]
fn burn_benchmark(
    pop_size: usize,
    input_size: usize,
    hidden_size: usize,
    output_size: usize,
    iterations: usize,
) -> String {
    use std::time::Instant;

    // Generate random data
    let architecture = vec![input_size, hidden_size, output_size];
    let weight_count = input_size * hidden_size + hidden_size + hidden_size * output_size + output_size;

    let weights_batch: Vec<Vec<f32>> = (0..pop_size)
        .map(|i| (0..weight_count).map(|j| ((i * j) % 100) as f32 / 100.0 - 0.5).collect())
        .collect();

    let inputs_batch: Vec<Vec<f32>> = (0..pop_size)
        .map(|i| (0..input_size).map(|j| ((i + j) % 100) as f32 / 100.0).collect())
        .collect();

    // Warmup
    let _ = batch_dense_forward::<B>(&weights_batch, &inputs_batch, &architecture);

    // Benchmark
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = batch_dense_forward::<B>(&weights_batch, &inputs_batch, &architecture);
    }
    let elapsed = start.elapsed();

    let total_forwards = pop_size * iterations;
    let forwards_per_sec = total_forwards as f64 / elapsed.as_secs_f64();
    let us_per_forward = elapsed.as_micros() as f64 / total_forwards as f64;

    #[cfg(feature = "cuda")]
    let backend = "CUDA RTX 4090";

    #[cfg(not(feature = "cuda"))]
    let backend = "NdArray CPU";

    format!(
        "Batch Forward Benchmark ({}):\n\
         - Population: {}\n\
         - Architecture: {:?}\n\
         - Iterations: {}\n\
         - Total time: {:.2}ms\n\
         - Forwards/sec: {:.0}\n\
         - μs/forward: {:.2}",
        backend,
        pop_size, architecture, iterations,
        elapsed.as_secs_f64() * 1000.0,
        forwards_per_sec,
        us_per_forward
    )
}

// =============================================================================
// BATCH PHYSICS SIMULATION
// =============================================================================

/// Batch physics simulation for sinuca tables
///
/// Simulates multiple billiards tables in parallel.
/// Uses Rayon for CPU parallelism (GPU tensor version planned).
///
/// Arguments:
/// - positions_x: [batch, 8] - ball X positions
/// - positions_z: [batch, 8] - ball Z positions
/// - velocities_x: [batch, 8] - ball X velocities
/// - velocities_z: [batch, 8] - ball Z velocities
/// - pocketed: [batch, 8] - pocketed flags (0 or 1)
/// - shots: [batch, 4] - shots as (angle, power, english, elevation) tuples
/// - max_steps: maximum simulation steps
///
/// Returns: (final_pos_x, final_pos_z, final_pocketed, steps_taken)
#[rustler::nif(schedule = "DirtyCpu")]
fn burn_batch_physics_simulate(
    positions_x: Vec<Vec<f32>>,
    positions_z: Vec<Vec<f32>>,
    velocities_x: Vec<Vec<f32>>,
    velocities_z: Vec<Vec<f32>>,
    pocketed: Vec<Vec<f32>>,
    shots: Vec<Vec<f32>>,  // Each shot is [angle, power, english, elevation]
    max_steps: usize,
) -> NifResult<(Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<usize>)> {
    // Convert shots from Vec<Vec<f32>> to Vec<(f32, f32, f32, f32)>
    let shots_tuples: Vec<(f32, f32, f32, f32)> = shots
        .iter()
        .map(|s| {
            if s.len() >= 4 {
                (s[0], s[1], s[2], s[3])
            } else if s.len() >= 2 {
                (s[0], s[1], 0.0, 0.0)
            } else {
                (0.0, 0.5, 0.0, 0.0)
            }
        })
        .collect();

    let (final_px, final_pz, final_pocketed, steps) = batch_physics::batch_simulate_cpu(
        positions_x,
        positions_z,
        velocities_x,
        velocities_z,
        pocketed,
        shots_tuples,
        max_steps,
    );

    Ok((final_px, final_pz, final_pocketed, steps))
}

/// Batch physics simulation WITH SPIN PHYSICS
///
/// This version uses english and elevation to apply realistic spin:
/// - English: side spin that causes curved ball paths (Magnus effect)
/// - Elevation: top/back spin that affects ball behavior after contact
///
/// Simulates multiple billiards tables in parallel with full spin physics.
///
/// Arguments:
/// - positions_x: [batch, 8] - ball X positions
/// - positions_z: [batch, 8] - ball Z positions
/// - velocities_x: [batch, 8] - ball X velocities
/// - velocities_z: [batch, 8] - ball Z velocities
/// - pocketed: [batch, 8] - pocketed flags (0 or 1)
/// - shots: [batch, 4] - shots as [angle, power, english, elevation]
/// - max_steps: maximum simulation steps
///
/// Returns: (final_pos_x, final_pos_z, final_pocketed, steps_taken)
#[rustler::nif(schedule = "DirtyCpu")]
fn burn_batch_physics_simulate_with_spin(
    positions_x: Vec<Vec<f32>>,
    positions_z: Vec<Vec<f32>>,
    velocities_x: Vec<Vec<f32>>,
    velocities_z: Vec<Vec<f32>>,
    pocketed: Vec<Vec<f32>>,
    shots: Vec<Vec<f32>>,  // Each shot is [angle, power, english, elevation]
    max_steps: usize,
) -> NifResult<(Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<usize>)> {
    // Convert shots from Vec<Vec<f32>> to Vec<(f32, f32, f32, f32)>
    let shots_tuples: Vec<(f32, f32, f32, f32)> = shots
        .iter()
        .map(|s| {
            if s.len() >= 4 {
                (s[0], s[1], s[2], s[3])
            } else if s.len() >= 2 {
                (s[0], s[1], 0.0, 0.0)
            } else {
                (0.0, 0.5, 0.0, 0.0)
            }
        })
        .collect();

    let (final_px, final_pz, final_pocketed, steps) = batch_physics::batch_simulate_cpu_with_spin(
        positions_x,
        positions_z,
        velocities_x,
        velocities_z,
        pocketed,
        shots_tuples,
        max_steps,
    );

    Ok((final_px, final_pz, final_pocketed, steps))
}

/// Create initial state for a batch of sinuca tables
///
/// Returns: (positions_x, positions_z, velocities_x, velocities_z, pocketed)
/// All shapes are [batch_size, 8]
#[rustler::nif]
fn burn_batch_physics_create_tables(batch_size: usize) -> (
    Vec<Vec<f32>>,  // positions_x
    Vec<Vec<f32>>,  // positions_z
    Vec<Vec<f32>>,  // velocities_x
    Vec<Vec<f32>>,  // velocities_z
    Vec<Vec<f32>>,  // pocketed
) {
    batch_physics::create_initial_batch(batch_size)
}

/// Calculate fitness for batch simulation results
///
/// Arguments:
/// - batch_size: number of simulations
/// - initial_pocketed: [batch, 8] initial pocketed flags
/// - final_pocketed: [batch, 8] final pocketed flags
/// - final_pos_x: [batch, 8] final X positions
/// - final_pos_z: [batch, 8] final Z positions
/// - initial_pos_x: [batch, 8] initial X positions
/// - initial_pos_z: [batch, 8] initial Z positions
/// - target_ball_idx: index of target ball (1 for Red in sinuca)
///
/// Returns: Vec<(fitness, hit_angle, scatter_ratio)>
#[rustler::nif(schedule = "DirtyCpu")]
fn burn_batch_physics_calculate_fitness(
    batch_size: usize,
    initial_pocketed: Vec<Vec<f32>>,
    final_pocketed: Vec<Vec<f32>>,
    final_pos_x: Vec<Vec<f32>>,
    final_pos_z: Vec<Vec<f32>>,
    initial_pos_x: Vec<Vec<f32>>,
    initial_pos_z: Vec<Vec<f32>>,
    target_ball_idx: usize,
) -> Vec<(f32, f32, f32)> {
    batch_physics::batch_calculate_fitness(
        batch_size,
        &initial_pocketed,
        &final_pocketed,
        &final_pos_x,
        &final_pos_z,
        &initial_pos_x,
        &initial_pos_z,
        target_ball_idx,
    )
}

/// Benchmark batch physics simulation
///
/// Compares sequential vs parallel simulation performance.
#[rustler::nif(schedule = "DirtyCpu")]
fn burn_batch_physics_benchmark(
    batch_size: usize,
    max_steps: usize,
    iterations: usize,
) -> String {
    use std::time::Instant;

    // Create initial batch
    let (px, pz, vx, vz, pocketed) = batch_physics::create_initial_batch(batch_size);

    // Generate random shots
    let shots: Vec<(f32, f32, f32, f32)> = (0..batch_size)
        .map(|i| {
            let angle = (i as f32 / batch_size as f32) * std::f32::consts::PI * 2.0;
            (angle, 0.5 + (i % 10) as f32 * 0.05, 0.0, 0.0)
        })
        .collect();

    // Warmup
    let _ = batch_physics::batch_simulate_cpu(
        px.clone(), pz.clone(), vx.clone(), vz.clone(), pocketed.clone(),
        shots.clone(), max_steps
    );

    // Benchmark
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = batch_physics::batch_simulate_cpu(
            px.clone(), pz.clone(), vx.clone(), vz.clone(), pocketed.clone(),
            shots.clone(), max_steps
        );
    }
    let elapsed = start.elapsed();

    let total_sims = batch_size * iterations;
    let sims_per_sec = total_sims as f64 / elapsed.as_secs_f64();
    let ms_per_batch = elapsed.as_secs_f64() * 1000.0 / iterations as f64;

    #[cfg(feature = "cuda")]
    let backend = "Rayon (CUDA batch planned)";

    #[cfg(not(feature = "cuda"))]
    let backend = "Rayon CPU";

    format!(
        "Batch Physics Benchmark ({}):\n\
         - Batch size: {}\n\
         - Max steps: {}\n\
         - Iterations: {}\n\
         - Total simulations: {}\n\
         - Total time: {:.2}ms\n\
         - Simulations/sec: {:.0}\n\
         - ms/batch: {:.2}\n\
         - us/simulation: {:.2}",
        backend,
        batch_size, max_steps, iterations, total_sims,
        elapsed.as_secs_f64() * 1000.0,
        sims_per_sec,
        ms_per_batch,
        elapsed.as_micros() as f64 / total_sims as f64
    )
}

// =============================================================================
// MULTI-SHOT EPISODE SIMULATION (KEY OPTIMIZATION)
// =============================================================================

/// Simulate complete multi-shot episodes for a population of neural networks
///
/// This is THE KEY OPTIMIZATION: reduces 4800 NIF calls per generation to 1.
///
/// Arguments:
/// - population_weights: [pop_size, weight_count] - all network weights
/// - architecture: [input, h1, h2, ..., output] - network layer sizes
/// - shots_per_episode: number of shots to simulate per episode
/// - max_steps_per_shot: max physics steps per shot
///
/// Returns: Vec of episode results as nested tuples (Rustler limit is 5-tuples)
/// Structure: ((fitness, shots_taken, balls_pocketed, hit_angle, scatter), (pos_x, pos_z, pocketed))
#[rustler::nif(schedule = "DirtyCpu")]
fn burn_batch_simulate_episodes(
    population_weights: Vec<Vec<f32>>,
    architecture: Vec<usize>,
    shots_per_episode: usize,
    max_steps_per_shot: usize,
) -> Vec<((f32, usize, usize, f32, f32), (Vec<f32>, Vec<f32>, Vec<f32>))> {
    let results = batch_physics::batch_simulate_episodes(
        &population_weights,
        &architecture,
        shots_per_episode,
        max_steps_per_shot,
    );

    // Convert EpisodeResult to nested NIF-friendly tuples (5 + 3 = 8 elements)
    results
        .into_iter()
        .map(|r| (
            (r.total_fitness, r.shots_taken, r.balls_pocketed, r.avg_hit_angle, r.avg_scatter_ratio),
            (r.final_pos_x, r.final_pos_z, r.final_pocketed),
        ))
        .collect()
}

/// Evaluate complete episodes and return only fitness + behavior
///
/// Simplified API for QD training - returns just what's needed for MAP-Elites.
///
/// Arguments:
/// - population_weights: [pop_size, weight_count] - all network weights
/// - architecture: [input, h1, h2, ..., output] - network layer sizes
/// - shots_per_episode: number of shots to simulate per episode
/// - max_steps_per_shot: max physics steps per shot
///
/// Returns: Vec<(fitness, hit_angle, scatter_ratio)>
#[rustler::nif(schedule = "DirtyCpu")]
fn burn_batch_evaluate_episodes(
    population_weights: Vec<Vec<f32>>,
    architecture: Vec<usize>,
    shots_per_episode: usize,
    max_steps_per_shot: usize,
) -> Vec<(f32, f32, f32)> {
    batch_physics::batch_evaluate_episodes(
        &population_weights,
        &architecture,
        shots_per_episode,
        max_steps_per_shot,
    )
}

// =============================================================================
// TENSOR PHYSICS SIMULATION (GPU)
// =============================================================================

/// GPU Tensor-based batch physics simulation
///
/// Uses pure tensor operations for maximum GPU utilization.
/// All collision detection and physics run on RTX 4090.
///
/// Arguments:
/// - positions_x: [batch * 8] flattened ball X positions
/// - positions_z: [batch * 8] flattened ball Z positions
/// - velocities_x: [batch * 8] flattened ball X velocities
/// - velocities_z: [batch * 8] flattened ball Z velocities
/// - pocketed: [batch * 8] flattened pocketed flags
/// - shots: [batch * 4] flattened shots (angle, power, english, elevation)
/// - batch_size: number of simulations
/// - max_steps: maximum physics steps
///
/// Returns: (final_pos_x, final_pos_z, final_pocketed, steps_taken)
#[cfg(feature = "cuda")]
#[rustler::nif(schedule = "DirtyCpu")]
fn burn_tensor_physics_simulate(
    positions_x: Vec<Vec<f32>>,
    positions_z: Vec<Vec<f32>>,
    velocities_x: Vec<Vec<f32>>,
    velocities_z: Vec<Vec<f32>>,
    pocketed: Vec<Vec<f32>>,
    shots: Vec<Vec<f32>>,
    max_steps: usize,
) -> NifResult<(Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<usize>)> {
    // Convert shots from Vec<Vec<f32>> to Vec<(f32, f32, f32, f32)>
    let shots_tuples: Vec<(f32, f32, f32, f32)> = shots
        .iter()
        .map(|s| {
            if s.len() >= 4 {
                (s[0], s[1], s[2], s[3])
            } else if s.len() >= 2 {
                (s[0], s[1], 0.0, 0.0)
            } else {
                (0.0, 0.5, 0.0, 0.0)
            }
        })
        .collect();

    let (final_px, final_pz, final_pocketed, steps) = batch_physics::batch_simulate_gpu_fast(
        positions_x,
        positions_z,
        velocities_x,
        velocities_z,
        pocketed,
        shots_tuples,
        max_steps,
    );

    Ok((final_px, final_pz, final_pocketed, steps))
}

/// Benchmark tensor physics (GPU) vs CPU physics
///
/// Compares:
/// - GPU: Pure tensor operations using Burn on RTX 4090
/// - CPU: Rayon parallel with scalar loops
#[cfg(feature = "cuda")]
#[rustler::nif(schedule = "DirtyCpu")]
fn burn_tensor_physics_benchmark(
    batch_size: usize,
    max_steps: usize,
    iterations: usize,
) -> String {
    tensor_physics::benchmark_tensor_physics(batch_size, max_steps, iterations)
}

// =============================================================================
// CMA-ES - COVARIANCE MATRIX ADAPTATION EVOLUTION STRATEGY
// =============================================================================

use std::sync::{Arc, Mutex};
use rustler::ResourceArc;

/// CMA-ES state wrapped for NIF usage
struct CmaEsResource {
    state: Mutex<cma_es::CmaEsState>,
}

/// Initialize CMA-ES optimizer with specified lambda
///
/// Arguments:
/// - initial_mean: starting point in search space [n]
/// - initial_sigma: initial step size (0.3 recommended for NN weights)
/// - lambda: population size
///
/// Returns: Resource handle to CMA-ES state
#[rustler::nif]
fn burn_cma_es_init(
    initial_mean: Vec<f32>,
    initial_sigma: f32,
    lambda: usize,
) -> ResourceArc<CmaEsResource> {
    let state = cma_es::CmaEsState::with_population(initial_mean, initial_sigma, lambda);

    ResourceArc::new(CmaEsResource {
        state: Mutex::new(state),
    })
}

/// Initialize CMA-ES optimizer with auto lambda (4 + 3*ln(n))
///
/// Arguments:
/// - initial_mean: starting point in search space [n]
/// - initial_sigma: initial step size (0.3 recommended for NN weights)
///
/// Returns: Resource handle to CMA-ES state
#[rustler::nif]
fn burn_cma_es_init_auto(
    initial_mean: Vec<f32>,
    initial_sigma: f32,
) -> ResourceArc<CmaEsResource> {
    let state = cma_es::CmaEsState::new(initial_mean, initial_sigma);

    ResourceArc::new(CmaEsResource {
        state: Mutex::new(state),
    })
}

/// Sample new population from CMA-ES distribution
///
/// Arguments:
/// - cma_state: Resource handle to CMA-ES state
/// - seed: Random seed for reproducibility
///
/// Returns: [lambda, n] population of candidate solutions
#[rustler::nif(schedule = "DirtyCpu")]
fn burn_cma_es_sample(
    cma_state: ResourceArc<CmaEsResource>,
    seed: u64,
) -> Vec<Vec<f32>> {
    let state = cma_state.state.lock().unwrap();
    cma_es::sample_population(&state, seed)
}

/// Update CMA-ES state with fitness evaluations
///
/// Arguments:
/// - cma_state: Resource handle to CMA-ES state
/// - population: [lambda, n] candidate solutions from sample
/// - fitnesses: [lambda] fitness values (higher = better)
///
/// Returns: true on success
#[rustler::nif(schedule = "DirtyCpu")]
fn burn_cma_es_update(
    cma_state: ResourceArc<CmaEsResource>,
    population: Vec<Vec<f32>>,
    fitnesses: Vec<f32>,
) -> bool {
    let mut state = cma_state.state.lock().unwrap();
    cma_es::update_state(&mut state, &population, &fitnesses);
    true
}

/// Complete CMA-ES step: sample -> (external eval) -> update
///
/// If population and fitnesses are provided, updates first then samples.
/// If empty, just samples from current distribution.
///
/// Arguments:
/// - cma_state: Resource handle to CMA-ES state
/// - population: Previous population (can be empty for first call)
/// - fitnesses: Previous fitnesses (can be empty for first call)
/// - seed: Random seed
///
/// Returns: New population to evaluate [lambda, n]
#[rustler::nif(schedule = "DirtyCpu")]
fn burn_cma_es_step(
    cma_state: ResourceArc<CmaEsResource>,
    population: Vec<Vec<f32>>,
    fitnesses: Vec<f32>,
    seed: u64,
) -> Vec<Vec<f32>> {
    let mut state = cma_state.state.lock().unwrap();
    cma_es::cma_es_step(&mut state, &fitnesses, &population, seed)
}

/// Get current mean (best estimate) from CMA-ES
///
/// Returns: [n] current mean vector
#[rustler::nif]
fn burn_cma_es_get_mean(cma_state: ResourceArc<CmaEsResource>) -> Vec<f32> {
    let state = cma_state.state.lock().unwrap();
    state.mean.clone()
}

/// Get current step size (sigma) from CMA-ES
///
/// Returns: current sigma value
#[rustler::nif]
fn burn_cma_es_get_sigma(cma_state: ResourceArc<CmaEsResource>) -> f32 {
    let state = cma_state.state.lock().unwrap();
    state.sigma
}

/// Get CMA-ES convergence diagnostics
///
/// Returns: (sigma, condition_number, normalized_ps_norm)
/// - sigma: current step size
/// - condition_number: ratio of largest/smallest eigenvalue (high = ill-conditioned)
/// - normalized_ps_norm: ||p_sigma|| / chi_n (should be ~1 when adapted)
#[rustler::nif]
fn burn_cma_es_get_diagnostics(cma_state: ResourceArc<CmaEsResource>) -> (f32, f32, f32) {
    let state = cma_state.state.lock().unwrap();
    cma_es::get_diagnostics(&state)
}

/// Benchmark CMA-ES operations
///
/// Arguments:
/// - n: dimension of search space
/// - lambda: population size
/// - iterations: number of iterations to benchmark
///
/// Returns: Benchmark report string
#[rustler::nif(schedule = "DirtyCpu")]
fn burn_cma_es_benchmark(
    n: usize,
    lambda: usize,
    iterations: usize,
) -> String {
    cma_es::benchmark_cma_es(n, lambda, iterations)
}

// Resource registration for CMA-ES
impl rustler::Resource for CmaEsResource {}

// =============================================================================
// CUSTOM CUDA KERNELS - ULTRA PERFORMANCE
// =============================================================================

/// Benchmark custom CUDA kernels (neural + physics)
///
/// This tests the hand-tuned CUDA kernels for RTX 4090:
/// - Fused neural forward pass (shared memory + fast tanh)
/// - Batched collision detection (warp-level primitives)
/// - Parallel fitness calculation
///
/// Expected performance: 3-5x speedup over burn-rs generic
#[cfg(feature = "cuda")]
#[rustler::nif(schedule = "DirtyCpu")]
fn burn_cuda_kernels_benchmark(
    batch_size: usize,
    iterations: usize,
) -> String {
    cuda_kernels::benchmark_cuda_kernels(batch_size, iterations)
}

#[cfg(not(feature = "cuda"))]
#[rustler::nif]
fn burn_cuda_kernels_benchmark(_batch_size: usize, _iterations: usize) -> String {
    "CUDA not available - compile with --features cuda".to_string()
}

/// Fused neural network forward pass using custom CUDA kernel
///
/// Single kernel launch for entire forward pass:
/// output = tanh(W2 @ tanh(W1 @ input + b1) + b2)
///
/// Arguments:
/// - weights_flat: [pop_size * total_weights] flattened
///   Layout per network: [W1, b1, W2, b2]
/// - inputs_flat: [pop_size * input_size]
/// - pop_size: number of networks
/// - input_size: input layer size
/// - hidden_size: hidden layer size
/// - output_size: output layer size
///
/// Returns: [pop_size * output_size] flattened outputs
#[cfg(feature = "cuda")]
#[rustler::nif(schedule = "DirtyCpu")]
fn burn_cuda_fused_forward(
    weights_flat: Vec<f32>,
    inputs_flat: Vec<f32>,
    pop_size: usize,
    input_size: usize,
    hidden_size: usize,
    output_size: usize,
) -> NifResult<Vec<f32>> {
    match cuda_kernels::CudaKernelManager::new() {
        Ok(kernels) => {
            kernels.fused_forward(
                &weights_flat, &inputs_flat,
                pop_size, input_size, hidden_size, output_size
            ).map_err(|e| rustler::Error::Term(Box::new(e)))
        }
        Err(e) => Err(rustler::Error::Term(Box::new(e)))
    }
}

#[cfg(not(feature = "cuda"))]
#[rustler::nif]
fn burn_cuda_fused_forward(
    _weights_flat: Vec<f32>,
    _inputs_flat: Vec<f32>,
    _pop_size: usize,
    _input_size: usize,
    _hidden_size: usize,
    _output_size: usize,
) -> NifResult<Vec<f32>> {
    Err(rustler::Error::Term(Box::new("CUDA not available")))
}

/// GPU physics simulation using custom CUDA kernels
///
/// All physics operations run on GPU with zero CPU transfers during simulation:
/// - Fused integration + friction
/// - Batched ball-ball collision (shared memory)
/// - Cushion collision
/// - Pocket detection
///
/// Arguments:
/// - pos_x: [batch * 8] ball X positions
/// - pos_z: [batch * 8] ball Z positions
/// - vel_x: [batch * 8] ball X velocities
/// - vel_z: [batch * 8] ball Z velocities
/// - pocketed: [batch * 8] pocketed flags
/// - batch_size: number of simulations
/// - max_steps: simulation steps
///
/// Returns: (final_pos_x, final_pos_z, final_pocketed, steps)
#[cfg(feature = "cuda")]
#[rustler::nif(schedule = "DirtyCpu")]
fn burn_cuda_physics_simulate(
    pos_x: Vec<f32>,
    pos_z: Vec<f32>,
    vel_x: Vec<f32>,
    vel_z: Vec<f32>,
    pocketed: Vec<f32>,
    batch_size: usize,
    max_steps: usize,
) -> NifResult<(Vec<f32>, Vec<f32>, Vec<f32>, usize)> {
    match cuda_kernels::GpuPhysicsSimulator::new() {
        Ok(mut simulator) => {
            simulator.simulate_batch(
                &pos_x, &pos_z, &vel_x, &vel_z, &pocketed,
                batch_size, max_steps
            ).map_err(|e| rustler::Error::Term(Box::new(e)))
        }
        Err(e) => Err(rustler::Error::Term(Box::new(e)))
    }
}

#[cfg(not(feature = "cuda"))]
#[rustler::nif]
fn burn_cuda_physics_simulate(
    _pos_x: Vec<f32>,
    _pos_z: Vec<f32>,
    _vel_x: Vec<f32>,
    _vel_z: Vec<f32>,
    _pocketed: Vec<f32>,
    _batch_size: usize,
    _max_steps: usize,
) -> NifResult<(Vec<f32>, Vec<f32>, Vec<f32>, usize)> {
    Err(rustler::Error::Term(Box::new("CUDA not available")))
}

/// Parallel fitness calculation on GPU
///
/// Computes fitness for all simulations in a single GPU kernel.
///
/// Arguments:
/// - initial_pocketed: [batch * 8] initial pocketed state
/// - final_pocketed: [batch * 8] final pocketed state
/// - final_pos_x: [batch * 8] final X positions
/// - final_pos_z: [batch * 8] final Z positions
/// - batch_size: number of simulations
/// - target_ball_idx: index of target ball (1 = red)
///
/// Returns: [batch_size] fitness values
#[cfg(feature = "cuda")]
#[rustler::nif(schedule = "DirtyCpu")]
fn burn_cuda_parallel_fitness(
    initial_pocketed: Vec<f32>,
    final_pocketed: Vec<f32>,
    final_pos_x: Vec<f32>,
    final_pos_z: Vec<f32>,
    batch_size: usize,
    target_ball_idx: usize,
) -> NifResult<Vec<f32>> {
    match cuda_kernels::CudaKernelManager::new() {
        Ok(kernels) => {
            kernels.parallel_fitness(
                &initial_pocketed, &final_pocketed,
                &final_pos_x, &final_pos_z,
                batch_size, target_ball_idx
            ).map_err(|e| rustler::Error::Term(Box::new(e)))
        }
        Err(e) => Err(rustler::Error::Term(Box::new(e)))
    }
}

#[cfg(not(feature = "cuda"))]
#[rustler::nif]
fn burn_cuda_parallel_fitness(
    _initial_pocketed: Vec<f32>,
    _final_pocketed: Vec<f32>,
    _final_pos_x: Vec<f32>,
    _final_pos_z: Vec<f32>,
    _batch_size: usize,
    _target_ball_idx: usize,
) -> NifResult<Vec<f32>> {
    Err(rustler::Error::Term(Box::new("CUDA not available")))
}
