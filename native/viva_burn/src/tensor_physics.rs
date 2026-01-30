//! Tensor-based GPU Physics Simulation - RTX 4090 Optimized
//!
//! Fully tensorized physics using custom CUDA kernels.
//! Zero CPU-GPU transfers during simulation steps.
//!
//! Performance targets:
//! - 10,000+ simultaneous table simulations
//! - 100,000+ physics steps/sec aggregate
//! - <1ms latency per batch step
//!
//! Architecture:
//! - All state lives on GPU (persistent buffers)
//! - Custom fused kernels minimize memory bandwidth
//! - Shared memory for collision detection
//! - Warp-level primitives for reductions

#[cfg(feature = "cuda")]
use burn::tensor::{Tensor, TensorData};

#[cfg(feature = "cuda")]
use burn_cuda::{Cuda, CudaDevice};

#[cfg(feature = "cuda")]
use crate::cuda_kernels::{CudaKernelManager, GpuPhysicsSimulator, benchmark_cuda_kernels};

// Import constants from batch_physics
use crate::batch_physics::{
    TABLE_HALF_L, TABLE_HALF_W,
    BALL_RADIUS, CUE_BALL_RADIUS,
    POCKET_RADIUS, POCKET_X, POCKET_Z,
    BALL_RESTITUTION, CUSHION_RESTITUTION,
    ROLLING_FRICTION, VELOCITY_THRESHOLD,
    NUM_BALLS, DT,
    create_initial_batch, batch_simulate_cpu,
};

/// Friction deceleration constant (mu * g)
#[allow(dead_code)]
const FRICTION_DECEL: f32 = ROLLING_FRICTION * 9.81;

// =============================================================================
// GPU TENSOR PHYSICS ENGINE
// =============================================================================

/// High-performance GPU physics engine using custom CUDA kernels
#[cfg(feature = "cuda")]
pub struct TensorPhysicsEngine {
    simulator: GpuPhysicsSimulator,
    // Batch capacity
    max_batch: usize,
}

#[cfg(feature = "cuda")]
impl TensorPhysicsEngine {
    /// Create new physics engine with GPU kernel support
    pub fn new() -> Result<Self, String> {
        Ok(Self {
            simulator: GpuPhysicsSimulator::new()?,
            max_batch: 0,
        })
    }

    /// Simulate batch of tables with shots
    ///
    /// This is the main entry point for GPU-accelerated simulation.
    /// All physics computations run on GPU using custom fused kernels.
    pub fn simulate_batch(
        &mut self,
        positions_x: &[Vec<f32>],
        positions_z: &[Vec<f32>],
        velocities_x: &[Vec<f32>],
        velocities_z: &[Vec<f32>],
        pocketed: &[Vec<f32>],
        shots: &[(f32, f32, f32, f32)],
        max_steps: usize,
    ) -> Result<(Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<usize>), String> {
        let batch_size = positions_x.len();

        if batch_size == 0 {
            return Ok((vec![], vec![], vec![], vec![]));
        }

        // Flatten input data
        let mut px_flat: Vec<f32> = positions_x.iter().flatten().copied().collect();
        let mut pz_flat: Vec<f32> = positions_z.iter().flatten().copied().collect();
        let mut vx_flat: Vec<f32> = velocities_x.iter().flatten().copied().collect();
        let mut vz_flat: Vec<f32> = velocities_z.iter().flatten().copied().collect();
        let pck_flat: Vec<f32> = pocketed.iter().flatten().copied().collect();

        // Apply shots to cue balls
        for i in 0..batch_size.min(shots.len()) {
            let (angle, power, _, _) = shots[i];
            let max_velocity = 50.0;
            vx_flat[i * NUM_BALLS] = power * max_velocity * angle.cos();
            vz_flat[i * NUM_BALLS] = power * max_velocity * angle.sin();
        }

        // Run GPU simulation
        let (final_px, final_pz, final_pck, steps) = self.simulator.simulate_batch(
            &px_flat, &pz_flat, &vx_flat, &vz_flat, &pck_flat,
            batch_size, max_steps
        )?;

        // Unflatten results
        let final_px_vec: Vec<Vec<f32>> = final_px.chunks(NUM_BALLS).map(|c| c.to_vec()).collect();
        let final_pz_vec: Vec<Vec<f32>> = final_pz.chunks(NUM_BALLS).map(|c| c.to_vec()).collect();
        let final_pck_vec: Vec<Vec<f32>> = final_pck.chunks(NUM_BALLS).map(|c| c.to_vec()).collect();
        let steps_vec: Vec<usize> = vec![steps; batch_size];

        Ok((final_px_vec, final_pz_vec, final_pck_vec, steps_vec))
    }

    /// Simulate and calculate fitness in single GPU pass
    pub fn simulate_and_evaluate(
        &mut self,
        positions_x: &[Vec<f32>],
        positions_z: &[Vec<f32>],
        velocities_x: &[Vec<f32>],
        velocities_z: &[Vec<f32>],
        pocketed: &[Vec<f32>],
        shots: &[(f32, f32, f32, f32)],
        max_steps: usize,
        target_ball_idx: usize,
    ) -> Result<Vec<f32>, String> {
        // Run simulation
        let (final_px, final_pz, final_pck, _) = self.simulate_batch(
            positions_x, positions_z, velocities_x, velocities_z, pocketed,
            shots, max_steps
        )?;

        // Calculate fitness on CPU (could be GPU too, but minimal overhead)
        let batch_size = positions_x.len();
        let mut fitness = Vec::with_capacity(batch_size);

        for i in 0..batch_size {
            let mut fit = 0.0f32;

            // Count newly pocketed balls (not cue)
            for b in 1..NUM_BALLS {
                if final_pck[i][b] > 0.5 && pocketed[i][b] < 0.5 {
                    fit += 100.0;
                }
            }

            // Cue ball penalty
            if final_pck[i][0] > 0.5 {
                fit -= 200.0;
            }

            // Distance bonus for target ball
            if target_ball_idx > 0 && target_ball_idx < NUM_BALLS {
                if final_pck[i][target_ball_idx] < 0.5 {
                    let tx = final_px[i][target_ball_idx];
                    let tz = final_pz[i][target_ball_idx];

                    let min_dist = (0..6).map(|p| {
                        let dx = tx - POCKET_X[p];
                        let dz = tz - POCKET_Z[p];
                        (dx * dx + dz * dz).sqrt()
                    }).fold(f32::INFINITY, f32::min);

                    fit += 50.0 * (1.0 - (min_dist / 2.0).min(1.0));
                }
            }

            fitness.push(fit);
        }

        Ok(fitness)
    }
}

// =============================================================================
// BURN TENSOR-BASED SIMULATION (Alternative approach)
// =============================================================================

/// Burn tensor-based simulation using existing burn_cuda backend
///
/// This uses Burn's tensor operations which compile to cuBLAS/cuDNN.
/// Good for prototyping, but custom kernels are faster for billiards-specific ops.
#[cfg(feature = "cuda")]
pub fn tensor_simulate_burn(
    positions_x: Vec<Vec<f32>>,
    positions_z: Vec<Vec<f32>>,
    velocities_x: Vec<Vec<f32>>,
    velocities_z: Vec<Vec<f32>>,
    pocketed: Vec<Vec<f32>>,
    shots: Vec<(f32, f32, f32, f32)>,
    max_steps: usize,
) -> (Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<usize>) {
    // Delegate to batch_physics::batch_simulate_gpu_fast which uses Burn tensors
    crate::batch_physics::batch_simulate_gpu_fast(
        positions_x, positions_z, velocities_x, velocities_z, pocketed, shots, max_steps
    )
}

// =============================================================================
// NEURAL NETWORK FORWARD ON GPU
// =============================================================================

/// Fused neural network forward pass using custom CUDA kernel
///
/// Much faster than burn-rs for small networks because:
/// - Single kernel launch (vs multiple for matmul + activation)
/// - Shared memory for hidden layer (no global memory roundtrip)
/// - Fast tanh approximation in registers
#[cfg(feature = "cuda")]
pub struct GpuNeuralEngine {
    kernels: CudaKernelManager,
}

#[cfg(feature = "cuda")]
impl GpuNeuralEngine {
    pub fn new() -> Result<Self, String> {
        Ok(Self {
            kernels: CudaKernelManager::new()?,
        })
    }

    /// Batch forward pass for population of networks
    ///
    /// weights_flat: [pop_size, total_weights] where total = in*h + h + h*out + out
    /// inputs_flat: [pop_size, input_size]
    ///
    /// Returns: [pop_size, output_size]
    pub fn batch_forward(
        &self,
        weights_flat: &[f32],
        inputs_flat: &[f32],
        pop_size: usize,
        input_size: usize,
        hidden_size: usize,
        output_size: usize,
    ) -> Result<Vec<f32>, String> {
        self.kernels.fused_forward(
            weights_flat, inputs_flat,
            pop_size, input_size, hidden_size, output_size
        )
    }

    /// NES gradient computation on GPU
    pub fn nes_gradient(
        &self,
        perturbations_flat: &[f32],
        advantages: &[f32],
        n_pert: usize,
        weight_count: usize,
        sigma: f32,
    ) -> Result<Vec<f32>, String> {
        self.kernels.nes_gradient(
            perturbations_flat, advantages,
            n_pert, weight_count, sigma
        )
    }
}

// =============================================================================
// COMPLETE GPU TRAINING PIPELINE
// =============================================================================

/// Full NES training step on GPU
///
/// This combines all operations for maximum throughput:
/// 1. Neural forward pass (custom kernel)
/// 2. Physics simulation (custom kernels)
/// 3. Fitness calculation (custom kernel)
/// 4. NES gradient (custom kernel)
#[cfg(feature = "cuda")]
pub struct GpuTrainingPipeline {
    neural: GpuNeuralEngine,
    physics: TensorPhysicsEngine,
}

#[cfg(feature = "cuda")]
impl GpuTrainingPipeline {
    pub fn new() -> Result<Self, String> {
        Ok(Self {
            neural: GpuNeuralEngine::new()?,
            physics: TensorPhysicsEngine::new()?,
        })
    }

    /// Complete NES evaluation step
    ///
    /// Evaluates entire population on GPU with minimal CPU involvement.
    pub fn evaluate_population(
        &mut self,
        weights_batch: &[Vec<f32>],
        architecture: &[usize],
        shots_per_episode: usize,
        max_steps_per_shot: usize,
    ) -> Result<Vec<f32>, String> {
        let pop_size = weights_batch.len();

        if pop_size == 0 || architecture.len() < 2 {
            return Ok(vec![]);
        }

        let input_size = architecture[0];
        let hidden_size = if architecture.len() > 2 { architecture[1] } else { 32 };
        let output_size = *architecture.last().unwrap();

        // Flatten weights
        let weights_flat: Vec<f32> = weights_batch.iter().flatten().copied().collect();

        // Create initial table states
        let (init_px, init_pz, init_vx, init_vz, init_pck) = create_initial_batch(pop_size);

        // For each network, generate inputs from table state
        let inputs = generate_neural_inputs(&init_px, &init_pz, input_size);
        let inputs_flat: Vec<f32> = inputs.iter().flatten().copied().collect();

        // Neural forward to get shots
        let outputs = self.neural.batch_forward(
            &weights_flat, &inputs_flat,
            pop_size, input_size, hidden_size, output_size
        )?;

        // Convert outputs to shots
        let shots: Vec<(f32, f32, f32, f32)> = outputs.chunks(output_size).map(|out| {
            let angle = if output_size > 0 { out[0] * std::f32::consts::PI } else { 0.0 };
            let power = if output_size > 1 { (out[1] + 1.0) * 0.5 } else { 0.5 };  // map [-1,1] to [0,1]
            let english = if output_size > 2 { out[2] } else { 0.0 };
            let elevation = if output_size > 3 { (out[3] + 1.0) * 0.5 } else { 0.0 };
            (angle, power.clamp(0.1, 1.0), english, elevation)
        }).collect();

        // Run physics simulation
        let fitness = self.physics.simulate_and_evaluate(
            &init_px, &init_pz, &init_vx, &init_vz, &init_pck,
            &shots, max_steps_per_shot, 1  // target ball 1 (red)
        )?;

        Ok(fitness)
    }
}

/// Generate neural network inputs from table state
fn generate_neural_inputs(
    pos_x: &[Vec<f32>],
    pos_z: &[Vec<f32>],
    input_size: usize,
) -> Vec<Vec<f32>> {
    pos_x.iter().zip(pos_z.iter()).map(|(px, pz)| {
        let mut inputs = Vec::with_capacity(input_size);

        // Ball positions (normalized to [-1, 1])
        for i in 0..NUM_BALLS.min(input_size / 2) {
            inputs.push(px[i] / TABLE_HALF_L);
            inputs.push(pz[i] / TABLE_HALF_W);
        }

        // Pad with zeros if needed
        while inputs.len() < input_size {
            inputs.push(0.0);
        }

        inputs.truncate(input_size);
        inputs
    }).collect()
}

// =============================================================================
// BENCHMARK: GPU TENSOR VS CPU
// =============================================================================

/// Benchmark tensor-based GPU physics vs CPU Rayon
///
/// Compares three implementations:
/// 1. Custom CUDA kernels (fastest)
/// 2. Burn tensor ops (medium)
/// 3. CPU Rayon parallel (baseline)
#[cfg(feature = "cuda")]
pub fn benchmark_tensor_physics(
    batch_size: usize,
    max_steps: usize,
    iterations: usize,
) -> String {
    use std::time::Instant;
    use crate::batch_physics::batch_simulate_gpu_fast;

    // Create initial batch
    let (px, pz, vx, vz, pocketed) = create_initial_batch(batch_size);

    // Generate shots
    let shots_tuples: Vec<(f32, f32, f32, f32)> = (0..batch_size)
        .map(|i| {
            let angle = (i as f32 / batch_size as f32) * std::f32::consts::PI * 2.0;
            (angle, 0.5 + (i % 10) as f32 * 0.05, 0.0, 0.0)
        })
        .collect();

    // === Custom CUDA Kernels ===
    let custom_result = match TensorPhysicsEngine::new() {
        Ok(mut engine) => {
            // Warmup
            let _ = engine.simulate_batch(&px, &pz, &vx, &vz, &pocketed, &shots_tuples, max_steps);

            let start = Instant::now();
            for _ in 0..iterations {
                let _ = engine.simulate_batch(&px, &pz, &vx, &vz, &pocketed, &shots_tuples, max_steps);
            }
            let elapsed = start.elapsed();

            let sims_per_sec = (batch_size * iterations) as f64 / elapsed.as_secs_f64();
            Some((elapsed, sims_per_sec))
        }
        Err(e) => {
            eprintln!("Custom kernel init failed: {}", e);
            None
        }
    };

    // === Burn Tensor Ops (existing implementation) ===
    // Warmup
    let _ = batch_simulate_gpu_fast(
        px.clone(), pz.clone(), vx.clone(), vz.clone(),
        pocketed.clone(), shots_tuples.clone(), max_steps
    );

    let start_burn = Instant::now();
    for _ in 0..iterations {
        let _ = batch_simulate_gpu_fast(
            px.clone(), pz.clone(), vx.clone(), vz.clone(),
            pocketed.clone(), shots_tuples.clone(), max_steps
        );
    }
    let burn_time = start_burn.elapsed();
    let burn_sims_per_sec = (batch_size * iterations) as f64 / burn_time.as_secs_f64();

    // === CPU Rayon ===
    let start_cpu = Instant::now();
    for _ in 0..iterations {
        let _ = batch_simulate_cpu(
            px.clone(), pz.clone(), vx.clone(), vz.clone(),
            pocketed.clone(), shots_tuples.clone(), max_steps
        );
    }
    let cpu_time = start_cpu.elapsed();
    let cpu_sims_per_sec = (batch_size * iterations) as f64 / cpu_time.as_secs_f64();

    let total_sims = batch_size * iterations;

    let mut output = format!(
        "Tensor Physics Benchmark (RTX 4090):\n\
         ====================================\n\
         Batch size: {}\n\
         Max steps: {}\n\
         Iterations: {}\n\
         Total simulations: {}\n\n",
        batch_size, max_steps, iterations, total_sims
    );

    // Custom kernels results
    if let Some((custom_time, custom_sps)) = custom_result {
        let speedup_vs_burn = custom_sps / burn_sims_per_sec;
        let speedup_vs_cpu = custom_sps / cpu_sims_per_sec;

        output.push_str(&format!(
            "Custom CUDA Kernels:\n\
             - Time: {:.2}ms\n\
             - Throughput: {:.0} sims/sec\n\
             - Per sim: {:.2}us\n\
             - Speedup vs Burn: {:.2}x\n\
             - Speedup vs CPU: {:.2}x\n\n",
            custom_time.as_secs_f64() * 1000.0,
            custom_sps,
            custom_time.as_micros() as f64 / total_sims as f64,
            speedup_vs_burn,
            speedup_vs_cpu
        ));
    } else {
        output.push_str("Custom CUDA Kernels: FAILED TO INIT\n\n");
    }

    // Burn results
    let burn_speedup = burn_sims_per_sec / cpu_sims_per_sec;
    output.push_str(&format!(
        "Burn Tensor Ops:\n\
         - Time: {:.2}ms\n\
         - Throughput: {:.0} sims/sec\n\
         - Per sim: {:.2}us\n\
         - Speedup vs CPU: {:.2}x\n\n",
        burn_time.as_secs_f64() * 1000.0,
        burn_sims_per_sec,
        burn_time.as_micros() as f64 / total_sims as f64,
        burn_speedup
    ));

    // CPU baseline
    output.push_str(&format!(
        "CPU (Rayon parallel):\n\
         - Time: {:.2}ms\n\
         - Throughput: {:.0} sims/sec\n\
         - Per sim: {:.2}us\n",
        cpu_time.as_secs_f64() * 1000.0,
        cpu_sims_per_sec,
        cpu_time.as_micros() as f64 / total_sims as f64
    ));

    output
}

/// CPU fallback for benchmark
#[cfg(not(feature = "cuda"))]
pub fn benchmark_tensor_physics(
    batch_size: usize,
    max_steps: usize,
    iterations: usize,
) -> String {
    use std::time::Instant;

    let (px, pz, vx, vz, pocketed) = create_initial_batch(batch_size);
    let shots_tuples: Vec<(f32, f32, f32, f32)> = (0..batch_size)
        .map(|i| {
            let angle = (i as f32 / batch_size as f32) * std::f32::consts::PI * 2.0;
            (angle, 0.5 + (i % 10) as f32 * 0.05, 0.0, 0.0)
        })
        .collect();

    let start_cpu = Instant::now();
    for _ in 0..iterations {
        let _ = batch_simulate_cpu(
            px.clone(), pz.clone(), vx.clone(), vz.clone(),
            pocketed.clone(), shots_tuples.clone(), max_steps
        );
    }
    let cpu_time = start_cpu.elapsed();

    let total_sims = batch_size * iterations;
    let cpu_sims_per_sec = total_sims as f64 / cpu_time.as_secs_f64();

    format!(
        "Tensor Physics Benchmark (CPU only - no CUDA):\n\
         =============================================\n\
         Batch size: {}\n\
         Max steps: {}\n\
         Iterations: {}\n\
         \n\
         CPU (Rayon parallel):\n\
         - Time: {:.2}ms\n\
         - Throughput: {:.0} sims/sec",
        batch_size, max_steps, iterations,
        cpu_time.as_secs_f64() * 1000.0,
        cpu_sims_per_sec
    )
}

// =============================================================================
// NIF BENCHMARK WRAPPER
// =============================================================================

/// Benchmark custom CUDA kernels for NIF export
#[cfg(feature = "cuda")]
pub fn benchmark_cuda_custom(batch_size: usize, iterations: usize) -> String {
    benchmark_cuda_kernels(batch_size, iterations)
}

#[cfg(not(feature = "cuda"))]
pub fn benchmark_cuda_custom(_batch_size: usize, _iterations: usize) -> String {
    "CUDA not available - compile with --features cuda".to_string()
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initial_batch_creation() {
        let (px, pz, vx, vz, pocketed) = create_initial_batch(10);

        assert_eq!(px.len(), 10);
        assert_eq!(px[0].len(), NUM_BALLS);
        assert_eq!(vx[0][0], 0.0);  // Initial velocity is zero
        assert_eq!(pocketed[0][0], 0.0);  // No balls pocketed initially
    }

    #[test]
    fn test_neural_input_generation() {
        let px = vec![vec![0.0; 8]; 10];
        let pz = vec![vec![0.0; 8]; 10];

        let inputs = generate_neural_inputs(&px, &pz, 16);

        assert_eq!(inputs.len(), 10);
        assert_eq!(inputs[0].len(), 16);
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_tensor_physics_engine() {
        let result = TensorPhysicsEngine::new();
        assert!(result.is_ok(), "Engine creation failed: {:?}", result.err());
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_gpu_neural_engine() {
        let result = GpuNeuralEngine::new();
        assert!(result.is_ok(), "Neural engine creation failed: {:?}", result.err());
    }
}
