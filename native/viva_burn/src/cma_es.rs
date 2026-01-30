//! CMA-ES - Covariance Matrix Adaptation Evolution Strategy
//!
//! GPU-accelerated implementation using Burn tensors for RTX 4090.
//! Optimized for neural network weight optimization with 867 dimensions
//! (architecture [8, 32, 16, 3]).
//!
//! Key operations on GPU:
//! - Eigendecomposition of covariance matrix
//! - Sampling from multivariate normal
//! - Covariance matrix updates (rank-mu, rank-one)
//!
//! Reference: Hansen & Ostermeier (2001) "Completely Derandomized Self-Adaptation
//!            in Evolution Strategies"
//!
//! Created at GATO-PC, Brazil, 2026.

use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};

#[cfg(feature = "cuda")]
use burn_cuda::{Cuda, CudaDevice};

#[cfg(not(feature = "cuda"))]
use burn_ndarray::NdArray;

use rayon::prelude::*;
use rand::{rngs::StdRng, SeedableRng, Rng};
use rand_distr::{Distribution, Normal, StandardNormal};

// =============================================================================
// CMA-ES STATE
// =============================================================================

/// CMA-ES optimizer state
/// Maintains covariance matrix, evolution paths, and step-size
#[derive(Clone)]
pub struct CmaEsState {
    /// Current mean (search center)
    pub mean: Vec<f32>,
    /// Step size (sigma)
    pub sigma: f32,
    /// Covariance matrix (flattened, n x n)
    pub covariance: Vec<f32>,
    /// Evolution path for covariance (p_c)
    pub pc: Vec<f32>,
    /// Evolution path for sigma (p_sigma)
    pub ps: Vec<f32>,
    /// Eigenvalues of C
    pub eigenvalues: Vec<f32>,
    /// Eigenvectors of C (flattened, n x n)
    pub eigenvectors: Vec<f32>,
    /// Dimension
    pub n: usize,
    /// Generation counter
    pub generation: usize,
    /// Population size (lambda)
    pub lambda: usize,
    /// Number of selected parents (mu)
    pub mu: usize,
    /// Weights for weighted recombination
    pub weights: Vec<f32>,
    /// Variance effective selection mass
    pub mu_eff: f32,
    /// Learning rate for covariance matrix (c_c)
    pub cc: f32,
    /// Learning rate for rank-one update (c_1)
    pub c1: f32,
    /// Learning rate for rank-mu update (c_mu)
    pub cmu: f32,
    /// Learning rate for sigma (c_sigma)
    pub csigma: f32,
    /// Damping for sigma
    pub dsigma: f32,
    /// Expected value of ||N(0,I)||
    pub chi_n: f32,
}

impl CmaEsState {
    /// Initialize CMA-ES state with default parameters
    pub fn new(initial_mean: Vec<f32>, initial_sigma: f32) -> Self {
        let n = initial_mean.len();

        // Default population size: 4 + floor(3 * ln(n))
        let lambda = 4 + (3.0 * (n as f32).ln()).floor() as usize;
        let mu = lambda / 2;

        // Compute weights for weighted recombination
        let weights: Vec<f32> = (0..mu)
            .map(|i| ((mu as f32 + 0.5).ln() - ((i + 1) as f32).ln()))
            .collect();
        let weights_sum: f32 = weights.iter().sum();
        let weights: Vec<f32> = weights.iter().map(|w| w / weights_sum).collect();

        // Variance effective selection mass
        let weights_sq_sum: f32 = weights.iter().map(|w| w * w).sum();
        let mu_eff = 1.0 / weights_sq_sum;

        // Learning rates (Hansen's recommended values)
        let cc = (4.0 + mu_eff / n as f32) / (n as f32 + 4.0 + 2.0 * mu_eff / n as f32);
        let csigma = (mu_eff + 2.0) / (n as f32 + mu_eff + 5.0);
        let c1 = 2.0 / ((n as f32 + 1.3).powi(2) + mu_eff);
        let cmu = f32::min(
            1.0 - c1,
            2.0 * (mu_eff - 2.0 + 1.0 / mu_eff) / ((n as f32 + 2.0).powi(2) + mu_eff)
        );
        let dsigma = 1.0 + 2.0 * f32::max(0.0, ((mu_eff - 1.0) / (n as f32 + 1.0)).sqrt() - 1.0) + csigma;

        // Expected value of ||N(0,I)||
        let chi_n = (n as f32).sqrt() * (1.0 - 1.0 / (4.0 * n as f32) + 1.0 / (21.0 * (n as f32).powi(2)));

        // Initialize covariance as identity matrix
        let mut covariance = vec![0.0f32; n * n];
        for i in 0..n {
            covariance[i * n + i] = 1.0;
        }

        // Initialize eigendecomposition (identity)
        let eigenvalues = vec![1.0f32; n];
        let mut eigenvectors = vec![0.0f32; n * n];
        for i in 0..n {
            eigenvectors[i * n + i] = 1.0;
        }

        Self {
            mean: initial_mean,
            sigma: initial_sigma,
            covariance,
            pc: vec![0.0; n],
            ps: vec![0.0; n],
            eigenvalues,
            eigenvectors,
            n,
            generation: 0,
            lambda,
            mu,
            weights,
            mu_eff,
            cc,
            c1,
            cmu,
            csigma,
            dsigma,
            chi_n,
        }
    }

    /// Create CMA-ES state with custom population size
    pub fn with_population(initial_mean: Vec<f32>, initial_sigma: f32, lambda: usize) -> Self {
        let mut state = Self::new(initial_mean, initial_sigma);

        // Adjust parameters for custom lambda
        state.lambda = lambda;
        state.mu = lambda / 2;

        // Recompute weights
        let weights: Vec<f32> = (0..state.mu)
            .map(|i| ((state.mu as f32 + 0.5).ln() - ((i + 1) as f32).ln()))
            .collect();
        let weights_sum: f32 = weights.iter().sum();
        state.weights = weights.iter().map(|w| w / weights_sum).collect();

        // Recompute mu_eff
        let weights_sq_sum: f32 = state.weights.iter().map(|w| w * w).sum();
        state.mu_eff = 1.0 / weights_sq_sum;

        state
    }
}

// =============================================================================
// GPU SAMPLING (Multivariate Normal)
// =============================================================================

/// Sample from multivariate normal N(mean, sigma^2 * C) on GPU
///
/// Uses eigendecomposition: C = B * D^2 * B^T
/// Sampling: x = mean + sigma * B * D * z, where z ~ N(0, I)
#[cfg(feature = "cuda")]
pub fn sample_population_gpu(
    state: &CmaEsState,
    seed: u64,
) -> Vec<Vec<f32>> {
    use burn::prelude::*;
    use burn::tensor::Distribution;

    type B = Cuda<f32, i32>;
    let device = CudaDevice::default();

    let n = state.n;
    let lambda = state.lambda;

    // Mean tensor [n]
    let mean_data = TensorData::new(state.mean.clone(), [n]);
    let mean_tensor: Tensor<B, 1> = Tensor::from_data(mean_data, &device);

    // Eigenvectors [n, n]
    let evec_data = TensorData::new(state.eigenvectors.clone(), [n, n]);
    let b_matrix: Tensor<B, 2> = Tensor::from_data(evec_data, &device);

    // Sqrt of eigenvalues (D matrix diagonal)
    let sqrt_evals: Vec<f32> = state.eigenvalues.iter().map(|e| e.sqrt()).collect();
    let d_data = TensorData::new(sqrt_evals, [n]);
    let d_diag: Tensor<B, 1> = Tensor::from_data(d_data, &device);

    // Sample standard normal [lambda, n]
    let z: Tensor<B, 2> = Tensor::random(
        [lambda, n],
        Distribution::Normal(0.0, 1.0),
        &device,
    );

    // Scale by D: [lambda, n] * [n] -> [lambda, n] (broadcast)
    let z_scaled = z.clone() * d_diag.unsqueeze_dim(0).repeat_dim(0, lambda);

    // Rotate by B: [lambda, n] @ [n, n]^T -> [lambda, n]
    let b_t = b_matrix.transpose();
    let y = z_scaled.matmul(b_t);

    // Scale by sigma and add mean
    let mean_broadcast = mean_tensor.unsqueeze_dim(0).repeat_dim(0, lambda);
    let samples = mean_broadcast + y * state.sigma;

    // Convert to Vec<Vec<f32>>
    let samples_data = samples.into_data();
    let samples_vec: Vec<f32> = samples_data.to_vec().unwrap();

    samples_vec
        .chunks(n)
        .map(|chunk| chunk.to_vec())
        .collect()
}

/// CPU fallback for sampling
pub fn sample_population_cpu(
    state: &CmaEsState,
    seed: u64,
) -> Vec<Vec<f32>> {
    let n = state.n;
    let lambda = state.lambda;

    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0f32, 1.0f32).unwrap();

    // Sample z ~ N(0, I)
    let mut samples = Vec::with_capacity(lambda);

    for _ in 0..lambda {
        // Generate standard normal
        let z: Vec<f32> = (0..n).map(|_| normal.sample(&mut rng)).collect();

        // Scale by sqrt(eigenvalues)
        let z_scaled: Vec<f32> = z.iter()
            .zip(state.eigenvalues.iter())
            .map(|(&zi, &ei)| zi * ei.sqrt())
            .collect();

        // Rotate by eigenvectors: B * z_scaled
        let mut y = vec![0.0f32; n];
        for i in 0..n {
            for j in 0..n {
                y[i] += state.eigenvectors[j * n + i] * z_scaled[j];
            }
        }

        // Scale by sigma and add mean
        let x: Vec<f32> = y.iter()
            .zip(state.mean.iter())
            .map(|(&yi, &mi)| mi + state.sigma * yi)
            .collect();

        samples.push(x);
    }

    samples
}

/// Auto-select GPU or CPU for sampling
#[cfg(feature = "cuda")]
pub fn sample_population(state: &CmaEsState, seed: u64) -> Vec<Vec<f32>> {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        sample_population_gpu(state, seed)
    })) {
        Ok(result) => result,
        Err(_) => sample_population_cpu(state, seed),
    }
}

#[cfg(not(feature = "cuda"))]
pub fn sample_population(state: &CmaEsState, seed: u64) -> Vec<Vec<f32>> {
    sample_population_cpu(state, seed)
}

// =============================================================================
// EIGENDECOMPOSITION (GPU-accelerated)
// =============================================================================

/// GPU-accelerated eigendecomposition using power iteration
/// Returns (eigenvalues, eigenvectors)
///
/// Note: For small matrices (n < 100), CPU is often faster.
/// We use hybrid approach: GPU for large, CPU for small.
#[cfg(feature = "cuda")]
pub fn eigendecomposition_gpu(
    covariance: &[f32],
    n: usize,
    max_iter: usize,
) -> (Vec<f32>, Vec<f32>) {
    use burn::prelude::*;

    type B = Cuda<f32, i32>;
    let device = CudaDevice::default();

    // For small matrices, use CPU
    if n < 64 {
        return eigendecomposition_cpu(covariance, n, max_iter);
    }

    // Convert to tensor
    let c_data = TensorData::new(covariance.to_vec(), [n, n]);
    let c_matrix: Tensor<B, 2> = Tensor::from_data(c_data, &device);

    // Power iteration for eigenvalues/eigenvectors
    // This is a simplified approach - for production, use proper SVD/eigendecomp

    let mut eigenvalues = vec![1.0f32; n];
    let mut eigenvectors = vec![0.0f32; n * n];

    // Initialize with identity
    for i in 0..n {
        eigenvectors[i * n + i] = 1.0;
    }

    // Deflation-based eigendecomposition
    let mut c_deflated = c_matrix.clone();

    for k in 0..n.min(10) {  // Compute top 10 eigenvalues for large n
        // Random starting vector
        let mut v: Tensor<B, 1> = Tensor::random(
            [n],
            burn::tensor::Distribution::Uniform(-1.0, 1.0),
            &device,
        );

        // Power iteration
        for _ in 0..max_iter.min(50) {
            // v = C * v
            let v_2d = v.clone().unsqueeze_dim(1);  // [n, 1]
            let av = c_deflated.clone().matmul(v_2d);  // [n, 1]
            let av_1d = av.squeeze(1);  // [n]

            // Normalize
            let norm = av_1d.clone().powf_scalar(2.0).sum().sqrt();
            v = av_1d / norm;
        }

        // Compute eigenvalue: lambda = v^T * C * v
        let v_2d = v.clone().unsqueeze_dim(1);  // [n, 1]
        let cv = c_deflated.clone().matmul(v_2d.clone());  // [n, 1]
        let vtcv = v_2d.transpose().matmul(cv);  // [1, 1]
        let lambda_data = vtcv.into_data();
        let lambda: f32 = lambda_data.to_vec::<f32>().unwrap()[0];

        eigenvalues[k] = lambda.abs().max(1e-10);

        // Store eigenvector
        let v_data = v.clone().into_data();
        let v_vec: Vec<f32> = v_data.to_vec().unwrap();
        for i in 0..n {
            eigenvectors[k * n + i] = v_vec[i];
        }

        // Deflate: C = C - lambda * v * v^T
        let vvt = v.clone().unsqueeze_dim(1).matmul(v.clone().unsqueeze_dim(0));
        c_deflated = c_deflated - vvt * lambda;
    }

    // Fill remaining with identity (for stability)
    for k in 10..n {
        eigenvalues[k] = 1.0;
        eigenvectors[k * n + k] = 1.0;
    }

    (eigenvalues, eigenvectors)
}

/// CPU eigendecomposition using Jacobi method (more accurate for small n)
pub fn eigendecomposition_cpu(
    covariance: &[f32],
    n: usize,
    max_iter: usize,
) -> (Vec<f32>, Vec<f32>) {
    // Copy covariance matrix
    let mut a = covariance.to_vec();

    // Initialize eigenvectors as identity
    let mut v = vec![0.0f32; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }

    // Jacobi rotation method
    for _ in 0..max_iter {
        // Find largest off-diagonal element
        let mut max_val = 0.0f32;
        let mut p = 0;
        let mut q = 0;

        for i in 0..n {
            for j in (i + 1)..n {
                let val = a[i * n + j].abs();
                if val > max_val {
                    max_val = val;
                    p = i;
                    q = j;
                }
            }
        }

        // Convergence check
        if max_val < 1e-10 {
            break;
        }

        // Compute rotation angle
        let app = a[p * n + p];
        let aqq = a[q * n + q];
        let apq = a[p * n + q];

        let theta = if (aqq - app).abs() < 1e-10 {
            std::f32::consts::PI / 4.0
        } else {
            0.5 * (2.0 * apq / (aqq - app)).atan()
        };

        let c = theta.cos();
        let s = theta.sin();

        // Apply rotation to A
        for i in 0..n {
            if i != p && i != q {
                let aip = a[i * n + p];
                let aiq = a[i * n + q];
                a[i * n + p] = c * aip - s * aiq;
                a[p * n + i] = a[i * n + p];
                a[i * n + q] = s * aip + c * aiq;
                a[q * n + i] = a[i * n + q];
            }
        }

        let new_pp = c * c * app - 2.0 * s * c * apq + s * s * aqq;
        let new_qq = s * s * app + 2.0 * s * c * apq + c * c * aqq;
        a[p * n + p] = new_pp;
        a[q * n + q] = new_qq;
        a[p * n + q] = 0.0;
        a[q * n + p] = 0.0;

        // Update eigenvectors
        for i in 0..n {
            let vip = v[i * n + p];
            let viq = v[i * n + q];
            v[i * n + p] = c * vip - s * viq;
            v[i * n + q] = s * vip + c * viq;
        }
    }

    // Extract eigenvalues from diagonal
    let eigenvalues: Vec<f32> = (0..n)
        .map(|i| a[i * n + i].abs().max(1e-10))
        .collect();

    (eigenvalues, v)
}

/// Auto-select GPU or CPU for eigendecomposition
#[cfg(feature = "cuda")]
pub fn eigendecomposition(covariance: &[f32], n: usize, max_iter: usize) -> (Vec<f32>, Vec<f32>) {
    if n >= 64 {
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            eigendecomposition_gpu(covariance, n, max_iter)
        })) {
            Ok(result) => result,
            Err(_) => eigendecomposition_cpu(covariance, n, max_iter),
        }
    } else {
        eigendecomposition_cpu(covariance, n, max_iter)
    }
}

#[cfg(not(feature = "cuda"))]
pub fn eigendecomposition(covariance: &[f32], n: usize, max_iter: usize) -> (Vec<f32>, Vec<f32>) {
    eigendecomposition_cpu(covariance, n, max_iter)
}

// =============================================================================
// CMA-ES UPDATE STEP (GPU-accelerated)
// =============================================================================

/// Update CMA-ES state given sorted population (best first)
///
/// Implements:
/// 1. Weighted mean update
/// 2. Evolution path update (p_sigma, p_c)
/// 3. Step size (sigma) update
/// 4. Covariance matrix update (rank-one + rank-mu)
#[cfg(feature = "cuda")]
pub fn update_state_gpu(
    state: &mut CmaEsState,
    population: &[Vec<f32>],
    fitnesses: &[f32],
) {
    use burn::prelude::*;

    type B = Cuda<f32, i32>;
    let device = CudaDevice::default();

    let n = state.n;
    let mu = state.mu;

    // Sort by fitness (descending for maximization)
    let mut indices: Vec<usize> = (0..population.len()).collect();
    indices.sort_by(|&a, &b| fitnesses[b].partial_cmp(&fitnesses[a]).unwrap());

    // Select top mu individuals
    let selected: Vec<&Vec<f32>> = indices.iter().take(mu).map(|&i| &population[i]).collect();

    // Old mean
    let old_mean = state.mean.clone();

    // 1. Update mean: weighted average of selected individuals
    let mut new_mean = vec![0.0f32; n];
    for (i, indiv) in selected.iter().enumerate() {
        for j in 0..n {
            new_mean[j] += state.weights[i] * indiv[j];
        }
    }
    state.mean = new_mean.clone();

    // Compute mean shift (normalized)
    let mean_shift: Vec<f32> = new_mean.iter()
        .zip(old_mean.iter())
        .map(|(new, old)| (new - old) / state.sigma)
        .collect();

    // Convert to tensors for GPU computation
    let shift_data = TensorData::new(mean_shift.clone(), [n]);
    let shift_tensor: Tensor<B, 1> = Tensor::from_data(shift_data, &device);

    // 2. Update p_sigma (evolution path for sigma)
    // p_sigma = (1 - c_sigma) * p_sigma + sqrt(c_sigma * (2 - c_sigma) * mu_eff) * B * D^-1 * B^T * mean_shift

    // First compute C^(-1/2) * mean_shift = B * D^(-1) * B^T * mean_shift
    let evec_data = TensorData::new(state.eigenvectors.clone(), [n, n]);
    let b_matrix: Tensor<B, 2> = Tensor::from_data(evec_data, &device);

    let inv_sqrt_evals: Vec<f32> = state.eigenvalues.iter()
        .map(|e| 1.0 / e.sqrt().max(1e-10))
        .collect();
    let dinv_data = TensorData::new(inv_sqrt_evals, [n]);
    let dinv_diag: Tensor<B, 1> = Tensor::from_data(dinv_data, &device);

    // B^T * shift
    let shift_2d = shift_tensor.clone().unsqueeze_dim(1);  // [n, 1]
    let bt_shift = b_matrix.clone().transpose().matmul(shift_2d.clone());  // [n, 1]

    // D^-1 * (B^T * shift)
    let bt_shift_1d = bt_shift.clone().squeeze(1);  // [n]
    let dinv_bt_shift = bt_shift_1d * dinv_diag;  // [n]

    // B * D^-1 * B^T * shift
    let dinv_bt_shift_2d = dinv_bt_shift.unsqueeze_dim(1);  // [n, 1]
    let invsqrt_c_shift = b_matrix.clone().matmul(dinv_bt_shift_2d);  // [n, 1]
    let invsqrt_c_shift_1d = invsqrt_c_shift.squeeze(1);  // [n]

    // Update p_sigma
    let ps_data = TensorData::new(state.ps.clone(), [n]);
    let ps_tensor: Tensor<B, 1> = Tensor::from_data(ps_data, &device);

    let factor_ps = (state.csigma * (2.0 - state.csigma) * state.mu_eff).sqrt();
    let new_ps_tensor = ps_tensor * (1.0 - state.csigma) + invsqrt_c_shift_1d * factor_ps;

    let new_ps_data = new_ps_tensor.clone().into_data();
    state.ps = new_ps_data.to_vec().unwrap();

    // 3. Update p_c (evolution path for covariance)
    // p_c = (1 - c_c) * p_c + h_sigma * sqrt(c_c * (2 - c_c) * mu_eff) * mean_shift

    // Compute h_sigma (indicator for step size adaptation)
    let ps_norm_sq: f32 = state.ps.iter().map(|x| x * x).sum();
    let ps_norm = ps_norm_sq.sqrt();
    let h_sigma_threshold = (1.0 - (1.0 - state.csigma).powi(2 * (state.generation as i32 + 1)))
        * (1.4 + 2.0 / (n as f32 + 1.0));
    let h_sigma = if ps_norm / state.chi_n < h_sigma_threshold.sqrt() { 1.0f32 } else { 0.0f32 };

    let pc_data = TensorData::new(state.pc.clone(), [n]);
    let pc_tensor: Tensor<B, 1> = Tensor::from_data(pc_data, &device);

    let factor_pc = h_sigma * (state.cc * (2.0 - state.cc) * state.mu_eff).sqrt();
    let new_pc_tensor = pc_tensor * (1.0 - state.cc) + shift_tensor.clone() * factor_pc;

    let new_pc_data = new_pc_tensor.clone().into_data();
    state.pc = new_pc_data.to_vec().unwrap();

    // 4. Update sigma (step size)
    // sigma = sigma * exp((c_sigma / d_sigma) * (||p_sigma|| / chi_n - 1))
    let sigma_update = (state.csigma / state.dsigma) * (ps_norm / state.chi_n - 1.0);
    state.sigma *= sigma_update.exp();

    // Clamp sigma to reasonable range
    state.sigma = state.sigma.clamp(1e-10, 1e10);

    // 5. Update covariance matrix
    // C = (1 - c_1 - c_mu) * C + c_1 * p_c * p_c^T + c_mu * sum(w_i * y_i * y_i^T)

    // Compute rank-one update: p_c * p_c^T
    let pc_2d = new_pc_tensor.unsqueeze_dim(1);  // [n, 1]
    let pc_outer = pc_2d.clone().matmul(pc_2d.transpose());  // [n, n]

    // Compute rank-mu update: sum(w_i * y_i * y_i^T)
    // where y_i = (x_i - old_mean) / sigma
    let mut rank_mu_flat = vec![0.0f32; n * n];

    for (i, indiv) in selected.iter().enumerate() {
        let y: Vec<f32> = indiv.iter()
            .zip(old_mean.iter())
            .map(|(x, m)| (x - m) / state.sigma)
            .collect();

        // Outer product
        for j in 0..n {
            for k in 0..n {
                rank_mu_flat[j * n + k] += state.weights[i] * y[j] * y[k];
            }
        }
    }

    let rank_mu_data = TensorData::new(rank_mu_flat, [n, n]);
    let rank_mu_tensor: Tensor<B, 2> = Tensor::from_data(rank_mu_data, &device);

    // Old covariance
    let c_data = TensorData::new(state.covariance.clone(), [n, n]);
    let c_tensor: Tensor<B, 2> = Tensor::from_data(c_data, &device);

    // Update: C = (1 - c1 - cmu) * C + c1 * pc * pc^T + cmu * rank_mu
    let decay = 1.0 - state.c1 - state.cmu;
    let new_c_tensor = c_tensor * decay + pc_outer * state.c1 + rank_mu_tensor * state.cmu;

    // Ensure symmetry
    let new_c_symmetric = (new_c_tensor.clone() + new_c_tensor.transpose()) * 0.5;

    let new_c_data = new_c_symmetric.into_data();
    state.covariance = new_c_data.to_vec().unwrap();

    // 6. Update eigendecomposition (every few generations)
    if state.generation % 5 == 0 {
        let (eigenvalues, eigenvectors) = eigendecomposition(&state.covariance, n, 100);
        state.eigenvalues = eigenvalues;
        state.eigenvectors = eigenvectors;
    }

    state.generation += 1;
}

/// CPU fallback for state update
pub fn update_state_cpu(
    state: &mut CmaEsState,
    population: &[Vec<f32>],
    fitnesses: &[f32],
) {
    let n = state.n;
    let mu = state.mu;

    // Sort by fitness (descending)
    let mut indices: Vec<usize> = (0..population.len()).collect();
    indices.sort_by(|&a, &b| fitnesses[b].partial_cmp(&fitnesses[a]).unwrap());

    let selected: Vec<&Vec<f32>> = indices.iter().take(mu).map(|&i| &population[i]).collect();
    let old_mean = state.mean.clone();

    // 1. Update mean
    let mut new_mean = vec![0.0f32; n];
    for (i, indiv) in selected.iter().enumerate() {
        for j in 0..n {
            new_mean[j] += state.weights[i] * indiv[j];
        }
    }
    state.mean = new_mean.clone();

    let mean_shift: Vec<f32> = new_mean.iter()
        .zip(old_mean.iter())
        .map(|(new, old)| (new - old) / state.sigma)
        .collect();

    // 2. Update p_sigma
    // Compute C^(-1/2) * mean_shift using eigendecomposition
    let mut invsqrt_c_shift = vec![0.0f32; n];

    // B^T * shift
    let mut bt_shift = vec![0.0f32; n];
    for i in 0..n {
        for j in 0..n {
            bt_shift[i] += state.eigenvectors[j * n + i] * mean_shift[j];
        }
    }

    // D^-1 * (B^T * shift)
    for i in 0..n {
        bt_shift[i] /= state.eigenvalues[i].sqrt().max(1e-10);
    }

    // B * result
    for i in 0..n {
        for j in 0..n {
            invsqrt_c_shift[i] += state.eigenvectors[i * n + j] * bt_shift[j];
        }
    }

    let factor_ps = (state.csigma * (2.0 - state.csigma) * state.mu_eff).sqrt();
    for i in 0..n {
        state.ps[i] = (1.0 - state.csigma) * state.ps[i] + factor_ps * invsqrt_c_shift[i];
    }

    // 3. Update p_c
    let ps_norm_sq: f32 = state.ps.iter().map(|x| x * x).sum();
    let ps_norm = ps_norm_sq.sqrt();
    let h_sigma_threshold = (1.0 - (1.0 - state.csigma).powi(2 * (state.generation as i32 + 1)))
        * (1.4 + 2.0 / (n as f32 + 1.0));
    let h_sigma = if ps_norm / state.chi_n < h_sigma_threshold.sqrt() { 1.0f32 } else { 0.0f32 };

    let factor_pc = h_sigma * (state.cc * (2.0 - state.cc) * state.mu_eff).sqrt();
    for i in 0..n {
        state.pc[i] = (1.0 - state.cc) * state.pc[i] + factor_pc * mean_shift[i];
    }

    // 4. Update sigma
    let sigma_update = (state.csigma / state.dsigma) * (ps_norm / state.chi_n - 1.0);
    state.sigma *= sigma_update.exp();
    state.sigma = state.sigma.clamp(1e-10, 1e10);

    // 5. Update covariance
    let decay = 1.0 - state.c1 - state.cmu;

    // Rank-one update
    for i in 0..n {
        for j in 0..n {
            state.covariance[i * n + j] *= decay;
            state.covariance[i * n + j] += state.c1 * state.pc[i] * state.pc[j];
        }
    }

    // Rank-mu update
    for (k, indiv) in selected.iter().enumerate() {
        let y: Vec<f32> = indiv.iter()
            .zip(old_mean.iter())
            .map(|(x, m)| (x - m) / state.sigma)
            .collect();

        for i in 0..n {
            for j in 0..n {
                state.covariance[i * n + j] += state.cmu * state.weights[k] * y[i] * y[j];
            }
        }
    }

    // Ensure symmetry
    for i in 0..n {
        for j in (i + 1)..n {
            let avg = (state.covariance[i * n + j] + state.covariance[j * n + i]) / 2.0;
            state.covariance[i * n + j] = avg;
            state.covariance[j * n + i] = avg;
        }
    }

    // 6. Update eigendecomposition
    if state.generation % 5 == 0 {
        let (eigenvalues, eigenvectors) = eigendecomposition_cpu(&state.covariance, n, 100);
        state.eigenvalues = eigenvalues;
        state.eigenvectors = eigenvectors;
    }

    state.generation += 1;
}

/// Auto-select GPU or CPU for state update
#[cfg(feature = "cuda")]
pub fn update_state(state: &mut CmaEsState, population: &[Vec<f32>], fitnesses: &[f32]) {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        update_state_gpu(state, population, fitnesses);
    })) {
        Ok(_) => {},
        Err(_) => update_state_cpu(state, population, fitnesses),
    }
}

#[cfg(not(feature = "cuda"))]
pub fn update_state(state: &mut CmaEsState, population: &[Vec<f32>], fitnesses: &[f32]) {
    update_state_cpu(state, population, fitnesses);
}

// =============================================================================
// CMA-ES STEP (Complete iteration)
// =============================================================================

/// Complete CMA-ES step: sample -> evaluate -> update
/// Returns new population for evaluation
pub fn cma_es_step(
    state: &mut CmaEsState,
    fitnesses: &[f32],
    population: &[Vec<f32>],
    seed: u64,
) -> Vec<Vec<f32>> {
    // Update state based on evaluations
    if !population.is_empty() && !fitnesses.is_empty() {
        update_state(state, population, fitnesses);
    }

    // Sample new population
    sample_population(state, seed + state.generation as u64)
}

/// Get current best solution from state
pub fn get_best_solution(state: &CmaEsState) -> Vec<f32> {
    state.mean.clone()
}

/// Get convergence diagnostics
pub fn get_diagnostics(state: &CmaEsState) -> (f32, f32, f32) {
    let condition = state.eigenvalues.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
        / state.eigenvalues.iter().cloned().fold(f32::INFINITY, f32::min);
    let sigma = state.sigma;
    let ps_norm: f32 = state.ps.iter().map(|x| x * x).sum::<f32>().sqrt();

    (sigma, condition, ps_norm / state.chi_n)
}

// =============================================================================
// BENCHMARKING
// =============================================================================

/// Benchmark CMA-ES operations
pub fn benchmark_cma_es(n: usize, lambda: usize, iterations: usize) -> String {
    use std::time::Instant;

    let initial_mean = vec![0.0f32; n];
    let mut state = CmaEsState::with_population(initial_mean, 0.3, lambda);

    // Warmup
    let pop = sample_population(&state, 42);
    let fitnesses: Vec<f32> = (0..lambda).map(|i| i as f32).collect();
    update_state(&mut state, &pop, &fitnesses);

    // Benchmark sampling
    let start = Instant::now();
    for i in 0..iterations {
        let _ = sample_population(&state, i as u64);
    }
    let sample_time = start.elapsed();

    // Benchmark update
    let pop = sample_population(&state, 42);
    let start = Instant::now();
    for _ in 0..iterations {
        update_state(&mut state, &pop, &fitnesses);
    }
    let update_time = start.elapsed();

    format!(
        "CMA-ES Benchmark (n: {}, lambda: {}, iterations: {}):\n\
         - Sampling: {:.2}ms ({:.0} samples/sec)\n\
         - Update: {:.2}ms ({:.0} updates/sec)",
        n, lambda, iterations,
        sample_time.as_secs_f64() * 1000.0,
        (iterations * lambda) as f64 / sample_time.as_secs_f64(),
        update_time.as_secs_f64() * 1000.0,
        iterations as f64 / update_time.as_secs_f64()
    )
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cma_es_init() {
        let mean = vec![0.0f32; 10];
        let state = CmaEsState::new(mean, 0.5);

        assert_eq!(state.n, 10);
        assert!(state.lambda > 0);
        assert!(state.mu > 0);
        assert!(state.sigma == 0.5);
    }

    #[test]
    fn test_sampling_cpu() {
        let mean = vec![0.0f32; 10];
        let state = CmaEsState::new(mean, 0.5);

        let samples = sample_population_cpu(&state, 42);

        assert_eq!(samples.len(), state.lambda);
        assert_eq!(samples[0].len(), 10);
    }

    #[test]
    fn test_eigendecomposition_cpu() {
        // Test with identity matrix
        let n = 5;
        let mut identity = vec![0.0f32; n * n];
        for i in 0..n {
            identity[i * n + i] = 1.0;
        }

        let (eigenvalues, eigenvectors) = eigendecomposition_cpu(&identity, n, 100);

        // All eigenvalues should be close to 1
        for ev in eigenvalues {
            assert!((ev - 1.0).abs() < 0.1);
        }
    }

    #[test]
    fn test_update_state_cpu() {
        let mean = vec![0.0f32; 10];
        let mut state = CmaEsState::new(mean, 0.5);

        let population = sample_population_cpu(&state, 42);
        let fitnesses: Vec<f32> = population.iter()
            .map(|x| -x.iter().map(|xi| xi * xi).sum::<f32>())  // Sphere function
            .collect();

        update_state_cpu(&mut state, &population, &fitnesses);

        assert_eq!(state.generation, 1);
    }
}
