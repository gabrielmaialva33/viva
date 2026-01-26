//! Standalone benchmark for viva_glands
//! Run: cargo run --release --bin benchmark

use std::time::Instant;
use rustfft::{FftPlanner, num_complex::Complex32};
use rayon::prelude::*;
use candle_core::{Device, Tensor, backend::BackendDevice};

const HRR_DIM: usize = 8192;
const LLM_DIM: usize = 4096;
const ITERATIONS: usize = 1000;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║        VIVA GLANDS - ULTRA Performance Benchmark             ║");
    println!("║        i9-13900K + RTX 4090 | AVX2 + FMA + CUDA              ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Check GPU
    let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
    println!("Device: {:?}", device);
    println!("Config: HRR_DIM={}, LLM_DIM={}, ITERATIONS={}\n", HRR_DIM, LLM_DIM, ITERATIONS);

    // Generate test vectors
    let a: Vec<f32> = (0..HRR_DIM).map(|i| (i as f32 * 0.001).sin()).collect();
    let b: Vec<f32> = (0..HRR_DIM).map(|i| (i as f32 * 0.001).cos()).collect();
    let embedding: Vec<f32> = (0..LLM_DIM).map(|i| (i as f32 * 0.001).sin()).collect();

    // Warmup
    println!("Warming up...");
    for _ in 0..100 {
        let _ = cosine_similarity_simd(&a, &b);
    }

    // ========================================================================
    // Benchmark: Cosine Similarity (SIMD)
    // ========================================================================
    print!("Cosine Similarity (SIMD AVX2)... ");
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        std::hint::black_box(cosine_similarity_simd(&a, &b));
    }
    let elapsed = start.elapsed();
    let sim_us = elapsed.as_nanos() as f64 / ITERATIONS as f64 / 1000.0;
    println!("{:.2} μs/op ({:.2}M ops/sec)", sim_us, 1.0 / (sim_us / 1_000_000.0));

    // ========================================================================
    // Benchmark: HRR Bind (FFT)
    // ========================================================================
    print!("HRR Bind (FFT + Rayon)... ");
    let mut workspace = HrrWorkspace::new(HRR_DIM);

    // Warmup
    for _ in 0..10 {
        let _ = workspace.bind(&a, &b);
    }

    let start = Instant::now();
    for _ in 0..ITERATIONS {
        std::hint::black_box(workspace.bind(&a, &b));
    }
    let elapsed = start.elapsed();
    let bind_us = elapsed.as_micros() as f64 / ITERATIONS as f64;
    println!("{:.2} μs/op ({:.2}K ops/sec)", bind_us, 1000.0 / bind_us);

    // ========================================================================
    // Benchmark: HRR Unbind (FFT)
    // ========================================================================
    print!("HRR Unbind (FFT + Rayon)... ");
    let trace = workspace.bind(&a, &b);

    let start = Instant::now();
    for _ in 0..ITERATIONS {
        std::hint::black_box(workspace.unbind(&trace, &a));
    }
    let elapsed = start.elapsed();
    let unbind_us = elapsed.as_micros() as f64 / ITERATIONS as f64;
    println!("{:.2} μs/op ({:.2}K ops/sec)", unbind_us, 1000.0 / unbind_us);

    // ========================================================================
    // Benchmark: Projection CPU (INT8 quantized)
    // ========================================================================
    print!("Projection CPU (INT8 quantized, {}→{})... ", LLM_DIM, HRR_DIM);
    let projection_cpu = QuantizedProjectionCPU::new(LLM_DIM, HRR_DIM, 42);

    // Warmup
    for _ in 0..10 {
        let _ = projection_cpu.project(&embedding);
    }

    let start = Instant::now();
    for _ in 0..100 {  // Less iterations for slow op
        std::hint::black_box(projection_cpu.project(&embedding));
    }
    let elapsed = start.elapsed();
    let proj_cpu_us = elapsed.as_micros() as f64 / 100.0;
    println!("{:.2} μs/op ({:.2} ops/sec)", proj_cpu_us, 1_000_000.0 / proj_cpu_us);

    // ========================================================================
    // Benchmark: Projection GPU (FP32 via Candle/CUDA)
    // ========================================================================
    if matches!(device, Device::Cuda(_)) {
        print!("Projection GPU (FP32 CUDA, {}→{})... ", LLM_DIM, HRR_DIM);
        let projection_gpu = ProjectionGPU::new(LLM_DIM, HRR_DIM, 42, &device).unwrap();

        // Warmup (important for GPU!)
        for _ in 0..50 {
            let _ = projection_gpu.project(&embedding);
        }

        // Sync before timing
        if let Device::Cuda(cuda) = &device {
            cuda.synchronize().unwrap();
        }

        let start = Instant::now();
        for _ in 0..ITERATIONS {
            std::hint::black_box(projection_gpu.project(&embedding));
        }

        // Sync after
        if let Device::Cuda(cuda) = &device {
            cuda.synchronize().unwrap();
        }

        let elapsed = start.elapsed();
        let proj_gpu_us = elapsed.as_micros() as f64 / ITERATIONS as f64;
        let speedup = proj_cpu_us / proj_gpu_us;
        println!("{:.2} μs/op ({:.2}K ops/sec) [{:.1}x faster than CPU]",
                 proj_gpu_us, 1000.0 / proj_gpu_us, speedup);
    } else {
        println!("Projection GPU... SKIPPED (no CUDA device)");
    }

    // ========================================================================
    // Benchmark: HRR Bind GPU (cuFFT)
    // ========================================================================
    if matches!(device, Device::Cuda(_)) {
        print!("HRR Bind GPU (cuFFT, {}d)... ", HRR_DIM);
        let hrr_gpu = HrrGPU::new(HRR_DIM, &device).unwrap();

        // Warmup
        for _ in 0..50 {
            let _ = hrr_gpu.bind(&a, &b);
        }

        if let Device::Cuda(cuda) = &device {
            cuda.synchronize().unwrap();
        }

        let start = Instant::now();
        for _ in 0..ITERATIONS {
            std::hint::black_box(hrr_gpu.bind(&a, &b));
        }

        if let Device::Cuda(cuda) = &device {
            cuda.synchronize().unwrap();
        }

        let elapsed = start.elapsed();
        let bind_gpu_us = elapsed.as_micros() as f64 / ITERATIONS as f64;
        let speedup = bind_us / bind_gpu_us;
        println!("{:.2} μs/op ({:.2}K ops/sec) [{:.1}x faster than CPU]",
                 bind_gpu_us, 1000.0 / bind_gpu_us, speedup);

        // Unbind GPU
        print!("HRR Unbind GPU (cuFFT, {}d)... ", HRR_DIM);
        let trace_for_gpu = hrr_gpu.bind(&a, &b).unwrap();

        if let Device::Cuda(cuda) = &device {
            cuda.synchronize().unwrap();
        }

        let start = Instant::now();
        for _ in 0..ITERATIONS {
            std::hint::black_box(hrr_gpu.unbind(&trace_for_gpu, &a));
        }

        if let Device::Cuda(cuda) = &device {
            cuda.synchronize().unwrap();
        }

        let elapsed = start.elapsed();
        let unbind_gpu_us = elapsed.as_micros() as f64 / ITERATIONS as f64;
        let speedup = unbind_us / unbind_gpu_us;
        println!("{:.2} μs/op ({:.2}K ops/sec) [{:.1}x faster than CPU]",
                 unbind_gpu_us, 1000.0 / unbind_gpu_us, speedup);
    }

    // ========================================================================
    // Benchmark: Batch Similarity (parallel)
    // ========================================================================
    print!("Batch Similarity (1000 vectors, parallel)... ");
    let vectors: Vec<Vec<f32>> = (0..1000)
        .map(|i| (0..HRR_DIM).map(|j| ((i * j) as f32 * 0.0001).sin()).collect())
        .collect();
    let query = a.clone();

    let start = Instant::now();
    for _ in 0..100 {
        let _: Vec<f32> = vectors.par_iter()
            .map(|v| cosine_similarity_simd(v, &query))
            .collect();
    }
    let elapsed = start.elapsed();
    let batch_ms = elapsed.as_millis() as f64 / 100.0;
    println!("{:.2} ms/batch ({:.2}M comparisons/sec)", batch_ms, 1000.0 / batch_ms);

    // ========================================================================
    // Summary
    // ========================================================================
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║                         SUMMARY                              ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║  Operation              │ Latency      │ Throughput          ║");
    println!("╠─────────────────────────┼──────────────┼─────────────────────╣");
    println!("║  Cosine Similarity      │ {:>8.2} μs  │ {:>8.2}M ops/sec   ║", sim_us, 1.0 / (sim_us / 1_000_000.0));
    println!("║  HRR Bind (8192d)       │ {:>8.2} μs  │ {:>8.2}K ops/sec   ║", bind_us, 1000.0 / bind_us);
    println!("║  HRR Unbind (8192d)     │ {:>8.2} μs  │ {:>8.2}K ops/sec   ║", unbind_us, 1000.0 / unbind_us);
    println!("║  Projection CPU         │ {:>8.2} μs  │ {:>8.2} ops/sec    ║", proj_cpu_us, 1_000_000.0 / proj_cpu_us);
    println!("║  Batch Sim (1K vectors) │ {:>8.2} ms  │ {:>8.2}M cmp/sec   ║", batch_ms, 1000.0 / batch_ms);
    println!("╚══════════════════════════════════════════════════════════════╝");

    // Verify correctness
    println!("\n[Correctness Check]");
    let sim = cosine_similarity_simd(&a, &a);
    println!("  self-similarity: {:.6} (expected: 1.0)", sim);

    let bound = workspace.bind(&a, &b);
    let recovered = workspace.unbind(&bound, &a);
    let recovery_sim = cosine_similarity_simd(&recovered, &b);
    println!("  bind→unbind recovery: {:.6} (higher = better)", recovery_sim);

    if matches!(device, Device::Cuda(_)) {
        let projection_gpu = ProjectionGPU::new(LLM_DIM, HRR_DIM, 42, &device).unwrap();
        let gpu_result = projection_gpu.project(&embedding).unwrap();
        let cpu_result = projection_cpu.project(&embedding);
        let gpu_cpu_sim = cosine_similarity_simd(&gpu_result, &cpu_result);
        println!("  GPU vs CPU projection: {:.6} (expected: ~1.0)", gpu_cpu_sim);
    }
}

// ============================================================================
// SIMD Operations
// ============================================================================

#[inline(always)]
fn dot_product_simd(a: &[f32], b: &[f32]) -> f32 {
    let chunks = a.len() / 8;
    let remainder = a.len() % 8;
    let mut sum = 0.0f32;

    for i in 0..chunks {
        let offset = i * 8;
        let a_chunk = &a[offset..offset + 8];
        let b_chunk = &b[offset..offset + 8];
        sum += a_chunk[0] * b_chunk[0]
             + a_chunk[1] * b_chunk[1]
             + a_chunk[2] * b_chunk[2]
             + a_chunk[3] * b_chunk[3]
             + a_chunk[4] * b_chunk[4]
             + a_chunk[5] * b_chunk[5]
             + a_chunk[6] * b_chunk[6]
             + a_chunk[7] * b_chunk[7];
    }

    let offset = chunks * 8;
    for i in 0..remainder {
        sum += a[offset + i] * b[offset + i];
    }
    sum
}

#[inline(always)]
fn norm_simd(v: &[f32]) -> f32 {
    dot_product_simd(v, v).sqrt()
}

#[inline(always)]
fn cosine_similarity_simd(a: &[f32], b: &[f32]) -> f32 {
    let dot = dot_product_simd(a, b);
    let norm_a = norm_simd(a);
    let norm_b = norm_simd(b);
    if norm_a == 0.0 || norm_b == 0.0 { return 0.0; }
    dot / (norm_a * norm_b)
}

// ============================================================================
// HRR Workspace
// ============================================================================

struct HrrWorkspace {
    fft_planner: FftPlanner<f32>,
    dim: usize,
    buffer_a: Vec<Complex32>,
    buffer_b: Vec<Complex32>,
    buffer_result: Vec<Complex32>,
}

impl HrrWorkspace {
    fn new(dim: usize) -> Self {
        let mut planner = FftPlanner::new();
        let _ = planner.plan_fft_forward(dim);
        let _ = planner.plan_fft_inverse(dim);
        Self {
            fft_planner: planner,
            dim,
            buffer_a: vec![Complex32::new(0.0, 0.0); dim],
            buffer_b: vec![Complex32::new(0.0, 0.0); dim],
            buffer_result: vec![Complex32::new(0.0, 0.0); dim],
        }
    }

    fn bind(&mut self, a: &[f32], b: &[f32]) -> Vec<f32> {
        let n = self.dim;
        let fft = self.fft_planner.plan_fft_forward(n);
        let ifft = self.fft_planner.plan_fft_inverse(n);

        for i in 0..n {
            self.buffer_a[i] = Complex32::new(a[i], 0.0);
            self.buffer_b[i] = Complex32::new(b[i], 0.0);
        }

        fft.process(&mut self.buffer_a);
        fft.process(&mut self.buffer_b);

        self.buffer_result.par_iter_mut()
            .zip(self.buffer_a.par_iter())
            .zip(self.buffer_b.par_iter())
            .for_each(|((r, a), b)| *r = a * b);

        ifft.process(&mut self.buffer_result);

        let scale = 1.0 / n as f32;
        self.buffer_result.iter().map(|c| c.re * scale).collect()
    }

    fn unbind(&mut self, trace: &[f32], key: &[f32]) -> Vec<f32> {
        let n = self.dim;
        let fft = self.fft_planner.plan_fft_forward(n);
        let ifft = self.fft_planner.plan_fft_inverse(n);

        for i in 0..n {
            self.buffer_a[i] = Complex32::new(trace[i], 0.0);
            self.buffer_b[i] = Complex32::new(key[i], 0.0);
        }

        fft.process(&mut self.buffer_a);
        fft.process(&mut self.buffer_b);

        self.buffer_result.par_iter_mut()
            .zip(self.buffer_a.par_iter())
            .zip(self.buffer_b.par_iter())
            .for_each(|((r, t), k)| *r = t * k.conj());

        ifft.process(&mut self.buffer_result);

        let scale = 1.0 / n as f32;
        self.buffer_result.iter().map(|c| c.re * scale).collect()
    }
}

// ============================================================================
// CPU Projection (INT8 quantized)
// ============================================================================

struct QuantizedProjectionCPU {
    weights_i8: Vec<i8>,
    scale: f32,
    llm_dim: usize,
    hrr_dim: usize,
}

impl QuantizedProjectionCPU {
    fn new(llm_dim: usize, hrr_dim: usize, seed: u64) -> Self {
        use std::hash::{Hash, Hasher};
        use std::collections::hash_map::DefaultHasher;

        let mut weights = Vec::with_capacity(llm_dim * hrr_dim);
        let mut hasher = DefaultHasher::new();
        let scale = 127.0 / (llm_dim as f32).sqrt();

        for i in 0..(llm_dim * hrr_dim) {
            (seed, i as u64).hash(&mut hasher);
            let hash = hasher.finish();
            let val = ((hash as f32 / u64::MAX as f32) * 2.0 - 1.0) * scale;
            weights.push((val * 127.0).clamp(-127.0, 127.0) as i8);
            hasher = DefaultHasher::new();
        }

        Self { weights_i8: weights, scale: 1.0 / 127.0, llm_dim, hrr_dim }
    }

    fn project(&self, embedding: &[f32]) -> Vec<f32> {
        let mut result = vec![0.0f32; self.hrr_dim];

        result.par_iter_mut().enumerate().for_each(|(j, out)| {
            let mut sum = 0i32;
            let row_offset = j * self.llm_dim;
            for i in 0..self.llm_dim {
                let w = self.weights_i8[row_offset + i] as i32;
                let x = (embedding[i] * 127.0) as i32;
                sum += w * x;
            }
            *out = sum as f32 * self.scale * self.scale;
        });

        let norm = norm_simd(&result);
        if norm > 0.0 {
            let inv_norm = 1.0 / norm;
            result.par_iter_mut().for_each(|x| *x *= inv_norm);
        }
        result
    }
}

// ============================================================================
// GPU HRR (cuFFT via cudarc low-level API)
// ============================================================================

use cudarc::cufft::{result as cufft, sys as cufft_sys};
use cudarc::driver::{result as cuda, sys as cuda_sys};
use cudarc::nvrtc::result as nvrtc;
use std::ffi::{c_int, c_void, CString};

// cuFFT direction constants
const CUFFT_FORWARD: c_int = -1;
const CUFFT_INVERSE: c_int = 1;

// CUDA kernel for complex multiplication (compiles at runtime via NVRTC)
const COMPLEX_MUL_KERNEL: &str = r#"
extern "C" __global__ void complex_mul(
    const float2* __restrict__ a,
    const float2* __restrict__ b,
    float2* __restrict__ out,
    int n,
    int conjugate_b
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float ar = a[idx].x;
        float ai = a[idx].y;
        float br = b[idx].x;
        float bi = conjugate_b ? -b[idx].y : b[idx].y;

        out[idx].x = ar * br - ai * bi;
        out[idx].y = ar * bi + ai * br;
    }
}
"#;

/// GPU-accelerated HRR using cuFFT + custom CUDA kernel
/// ZERO intermediate copies - everything stays on GPU!
struct HrrGPU {
    dim: usize,
    plan: cufft_sys::cufftHandle,
    // GPU memory pointers for cuFFT
    d_a: cuda_sys::CUdeviceptr,
    d_b: cuda_sys::CUdeviceptr,
    d_result: cuda_sys::CUdeviceptr,
    // CUDA kernel
    kernel_func: cuda_sys::CUfunction,
    _module: cuda_sys::CUmodule,
}

impl HrrGPU {
    fn new(dim: usize, _candle_device: &Device) -> Result<Self, Box<dyn std::error::Error>> {
        // Initialize CUDA driver
        cuda::init()?;

        // Get device and create context
        let cuda_dev = cuda::device::get(0)?;
        let ctx = unsafe { cuda::primary_ctx::retain(cuda_dev)? };
        unsafe { cuda::ctx::set_current(ctx)?; }

        // Create cuFFT plan for C2C transform
        let plan = cufft::plan_1d(dim as c_int, cufft_sys::cufftType::CUFFT_C2C, 1)?;

        // Allocate GPU memory (complex = 2 floats per element)
        let size_bytes = dim * 2 * std::mem::size_of::<f32>();
        let d_a = unsafe { cuda::malloc_sync(size_bytes)? };
        let d_b = unsafe { cuda::malloc_sync(size_bytes)? };
        let d_result = unsafe { cuda::malloc_sync(size_bytes)? };

        // Compile CUDA kernel at runtime using NVRTC
        let src = CString::new(COMPLEX_MUL_KERNEL)?;
        let program = nvrtc::create_program(&src, Some(c"complex_mul.cu"))?;

        // Compile (empty options for default)
        let opts: Vec<String> = vec![];
        unsafe {
            match nvrtc::compile_program(program, &opts) {
                Ok(_) => {}
                Err(e) => {
                    let log = nvrtc::get_program_log(program)?;
                    let log_str: String = log.iter().map(|&c| c as u8 as char).collect();
                    eprintln!("NVRTC compile error: {:?}\nLog: {}", e, log_str);
                    return Err(format!("NVRTC compile failed: {}", log_str).into());
                }
            }
        }

        let ptx = unsafe { nvrtc::get_ptx(program)? };
        unsafe { nvrtc::destroy_program(program)?; }

        // Load the compiled module (PTX as null-terminated string)
        let module = unsafe { cuda::module::load_data(ptx.as_ptr() as *const c_void)? };
        let func_name = CString::new("complex_mul")?;
        let kernel_func = unsafe { cuda::module::get_function(module, func_name)? };

        Ok(Self {
            dim,
            plan,
            d_a,
            d_b,
            d_result,
            kernel_func,
            _module: module,
        })
    }

    /// Launch custom CUDA kernel for complex multiplication - ZERO copies!
    fn complex_mul_gpu(&self, conjugate_b: bool) -> Result<(), Box<dyn std::error::Error>> {
        let n = self.dim as c_int;
        let conj = if conjugate_b { 1i32 } else { 0i32 };

        // Kernel launch parameters
        let block_size = 256u32;
        let grid_size = ((self.dim as u32) + block_size - 1) / block_size;

        // Kernel arguments - must be mutable pointers
        let mut args: Vec<*mut c_void> = vec![
            &self.d_a as *const _ as *mut _,
            &self.d_b as *const _ as *mut _,
            &self.d_result as *const _ as *mut _,
            &n as *const _ as *mut _,
            &conj as *const _ as *mut _,
        ];

        unsafe {
            cuda::launch_kernel(
                self.kernel_func,
                (grid_size, 1, 1),        // grid dimensions
                (block_size, 1, 1),       // block dimensions
                0,                         // shared memory
                std::ptr::null_mut(),      // default stream
                &mut args,
            )?;
        }

        Ok(())
    }

    fn bind(&self, a: &[f32], b: &[f32]) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let n = self.dim;

        // Convert to complex (interleaved real/imag)
        let mut a_complex: Vec<f32> = Vec::with_capacity(n * 2);
        let mut b_complex: Vec<f32> = Vec::with_capacity(n * 2);
        for i in 0..n {
            a_complex.push(a[i]);
            a_complex.push(0.0);
            b_complex.push(b[i]);
            b_complex.push(0.0);
        }

        // Copy to GPU (only transfer at boundary)
        unsafe {
            cuda::memcpy_htod_sync(self.d_a, &a_complex)?;
            cuda::memcpy_htod_sync(self.d_b, &b_complex)?;
        }

        // Forward FFT (in-place) on GPU
        unsafe {
            cufft::exec_c2c(
                self.plan,
                self.d_a as *mut cufft_sys::cufftComplex,
                self.d_a as *mut cufft_sys::cufftComplex,
                CUFFT_FORWARD,
            )?;
            cufft::exec_c2c(
                self.plan,
                self.d_b as *mut cufft_sys::cufftComplex,
                self.d_b as *mut cufft_sys::cufftComplex,
                CUFFT_FORWARD,
            )?;
        }

        // Complex multiplication on GPU (custom kernel - ZERO COPIES!)
        self.complex_mul_gpu(false)?;

        // Inverse FFT on GPU
        unsafe {
            cufft::exec_c2c(
                self.plan,
                self.d_result as *mut cufft_sys::cufftComplex,
                self.d_result as *mut cufft_sys::cufftComplex,
                CUFFT_INVERSE,
            )?;
        }

        // Copy result back (only transfer at boundary)
        let mut result = vec![0.0f32; n * 2];
        unsafe {
            cuda::memcpy_dtoh_sync(&mut result, self.d_result)?;
        }

        let scale = 1.0 / n as f32;
        let output: Vec<f32> = (0..n).map(|i| result[i * 2] * scale).collect();

        Ok(output)
    }

    fn unbind(&self, trace: &[f32], key: &[f32]) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let n = self.dim;

        // Convert to complex
        let mut trace_complex: Vec<f32> = Vec::with_capacity(n * 2);
        let mut key_complex: Vec<f32> = Vec::with_capacity(n * 2);
        for i in 0..n {
            trace_complex.push(trace[i]);
            trace_complex.push(0.0);
            key_complex.push(key[i]);
            key_complex.push(0.0);
        }

        // Copy to GPU
        unsafe {
            cuda::memcpy_htod_sync(self.d_a, &trace_complex)?;
            cuda::memcpy_htod_sync(self.d_b, &key_complex)?;
        }

        // Forward FFT on GPU
        unsafe {
            cufft::exec_c2c(
                self.plan,
                self.d_a as *mut cufft_sys::cufftComplex,
                self.d_a as *mut cufft_sys::cufftComplex,
                CUFFT_FORWARD,
            )?;
            cufft::exec_c2c(
                self.plan,
                self.d_b as *mut cufft_sys::cufftComplex,
                self.d_b as *mut cufft_sys::cufftComplex,
                CUFFT_FORWARD,
            )?;
        }

        // Complex multiplication with conjugate on GPU (custom kernel)
        self.complex_mul_gpu(true)?;

        // Inverse FFT on GPU
        unsafe {
            cufft::exec_c2c(
                self.plan,
                self.d_result as *mut cufft_sys::cufftComplex,
                self.d_result as *mut cufft_sys::cufftComplex,
                CUFFT_INVERSE,
            )?;
        }

        // Copy result back
        let mut result = vec![0.0f32; n * 2];
        unsafe {
            cuda::memcpy_dtoh_sync(&mut result, self.d_result)?;
        }

        let scale = 1.0 / n as f32;
        let output: Vec<f32> = (0..n).map(|i| result[i * 2] * scale).collect();

        Ok(output)
    }
}

impl Drop for HrrGPU {
    fn drop(&mut self) {
        unsafe {
            let _ = cufft::destroy(self.plan);
            let _ = cuda::memory_free(self.d_a);
            let _ = cuda::memory_free(self.d_b);
            let _ = cuda::memory_free(self.d_result);
            let _ = cuda::module::unload(self._module);
            if let Ok(dev) = cuda::device::get(0) {
                let _ = cuda::primary_ctx::release(dev);
            }
        }
    }
}

// ============================================================================
// GPU Projection (FP32 via Candle CUDA)
// ============================================================================

struct ProjectionGPU {
    weights: Tensor,  // [hrr_dim, llm_dim] on GPU
    device: Device,
    llm_dim: usize,
    hrr_dim: usize,
}

impl ProjectionGPU {
    fn new(llm_dim: usize, hrr_dim: usize, seed: u64, device: &Device) -> Result<Self, candle_core::Error> {
        use std::hash::{Hash, Hasher};
        use std::collections::hash_map::DefaultHasher;

        // Generate weights on CPU then transfer to GPU
        let mut weights_data = Vec::with_capacity(hrr_dim * llm_dim);
        let mut hasher = DefaultHasher::new();
        let scale = 1.0 / (llm_dim as f32).sqrt();

        for i in 0..(hrr_dim * llm_dim) {
            (seed, i as u64).hash(&mut hasher);
            let hash = hasher.finish();
            let val = ((hash as f32 / u64::MAX as f32) * 2.0 - 1.0) * scale;
            weights_data.push(val);
            hasher = DefaultHasher::new();
        }

        let weights = Tensor::from_vec(weights_data, (hrr_dim, llm_dim), device)?;

        Ok(Self {
            weights,
            device: device.clone(),
            llm_dim,
            hrr_dim,
        })
    }

    fn project(&self, embedding: &[f32]) -> Result<Vec<f32>, candle_core::Error> {
        // Transfer embedding to GPU
        let input = Tensor::from_vec(embedding.to_vec(), (1, self.llm_dim), &self.device)?;

        // Matrix multiply: [1, llm_dim] x [llm_dim, hrr_dim]^T = [1, hrr_dim]
        // weights is [hrr_dim, llm_dim], input is [1, llm_dim]
        // result = input @ weights.T
        let result = input.matmul(&self.weights.t()?)?;

        // Normalize on GPU
        let norm = result.sqr()?.sum_all()?.sqrt()?;
        let normalized = result.broadcast_div(&norm)?;

        // Transfer back to CPU
        let output = normalized.flatten_all()?.to_vec1::<f32>()?;

        Ok(output)
    }
}
