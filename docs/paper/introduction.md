# 1. Introduction

## 1.1 Motivation

Neuroevolution has demonstrated remarkable success in evolving neural network topologies and weights for complex control tasks. However, traditional evolutionary approaches face two fundamental challenges: (1) computational bottlenecks that limit evaluation throughput, and (2) premature convergence to local optima due to fitness-only selection pressure.

Quality-Diversity (QD) algorithms address the second challenge by maintaining archives of diverse, high-performing solutions across behavioral dimensions. MAP-Elites (Mouret & Clune, 2015) has become the dominant QD algorithm, but its practical application has been limited by the computational cost of evaluating thousands of solutions per generation.

Recent advances in GPU computing have accelerated deep learning dramatically, yet neuroevolution has not fully benefited from these improvements. The primary obstacle is the inherent sequential nature of most neuroevolution algorithms: topology mutation, speciation, and crossover operations are difficult to parallelize efficiently.

We present VIVA-QD, a framework that overcomes these limitations through a novel architectural approach: leveraging the Erlang BEAM virtual machine for orchestration and fault tolerance while offloading compute-intensive evaluation to Rust NIFs with GPU acceleration.

## 1.2 Contributions

This paper makes the following contributions:

1. **Gleam-Erlang-Rust-CUDA Pipeline**: We introduce a four-layer architecture that combines:
   - Gleam (statically-typed Erlang) for type-safe algorithm implementation
   - Erlang OTP for fault-tolerant supervision and actor-based parallelism
   - Rust NIFs for zero-copy data transfer and memory safety
   - CUDA kernels for parallel neural network evaluation and HRR operations

2. **HoloMAP-Elites**: An extension of MAP-Elites that uses Holographic Reduced Representations (Plate, 1995) for:
   - Compact genome encoding in holographic space (8192 dimensions)
   - Behavior descriptor extraction from HRR vector projections
   - Circular convolution-based crossover in frequency domain (cuFFT)

3. **Decoupled QD-NEAT Hybrid**: A novel selection mechanism that:
   - Separates fitness evaluation from selection pressure
   - Implements linear weight annealing (w=0.7 to w=0.3) over generations
   - Maintains cell-wise elitism with archive-based parent selection

4. **Performance Benchmarks**: Empirical evaluation demonstrating:
   - 320,000 evaluations/second sustained throughput
   - 50,000x speedup over sequential baseline
   - 4,800 parallel physics simulations per batch
   - 57.9% MAP-Elites coverage in 50 generations

## 1.3 Problem Domain

We evaluate VIVA-QD on Brazilian sinuca (snooker variant), a challenging billiards domain requiring:

- **Geometric Reasoning**: Computing optimal cue ball angles to pocket target balls
- **Multi-step Planning**: Positioning the cue ball for subsequent shots
- **Continuous Control**: 3 continuous outputs (angle, power, english)
- **Partial Observability**: Limited information about table friction and ball interactions

The domain provides a rich behavioral space with clear metrics (balls pocketed, position quality, combo length) while maintaining computational tractability for high-throughput evaluation.

## 1.4 Paper Organization

Section 2 reviews related work in QD algorithms, NEAT, and GPU-accelerated evolution. Section 3 details our methodology including the HoloMAP-Elites algorithm and GPU pipeline. Section 4 presents experimental results on the sinuca domain. Section 5 discusses implications and limitations. Section 6 concludes with future research directions.
