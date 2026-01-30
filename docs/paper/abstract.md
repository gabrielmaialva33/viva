# Abstract

**VIVA-QD: A Hybrid Quality-Diversity Neuroevolution Framework with GPU-Accelerated Evaluation on Erlang/OTP**

We present VIVA-QD, a novel neuroevolution framework that combines Quality-Diversity (QD) algorithms with NEAT topology evolution, achieving 320,000 evaluations per second on a single RTX 4090 GPU. Our system introduces three key innovations: (1) a unique Gleam-to-Rust-to-CUDA pipeline that bridges functional programming with GPU compute, (2) HoloMAP-Elites, an extension of MAP-Elites using Holographic Reduced Representations (HRR) for compact behavioral encoding, and (3) a decoupled fitness-selection mechanism with linear weight annealing that balances exploration and exploitation.

We evaluate VIVA-QD on a Brazilian sinuca (billiards) domain, demonstrating that our approach achieves 57.9% MAP-Elites grid coverage and a QD-Score of 908.3 within 50 generations. The framework processes 4,800 parallel physics simulations through a custom Jolt Physics integration, delivering a 50,000x speedup over sequential CPU evaluation.

Our implementation leverages the Erlang BEAM virtual machine for fault-tolerant supervision while offloading compute-intensive operations to Rust Native Implemented Functions (NIFs) that utilize cuFFT for HRR circular convolution and burn-rs for batched neural network inference. This architecture enables the first practical deployment of Quality-Diversity neuroevolution on functional actor-based systems, opening new research directions for robust, self-healing evolutionary systems.

**Keywords:** Quality-Diversity, NEAT, Neuroevolution, MAP-Elites, GPU Acceleration, Functional Programming, Erlang, Gleam, Holographic Reduced Representations
