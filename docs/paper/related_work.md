# 2. Related Work

## 2.1 Quality-Diversity Algorithms

Quality-Diversity (QD) optimization represents a paradigm shift from single-objective optimization toward maintaining a diverse collection of high-performing solutions. The field emerged from the observation that diversity maintenance can improve both exploration and final solution quality.

**Novelty Search** (Lehman & Stanley, 2011) demonstrated that optimizing purely for behavioral novelty can outperform fitness-based search in deceptive domains. However, novelty search alone does not guarantee high-quality solutions.

**MAP-Elites** (Mouret & Clune, 2015) discretizes the behavioral space into a grid and maintains the highest-fitness solution in each cell. The algorithm has been successfully applied to robotics (Cully et al., 2015), game playing (Fontaine et al., 2020), and procedural content generation (Gravina et al., 2019).

**CVT-MAP-Elites** (Vassiliades et al., 2018) replaces the uniform grid with Centroidal Voronoi Tessellation, improving coverage in high-dimensional behavior spaces. Our HoloMAP extension uses HRR projections to achieve similar benefits with fixed grid resolution.

**CMA-ME** (Fontaine & Nikolaidis, 2021) combines MAP-Elites with CMA-ES for continuous optimization, achieving state-of-the-art results on QD benchmarks. VIVA-QD incorporates similar principles through NES gradient updates within archive cells.

## 2.2 NEAT and Topology Evolution

**NEAT** (Stanley & Miikkulainen, 2002) revolutionized neuroevolution by evolving both network topology and weights simultaneously. Key innovations include:

- Historical markings (innovation numbers) for crossover alignment
- Speciation to protect structural innovations
- Minimal network initialization with complexification

**HyperNEAT** (Stanley et al., 2009) extends NEAT with indirect encoding through Compositional Pattern-Producing Networks (CPPNs), enabling evolution of large-scale networks with geometric regularities.

**NEAT-Python** and **neat-gru** provide widely-used implementations, but none leverage GPU acceleration for the core NEAT operations. Our work maintains NEAT's topology evolution while accelerating fitness evaluation.

## 2.3 GPU-Accelerated Neuroevolution

**EvoJAX** (Tang et al., 2022) implements neuroevolution algorithms in JAX, achieving significant speedups through vectorized operations. However, it focuses on fixed-topology networks and does not support NEAT-style structural mutation.

**PGPE-GPU** and **OpenAI-ES** implementations demonstrate that population-based methods can benefit enormously from GPU parallelism when network architectures are fixed.

**Accelerated Differential Evolution** (Wang et al., 2020) achieves 100x speedups using CUDA, but the approach is limited to direct weight encoding.

Our contribution differs by:
1. Supporting variable-topology networks (NEAT)
2. Integrating with functional programming (Erlang/Gleam)
3. Providing fault tolerance through OTP supervision

## 2.4 Holographic Reduced Representations

**HRR** (Plate, 1995) provides a distributed representation for symbolic structures using circular convolution. Key properties include:

- Binding: Two vectors can be bound via circular convolution
- Unbinding: Approximate retrieval via circular correlation
- Superposition: Multiple bindings can be summed
- Fixed dimensionality regardless of structure depth

HRRs have been applied to:
- Analogical reasoning (Eliasmith, 2013)
- Neural semantic memory (Gayler, 2003)
- Cognitive architectures (Rachkovskij, 2001)

To our knowledge, VIVA-QD is the first application of HRR to Quality-Diversity genome encoding.

## 2.5 Erlang/OTP in Machine Learning

The Erlang ecosystem has seen limited adoption in ML due to historical performance concerns. Recent developments include:

**Nx** (Numerical Elixir) provides tensor operations with backend-agnostic compilation to CPU/GPU. However, Nx targets direct computation rather than evolutionary algorithms.

**Axon** builds neural network abstractions on Nx but lacks neuroevolution support.

**Rustler** enables Rust NIFs with automatic NIF safety checks, forming the bridge between BEAM and native code that VIVA-QD exploits.

Our work demonstrates that the BEAM's strengths in fault tolerance and concurrency can complement GPU acceleration rather than compete with it.

## 2.6 Positioning VIVA-QD

| System | QD Support | NEAT Topology | GPU Accel | Fault Tolerant |
|--------|-----------|---------------|-----------|----------------|
| NEAT-Python | No | Yes | No | No |
| EvoJAX | Limited | No | Yes | No |
| CMA-ME | Yes | No | CPU only | No |
| VIVA-QD | Yes | Yes | Yes | Yes |

VIVA-QD uniquely combines all four properties, enabled by its heterogeneous Gleam-Rust-CUDA architecture.
