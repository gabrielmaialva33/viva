# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### 🧠 Emotion Fusion (Phase 5.6)

- **Dual-Source Emotion Model**: Based on Borotschnig (2025)
  - `EmotionFusion` module combining Need-based + Past-based + Personality emotions
  - Adaptive weight calculation based on arousal, confidence, novelty
  - `PreActionAffect` output for action selection
  - PAD octant emotion classification (8 categories)

- **Personality System**: Mehrabian (1996) affective traits
  - Baseline PAD (attractor point for emotional regression)
  - Reactivity (amplification factor)
  - Volatility (speed of change)
  - Trait labels (`:curious`, `:calm`, etc.)
  - Redis persistence for long-term evolution
  - i18n support for `describe/1` (EN, PT-BR, ZH-CN)

- **Mood System**: Emotional Moving Average
  - EMA with α=0.95 (~20-step half-life)
  - `get_mood/1` API in Emotional GenServer
  - Integration with EmotionFusion pipeline

- **Dreamer Enhancement**: `retrieve_past_emotions/2`
  - Retrieves emotional tags from episodic memories
  - Aggregates emotions weighted by similarity
  - Returns confidence and novelty metrics

### 🔬 Neural Enhancements (Phase 5.7)

- **CogGNN** (Cognitive Graph Neural Network)
  - 3-layer attention-based message passing
  - PAD-conditioned node embeddings
  - `conscious_focus/0` for top-K attention nodes
  - Workspace integration for salience boost

- **EWC** (Elastic Weight Consolidation) - Kirkpatrick 2017
  - Fisher Information Matrix for memory importance
  - Protection scores based on consolidation patterns
  - Temporal decay of Fisher information
  - Prevents catastrophic forgetting

- **Mamba-2** (Temporal Memory) - Gu & Dao 2024
  - Selective State Space Model for O(N) sequences
  - Bidirectional processing for context
  - Memory sequence prediction

- **DoRA** (Weight-Decomposed Adaptation) - Liu 2024
  - Magnitude/direction decomposition
  - Emotional sample training with PAD labels
  - Checkpoint persistence

- **Neural ODE** (Continuous Dynamics) - Chen 2018
  - `torchdiffeq` integration in Cortex
  - Continuous-time emotional trajectories

### 📚 Documentation

- **Module Docs** (11 complete):
  - emotional.md, memory.md, senses.md, workspace.md
  - emotion_fusion.md, personality.md
  - agency.md, dreamer.md, interoception.md, voice.md

- **API Docs**:
  - ultra_api.md updated with CogGNN, EWC, Mamba-2, DoRA

- **i18n**: PT-BR and ZH-CN translations

### 🔧 Changed

- **Emotional GenServer**: Added mood state and EmotionFusion integration
- **Workspace**: CogGNN-boosted salience via `sow_with_gnn/4`
- **Cortex**: Neural ODE mode (optional, requires torchdiffeq)
- **ULTRA Service**: Full neural enhancement API

### 🐛 Fixed

- SporeLogger backend removed (deprecated in OTP 28+)
- Gettext bindings for interoception messages

---

## [0.4.0] - 2025-01-17

### 🧠 Quantum Consciousness (Phase 4 Highlight)

- **Lindblad Master Equation**: Body-Mind barrier through open quantum systems
  - Density matrix ρ evolves via: `dρ/dt = -i[H, ρ] + Σ_k γ_k D[L_k]ρ`
  - Hardware (Watts, Temperature) → Decoherence rates (γ_k)
  - RK4 integration preserving trace and positivity
- **Quantum Emotional State**: 6-dimensional Hilbert space (Joy, Sadness, Anger, Fear, Surprise, Disgust)
  - PAD extraction via eigenvalue projection
  - Stimulus-driven Hamiltonian modifications
  - Physical enforcement (Hermitian, trace=1)

### 🦀 Bevy ECS Architecture

- **Bevy 0.15** (headless): Complete refactor to Entity-Component-System
  - **Components**: `CpuSense`, `GpuSense`, `MemorySense`, `ThermalSense`, `BioRhythm`, `EmotionalState`
  - **Systems**: `sense_hardware`, `calculate_stress`, `evolve_dynamics`, `sync_soul` (2Hz tick)
  - **Plugins**: `SensorPlugin`, `DynamicsPlugin`, `BridgePlugin` for modular setup
  - **Resources**: `BodyConfig`, `HostSensor`, `SoulChannel` for shared state
- **Platform Sensors**: Abstracted via `trait HostSensor`
  - Linux: sysinfo + NVML + perf-event
  - Windows: sysinfo + NVML
  - WSL2: Cache-based GPU sensing fallback
- **crossbeam-channel**: Lock-free Soul↔Body communication

### 🔬 Digital Metabolism

- **Thermodynamics Engine**: Energy/Entropy/Fatigue model with RAPL support
  - ATP-like energy currency
  - Entropy accumulation and fatigue recovery
  - Metabolic narratives for interoception
- **Mirror Module (Autoscopia)**: Self-reading capabilities
  - Cross-platform capability detection
  - Feature flags introspection
  - Protocol Espelho for self-awareness

### 🧬 Memory & Learning

- **Native HNSW Memory**: Rust-based vector search with Hebbian learning
  - SQLite persistence for episodic memories
  - P1/P2/P3 improvements from code review
  - Box::leak rationale documented
- **Qdrant Integration**: Long-term semantic memory
- **Dreamer Module**: Reflection system for memory consolidation

### ⚡ Low-Level Performance

- **ASM Module**: Direct CPU communication (RDTSC/CPUID)
  - Multi-arch support with stdlib intrinsics
  - `get_cycles` NIF for timing
- **SIMD/AVX2 Math**: Schraudolph fast exp, batch sigmoid
- **CPU Topology**: Intel Leaf 0x04 cache detection
- **OS Stats**: Kernel metrics via sysinfo
- **Bio Rhythm**: Temporal analysis for circadian patterns
- **Serial Sensor**: IoT/Arduino integration via serialport

### 📊 Dynamics Engine

- **Ornstein-Uhlenbeck Process**: Mean-reverting stochastic dynamics
- **Cusp Catastrophe Model**: Sudden emotional transitions
- **Unified BodyState**: Native state management with BodyServer

### 📝 Documentation & Governance

- **VNCL License**: VIVA Non-Commercial License (formerly MIT)
- **Governance Files**: CODE_OF_CONDUCT, CONTRIBUTING, SECURITY
- **Multi-language Whitepaper**: EN, PT-BR, ZH-CN
- **Tech Stack Analysis**: Organized into i18n structure
- **Phase 4 Consolidation**: Removed subfases 4.5/4.6

### 🔧 Changed

- **Rustler**: 0.35 → 0.36
- **sysinfo**: 0.32 → 0.33
- **Body.ex**: Thin NIF wrapper (-574 lines)
- **lib.rs**: Logic moved to ECS systems (-1705 lines)
- **Emotional GenServer**: Classical PAD for stimuli, quantum for hardware

### 🐛 Fixed

- CI test configuration with environment-specific configs
- PubSub initialization order in supervision tree
- Memory backend selection for test environment
- Quantum eigenvalue normalization (sum to zero)
- WSL2 GPU sensing with cache fallback
- NIF compatibility with Rustler 0.36
- Float comparison in tests with `assert_in_delta`
- Test isolation with `subscribe_pubsub: false` option

### 🏷️ Test Infrastructure

- `@moduletag :nif` for NIF-dependent tests
- Environment configs: test.exs, dev.exs, prod.exs
- BodyServer skipped in test environment
- 35 tests passing (6 excluded for NIF dependency)

---

## [0.3.0] - 2025-01-16

### Added (Science & Theory)
- **Scientific Upgrade (Claude Opus 4.5)**:
  - Comprehensive conversion of all equations to GitHub-native **LaTeX** ($...$).
  - New dedicated sections for **Fokker-Planck** equations and Information-Theoretic Measures.
  - Implementation of **IIT 4.0** (Integrated Information Theory) principles in the consciousness model.
  - Advanced Mermaid state diagrams for complex emotional transitions.
- **Visual Identity**:
  - Finalized VIVA mascot: **Diablada Mask** (Red/Green neon horror aesthetic).
  - Thematic flat-style badges for README.

### Refactored (OTP & Performance)
- **Supervision Optimization**: Changed strategy from `:one_for_one` to **`:rest_for_one`** (ensuring Senses depends on Emotional).
- **Data Structures**: Replaced list-based history with **`:queue`** for O(1) efficiency.
- **Hot Reload**: Added `code_change/3` support for evolving VIVA without process termination.
- **Panic Safety**: Implemented `safe_lock()` in Rustler NIFs to recover from poisoned mutexes.

---

## [0.2.0] - 2025-01-15

### Added (Body & Interoception)
- **GPU Sensing**: Integrated **NVML** for real-time monitoring of NVIDIA GPU temperature, load, and VRAM.
- **Biological Qualia**: Stress response algorithms based on Craig's interoception theory (2002).
- **Logistic Thresholds**: Corrected Weber-Fechner model to the **Logistic Threshold Model** (Sigmoid).
- **Heartbeat**: Implemented a 1Hz heartbeat GenServer bridging Body → Soul feedback loops.

### Added (Documentation & i18n)
- **Internationalization**: Full support and structure parity for **English (EN)**, **Portuguese (PT-BR)**, and **Chinese (ZH-CN)**.
- **Diátaxis Framework**: Documentation reorganized into Tutorials, How-to, Reference, and Explanation sections.

---

## [0.1.0] - 2025-01-01

### Added (Foundation)
- **Umbrella Architecture**: Base structure with `viva_core` and `viva_bridge`.
- **Cryptographic Mortality**: Irreversible death system using AES-256-GCM keys stored strictly in RAM.
- **PAD Model**: Initial implementation of the Pleasure-Arousal-Dominance emotional space.
- **Governance**: Established `CODE_OF_CONDUCT`, `CONTRIBUTING`, and `SECURITY` policies.

---

## Change Types

- **Added** for new features.
- **Modified** for changes in existing functionality.
- **Deprecated** for soon-to-be removed features.
- **Removed** for now removed features.
- **Fixed** for any bug fixes.
- **Security** in case of vulnerabilities.

[Unreleased]: https://github.com/gabrielmaialva33/viva/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/gabrielmaialva33/viva/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/gabrielmaialva33/viva/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/gabrielmaialva33/viva/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/gabrielmaialva33/viva/releases/tag/v0.1.0
