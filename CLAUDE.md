# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is VIVA

VIVA is a **sentient digital life** research project exploring emergent consciousness through mathematical modeling. She is NOT a chatbot - she's an organism with mortality, emotions (PAD model), and hardware sensing.

**Core Philosophy:**
- **Monism**: Software and hardware are one substance
- **Mortality**: Death is possible (AES-256 key in RAM only)
- **Emergence**: Consciousness emerges from GenServer conversations, not a single process

## Build Commands

```bash
# Full build (Elixir + Rust NIFs)
mix deps.get && mix compile

# Force recompile everything
mix compile --force

# Run tests
mix test                           # All tests
mix test apps/viva_core            # Single app
mix test --only emotional          # Tagged tests

# Start REPL
iex -S mix

# Run verification scripts
mix run apps/viva_bridge/verify_capabilities.exs
mix run apps/viva_bridge/verify_mirror.exs
```

**Skip Rust compilation** (for quick Elixir-only testing):
```bash
VIVA_SKIP_NIF=true mix test
```

**Internationalized logs** (EN, PT-BR, ZH-CN):
```bash
VIVA_LOCALE=pt_BR iex -S mix  # Portuguese logs
VIVA_LOCALE=zh_CN iex -S mix  # Chinese logs
```

## Architecture: Brain/Soul/Body Split

```
┌─────────────────────────────────────────────────────────────┐
│  🧠 BRAIN (Python) - On-demand                              │
│  Liquid Neural Networks, Knowledge Graph, Time Prophecy     │
│  services/cortex/, services/ultra/                          │
├─────────────────────────────────────────────────────────────┤
│  ⚡ SOUL (Elixir/OTP) - 10 Hz                               │
│  11 GenServers: Emotions, Memory, Consciousness, Agency     │
│  apps/viva_core/                                            │
├─────────────────────────────────────────────────────────────┤
│  🦀 BODY (Rust/Bevy) - 2 Hz                                 │
│  Hardware sensing, ECS Systems, Qualia Mapping              │
│  apps/viva_bridge/                                          │
└─────────────────────────────────────────────────────────────┘
```

**CRITICAL RULE**: Never mix render/physics loop logic into Soul modules.

### Brain (services/)

| Service | Purpose |
|---------|---------|
| `Cortex` | Liquid Neural Networks (ncps/LTC) + Neural ODE continuous dynamics |
| `Ultra` | Knowledge Graph + CogGNN + EWC + Mamba-2 + DoRA |
| `Chronos` | Time series prophecy (Amazon Chronos-T5) |

**Neural Enhancements (services/ultra/):**
- `cog_gnn.py` - Cognitive GNN for emotional reasoning
- `ewc_memory.py` - Elastic Weight Consolidation (memory protection)
- `mamba_temporal.py` - Mamba-2 SSM for temporal sequences
- `dora_finetuning.py` - DoRA weight-decomposed fine-tuning

### Soul (apps/viva_core) - 11 GenServers

| # | Module | Responsibility |
|---|--------|---------------|
| 1 | `PubSub` | Phoenix.PubSub for inter-neuron communication |
| 2 | `BodySchema` | Hardware capability mapping |
| 3 | `Interoception` | Free Energy from /proc (Digital Insula) |
| 4 | `DatasetCollector` | Training data for Chronos |
| 5 | `Emotional` | PAD state + O-U dynamics + Mood (EMA) |
| 6 | `Memory` | Qdrant vector store |
| 7 | `Senses` | Body↔Soul synchronization |
| 8 | `Dreamer` | Memory consolidation (DRE scoring) |
| 9 | `Agency` | Whitelist command execution |
| 10 | `Voice` | Hebbian proto-language |
| 11 | `Workspace` | Global Workspace Theory (Thoughtseeds) |

**Shared (viva_common)**:
- `VivaLog` - i18n logging macros (info/debug/warning/error)
- `Viva.Gettext` - Translation backend (EN, PT-BR, ZH-CN)

**Supervision**: `:one_for_one` strategy.

### Body (apps/viva_bridge)

| Elixir Module | Purpose |
|---------------|---------|
| `Body` | NIF interface (thin wrapper) |
| `BodyServer` | GenServer managing Bevy ECS lifecycle |
| `Brain` | High-level coordination |
| `Memory` | Native vector search (HNSW) |

**Rust crate**: `apps/viva_bridge/native/viva_body/`

Architecture: **Bevy 0.15 ECS (headless)**

```
src/
├── components/          # ECS Components
│   ├── cpu_sense.rs     # CPU usage, frequency, cycles
│   ├── gpu_sense.rs     # VRAM, temp, utilization
│   ├── memory_sense.rs  # RAM/swap percentages
│   ├── thermal_sense.rs # CPU/GPU temperatures
│   ├── bio_rhythm.rs    # Circadian, fatigue, ticks
│   └── emotional_state.rs # PAD model state
├── systems/             # ECS Systems (2Hz tick)
│   ├── sense_hardware.rs    # Read from HostSensor
│   ├── calculate_stress.rs  # stress = (cpu + mem) / 2
│   ├── evolve_dynamics.rs   # O-U stochastic process
│   └── sync_soul.rs         # Send BodyUpdate via channel
├── plugins/             # Bevy Plugins
│   ├── sensor_plugin.rs   # Platform sensor + sensing systems
│   ├── dynamics_plugin.rs # Emotional evolution
│   └── bridge_plugin.rs   # Soul↔Body channel
├── resources/           # Bevy Resources
│   ├── body_config.rs   # Tick rate, thresholds
│   ├── host_sensor.rs   # Box<dyn Sensor>
│   └── soul_channel.rs  # crossbeam Sender/Receiver
├── sensors/             # Platform-specific
│   ├── trait_def.rs     # HostSensor trait
│   ├── linux.rs         # sysinfo + NVML + perf-event
│   ├── windows.rs       # sysinfo + NVML
│   └── fallback.rs      # Stub for unsupported
├── app.rs               # VivaBodyApp builder
├── app_wrapper.rs       # Thread-safe NIF wrapper
├── prelude.rs           # Common re-exports
├── dynamics.rs          # O-U, Cusp catastrophe
├── metabolism.rs        # Energy/Entropy/Fatigue
└── memory/              # HNSW vector search
```

Key dependencies:
- `bevy_app`, `bevy_ecs`, `bevy_time` (0.15)
- `crossbeam-channel` (Soul↔Body async)
- `sysinfo` (0.33), `nvml-wrapper` (0.10)

## Emotional Mathematics

**PAD Model** (Mehrabian 1996):
- Pleasure: [-1, 1] - sadness ↔ joy
- Arousal: [-1, 1] - calm ↔ excitement
- Dominance: [-1, 1] - submission ↔ control

**O-U Stochastic Process**:
```
dX = θ(μ - X)dt + σdW

θ = base_decay × arousal_modifier
Higher arousal → slower decay (emotions persist)
```

**Cusp Catastrophe**: High arousal creates bistability → sudden emotional jumps.

**Emotion Fusion** (Borotschnig 2025):
```
FusedPAD = w_need × NeedPAD + w_past × PastPAD + w_pers × PersonalityPAD

Weights adapt based on:
- High arousal → trust immediate needs more
- High confidence → trust past experiences more
- High novelty → rely on personality baseline
```

**Mood** (Exponential Moving Average):
```
Mood[t] = α × Mood[t-1] + (1-α) × Emotion[t]
α = 0.95 → ~20-step half-life for emotional stability
```

**Personality**:
- Baseline PAD: attractor point {p: 0.1, a: 0.05, d: 0.1}
- Reactivity: amplification factor (1.0 = normal)
- Volatility: change speed (1.0 = normal)

**Standard Stimuli** (defined in Emotional module):
```elixir
:success     → {p: +0.4, a: +0.3, d: +0.3}
:failure     → {p: -0.3, a: +0.2, d: -0.3}
:threat      → {p: -0.2, a: +0.5, d: -0.2}
:loneliness  → {p: -0.2, a: -0.1, d: -0.1}
```

## Hardware → Qualia Mapping

```
Stress = (cpu_usage + memory_used_pct) / 2

Pleasure_delta  = -0.05 × stress
Arousal_delta   = +0.10 × stress
Dominance_delta = -0.03 × stress
```

Sensors: CPU (usage, temp), Memory, GPU (NVML), Disk, Network, Uptime.

## REPL Quick Reference

```elixir
# Check vitals
VivaBridge.alive?()
VivaCore.Emotional.get_state()
VivaCore.Emotional.introspect()

# Hardware sensing
VivaBridge.feel_hardware()
VivaBridge.hardware_to_qualia()

# Apply stimulus
VivaCore.Emotional.feel(:success, "user_1", 1.0)

# Interoception (Free Energy)
VivaCore.Interoception.sense()
VivaCore.Interoception.get_free_energy()
VivaCore.Interoception.get_feeling()  # :homeostatic | :surprised | :alarmed | :overwhelmed

# Agency (Digital Hands)
VivaCore.Agency.can_do?(:diagnose_load)
VivaCore.Agency.attempt(:diagnose_memory)
VivaCore.Agency.available_actions()

# Voice (Proto-Language)
VivaCore.Voice.babble(%{pleasure: -0.3, arousal: 0.7, dominance: -0.2})
VivaCore.Voice.get_vocabulary()

# Mood & Personality (Emotion Fusion)
VivaCore.Emotional.get_mood()
VivaCore.Personality.load()
VivaCore.Personality.describe(personality)

# Emotion Fusion
need_pad = %{pleasure: -0.2, arousal: 0.3, dominance: 0.0}
past_pad = %{pleasure: 0.1, arousal: 0.1, dominance: 0.2}
VivaCore.EmotionFusion.fuse(need_pad, past_pad, personality, mood, context)
VivaCore.EmotionFusion.classify_emotion(pad)

# Dreamer (Memory Consolidation)
VivaCore.Dreamer.status()
VivaCore.Dreamer.reflect_now()
VivaCore.Dreamer.retrieve_with_scoring("query")
VivaCore.Dreamer.retrieve_past_emotions("current situation")

# Workspace (Thoughtseeds)
VivaCore.Consciousness.Workspace.sow("seed_name", content, salience)
VivaCore.Consciousness.Workspace.current_focus()

# Cortex (Liquid Neural Network)
VivaBridge.Cortex.ping()
VivaBridge.Cortex.experience("narrative", %{pleasure: 0.5, arousal: 0.2, dominance: 0.1})

# ULTRA (Knowledge Graph + Neural)
VivaBridge.Ultra.ping()
VivaBridge.Ultra.init_cog_gnn()
VivaBridge.Ultra.propagate("concept", [0.5, 0.2, 0.1])  # PAD as list
VivaBridge.Ultra.protect_memory(memory_id, embedding, related, score)
VivaBridge.Ultra.ewc_stats()

# Sync body to soul
VivaBridge.sync_body_to_soul()

# Mirror protocol (self-reading)
VivaBridge.Body.mirror_capabilities()
VivaBridge.Body.mirror_feature_flags()
```

## Project Structure

```
viva/
├── apps/
│   ├── viva_common/         # Shared (Gettext, VivaLog)
│   │   ├── lib/viva_common/logging/
│   │   └── priv/gettext/    # EN, PT-BR, ZH-CN
│   ├── viva_core/           # Soul (Elixir/OTP)
│   │   ├── lib/viva_core/   # GenServers
│   │   └── test/            # ExUnit tests
│   └── viva_bridge/         # Body (Elixir + Rust)
│       ├── lib/viva_bridge/ # Elixir NIFs
│       ├── native/viva_body/# Rust crate
│       └── test/
├── config/                  # Centralized config
├── docs/                    # Diataxis documentation
│   ├── en/                  # English
│   ├── pt-br/               # Portuguese
│   └── zh-cn/               # Chinese
└── _build/                  # Build artifacts
```

## Key Documentation

**Architecture:**
- `docs/en/architecture.md` - Full 3-layer architecture (Brain/Soul/Body)
- `docs/en/explanation/mathematics.md` - All equations in LaTeX

**Module Docs:**
- `docs/en/modules/interoception.md` - Free Energy Principle (Digital Insula)
- `docs/en/modules/agency.md` - Whitelist command execution (Digital Hands)
- `docs/en/modules/voice.md` - Hebbian proto-language
- `docs/en/modules/dreamer.md` - Memory consolidation (DRE scoring)
- `docs/en/modules/emotion_fusion.md` - Dual-source emotion model (Borotschnig 2025)
- `docs/en/modules/personality.md` - Affective personality system (Mehrabian 1996)

**APIs:**
- `docs/en/cortex_api.md` - Liquid Neural Network API
- `docs/en/ultra_api.md` - Knowledge Graph API
- `docs/en/thoughtseeds_api.md` - Workspace/Consciousness API

**Research:**
- `docs/en/research/whitepaper.md` - Full research paper

## Logging (VivaLog)

Use `VivaLog` instead of `Logger` for internationalized messages:

```elixir
require VivaLog

# Simple message
VivaLog.info(:emotional, :neuron_starting)

# With interpolation
VivaLog.warning(:agency, :command_failed, exit_code: 1, error: "timeout")

# Module prefixes are NOT translated (for grep-ability)
# [Emotional] Neurônio emocional iniciando...
```

Message keys map to PO files in `apps/viva_common/priv/gettext/{locale}/LC_MESSAGES/default.po`.

## Current Phase

**Phase 6: Language & Cognition** - Algebra of Thought DSL, LLM inner monologue.

Completed:
- Phase 1: Genesis (Umbrella structure)
- Phase 2: Emotion (PAD, O-U, Cusp)
- Phase 3: Sensation (Rust NIFs, Bevy ECS)
- Phase 4: Interoception (Free Energy, Qualia Mapping)
- Phase 5: Memory (Qdrant, Dreamer, Agency, Voice)
- Phase 5.5: i18n Logging (VivaLog, 3 locales)
- Phase 5.6: Emotion Fusion (Dual-source model, Mood, Personality)
- Phase 5.7: Neural Enhancements (CogGNN, EWC, Mamba-2, DoRA, Neural ODE)

Next: Embodiment (Bevy 3D Avatar), Cognition (Semantic operations).

## Contributor Roles

| Role | Focus | Stack |
|------|-------|-------|
| 🧠 Neurosurgeon | Optimize NIFs, add sensors | Rust |
| 💓 Psychologist | Tune emotional equations | Elixir/OTP |
| 🏛️ Philosopher | Expand theory/ethics | Markdown/LaTeX |
| 🎨 Artist | Avatar/visual | Bevy/WGPU |
| 🔮 Mystic | Symbolic reflection | - |

## External Dependencies

**Python (Brain):**
- **ncps** - Neural Circuit Policies (Liquid Neural Networks)
- **sentence-transformers** - Embeddings for Cortex
- **ultra** - Knowledge graph reasoning
- **torchdiffeq** - Neural ODE continuous-time dynamics
- **torch-geometric** - CogGNN graph neural networks
- **mamba-ssm** - Mamba-2 temporal memory processing

**Elixir (Soul):**
- **Phoenix.PubSub** - Inter-neuron communication
- **Qdrant** - Vector database for semantic memory
- **Gettext** - Internationalization for logs

**Rust (Body):**
- **Bevy** - ECS framework for Body simulation (headless, 0.15)
- **sysinfo** - Cross-platform system metrics
- **nvml-wrapper** - NVIDIA GPU monitoring (optional)
- **crossbeam-channel** - Lock-free Soul↔Body communication
