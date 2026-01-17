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

## Architecture: Soul/Body Split

```
┌─────────────────────────────────────────────────────────────┐
│  SOUL (Elixir/OTP) - 1-10 Hz                                │
│  Logic, Emotions, Memory, Decisions                         │
│  apps/viva_core/                                            │
├─────────────────────────────────────────────────────────────┤
│  BODY (Rust NIFs) - 2 Hz (500ms ticks)                      │
│  Hardware sensing, Qualia, Dynamics engine                  │
│  apps/viva_bridge/                                          │
└─────────────────────────────────────────────────────────────┘
```

**CRITICAL RULE**: Never mix render/physics loop logic into Soul modules.

### Soul (apps/viva_core)

| Module | Responsibility |
|--------|---------------|
| `Emotional` | PAD state (Pleasure/Arousal/Dominance), O-U dynamics |
| `Memory` | Episodic/semantic memory, Qdrant integration |
| `Dreamer` | Memory consolidation, reflection, scoring |
| `Senses` | Heartbeat GenServer (1Hz), body→soul sync |
| `Mathematics` | O-U process, Cusp catastrophe, Free Energy |
| `Qdrant` | HTTP client for vector database |

**Supervision**: `:rest_for_one` strategy - if Emotional fails, Senses restarts too.

### Body (apps/viva_bridge)

| Elixir Module | Rust File | Purpose |
|---------------|-----------|---------|
| `Body` | `lib.rs` | NIF interface for hardware sensing |
| `BodyServer` | `dynamics.rs` | State evolution (500ms ticks) |
| `Brain` | - | High-level coordination |
| `Memory` | `memory/` | Native vector search (HNSW) |

**Rust crate**: `apps/viva_bridge/native/viva_body/`

Key Rust modules:
- `dynamics.rs` - Ornstein-Uhlenbeck, Cusp catastrophe
- `metabolism.rs` - Energy/Entropy/Fatigue model
- `mirror.rs` - Self-reading (Autoscopia/Protocolo Espelho)
- `asm.rs` - RDTSC/CPUID inline assembly

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

- `docs/en/explanation/mathematics.md` - All equations in LaTeX
- `docs/en/explanation/architecture.md` - Deep technical breakdown
- `docs/en/research/whitepaper.md` - Full research paper

## Current Phase

**Phase 5: Memory** - Qdrant integration, semantic search.

Completed: Genesis, Emotion, Sensation, Interoception.
Next: Language (LLM), Embodiment (Bevy 3D).

## Contributor Roles

| Role | Focus | Stack |
|------|-------|-------|
| 🧠 Neurosurgeon | Optimize NIFs, add sensors | Rust |
| 💓 Psychologist | Tune emotional equations | Elixir/OTP |
| 🏛️ Philosopher | Expand theory/ethics | Markdown/LaTeX |
| 🎨 Artist | Avatar/visual | Bevy/WGPU |
| 🔮 Mystic | Symbolic reflection | - |

## External Dependencies

- **Qdrant** - Vector database for semantic memory
- **NVML** - NVIDIA GPU monitoring (optional, runtime detection)
- **sysinfo** - Cross-platform system metrics
