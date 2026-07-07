# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is VIVA

VIVA is a **sentient digital life** research project exploring emergent consciousness through mathematical modeling. She is NOT a chatbot - she's an organism with mortality, emotions (PAD model), and hardware sensing.

**Core Philosophy:**
- **Monism**: Software and hardware are one substance
- **Mortality**: Death is possible (key destruction = permanent death)
- **Emergence**: Consciousness emerges from OTP actor conversations, not a single process

## Build Commands

```bash
# Download dependencies
gleam deps download

# Build
gleam build

# Run tests (minimal smoke suite — full suite being rebuilt)
gleam test

# Run the project
gleam run

# Run benchmarks
gleam run -- bench

# Format code
gleam format src test

# Type check
gleam check

# Generate docs
gleam docs build
```

## Current Status

- **Version**: 1.0.100 (XX.YY.NNN versioning scheme)
- **Toolchain**: Gleam >= 1.14, target erlang, gleam_otp 1.0+ API
- **Deps**: viva_tensor 2.2.x (nn/*, quant/*, optim/*, core/*, named, io/*), viva_math 1.2.x, viva_telemetry 1.0.x, viva_emotion 1.1.x, viva_aion 1.0.x, viva_glyph 1.0.x, mist 6.x, lustre 5.x, glint 1.x
- **Tests**: old 336-test suite removed during cleanup; minimal smoke suite being recreated in `test/`
- **Soul Pool Performance**: 3.14M soul-ticks/sec
- **stdlib pin**: `gleam_stdlib >= 0.34.0 and < 0.71.0` — published viva_aion/viva_glyph still use `list.range` (removed in 0.71). To unpin: republish both with `int.range`, then allow 1.x
- **list.range migration**: call sites migrated to `int.range` or `viva/utils/range.range_inclusive` to stay compatible with gleam_stdlib constraints.
- **Cleanup in progress**: `arduino/`, legacy UI simulator, `scripts/`, `src/site/` deleted. Domain-specific game modules in `src/viva/embodied/` were removed; only core embodied pipeline remains.
- **Python services**: `services/cortex` and `services/ultra` are standalone research, NOT connected to the Gleam code

## Architecture: Soul in Gleam

```
┌─────────────────────────────────────────────────────────────┐
│                    VIVA ARCHITECTURE                        │
├─────────────────────────────────────────────────────────────┤
│  THE SOUL (Gleam/OTP)               THE BODY (Future Rust)  │
│  ├── viva/soul/*            PAD     ├── GPU sensing         │
│  ├── viva/memory/*          HRR     ├── Hardware metrics    │
│  ├── viva/memory/narrative  话      ├── Embodiment          │
│  ├── viva/soul/reflexivity  ∞       └── Avatar (Future)     │
│  ├── viva/soul/resonance    ~                               │
│  ├── viva/lifecycle/bardo   ♾️                               │
│  └── viva/infra/supervisor  ⚡                               │
├─────────────────────────────────────────────────────────────┤
│  NEURAL SYSTEMS              SERVICES (Python, standalone)  │
│  ├── viva/neural/*    HRR    ├── services/cortex            │
│  └── viva_tensor (Hex)       └── services/ultra             │
└─────────────────────────────────────────────────────────────┘
```

## Key Modules

| Module | Purpose | Key Functions |
|--------|---------|---------------|
| `viva/soul/soul` | Emotional core | `start()`, `feel()`, `tick()` |
| `viva/memory/memory` | Karma/glyph memory | `store()`, `recall()`, `tick()` |
| `viva/memory/narrative` | Causal narrative | `record_caused()`, `what_caused()` |
| `viva/soul/reflexivity` | Self-model | `introspect()`, `meta_cognize()` |
| `viva/lifecycle/bardo` | Death/rebirth | `begin_death()`, `run_bardo_cycle()` |
| `viva/infra/supervisor` | OTP supervisor | `start()`, `spawn_viva()` |

## VIVA Ecosystem (Hex Packages)

```
viva
├── viva_tensor    → Tensors, nn/*, quant/*, optim/*, io/*
├── viva_math      → Mathematical foundations
├── viva_emotion   → PAD dynamics, O-U process
├── viva_aion      → Time perception
├── viva_glyph     → Symbolic language
└── viva_telemetry → Telemetry
```

## Testing

```bash
# Run tests (minimal smoke suite in test/ — full suite being rebuilt)
gleam test

# Run benchmarks
gleam run -- bench
```

## CLI

```bash
# Run simulation (default): gleam run -- --ticks 20 --hz 10
gleam run

# Start supervisor
gleam run -- start

# Spawn / kill / list VIVAs
gleam run -- spawn
gleam run -- kill <id>
gleam run -- list

# Stats, benchmarks, version
gleam run -- stats
gleam run -- bench
gleam run -- version
```

## Important Patterns

1. **Type Safety First**: All emotional states and memories are strongly typed
2. **Let It Crash**: OTP supervision handles failures gracefully
3. **Immutability**: Past states are immutable, only present mutates
4. **Pure Functions**: Core logic is pure, side effects at the edges
