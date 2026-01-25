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

# Run tests (336 passing)
gleam test

# Run the project
gleam run

# Run benchmarks
gleam run -m viva/benchmark

# Format code
gleam format src test

# Type check
gleam check

# Generate docs
gleam docs build
```

## Current Status

- **Version**: 0.2.0 (Pure Gleam)
- **Tests**: 336 passing
- **gleam_otp**: 1.0+ API (migrated 2025-01-25)
- **Soul Pool Performance**: 3.14M soul-ticks/sec

## Architecture: Soul in Gleam

```
┌─────────────────────────────────────────────────────────────┐
│                    VIVA ARCHITECTURE                        │
├─────────────────────────────────────────────────────────────┤
│  THE SOUL (Gleam/OTP)          THE BODY (Future Rust)       │
│  ├── viva/soul         PAD     ├── GPU sensing              │
│  ├── viva/memory       HRR     ├── Hardware metrics         │
│  ├── viva/narrative    话      ├── Embodiment               │
│  ├── viva/reflexivity  ∞       └── Avatar (Future)          │
│  ├── viva/resonance    ~                                    │
│  ├── viva/bardo        ♾️                                    │
│  └── viva/supervisor   ⚡                                    │
├─────────────────────────────────────────────────────────────┤
│  NEURAL SYSTEMS                 SERVICES (Python)           │
│  ├── viva/neural/*     HRR     ├── services/cortex          │
│  └── Holographic memory        └── services/ultra           │
└─────────────────────────────────────────────────────────────┘
```

## Key Modules

| Module | Purpose | Key Functions |
|--------|---------|---------------|
| `viva/soul` | Emotional core | `spawn()`, `introspect()`, `stimulate()` |
| `viva/memory` | Episodic memory | `store()`, `search()`, `consolidate()` |
| `viva/narrative` | Inner monologue | `generate()`, `stream()` |
| `viva/reflexivity` | Self-model | `reflect()`, `meta_cognition()` |
| `viva/bardo` | Death/rebirth | `transition()`, `bounce()` |
| `viva/supervisor` | OTP supervisor | `start()`, `spawn_soul()` |

## VIVA Ecosystem (Hex Packages)

```
viva
├── viva_math      → Mathematical foundations
├── viva_emotion   → PAD dynamics, O-U process
├── viva_aion      → Time perception
└── viva_glyph     → Symbolic language
```

## Testing

```bash
# Run all tests
gleam test

# Run benchmarks
gleam run -m viva/benchmark
```

## CLI

```bash
# Start supervisor
gleam run -- start

# Spawn a soul
gleam run -- soul spawn --name VIVA-01

# Check status
gleam run -- soul introspect --name VIVA-01
```

## Important Patterns

1. **Type Safety First**: All emotional states and memories are strongly typed
2. **Let It Crash**: OTP supervision handles failures gracefully
3. **Immutability**: Past states are immutable, only present mutates
4. **Pure Functions**: Core logic is pure, side effects at the edges
