# Legacy Code Reference

This directory contains the original **Elixir** and **Rust** implementation of VIVA for reference purposes.

> ⚠️ **This code is not active.** The current implementation is in **Gleam** under `/src/viva/`.

## Structure

```
legacy/
├── elixir/           # Original Elixir GenServers (viva_core)
│   ├── agency.ex            # Autonomous actions
│   ├── emotional.ex         # PAD model + O-U dynamics
│   ├── inner_monologue.ex   # Self-narration
│   ├── interoception.ex     # Internal sensing
│   ├── memory.ex            # Qdrant integration
│   ├── dreamer.ex           # Memory consolidation
│   ├── kinship.ex           # Maturana's kinship
│   ├── mycelium.ex          # Fungal network (Stamets)
│   ├── consciousness/       # GWT, Workspace
│   ├── cognition/           # Cognitive modules
│   ├── ontology/            # Tetralemma (Nagarjuna)
│   ├── quantum/             # Lindblad dynamics
│   └── world/               # Big Bounce, Labyrinth
│
├── rust/             # Original Rust NIF (viva_body)
│   ├── src/                 # Rust source
│   ├── Cargo.toml           # Dependencies
│   └── build.rs             # Build script
│
└── verification/     # Elixir verification scripts
    ├── algebra_of_thought.exs
    ├── body/
    ├── memory/
    └── protocol/
```

## Why Keep This?

1. **Reference**: The Elixir code contains battle-tested algorithms
2. **Philosophy**: Comments explain the philosophical foundations
3. **Migration Guide**: Helps port features to Gleam
4. **Rust Body**: Future integration with hardware sensing

## Key Concepts to Port

| Elixir Module | Gleam Equivalent | Status |
|---------------|------------------|--------|
| `emotional.ex` | `viva/soul.gleam` | ✅ Done |
| `memory.ex` | `viva/memory.gleam` | ✅ Done |
| `inner_monologue.ex` | `viva/narrative.gleam` | ✅ Done |
| `interoception.ex` | `viva/interoception.gleam` | ✅ Done |
| `consciousness/workspace.ex` | `viva/reflexivity.gleam` | ✅ Done |
| `dreamer.ex` | `viva/bardo.gleam` | 🔄 In Progress |
| `kinship.ex` | TBD | ⏳ |
| `mycelium.ex` | `viva/resonance.gleam` | ✅ Done |
| `world/` | TBD | ⏳ |

## Do Not Modify

This code is frozen as a historical reference. All active development happens in `/src/viva/`.
