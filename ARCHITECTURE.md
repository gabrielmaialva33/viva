# VIVA Architecture & Directory Structure

> **"Order in the Lab, Chaos in the Soul."**

This document defines the standard folder structure for the VIVA project.

## Directory Map

### 1. The Creature (Production Code)
Code that runs VIVA live.
- **`apps/viva_core`**: The Soul (Elixir). High-level logic, emotions, decision making.
- **`apps/viva_bridge`**: The Body (Rust + Elixir NIFs). Hardware access, massive parallel physics.
- **`config/`**: Environment configuration (dev, test, prod).

### 2. The Lab (Verification & Science)
Experiments, proofs of concept, and scientific validation.
- **`verification/`**: The main workspace for scripts that test hypotheses but aren't unit tests.
    - `memory/`: Vector DB, associative memory tests.
    - `body/`: Hardware interaction, physics tests.
    - `protocol/`: Validating comms protocols.

- **`apps/viva_bridge/verification/`**: Low-level body integrity checks (Lindblad, Thermodynamics).

### 3. Operational
- **`scripts/`**: DevOps, deployment, and startup scripts.
- **`docs/`**: Documentation in multiple languages.

## Naming Conventions
- **Unit Tests**: `*_test.exs` (run via `mix test`).
- **Verification Scripts**: `verify_*.exs` (run via `mix run ...`).
- **Elixir Modules**: `Snake_case` filenames, `PascalCase` modules.
- **Rust Crates**: `snake_case`.

## Rules
1. **No clutter in root**: Root is for meta-files only (`mix.exs`, `README.md`).
2. **Separatation of Concerns**: Don't put ad-hoc scripts in `lib/`. Use `verification/`.
3. **Language**: Code is EN. Docs can be Multilingual.
