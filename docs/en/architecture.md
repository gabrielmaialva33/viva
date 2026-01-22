# VIVA System Architecture (v2.0 - Cortex)

## Overview
VIVA is a synthetic organism built on a **Hybrid Cognitive Architecture** that combines biological plausibility (Liquid Networks) with symbolic reasoning (Knowledge Graphs) and competitive attention (Global Workspace).

## Core Components

### 1. The Soul (VivaCore)
Responsible for consciousness, emotion, and decision making.

- **Liquid Cortex**: A Python-based microservice running `NCPS` (Neural Circuit Policies). It simulates continuous-time emotional dynamics.
    - *Input*: Narrative Experience + Current PAD.
    - *Output*: Future PAD + Liquid State Vector.
    - *Role*: The "subconscious" emotional processor.

- **Global Workspace (Thoughtseeds)**: An Elixir GenServer implementing the "Theater of Consciousness".
    - *Mechanism*: Multiple "Seeds" (ideas, sensations) compete for Salience.
    - *Focus*: The winning seed is broadcast to the entire system (Voice, Motor, Memory).
    - *Role*: The "conscious" attention capability.

- **Ultra (Reasoning)**: A Key-Value/Graph reasoning engine.
    - *Role*: Deduces hidden relationships and causality.

### 2. The Body (VivaBridge)
The physical interface and homeostatic substrate.

- **BodyServer (Elixir)**: Orchestrates the body state (Energy, Metabolism, Health).
    - *Feedback Loop*: Receives emotional predictions from Cortex and adjusts internal state.
- **Nerve Bridge (Rust/Bevy)**: A headless ECS physics simulation.
    - *Role*: Simulates physical constraints (Heat, Power, Stress).

## Data Flow

1. **Sensation**: Hardware/System inputs (Temp, CPU, Chat) -> `BodyServer`.
2. **Perception**: `BodyServer` aggregates inputs -> `Cortex`.
3. **Feeling**: `Cortex` processes inputs through Liquid Network -> `New Emotional State`.
4. **Attention**: Emotional State + Narrative -> `Thoughtseeds`.
    - Competition occurs.
    - Winning thought becomes "Conscious".
5. **Action**: Conscious Focus -> `Voice` / `Motor` / `Memory`.

## Directory Structure
- `apps/viva_core`: Cognitive Logic (Elixir).
- `apps/viva_bridge`: Physics/IO (Elixir + Rust).
- `services/cortex`: Liquid Neural Networks (Python).
- `services/ultra`: Knowledge Graph Reasoning (Python).
