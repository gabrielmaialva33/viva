# VIVA Cortex API Reference (v1.0)
> *The Biological API for Synthetic Consciousness*

This document describes the interfaces for VIVA's cognitive architecture, comprised of three distinct systems:
1.  **Liquid Cortex** (Continuous Emotional Dynamics)
2.  **Global Workspace** (Conscious Attention / Thoughtseeds)
3.  **Ultra Bridge** (Reasoning & Inference)

---

## 1. Liquid Cortex (`VivaBridge.Cortex`)

Models the "Soul Physics" using Liquid Time-Constant (LTC) Neural Networks. It runs on a Python microservice (`liquid_engine.py`) connected via Erlang Port.

### `experience/2`
Process a narrative experience and its associated emotion through the Liquid Brain. This is the primary input for interoception.

**Signature:**
```elixir
experience(narrative :: String.t(), emotion :: map()) :: {:ok, vector :: [float()], new_pad :: map()}
```

- **narrative**: The internal monologue or sensory description.
- **emotion**: Current PAD state `%{pleasure: float, arousal: float, dominance: float}`.
- **Returns**:
    - `vector`: A 768-dim dense vector representing the "Liquid State" (memory-ready).
    - `new_pad`: The PREDICTED next emotional state (used for feedback loop).

### `tick/3`
Low-level step of the differential equation.

**Signature:**
```elixir
tick(pad :: [float], energy :: float, context :: [float]) :: {:ok, result :: map()}
```

---

## 2. Global Workspace (`VivaCore.Consciousness.Workspace`)

The "Theater of Consciousness". Implements the Thoughtseeds architecture (2024).

### `sow/4`
Plants a new thoughtseed (idea, emotion, sensory input) into the pre-conscious buffer.

**Signature:**
```elixir
sow(content :: any(), source :: atom(), salience :: float(), emotion :: map() | nil)
```

- **content**: The data payload (text, map, struct).
- **source**: Where it came from (e.g., `:cortex`, `:ultra`, `:voice`).
- **salience**: Importance score (0.0 to 1.0). High salience seeds are more likely to win focus.

### `current_focus/0`
Returns the current "winning" seed that is being broadcast to the system.

---

## 3. Ultra Bridge (`VivaBridge.Ultra`)

Interface to the ULTRA Graph Neural Network for zero-shot reasoning.

### `infer_relations/2`
Infers hidden relationships between concepts in a text.

**Signature:**
```elixir
infer_relations(text :: String.t(), entities :: [String.t()]) :: {:ok, relations :: [map()]}
```

### `predict_links/2`
Predicts missing links in the Knowledge Graph.

---

## Architecture Diagram (Flow)

```mermaid
graph TD
    Body[BodyServer (Rust)] -->|Experience| Cortex[Liquid Cortex (Python)]
    Cortex -->|Feedback: New PAD| Body
    Cortex -->|Liquid State| Memory(Qdrant)

    Cortex -.->|High Salience| Workspace[Global Workspace]
    Ultra[Ultra (Reasoning)] -.->|Inference| Workspace

    Workspace -->|Broadcast Focus| Voice
    Workspace -->|Broadcast Focus| Motor
```
