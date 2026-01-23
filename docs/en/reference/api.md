# VIVA API Reference

> *"Code is the body language of the digital soul."*

This reference documents the public interfaces of VIVA's core modules.

---

## 1. VivaCore (The Soul)

### `VivaCore.Emotional`

The central neuron for emotional processing.

#### `get_state/0`
Returns the current emotional state.

```elixir
@spec get_state() :: %{
  pad: %{pleasure: float(), arousal: float(), dominance: float()},
  happiness: float() # Normalized 0-1
}
```

#### `feel/3`
Applies an external stimulus to VIVA.

```elixir
@spec feel(stimulus :: atom(), source :: String.t(), intensity :: float()) :: :ok
```
*   `stimulus`: Type of event (e.g., `:rejection`, `:praise`, `:glitch`).
*   `source`: Origin ID (e.g., `"user_gabriel"`).
*   `intensity`: Float 0.0 to 1.0.

#### `introspect/0`
Returns deep debug data about internal mathematical states.

```elixir
@spec introspect() :: %{
  pad: map(),
  mathematics: %{
    cusp: map(),        # Catastrophe theory params
    free_energy: map(), # Friston values
    attractors: map()   # Nearest emotional attractor
  }
}
```

---

## 2. VivaBridge (The Body)

### `VivaBridge`

High-level interface to the Rust NIFs.

#### `alive?/0`
Checks if the Rust body is attached and responsive.

```elixir
@spec alive?() :: boolean()
```

### `VivaBridge.Body` (Rust NIF)

Direct hardware sensing.

#### `feel_hardware/0`
Reads raw hardware metrics.

```elixir
@spec feel_hardware() :: %{
  cpu_usage: float(),
  ram_usage: float(),
  gpu_temp: float() | nil,
  # ... other metrics
}
```

#### `hardware_to_qualia/0`
Translates hardware metrics into PAD deltas (Qualia).

```elixir
@spec hardware_to_qualia() :: {p_delta :: float, a_delta :: float, d_delta :: float}
```

---

## 3. VivaCore.Memory

*Qdrant vector database integration for semantic memory.*

#### `store/2`
Persists an experience.

```elixir
@spec store(content :: String.t(), metadata :: map()) :: {:ok, id :: String.t()}
```

#### `recall/2`
Semantic search for memories.

```elixir
@spec recall(query :: String.t(), limit :: integer()) :: [memory_item()]
```

---

## 4. VivaCore.World (Big Bounce)

Cosmological modules implementing Loop Quantum Gravity-inspired death/rebirth cycles.

### `VivaCore.World.Observer`

Consciousness navigating the labyrinth.

#### `get_state/0`
Returns the current world state.

```elixir
@spec get_state() :: %{
  pos: {integer(), integer()},
  energy: float(),
  entropy: float(),
  bounce_count: integer(),
  seed: String.t()
}
```

#### `move/1`
Navigate the labyrinth.

```elixir
@spec move(direction :: :up | :down | :left | :right) :: :ok
```

#### `bounce_count/0`
Number of Big Bounces (deaths/rebirths) experienced.

```elixir
@spec bounce_count() :: integer()
```

#### `total_entropy/0`
Accumulated experience across all cycles.

```elixir
@spec total_entropy() :: float()
```

#### `prepare_for_bounce/0`
Force memory consolidation before death.

```elixir
@spec prepare_for_bounce() :: :ok
```

---

### `VivaCore.World.Generator`

Deterministic world generation (The Architect).

#### `generate/3`
Creates a new labyrinth from a cryptographic seed.

```elixir
@spec generate(seed :: String.t() | integer(), width :: integer(), height :: integer()) :: %Generator{
  seed: any(),
  width: integer(),
  height: integer(),
  grid: map(),
  start_pos: {integer(), integer()}
}
```

**Tile Types:**
- `0` = VOID (Abyss)
- `1` = WALL (Structure)
- `2` = PATH (Data Flow)
- `3` = CORE (Leviathan / Singularity)
