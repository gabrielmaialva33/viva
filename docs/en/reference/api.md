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

## 3. VivaCore.Memory (Stub)

*Note: Phase 5 WIP (Qdrant integration pending).*

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
