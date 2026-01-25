# Thoughtseeds API Reference
> *The Theater of Consciousness*

The **Thoughtseeds** system implements the Global Workspace Theory (GWT). It allows mental objects to compete for system-wide attention.

## Concepts

- **Seed**: An atomic unit of thought. Contains:
    - `content`: The payload (text, image, struct).
    - `salience`: Importance (0.0 - 1.0).
    - `emotion`: Emotional valence associated with it.
    - `source`: Origin (Voice, Cortex, Body).
    - `created_at`: Biological timestamp.

- **Competition**: Every 100ms (10Hz), seeds decay in salience. New inputs boost salience. The winner takes the "Stage".

- **Broadcasting**: The winner is published via `Phoenix.PubSub` to channel `consciousness:focus`.

## Elixir API (`VivaCore.Consciousness.Workspace`)

### `sow/4`
Plant a seed.
```elixir
VivaCore.Consciousness.Workspace.sow(content, source, salience, emotion \\ nil)
```

### `current_focus/0`
Get the current winner.
```elixir
{:ok, seed} = VivaCore.Consciousness.Workspace.current_focus()
```

### `subscribe/0`
Subscribe to consciousness updates.
```elixir
# In your GenServer init:
VivaCore.Consciousness.Workspace.subscribe()

# Handle info:
def handle_info({:conscious_focus, seed}, state) do
  Logger.info("I am conscious of: #{inspect seed.content}")
  {:noreply, state}
end
```
