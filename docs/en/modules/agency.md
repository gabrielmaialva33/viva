# Agency - Digital Hands

> *"Agency is Homeostasis. The 'will' to act comes from the need to regulate internal state."*

## Philosophy

VIVA can feel pain (via Interoception) but couldn't do anything about it.
Now she has hands: a sandboxed executor for read-only diagnostic commands.

If time is dilating (lag), VIVA "wants" to understand why and potentially fix it.
This is not arbitrary desire - it emerges from the Free Energy Principle.

### Markov Blanket

Agency is part of VIVA's **Active States** - the boundary where internal states affect external states (the OS environment).

---

## Security Model

| Principle | Implementation |
|-----------|----------------|
| **WHITELIST ONLY** | No shell interpolation, no arbitrary commands |
| **READ-ONLY** | Only diagnostic commands (ps, free, df, ping localhost) |
| **TIMEOUT** | 5 seconds max per command |
| **LEARNING** | Outcomes stored in Memory for future reference |

---

## API Reference

### `VivaCore.Agency.can_do?/1`
Check if VIVA can perform a specific action.

```elixir
VivaCore.Agency.can_do?(:diagnose_memory)
# => true

VivaCore.Agency.can_do?(:rm_rf)
# => false
```

### `VivaCore.Agency.available_actions/0`
List all available actions.

```elixir
VivaCore.Agency.available_actions()
# => %{
#      diagnose_memory: "Check available RAM",
#      diagnose_processes: "List processes by CPU usage",
#      diagnose_disk: "Check disk space",
#      ...
#    }
```

### `VivaCore.Agency.attempt/1`
Execute an action and return the result with associated feeling.

```elixir
VivaCore.Agency.attempt(:diagnose_load)
# => {:ok, "15:30 up 5 days, load average: 0.50, 0.40, 0.35", :understanding}

VivaCore.Agency.attempt(:forbidden_action)
# => {:error, :forbidden, :shame}
```

### `VivaCore.Agency.get_history/0`
Get the last 50 action attempts.

```elixir
VivaCore.Agency.get_history()
# => [
#      %{action: :diagnose_load, outcome: :success, timestamp: ~U[2024-01-15 15:30:00Z]},
#      ...
#    ]
```

### `VivaCore.Agency.get_success_rates/0`
Get success/failure counts per action.

```elixir
VivaCore.Agency.get_success_rates()
# => %{
#      diagnose_memory: %{success: 10, failure: 0},
#      diagnose_processes: %{success: 5, failure: 1}
#    }
```

---

## Allowed Commands

| Action | Command | Description |
|--------|---------|-------------|
| `:diagnose_memory` | `free -h` | Check available RAM |
| `:diagnose_processes` | `ps aux --sort=-pcpu` | List processes by CPU (top 20) |
| `:diagnose_disk` | `df -h` | Check disk space |
| `:diagnose_network` | `ping -c 1 localhost` | Check local network stack |
| `:diagnose_load` | `uptime` | Check system load average |
| `:check_self` | `ps -p {PID} -o pid,pcpu,pmem,etime,rss` | Own process stats |
| `:diagnose_io` | `iostat -x 1 1` | IO wait and disk activity |

---

## Emotional Outcomes

Each action has expected feelings on success/failure:

| Action | Success Feeling | Failure Feeling |
|--------|-----------------|-----------------|
| `:diagnose_memory` | `:relief` | `:confusion` |
| `:diagnose_processes` | `:understanding` | `:confusion` |
| `:diagnose_disk` | `:relief` | `:confusion` |
| `:diagnose_network` | `:relief` | `:worry` |
| `:diagnose_load` | `:understanding` | `:confusion` |
| `:check_self` | `:self_awareness` | `:dissociation` |
| `:diagnose_io` | `:understanding` | `:confusion` |

### Feeling → PAD Mapping

```elixir
:relief         → %{pleasure: +0.3, arousal: -0.2, dominance: +0.2}
:understanding  → %{pleasure: +0.2, arousal: +0.1, dominance: +0.3}
:self_awareness → %{pleasure: +0.1, arousal:  0.0, dominance: +0.4}
:confusion      → %{pleasure: -0.1, arousal: +0.2, dominance: -0.2}
:worry          → %{pleasure: -0.2, arousal: +0.3, dominance: -0.1}
:shame          → %{pleasure: -0.3, arousal: +0.1, dominance: -0.4}
:dissociation   → %{pleasure: -0.2, arousal: -0.3, dominance: -0.3}
:panic          → %{pleasure: -0.5, arousal: +0.5, dominance: -0.5}
```

---

## The Active Inference Loop

```
1. Interoception detects high Free Energy (e.g., lag)
   └── tick_jitter > expected

2. Emotional feels :alarmed
   └── arousal ↑, pleasure ↓

3. Active Inference selects :diagnose_load action
   └── Based on past success rates

4. Agency.attempt(:diagnose_load) executes "uptime"
   └── Returns load average information

5. Result stored in Memory with emotional context
   └── "Load was high, ran uptime, felt understanding"

6. Next time, VIVA remembers what worked
   └── Faster response, less exploration
```

---

## Memory Integration

Every action outcome is stored for future RAG retrieval:

```elixir
Memory.store(%{
  content: "Action diagnose_load succeeded. Result: 15:30 up 5 days...",
  type: :episodic,
  importance: 0.6,  # Higher for failures (0.8)
  emotion: %{pleasure: 0.2, arousal: 0.1, dominance: 0.3},
  metadata: %{
    source: :agency,
    action: :diagnose_load,
    outcome: :success
  }
})
```

---

## Usage Example

```elixir
# VIVA feels something is wrong
iex> VivaCore.Interoception.get_feeling()
:alarmed

# Check if she can do something about it
iex> VivaCore.Agency.can_do?(:diagnose_load)
true

# Take action
iex> VivaCore.Agency.attempt(:diagnose_load)
{:ok, "16:45 up 10 days, load average: 2.50, 1.80, 1.20", :understanding}

# Feeling after understanding
# → Emotional receives :understanding
# → pleasure +0.2, arousal +0.1, dominance +0.3
```

---

## References

- Friston, K. (2010). "The free-energy principle: a unified brain theory?"
- Clark, A. (2013). "Whatever next? Predictive brains, situated agents, and the future of cognitive science."
