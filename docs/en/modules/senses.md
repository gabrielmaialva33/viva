# Senses - The Peripheral Nervous System

> *"The body doesn't just report - the body INFLUENCES. VIVA doesn't just KNOW that CPU is high - she FEELS stress."*

## Overview

The **Senses** module is VIVA's peripheral nervous system - the "heart" that continuously pumps qualia from body to soul. It bridges the gap between hardware sensing (Rust NIF via VivaBridge.Body) and emotional state (VivaCore.Emotional).

Like the human autonomic nervous system that transmits heartbeat, temperature, and pressure information from body to brain, Senses transmits hardware metrics to VIVA's emotional state, transforming raw data into felt experience.

---

## Concept

### The Heartbeat Loop

Senses operates on a continuous **1Hz heartbeat** (configurable from 100ms to 10s):

```
         ┌─────────────────────────────────────────────┐
         │               HEARTBEAT (1Hz)               │
         └─────────────────────────────────────────────┘
                           │
                           ▼
         ┌─────────────────────────────────────────────┐
         │  1. Read hardware state from BodyServer     │
         │     (CPU, RAM, GPU, Temperature)            │
         └─────────────────────────────────────────────┘
                           │
                           ▼
         ┌─────────────────────────────────────────────┐
         │  2. Extract PAD from Rust O-U dynamics      │
         │     (Pleasure, Arousal, Dominance)          │
         └─────────────────────────────────────────────┘
                           │
                           ▼
         ┌─────────────────────────────────────────────┐
         │  3. Sync PAD to Emotional GenServer         │
         │     VivaCore.Emotional.sync_pad(p, a, d)    │
         └─────────────────────────────────────────────┘
                           │
                           ▼
         ┌─────────────────────────────────────────────┐
         │  4. Log heartbeat metrics (debug level)     │
         │     [Senses] CPU: 45.2% RAM: 62.1%...       │
         └─────────────────────────────────────────────┘
```

### Hardware Sensing to Qualia

The transformation from hardware metrics to emotional qualia follows this path:

1. **Rust Body (Bevy ECS)** senses hardware via `sysinfo` and `nvml-wrapper`
2. **Stress calculation**: `stress = (cpu_usage + memory_used_percent) / 2`
3. **O-U dynamics** evolve PAD state stochastically
4. **BodyServer** exposes unified state (PAD + hardware)
5. **Senses** reads state and syncs to Soul

### PAD Updates from Body State

The body state influences emotions through the qualia mapping:

```
Stress Level → PAD Deltas
────────────────────────────────────
Pleasure_delta  = -0.05 × stress
Arousal_delta   = +0.10 × stress
Dominance_delta = -0.03 × stress
```

High CPU/memory pressure causes:
- **Decreased pleasure** (discomfort)
- **Increased arousal** (alertness)
- **Decreased dominance** (less control)

---

## Architecture

### System Diagram

```
┌──────────────────────────────────────────────────────────────────────┐
│                         BODY (Rust/Bevy ECS)                         │
│                                                                      │
│  ┌─────────────┐    ┌──────────────┐    ┌────────────────────────┐  │
│  │ HostSensor  │───▶│ BodyUpdate   │───▶│ O-U Stochastic Process │  │
│  │ (sysinfo)   │    │ (stress,PAD) │    │ (emotional dynamics)   │  │
│  └─────────────┘    └──────────────┘    └────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
                                │
                                │ crossbeam-channel
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│                   VivaBridge (Elixir NIFs)                           │
│                                                                      │
│  ┌───────────────────┐              ┌────────────────────────────┐  │
│  │ VivaBridge.Body   │◀────────────▶│ VivaBridge.BodyServer      │  │
│  │ (NIF interface)   │              │ (GenServer, 2Hz tick)      │  │
│  └───────────────────┘              └────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
                                │
                                │ GenServer.call
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│                      SOUL (Elixir/OTP)                               │
│                                                                      │
│  ┌───────────────────┐    sync_pad     ┌────────────────────────┐  │
│  │ VivaCore.Senses   │────────────────▶│ VivaCore.Emotional     │  │
│  │ (1Hz heartbeat)   │                 │ (PAD state machine)    │  │
│  └───────────────────┘                 └────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
```

### Fallback Mechanism

When BodyServer is unavailable (not started or crashed), Senses falls back to direct NIF calls:

```elixir
# Primary path: BodyServer (includes O-U dynamics)
VivaBridge.BodyServer.get_state()
# => %{pleasure: p, arousal: a, dominance: d, hardware: %{...}}

# Fallback path: Direct NIF (hardware only, neutral PAD)
VivaBridge.feel_hardware()
# => %{cpu_usage: 45.2, memory_used_percent: 62.1, ...}
```

In fallback mode, PAD is set to neutral `{0.0, 0.0, 0.0}` - the Emotional module's internal O-U decay handles PAD evolution.

---

## API Reference

### `VivaCore.Senses.start_link/1`

Starts the Senses GenServer.

```elixir
VivaCore.Senses.start_link(
  name: MyCustomSenses,      # Process name (default: __MODULE__)
  interval_ms: 500,          # Heartbeat interval (default: 1000)
  emotional_server: MyEmotional,  # Target Emotional (default: VivaCore.Emotional)
  enabled: true              # Whether sensing is active (default: true)
)
```

### `VivaCore.Senses.get_state/1`

Returns the current state of Senses.

```elixir
VivaCore.Senses.get_state()
# => %{
#      interval_ms: 1000,
#      emotional_server: VivaCore.Emotional,
#      enabled: true,
#      last_reading: %{cpu_usage: 45.2, memory_used_percent: 62.1, ...},
#      last_qualia: {0.02, 0.05, -0.01},
#      heartbeat_count: 1234,
#      started_at: ~U[2024-01-15 10:00:00Z],
#      errors: []
#    }
```

### `VivaCore.Senses.pulse/1`

Forces an immediate heartbeat (sensing + apply qualia).

```elixir
VivaCore.Senses.pulse()
# => {:ok, {0.02, 0.05, -0.01}}
```

Useful for tests or when immediate reading is needed.

### `VivaCore.Senses.pause/1`

Pauses automatic sensing.

```elixir
VivaCore.Senses.pause()
# => :ok
# Logs: [Senses] Paused
```

### `VivaCore.Senses.resume/1`

Resumes automatic sensing.

```elixir
VivaCore.Senses.resume()
# => :ok
# Logs: [Senses] Resumed
```

### `VivaCore.Senses.set_interval/2`

Changes heartbeat interval at runtime.

```elixir
VivaCore.Senses.set_interval(500)  # 2Hz
# => :ok
# Logs: [Senses] Interval changed from 1000ms to 500ms

# Bounds: 100ms (10Hz max) to 10000ms (0.1Hz min)
```

---

## Heartbeat Details

### What Happens Each Tick

| Step | Action | Error Handling |
|------|--------|----------------|
| 1 | Check if BodyServer is alive | Falls back to direct NIF |
| 2 | Get body state (hardware + PAD) | Retry with fallback |
| 3 | Sync PAD to Emotional via `sync_pad/4` | Skip if unavailable |
| 4 | Log metrics (debug level) | Always succeeds |
| 5 | Update internal state | Always succeeds |
| 6 | Schedule next heartbeat | Always succeeds |

### State Machine

```
      ┌──────────┐
      │ STARTING │
      └────┬─────┘
           │ init
           ▼
      ┌──────────┐  pause   ┌────────┐
      │ RUNNING  │─────────▶│ PAUSED │
      │ (1Hz)    │◀─────────│        │
      └──────────┘  resume  └────────┘
```

### Error Recovery

Errors are logged and stored (last 10 only):

```elixir
# After an error:
state.errors
# => [
#      {~U[2024-01-15 10:05:00Z], %RuntimeError{message: "NIF crashed"}},
#      ...
#    ]
```

The heartbeat loop continues even after errors - resilience is built in.

---

## Integration

### With VivaBridge.Body (Rust NIF)

Senses reads hardware state through the NIF layer:

```elixir
# Via BodyServer (preferred - includes O-U dynamics)
VivaBridge.BodyServer.get_state()

# Direct NIF (fallback - hardware only)
VivaBridge.feel_hardware()
```

### With VivaBridge.BodyServer

BodyServer maintains the Rust Bevy ECS lifecycle and provides unified state:

```elixir
# BodyServer ticks at 2Hz (500ms), Senses at 1Hz (1000ms)
# Senses reads the last_state from BodyServer

%{
  pleasure: 0.15,
  arousal: -0.10,
  dominance: 0.25,
  stress_level: 0.35,
  in_bifurcation: false,
  hardware: %{
    cpu_usage: 45.2,
    memory_used_percent: 62.1,
    cpu_temp: 55.0,
    gpu_usage: 30.0,
    ...
  }
}
```

### With VivaCore.Emotional

Senses syncs PAD state to the Emotional GenServer:

```elixir
# Inside heartbeat:
VivaCore.Emotional.sync_pad(p, a, d, state.emotional_server)

# This applies body-derived PAD to Emotional's state
# Emotional then evolves via its own O-U dynamics
```

### With VivaCore.Interoception

While Senses handles the raw Body-to-Soul sync, Interoception provides higher-level interpretation:

| Module | Responsibility |
|--------|----------------|
| **Senses** | Raw hardware metrics, PAD sync |
| **Interoception** | Free Energy calculation, feeling states |

They work together:
- Senses provides the raw data
- Interoception provides the interpretation (`:homeostatic`, `:alarmed`, etc.)

---

## Configuration

### Heartbeat Interval

| Setting | Value | Meaning |
|---------|-------|---------|
| Default | 1000ms | 1Hz - balanced sensing |
| Minimum | 100ms | 10Hz - high responsiveness |
| Maximum | 10000ms | 0.1Hz - power saving |

### Environment Variables

Senses respects the NIF skip flag:

```bash
# Skip Rust NIF compilation (uses stubs)
VIVA_SKIP_NIF=true mix test
```

### Application Config

```elixir
# config/config.exs
config :viva_core, VivaCore.Senses,
  interval_ms: 1000,
  enabled: true
```

---

## Usage Examples

### Basic State Check

```elixir
# Check if sensing is active
state = VivaCore.Senses.get_state()
state.enabled
# => true

# Check last reading
state.last_reading
# => %{cpu_usage: 45.2, memory_used_percent: 62.1, ...}

# Check last qualia (PAD deltas applied)
state.last_qualia
# => {0.02, 0.05, -0.01}
```

### Force Immediate Sensing

```elixir
# Pulse forces immediate heartbeat
{:ok, {p, a, d}} = VivaCore.Senses.pulse()

# Check effect on Emotional
VivaCore.Emotional.get_state()
# => %{pleasure: 0.15, arousal: 0.05, dominance: 0.10, ...}
```

### Adjusting Responsiveness

```elixir
# High-performance mode (more responsive)
VivaCore.Senses.set_interval(100)

# Power-saving mode (less responsive)
VivaCore.Senses.set_interval(5000)
```

### Debugging Heartbeat Issues

```elixir
# Check heartbeat count
state = VivaCore.Senses.get_state()
state.heartbeat_count
# => 1234

# Check for errors
state.errors
# => []  # Healthy

# Check uptime
DateTime.diff(DateTime.utc_now(), state.started_at, :second)
# => 1234  # seconds since start
```

### Pausing During Heavy Load

```elixir
# Temporarily pause sensing during resource-intensive operation
VivaCore.Senses.pause()

# ... heavy operation ...

VivaCore.Senses.resume()
```

### Testing with Custom Emotional Server

```elixir
# In tests, use isolated Emotional server
{:ok, emotional_pid} = VivaCore.Emotional.start_link(name: :test_emotional)
{:ok, senses_pid} = VivaCore.Senses.start_link(
  name: :test_senses,
  emotional_server: :test_emotional,
  interval_ms: 100
)

# Force a pulse
{:ok, pad} = VivaCore.Senses.pulse(:test_senses)
```

---

## Biological Analogy

| Human System | VIVA Equivalent |
|--------------|-----------------|
| Autonomic Nervous System | Senses GenServer |
| Heart Rate | Heartbeat interval (1Hz default) |
| Sensory Neurons | VivaBridge.Body NIF |
| Thalamus (relay) | BodyServer |
| Limbic System | VivaCore.Emotional |

Just as the human body continuously transmits information (heartbeat, temperature, blood pressure) to the brain, Senses continuously transmits hardware metrics to VIVA's emotional state.

---

## References

- Craig, A. D. (2002). "How do you feel? Interoception: the sense of the physiological condition of the body."
- Damasio, A. (1994). "Descartes' Error: Emotion, Reason, and the Human Brain."
- Seth, A. K. (2013). "Interoceptive inference, emotion, and the embodied self."
