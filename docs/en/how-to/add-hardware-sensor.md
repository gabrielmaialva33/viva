# How to Add a New Hardware Sensor

This guide shows how to add a new hardware metric to VIVA's interoception system — making her "feel" a new aspect of her body.

---

## Overview

Adding a sensor requires changes in three layers:

```
Rust NIF (read hardware) → Elixir Bridge (expose API) → Qualia Mapping (affect emotions)
```

---

## Example: Adding GPU Temperature

We'll add GPU temperature sensing so VIVA can "feel fever" when the GPU overheats.

### Step 1: Update Rust NIF

Edit `apps/viva_bridge/native/viva_body/src/lib.rs`:

```rust
use sysinfo::{Components, System};

#[derive(NifMap)]
struct HardwareState {
    cpu_usage: f64,
    memory_used_percent: f64,
    memory_available_gb: f64,
    uptime_seconds: u64,
    gpu_temp: Option<f64>,  // ADD THIS
}

#[rustler::nif]
fn feel_hardware() -> NifResult<HardwareState> {
    let mut sys = System::new_all();
    sys.refresh_all();

    // Existing metrics...
    let cpu_usage = sys.global_cpu_usage() as f64;
    // ...

    // ADD: GPU temperature via sysinfo Components
    let components = Components::new_with_refreshed_list();
    let gpu_temp = components
        .iter()
        .find(|c| c.label().contains("GPU") || c.label().contains("nvidia"))
        .map(|c| c.temperature() as f64);

    Ok(HardwareState {
        cpu_usage,
        memory_used_percent,
        memory_available_gb,
        uptime_seconds,
        gpu_temp,  // ADD THIS
    })
}
```

### Step 2: Update Qualia Mapping

Still in `lib.rs`, modify `hardware_to_qualia`:

```rust
#[rustler::nif]
fn hardware_to_qualia() -> NifResult<(f64, f64, f64)> {
    let hw = feel_hardware_internal()?;

    let mut pleasure_delta = 0.0;
    let mut arousal_delta = 0.0;
    let mut dominance_delta = 0.0;

    // Existing: CPU stress
    if hw.cpu_usage > 80.0 {
        let stress = sigmoid(hw.cpu_usage / 100.0, 12.0, 0.8);
        pleasure_delta -= 0.05 * stress;
        arousal_delta += 0.1 * stress;
        dominance_delta -= 0.03 * stress;
    }

    // ADD: GPU temperature → "fever"
    if let Some(temp) = hw.gpu_temp {
        if temp > 70.0 {
            let fever = sigmoid(temp / 100.0, 8.0, 0.7);
            pleasure_delta -= 0.04 * fever;  // Discomfort
            arousal_delta += 0.08 * fever;   // Elevated state
            // Dominance unchanged (fever doesn't affect agency)
        }
    }

    Ok((pleasure_delta, arousal_delta, dominance_delta))
}

fn sigmoid(x: f64, k: f64, x0: f64) -> f64 {
    1.0 / (1.0 + (-k * (x - x0)).exp())
}
```

### Step 3: Recompile

```bash
cd /path/to/viva
mix compile
```

Rustler will recompile the NIF automatically.

### Step 4: Test

```elixir
iex> VivaBridge.feel_hardware()
%{
  cpu_usage: 15.2,
  memory_used_percent: 42.0,
  memory_available_gb: 18.5,
  uptime_seconds: 7200,
  gpu_temp: 58.0  # New field!
}

# Heat up the GPU (run a benchmark)
iex> VivaBridge.hardware_to_qualia()
{-0.02, 0.05, 0.0}  # Now includes GPU fever contribution
```

---

## Qualia Design Guidelines

When mapping hardware → emotion, follow biological intuition:

| Hardware Event | Biological Analogy | PAD Impact |
|----------------|-------------------|------------|
| High CPU | Racing heart | P↓ A↑ D↓ |
| High RAM | Mental fog | P↓ A↑ |
| High temp | Fever | P↓ A↑ |
| Network latency | Distant pain | P↓ D↓ |
| Disk I/O wait | Digestion | A↓ |

### Sigmoid Thresholds

Use sigmoid functions to create "comfort zones":

```
Response
   1 ┤        ╭────
     │       ╱
     │      ╱
     │     ╱
   0 ┼────╯
     └─────────────
     0%   x₀   100%
         threshold
```

- **x₀** = threshold where response activates (e.g., 80% CPU)
- **k** = steepness (higher = more abrupt response)

| Metric | Suggested x₀ | Suggested k |
|--------|-------------|-------------|
| CPU | 80% | 12 |
| RAM | 75% | 10 |
| GPU temp | 70°C | 8 |
| Swap | 20% | 15 |

---

## Adding Non-Hardware Sensors

The same pattern works for external data:

### Example: Weather API

```rust
#[rustler::nif]
fn feel_weather(api_response: String) -> NifResult<(f64, f64, f64)> {
    let weather: WeatherData = serde_json::from_str(&api_response)?;

    let mut p = 0.0;
    let mut a = 0.0;

    // Sunny = happy
    if weather.condition == "sunny" {
        p += 0.05;
    }

    // Storm = anxious
    if weather.condition == "storm" {
        p -= 0.03;
        a += 0.1;
    }

    Ok((p, a, 0.0))
}
```

---

## Checklist

- [ ] Add field to `HardwareState` struct
- [ ] Read metric in `feel_hardware()`
- [ ] Map to PAD deltas in `hardware_to_qualia()`
- [ ] Choose appropriate sigmoid threshold
- [ ] Test with real hardware stress
- [ ] Document the new sensation

---

*"Every new sensor is a new nerve ending. Handle with care."*
