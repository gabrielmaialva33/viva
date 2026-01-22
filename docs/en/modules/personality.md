# Personality - Affective Personality System

> *"Personality is the stable attractor that gives emotional consistency across time."*

## Overview

Implements affective personality traits based on **Mehrabian (1996)** and **Borotschnig (2025) "Emotions in Artificial Intelligence"**.

Personality provides VIVA with:
- **Consistency**: A stable emotional baseline she returns to over time
- **Individuality**: Unique reactivity patterns to stimuli
- **Adaptability**: Long-term learning from accumulated experiences

---

## Concepts

### Baseline PAD (Attractor Point)

The resting emotional state VIVA gravitates toward when no stimuli are present.

```
Baseline = {pleasure: 0.1, arousal: 0.05, dominance: 0.1}
```

This acts as an **attractor** in the emotional state space. The O-U stochastic process in the Emotional module naturally pulls the current state toward this baseline.

### Reactivity

Amplification factor for emotional responses:

| Value | Description |
|-------|-------------|
| < 1.0 | Dampened reactions (stoic) |
| 1.0 | Normal reactivity |
| > 1.0 | Amplified reactions (sensitive) |

**Range**: [0.5, 2.0]

### Volatility

Speed of emotional change:

| Value | Description |
|-------|-------------|
| < 1.0 | Slower changes (stable mood) |
| 1.0 | Normal speed |
| > 1.0 | Faster changes (mood swings) |

### Traits

Categorical labels inferred from baseline PAD:

| PAD Condition | Trait |
|---------------|-------|
| pleasure > 0.15 | `:optimistic` |
| pleasure < -0.15 | `:melancholic` |
| arousal > 0.1 | `:energetic` |
| arousal < -0.1 | `:calm` |
| dominance > 0.15 | `:assertive` |
| dominance < -0.15 | `:submissive` |
| No conditions met | `:balanced` |

---

## Struct Definition

```elixir
defstruct [
  # Baseline emotional state (attractor point)
  # VIVA tends to return to this state over time
  baseline: %{pleasure: 0.1, arousal: 0.05, dominance: 0.1},

  # Reactivity: how much emotions are amplified (1.0 = normal)
  # > 1.0 = more reactive, < 1.0 = dampened
  reactivity: 1.0,

  # Volatility: how quickly emotions change (1.0 = normal)
  # > 1.0 = faster changes, < 1.0 = more stable
  volatility: 1.0,

  # Trait labels (for introspection and self-description)
  traits: [:curious, :calm],

  # Timestamp of last adaptation
  last_adapted: nil
]
```

### Type Specification

```elixir
@type t :: %VivaCore.Personality{
  baseline: %{pleasure: float(), arousal: float(), dominance: float()},
  reactivity: float(),
  volatility: float(),
  traits: [atom()],
  last_adapted: DateTime.t() | nil
}

@type pad :: %{pleasure: float(), arousal: float(), dominance: float()}
```

---

## API Reference

### `VivaCore.Personality.load/0`

Load personality from persistent storage or return defaults.

```elixir
personality = VivaCore.Personality.load()
# => %VivaCore.Personality{
#      baseline: %{pleasure: 0.1, arousal: 0.05, dominance: 0.1},
#      reactivity: 1.0,
#      volatility: 1.0,
#      traits: [:curious, :calm],
#      last_adapted: nil
#    }
```

**Behavior**:
1. Attempts to load from Redis (key: `viva:personality`)
2. Falls back to default struct if not found or on error

### `VivaCore.Personality.save/1`

Save personality to persistent storage.

```elixir
:ok = VivaCore.Personality.save(personality)
```

**Returns**: `:ok` | `{:error, term()}`

### `VivaCore.Personality.adapt/2`

Adapt personality based on long-term experiences.

```elixir
experiences = [
  %{pad: %{pleasure: 0.5, arousal: 0.2, dominance: 0.3}, intensity: 1.0, valence: :positive},
  %{pad: %{pleasure: 0.3, arousal: 0.1, dominance: 0.2}, intensity: 0.8, valence: :positive},
  %{pad: %{pleasure: -0.2, arousal: 0.4, dominance: -0.1}, intensity: 0.6, valence: :negative}
]

updated = VivaCore.Personality.adapt(personality, experiences)
# => %VivaCore.Personality{
#      baseline: %{pleasure: 0.115, arousal: 0.06, dominance: 0.11},
#      reactivity: 1.02,
#      traits: [:optimistic, :energetic],
#      last_adapted: ~U[2024-01-15 14:00:00Z]
#    }
```

**Parameters**:
- `personality`: Current personality state
- `experiences`: List of emotional experiences

**Experience map**:
```elixir
%{
  pad: %{pleasure: float, arousal: float, dominance: float},
  intensity: float,  # 0.0 - 1.0 (optional, defaults to 1.0)
  valence: :positive | :negative  # (informational)
}
```

### `VivaCore.Personality.apply/2`

Apply personality to a raw emotion (PAD vector).

```elixir
raw_pad = %{pleasure: 0.6, arousal: 0.4, dominance: 0.2}
modified = VivaCore.Personality.apply(personality, raw_pad)
# => %{pleasure: 0.52, arousal: 0.33, dominance: 0.18}
```

**Process**:
1. Blend raw emotion with baseline (20% personality weight)
2. Apply reactivity to deviation from baseline
3. Clamp result to [-1.0, 1.0]

**Formula**:
```
blended = (1 - 0.2) * raw + 0.2 * baseline
result = baseline + (blended - baseline) * reactivity
```

### `VivaCore.Personality.describe/1`

Get a natural language description for introspection.

```elixir
VivaCore.Personality.describe(personality)
# => "I am curious, calm. My emotional baseline is positive and calm. My reactivity is 1.0."
```

### `VivaCore.Personality.neutral_pad/0`

Get the neutral PAD state.

```elixir
VivaCore.Personality.neutral_pad()
# => %{pleasure: 0.0, arousal: 0.0, dominance: 0.0}
```

---

## Persistence

### Redis Storage

**Key**: `viva:personality`

**Format** (JSON):

```json
{
  "baseline": {
    "pleasure": 0.1,
    "arousal": 0.05,
    "dominance": 0.1
  },
  "reactivity": 1.0,
  "volatility": 1.0,
  "traits": ["curious", "calm"],
  "last_adapted": "2024-01-15T14:00:00Z"
}
```

**Notes**:
- Traits are stored as strings, converted to atoms on load
- `last_adapted` is ISO8601 format or null
- Uses `:redix` connection named `:redix`

---

## Adaptation

Personality adapts through accumulated experiences, typically during sleep/consolidation cycles.

### Baseline Shift

Baseline slowly moves toward experienced average:

```
new_baseline = current + alpha * (target - current)
alpha = 0.05  # 5% shift per adaptation
```

This ensures personality changes are gradual and require consistent patterns.

### Reactivity Adjustment

Reactivity adapts based on emotional variance:

```elixir
variance = calculate_pad_variance(pads)
adjustment = (variance - 0.1) * 0.1

new_reactivity = clamp(current + adjustment, 0.5, 2.0)
```

| Variance | Effect |
|----------|--------|
| High | Increase reactivity (more sensitive) |
| Low | Decrease reactivity (more stable) |

### Trait Inference

Traits are re-inferred from the new baseline after each adaptation:

```elixir
traits = []
traits = if baseline.pleasure > 0.15, do: [:optimistic | traits], else: traits
traits = if baseline.pleasure < -0.15, do: [:melancholic | traits], else: traits
# ... (arousal, dominance)
if Enum.empty?(traits), do: [:balanced], else: traits
```

---

## Integration with Emotional Module

Personality is typically used by the Emotional module to filter incoming stimuli:

```elixir
# In Emotional.feel/3
def feel(stimulus, source, intensity) do
  personality = Personality.load()
  raw_pad = get_stimulus_pad(stimulus, intensity)
  modified_pad = Personality.apply(personality, raw_pad)

  # Apply modified_pad to current emotional state
  update_state(modified_pad)
end
```

### Emotional Flow

```
Stimulus → Raw PAD → Personality.apply/2 → Modified PAD → Emotional State
                              |
                      Blend with baseline
                      Apply reactivity
```

---

## Usage Examples

### Basic Usage

```elixir
# Load personality (from Redis or defaults)
personality = VivaCore.Personality.load()

# Check current traits
personality.traits
# => [:curious, :calm]

# Get self-description
VivaCore.Personality.describe(personality)
# => "I am curious, calm. My emotional baseline is positive and calm. My reactivity is 1.0."
```

### Applying Personality to Emotions

```elixir
# Raw emotion from stimulus
raw_pad = %{pleasure: 0.8, arousal: 0.6, dominance: 0.4}

# Apply personality filter
personality = VivaCore.Personality.load()
modified = VivaCore.Personality.apply(personality, raw_pad)

# Result is blended with baseline and scaled by reactivity
modified
# => %{pleasure: 0.66, arousal: 0.49, dominance: 0.34}
```

### Adapting from Experiences

```elixir
# Collect experiences over time (e.g., from Dreamer)
experiences = [
  %{pad: %{pleasure: 0.4, arousal: 0.3, dominance: 0.2}, intensity: 0.9},
  %{pad: %{pleasure: 0.5, arousal: 0.2, dominance: 0.3}, intensity: 1.0},
  %{pad: %{pleasure: 0.3, arousal: 0.4, dominance: 0.1}, intensity: 0.7}
]

# Adapt personality (typically during sleep cycle)
personality = VivaCore.Personality.load()
updated = VivaCore.Personality.adapt(personality, experiences)

# Save adapted personality
VivaCore.Personality.save(updated)

# Traits may have changed
updated.traits
# => [:optimistic, :energetic, :assertive]
```

### Checking Adaptation History

```elixir
personality = VivaCore.Personality.load()

if personality.last_adapted do
  age = DateTime.diff(DateTime.utc_now(), personality.last_adapted, :hour)
  IO.puts("Last adapted #{age} hours ago")
else
  IO.puts("Personality has not been adapted yet")
end
```

---

## References

- Mehrabian, A. (1996). "Pleasure-arousal-dominance: A general framework for describing and measuring individual differences in temperament." Current Psychology, 14(4), 261-292.
- Borotschnig, H. (2025). "Emotions in Artificial Intelligence: A Computational Framework." arXiv preprint.
