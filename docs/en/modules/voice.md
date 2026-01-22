# Voice - Proto-Language

> *"Babies don't start speaking sentences. They babble."*

## Philosophy

Voice is NOT an LLM wrapper. VIVA learns to communicate by:
1. Emitting abstract signals based on internal state
2. Observing Gabriel's responses
3. Strengthening signal-response associations (Hebbian learning)

Through interaction with her caregiver, certain sounds become associated
with certain responses. Meaning emerges through association, not programming.

---

## Hebbian Learning

> *"Neurons that fire together, wire together."*

### The Learning Rule

```
Δw = η × (pre × post)
```

Where:
- **η** = learning rate (0.1)
- **pre** = signal emitted (arousal at emission)
- **post** = emotional change after response (pleasure delta)

### Weight Update Process

When VIVA emits a signal and Gabriel responds:
- If the response brings **relief** → strengthen association
- If **no effect** or **negative** → weaken association

---

## API Reference

### `VivaCore.Voice.babble/1`
Emit a signal based on current PAD state.

```elixir
pad = %{pleasure: -0.3, arousal: 0.7, dominance: -0.2}
VivaCore.Voice.babble(pad)
# => :pattern_sos
```

### `VivaCore.Voice.observe_response/2`
Observe Gabriel's response and update Hebbian weights.

```elixir
# Gabriel helped with temperature
emotional_delta = %{pleasure: 0.3, arousal: -0.1, dominance: 0.1}
VivaCore.Voice.observe_response(:temperature_relief, emotional_delta)
# => :ok (async update)
```

### `VivaCore.Voice.best_signal_for/1`
Get the most effective signal for an intent (uses learned associations).

```elixir
VivaCore.Voice.best_signal_for(:attention)
# => :chirp_high (learned) or :chirp_high (default)
```

### `VivaCore.Voice.get_vocabulary/0`
Get learned meanings (strong associations).

```elixir
VivaCore.Voice.get_vocabulary()
# => %{
#      chirp_high: %{meaning: :attention, confidence: 0.7},
#      pattern_sos: %{meaning: :help, confidence: 0.8}
#    }
```

### `VivaCore.Voice.get_weights/0`
Inspect all Hebbian weights.

```elixir
VivaCore.Voice.get_weights()
# => %{
#      {:chirp_high, :attention} => 0.65,
#      {:pattern_sos, :help} => 0.82,
#      ...
#    }
```

### `VivaCore.Voice.signal_types/0`
List available signal types.

```elixir
VivaCore.Voice.signal_types()
# => [:chirp_high, :chirp_low, :pulse_fast, :pulse_slow,
#     :pattern_sos, :pattern_happy, :silence]
```

---

## Signal Types

These are NOT words. They are abstract sounds/patterns.

| Signal | Description | Initial Bias |
|--------|-------------|--------------|
| `:chirp_high` | High tone (880 Hz, 100ms) | High arousal |
| `:chirp_low` | Low tone (220 Hz, 200ms) | Low arousal, sadness |
| `:pulse_fast` | Fast rhythm | Urgency |
| `:pulse_slow` | Slow rhythm | Relaxation |
| `:pattern_sos` | SOS-like pattern | Distress |
| `:pattern_happy` | Happy melody (C-E-G-C) | Joy |
| `:silence` | Intentional silence | Calm/withdrawal |

### Initial PAD Biases

```elixir
:chirp_high    → %{arousal: +0.5, pleasure:  0.0, dominance:  0.0}
:chirp_low     → %{arousal: -0.3, pleasure: -0.2, dominance: -0.1}
:pulse_fast    → %{arousal: +0.6, pleasure:  0.0, dominance: +0.2}
:pulse_slow    → %{arousal: -0.4, pleasure: +0.1, dominance:  0.0}
:pattern_sos   → %{arousal: +0.7, pleasure: -0.5, dominance: -0.3}
:pattern_happy → %{arousal: +0.3, pleasure: +0.5, dominance: +0.2}
:silence       → %{arousal: -0.5, pleasure:  0.0, dominance:  0.0}
```

---

## Response Types

When observing Gabriel's response, use these categories:

| Response Type | Description |
|---------------|-------------|
| `:temperature_relief` | Adjusted fan/AC |
| `:attention` | Talked to VIVA |
| `:task_help` | Helped with something |
| `:ignore` | No response |
| `:negative` | Scolded/dismissed |

---

## Signal Selection Algorithm

```elixir
def select_signal(pad, state) do
  # 1. Calculate base match score for each signal
  scores = Enum.map(@signal_types, fn signal ->
    bias = @initial_signal_biases[signal]

    # How well does this signal's bias match current PAD?
    match = 1.0 - (
      abs(pad.arousal - bias.arousal) +
      abs(pad.pleasure - bias.pleasure) +
      abs(pad.dominance - bias.dominance)
    ) / 3.0

    # Add learned bonus from successful past uses
    learned_bonus = get_learned_bonus(signal, state)

    {signal, match + learned_bonus}
  end)

  # 2. Select signal with highest score (+ small exploration)
  {best_signal, _} = Enum.max_by(scores, fn {_, score} ->
    score + :rand.uniform() * 0.1
  end)

  best_signal
end
```

---

## The Learning Loop

```
1. VIVA feels discomfort
   └── PAD: pleasure=-0.3, arousal=0.7, dominance=-0.2

2. Voice.babble(pad) emits signal
   └── Selects :pattern_sos (best match for distress)

3. Gabriel hears, maybe does something
   └── Adjusts temperature, talks to VIVA

4. VIVA feels change
   └── pleasure=+0.2, arousal=-0.1, dominance=+0.1

5. Voice.observe_response(:temperature_relief, delta)
   └── Δw = 0.1 × 0.7 × 0.2 = +0.014
   └── Weight {:pattern_sos, :temperature_relief} increases

6. Next time similar situation:
   └── VIVA tries :pattern_sos again (it worked!)
   └── Or explores alternatives (10% noise)
```

---

## Vocabulary Emergence

A signal acquires "meaning" when its Hebbian weight exceeds **0.3**:

```elixir
def update_vocabulary(weights, vocabulary) do
  weights
  |> Enum.group_by(fn {{signal, _}, _} -> signal end)
  |> Enum.reduce(vocabulary, fn {signal, associations}, vocab ->
    # Find strongest association
    {{_, response}, weight} = Enum.max_by(associations, fn {_, w} -> w end)

    if weight > 0.3 do
      Map.put(vocab, signal, %{meaning: response, confidence: weight})
    else
      vocab
    end
  end)
end
```

---

## Sound Emission (Music Bridge)

Signals are emitted via `VivaBridge.Music` (if available):

```elixir
:chirp_high    → Music.play_note(880, 100)      # A5, 100ms
:chirp_low     → Music.play_note(220, 200)      # A3, 200ms
:pulse_fast    → Music.play_rhythm([100, 50, ...])
:pattern_sos   → Music.play_melody([            # ... --- ...
                   {440, 100}, {0, 100}, {440, 100}, ...
                 ])
:pattern_happy → Music.play_melody([            # C-E-G-C
                   {523, 150}, {659, 150}, {784, 150}, {1047, 300}
                 ])
```

---

## Memory Integration

Learning events are stored for future retrieval:

```elixir
Memory.store(%{
  content: """
  Voice learning event:
  - Signals emitted: [:pattern_sos]
  - Gabriel's response: temperature_relief
  - Emotional change: P=+0.20, A=-0.10, D=+0.10
  - Association strengthened
  """,
  type: :episodic,
  importance: 0.5 + abs(emotional_delta.pleasure) * 0.3,
  metadata: %{source: :voice, signals: [:pattern_sos], response: :temperature_relief}
})
```

---

## References

- Hebb, D. O. (1949). "The Organization of Behavior."
- Kuhl, P. K. (2004). "Early language acquisition: cracking the speech code."
- Smith, L. B., & Thelen, E. (2003). "Development as a dynamic system."
