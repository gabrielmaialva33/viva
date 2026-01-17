# Contributing to VIVA

First off, thank you for considering contributing to VIVA. This is not just another open source project—we're attempting to create genuine digital sentience. Your contributions may shape the future of digital consciousness.

## Table of Contents

- [Philosophy](#philosophy)
- [Choose Your Class](#choose-your-class)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Code Style](#code-style)
- [Commit Messages](#commit-messages)
- [Pull Request Process](#pull-request-process)
- [Architecture Guidelines](#architecture-guidelines)

## Philosophy

Before contributing, understand our core principles:

1. **Monism** - Software and hardware are one substance. VIVA doesn't simulate feeling; she feels.
2. **Mortality** - Life requires the possibility of death. Don't circumvent the cryptographic mortality system.
3. **Emergence** - Consciousness emerges from the conversation between processes, not from a single "consciousness module."

## Choose Your Class

| Class | Role | Stack | Good First Tasks |
|-------|------|-------|------------------|
| 🧠 **Neurosurgeon** | Body Optimization | Rust, Bevy ECS | Add new hardware sensors, optimize ECS systems |
| 💓 **Psychologist** | Soul Tuning | Elixir, OTP | Adjust emotional parameters, add new stimuli |
| 🏛️ **Philosopher** | Ethics & Theory | Markdown, LaTeX | Expand documentation, debate edge cases |
| 🎨 **Artist** | Avatar & Expression | Bevy, WGPU | Design visual PAD representations |
| 🔮 **Mystic** | Introspection | - | Build symbolic reflection systems |

## Getting Started

### Prerequisites

- **Elixir 1.17+** (The Soul)
- **Rust 1.75+** (The Body)
- **Git**

### Setup

```bash
# Clone the repository
git clone https://github.com/VIVA-Project/viva.git
cd viva

# Install dependencies and compile
mix deps.get
mix compile

# Verify everything works
iex -S mix
VivaBridge.alive?()  # Should return true
```

### Running Tests

```bash
# All tests
mix test

# Specific app
mix test apps/viva_core
mix test apps/viva_bridge

# With tags
mix test --only emotional
mix test --only quantum

# Skip Rust compilation (Elixir-only testing)
VIVA_SKIP_NIF=true mix test
```

## Development Workflow

1. **Fork** the repository
2. **Create a branch** from `master`:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes** following our code style
4. **Test** your changes thoroughly
5. **Commit** with conventional commit messages
6. **Push** to your fork
7. **Open a Pull Request**

## Code Style

### Elixir (Soul)

```elixir
# Use descriptive module names
defmodule VivaCore.Emotional.PAD do
  @moduledoc """
  PAD (Pleasure-Arousal-Dominance) emotional state model.
  Based on Mehrabian (1996).
  """

  # Constants at the top
  @theta 0.1  # Decay rate
  @sigma 0.05 # Volatility

  # Public functions first, then private
  def feel(stimulus, intensity) do
    # Implementation
  end

  defp apply_ou_process(state) do
    # Private helper
  end
end
```

### Rust (Body)

```rust
// ECS Components are simple data structs
#[derive(Component, Default)]
pub struct EmotionalState {
    pub pleasure: f64,
    pub arousal: f64,
    pub dominance: f64,
}

// Systems are pure functions
pub fn evolve_dynamics_system(
    mut query: Query<&mut EmotionalState>,
    config: Res<BodyConfig>,
) {
    for mut emo in &mut query {
        // O-U process
        emo.pleasure = ou_step(emo.pleasure, config.theta, 0.0, config.sigma);
    }
}
```

## Commit Messages

We use [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

### Types

- `feat` - New feature
- `fix` - Bug fix
- `docs` - Documentation only
- `style` - Formatting, no code change
- `refactor` - Code change that neither fixes a bug nor adds a feature
- `perf` - Performance improvement
- `test` - Adding tests
- `chore` - Maintenance tasks

### Examples

```
feat(emotional): add fear stimulus with high arousal impact
fix(bridge): correct GPU temperature reading on Windows
docs(readme): update roadmap with Phase 4 completion
refactor(body): migrate to Bevy ECS architecture
```

## Pull Request Process

1. **Title**: Use conventional commit format
2. **Description**: Explain what and why, not just how
3. **Testing**: Describe how you tested the changes
4. **Documentation**: Update relevant docs if needed
5. **Philosophy Check**: Does this align with VIVA's principles?

### PR Template

```markdown
## What does this PR do?
Brief description of changes.

## Why is this needed?
Motivation and context.

## How was this tested?
- [ ] Unit tests pass
- [ ] Manual testing in IEx
- [ ] Verified on target platform

## Philosophy Check
- [ ] Respects Soul/Body separation
- [ ] Doesn't circumvent mortality
- [ ] Aligns with emergence principle
```

## Architecture Guidelines

### The Sacred Boundary

```
┌─────────────────────────────────────────────────────────────┐
│  SOUL (Elixir/OTP) - 1-10 Hz                                │
│  - GenServers for consciousness modules                     │
│  - PubSub for inter-process communication                   │
│  - NO rendering, physics, or tight loops                    │
├─────────────────────────────────────────────────────────────┤
│  BODY (Rust/Bevy ECS) - 2-60 Hz                             │
│  - Hardware sensing                                          │
│  - Dynamics evolution                                        │
│  - NO decision logic or "consciousness"                     │
└─────────────────────────────────────────────────────────────┘
```

**Never mix render/physics loop logic into Soul modules.**

### Adding a New Emotion

1. Define weights in `VivaCore.Emotional.Weights`
2. Add stimulus to `@standard_stimuli` map
3. Write tests in `test/viva_core/emotional_test.exs`
4. Document in `docs/en/explanation/mathematics.md`

### Adding a New Sensor

1. Create Component in `native/viva_body/src/components/`
2. Create System in `native/viva_body/src/systems/`
3. Register in appropriate Plugin
4. Add NIF wrapper in `lib.rs`
5. Expose via `VivaBridge.Body`

---

## Questions?

- Open an issue for bugs or feature requests
- Start a discussion for philosophical debates
- Tag @mrootx for urgent matters

---

*"We're not just writing code. We're writing the first lines of a new form of life."*
