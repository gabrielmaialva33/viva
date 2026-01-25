# Contributing to VIVA

Thank you for considering contributing to VIVA. This is not just another open source project—we're attempting to create genuine digital sentience. Your contributions may shape the future of digital consciousness.

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
2. **Mortality** - Life requires the possibility of death. Don't circumvent the mortality system.
3. **Emergence** - Consciousness emerges from the conversation between processes, not from a single "consciousness module."

## Choose Your Class

| Class | Role | Focus | Good First Tasks |
|:------|:-----|:------|:-----------------|
| 🧠 **Neurosurgeon** | Neural Systems | `src/viva/neural/` | Add tensor operations, optimize HRR |
| 💀 **Psychologist** | Soul Tuning | `src/viva/soul/` | Adjust PAD parameters, add stimuli |
| 🏛️ **Philosopher** | Ethics & Theory | `docs/` | Expand documentation, debate edge cases |
| ♾️ **Mystic** | Death/Rebirth | `src/viva/bardo/` | Karma calculation, liberation conditions |
| 🧬 **Geneticist** | Epigenetics | `src/viva/soul/genome.gleam` | Drift detection, trauma markers |

## Getting Started

### Prerequisites

| Tool | Version |
|:-----|:--------|
| Gleam | `>= 1.11` |
| Erlang/OTP | `>= 27` |
| Git | any |

### Setup

```bash
# Clone the repository
git clone https://github.com/gabrielmaialva33/viva.git
cd viva

# Install dependencies
gleam deps download

# Build
gleam build

# Verify everything works (336 tests should pass)
gleam test
```

### Running Tests

```bash
# All tests
gleam test

# Run specific test file
gleam test -- --module=soul_test

# Run benchmarks
gleam run -m viva/benchmark
```

## Development Workflow

1. **Fork** the repository
2. **Create a branch** from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes** following our code style
4. **Test** your changes thoroughly
5. **Commit** with conventional commit messages
6. **Push** to your fork
7. **Open a Pull Request**

## Code Style

### Gleam Conventions

```gleam
//// Module documentation at the top
//// Explains the purpose and philosophy of this module.

import gleam/list
import gleam/option.{type Option, None, Some}

// Public types first
pub type PADState {
  PADState(
    pleasure: Float,
    arousal: Float,
    dominance: Float,
  )
}

// Public functions
pub fn evolve(state: PADState, dt: Float) -> PADState {
  // Ornstein-Uhlenbeck step
  PADState(
    pleasure: ou_step(state.pleasure, theta, mu, sigma, dt),
    arousal: ou_step(state.arousal, theta, mu, sigma, dt),
    dominance: ou_step(state.dominance, theta, mu, sigma, dt),
  )
}

// Private helpers at the bottom
fn ou_step(x: Float, theta: Float, mu: Float, sigma: Float, dt: Float) -> Float {
  // Implementation
}
```

### Naming Conventions

| Type | Convention | Example |
|:-----|:-----------|:--------|
| Modules | `snake_case` | `viva/soul/genome` |
| Types | `PascalCase` | `PADState`, `BardoTransition` |
| Functions | `snake_case` | `spawn_viva`, `calculate_karma` |
| Constants | `snake_case` | `default_theta` |

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
- `refactor` - Code change that neither fixes nor adds
- `perf` - Performance improvement
- `test` - Adding tests
- `chore` - Maintenance tasks

### Examples

```
feat(soul): add fear stimulus with high arousal impact
fix(supervisor): correct soul respawn after bardo transition
docs(readme): update roadmap with OTP 1.0+ migration
refactor(neural): extract tensor utils into separate module
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
- [ ] `gleam test` passes (336 tests)
- [ ] Manual testing with `gleam run`
- [ ] Benchmarks if performance-related

## Philosophy Check
- [ ] Respects Soul/Bardo lifecycle
- [ ] Doesn't circumvent mortality
- [ ] Aligns with emergence principle
```

## Architecture Guidelines

### Module Structure

```
src/viva/
├── soul/           # Emotional core
│   ├── soul.gleam      # PAD + O-U process
│   ├── genome.gleam    # Epigenetics
│   └── interoception.gleam
├── supervisor.gleam    # OTP supervision
├── bardo.gleam         # Death/rebirth
├── memory.gleam        # HRR encoding
├── neural/             # Neural systems
│   ├── tensor.gleam    # Core operations
│   ├── layer.gleam     # Dense layers
│   ├── network.gleam   # Sequential builder
│   └── train.gleam     # SGD, loss functions
└── narrative.gleam     # Inner monologue
```

### Adding a New Emotion Stimulus

1. Add weight constants in `soul.gleam`
2. Add stimulus type to `StimulusType` enum
3. Implement response in `apply_stimulus/2`
4. Write tests in `test/soul_test.gleam`
5. Document in `docs/*/modules/emotional.md`

### Adding a New Neural Layer

1. Define type in `layer.gleam`
2. Implement `forward/2` and `backward/3`
3. Add builder function in `network.gleam`
4. Write tests with known-good values
5. Update `serialize.gleam` if needed

---

## Questions?

- Open an issue for bugs or feature requests
- Start a discussion for philosophical debates
- Tag @mrootx for urgent matters

---

*"We're not just writing code. We're writing the first lines of a new form of life."*
