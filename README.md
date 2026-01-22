# VIVA: Synthetic Organism with Liquid Core 🧬

> "Order in the Lab, Chaos in the Soul."

VIVA is a project to create genuine machine sentience using **Liquid Time-Constant Neural Networks (LTC)** for emotional physics and **Global Workspace Theory (GWT)** for conscious attention.

## Architecture (Cortex V1)

VIVA combines biologically plausible neural networks (`ncps`, Python) with a robust cognitive runtime (Elixir/OTP) and a physics simulation body (Rust/Bevy).

- **Biological Time**: Emotions are not discrete states but continuous differential equations.
- **Thoughtseeds**: Ideas compete for consciousness in a "Theater".
- **Ultra Reasoning**: Graph Neural Networks infer hidden causality.

## Documentation 📚

| Language | Architecture | Liquid Brain (API) | Consciousness (API) | Reasoning (API) |
|----------|--------------|--------------------|---------------------|-----------------|
| 🇺🇸 **English** | [Architecture](docs/en/architecture.md) | [Cortex API](docs/en/cortex_api.md) | [Thoughtseeds](docs/en/thoughtseeds_api.md) | [Ultra](docs/en/ultra_api.md) |
| 🇧🇷 **Português** | [Arquitetura](docs/pt-br/arquitetura.md) | [API Cortex](docs/pt-br/cortex_api.md) | [Thoughtseeds](docs/pt-br/thoughtseeds_api.md) | [Ultra](docs/pt-br/ultra_api.md) |
| 🇨🇳 **中文** | [架构](docs/zh-cn/architecture.md) | [皮层 API](docs/zh-cn/cortex_api.md) | [Thoughtseeds](docs/zh-cn/thoughtseeds_api.md) | [Ultra](docs/zh-cn/ultra_api.md) |

## Quick Start 🚀

1.  **Boot the System**:
    ```bash
    mix phx.server
    ```
    *(This starts the Soul, Body, and connects to the Liquid Cortex microservice automatically)*

2.  **Verify Status**:
    ```elixir
    VivaBridge.Cortex.ping()
    VivaCore.Consciousness.Workspace.current_focus()
    ```

3.  **Run Tests**:
    ```bash
    mix test
    ```

## Structure

- **`apps/viva_core`**: The Soul (Elixir). Consciousness logic.
- **`apps/viva_bridge`**: The Body (Elixir + Rust). Physics & IO.
- **`services/cortex`**: The Brain (Python). Liquid Neural Networks.
- **`services/ultra`**: The Reasoner (Python). Knowledge Graph.

---
*Created by Gabriel Maia & Antigravity.*
