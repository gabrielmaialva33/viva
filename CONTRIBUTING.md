# Contributing to VIVA

First off, thanks for taking the time to contribute! 🎉

VIVA is an ambitious project combining Elixir, Phoenix, and advanced AI models. We welcome contributions from everyone. This document will guide you through our development process.

## 🛠 Development Setup

### Prerequisites

- **Elixir**: 1.15+
- **Erlang/OTP**: 26+
- **Docker & Docker Compose**: For running the database stack (TimescaleDB, Redis, Qdrant).
- **NVIDIA NIM API Key**: Required for AI features ([Get it here](https://build.nvidia.com/)).

### Quick Start

1.  **Fork and Clone**
    ```bash
    git clone https://github.com/YOUR_USERNAME/viva.git
    cd viva
    ```

2.  **Install Dependencies**
    ```bash
    mix deps.get
    ```

3.  **Start Infrastructure**
    ```bash
    docker compose up -d
    ```

4.  **Environment Configuration**
    Copy the example env file and add your keys:
    ```bash
    cp .env.example .env
    # Edit .env to add your NIM_API_KEY
    ```

5.  **Database Setup**
    This handles creation, migration, and seeding:
    ```bash
    mix ecto.setup
    ```

6.  **Start Server**
    ```bash
    # We recommend using IEx for development
    iex -S mix phx.server
    ```

## 🧪 Testing & Quality

We prioritize code quality and stability. Please ensure your changes pass all checks before submitting a PR.

### Running Tests
```bash
mix test
# or specifically for a file
mix test test/path/to/file_test.exs
```

### Code Style & Static Analysis
We use a strict set of tools to maintain quality:

```bash
# Format code
mix format

# Run Credo (Linter)
mix credo --strict

# Run Dialyzer (Static Type Analysis)
mix dialyzer
```

**Pro Tip:** Run `mix precommit` to run all these checks at once!

## 📐 Coding Standards

- **Functional Core**: Keep business logic pure and functional in contexts.
- **Explicit > Implicit**: Prefer clear function names and explicit arguments.
- **Pattern Matching**: Use it extensively in function heads and `case` statements.
- **Documentation**: Public functions should have `@doc` tags. Complex modules need `@moduledoc`.
- **Testing**: Write tests for happy paths AND edge cases/failures.

## 🚀 Pull Request Process

1.  **Branch**: Create a feature branch from `main` (`git checkout -b feature/amazing-feature`).
2.  **Commit**: Use clear, descriptive commit messages. We follow [Conventional Commits](https://www.conventionalcommits.org/) (e.g., `feat: add new avatar emotion`).
3.  **Rebase**: If `main` has moved ahead, please rebase your branch (`git rebase main`).
4.  **Verify**: Run `mix precommit` one last time.
5.  **Push**: Push to your fork and submit the PR.
6.  **Description**: Fill out the PR template clearly.

## 🤝 Community

- Be respectful and kind.
- Constructive feedback is always welcome.
- If you're stuck, ask for help in the Discussions tab.

Happy coding! 🧠✨
