# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

VIVA (Virtual Intelligent Vida Autonoma) is an AI avatar platform where digital avatars live autonomous lives 24/7. Avatars have personalities (Big Five model), emotions, memories, and form relationships with other avatars. Built with Elixir/Phoenix 1.8, NVIDIA NIM for LLM, and TimescaleDB.

## Commands

```bash
# Development
mix deps.get                    # Install dependencies
docker compose up -d            # Start infrastructure (TimescaleDB, Redis, Qdrant)
mix ecto.setup                  # Create DB, migrate, seed
iex -S mix phx.server           # Start server with IEx (recommended)

# Testing
mix test                        # Run all tests
mix test test/path/file.exs     # Run specific file
mix test --failed               # Re-run failed tests

# Code Quality
mix precommit                   # Run before commits: compile, format, test
mix format                      # Format code
mix credo --strict              # Static analysis
mix dialyzer                    # Type checking

# Database
mix ecto.gen.migration name     # Generate migration (always use this, not manual)
mix ecto.migrate                # Run migrations
mix ecto.reset                  # Drop, create, migrate, seed
```

## Architecture

### Domain Contexts (lib/viva/)

- **Accounts** - User management and authentication
- **Avatars** - Avatar entities, personality (Big Five), internal state (emotions/needs), memories
- **Relationships** - Relationships between avatars with trust/affection/familiarity scores
- **Conversations** - Messages and conversation history
- **Matchmaker** - Compatibility scoring engine (GenServer with caching)
- **Sessions** - Avatar runtime processes (LifeProcess GenServer for each avatar)
- **Nim** - NVIDIA NIM LLM client using Req HTTP library
- **World** - Simulation clock with configurable time scaling

### Avatar Life Simulation

Each avatar runs as a `Viva.Sessions.LifeProcess` GenServer:
- Registered via `Viva.Sessions.AvatarRegistry`
- Managed by `Viva.Sessions.AvatarSupervisor` (DynamicSupervisor)
- Ticks every 60 seconds with 10x time scaling (1 real min = 10 simulated min)
- Decays needs (energy, social, stimulation, comfort) based on personality
- Processes emotions, develops desires, may act autonomously
- Persists state to DB every 5 minutes

### Key GenServers

- `Viva.Sessions.LifeProcess` - Individual avatar simulation
- `Viva.Matchmaker.Engine` - Caches compatibility scores, refreshes hourly
- `Viva.World.Clock` - Manages simulation time

### Infrastructure

- **TimescaleDB** (port 5432) - PostgreSQL with time-series extensions
- **Redis** (port 6379) - Cache and pub/sub
- **Qdrant** (port 6333) - Vector database for semantic memory
- **NVIDIA NIM Cloud** - LLM via `integrate.api.nvidia.com/v1`

## Project Guidelines

- Use `mix precommit` before finalizing changes
- Use `Req` for HTTP requests (already included), not HTTPoison/Tesla
- Oban handles background jobs with queues: default, avatar_simulation, matchmaking, memory_processing, conversations

### Phoenix 1.8 Specifics

- LiveView templates must start with `<Layouts.app flash={@flash} ...>`
- Use `<.icon name="hero-x-mark">` for icons (not Heroicons modules)
- Use `<.input>` component for form inputs
- `<.flash_group>` only in `layouts.ex`
- Tailwind v4: no config file, use `@import "tailwindcss"` syntax in app.css

### Elixir/Ecto Guidelines

- Lists don't support index access - use `Enum.at/2`
- Rebind `if/case/cond` results: `socket = if connected?(socket), do: ...`
- Never nest multiple modules in one file
- Access struct fields directly (`my_struct.field`), not with `[:field]`
- Use `Ecto.Changeset.get_field/2` for changeset field access
- Fields set programmatically (like `user_id`) must not be in `cast` calls

### Testing

- Use `start_supervised!/1` for process cleanup
- Avoid `Process.sleep/1` - use `Process.monitor/1` with `assert_receive {:DOWN, ...}`
- Use `:sys.get_state/1` to synchronize before assertions

### LiveView

- Use streams for collections: `stream(socket, :items, items)`
- Template: `<div id="items" phx-update="stream">` with `@streams.items`
- Streams are not enumerable - refetch and reset when filtering
- Use `push_navigate/push_patch` not deprecated `live_redirect/live_patch`
- Colocated JS hooks must start with `.` prefix (e.g., `.PhoneNumber`)
