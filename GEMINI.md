# GEMINI.md

This file provides context and guidance for Gemini CLI and Google Antigravity when working with the VIVA project.

## Project Overview

**VIVA (Virtual Intelligent Vida Autonoma)** is an AI platform where digital avatars live autonomous lives 24/7.

- **Core Concept:** Avatars have unique personalities (Big Five + Enneagram), emotions, and memories. They form relationships and interact even when their owners are offline.
- **Language:** Brazilian Portuguese (pt-BR) is the primary language for avatar interactions and UI.

## Tech Stack

| Layer | Technology |
|-------|------------|
| **Backend** | Elixir 1.15+ / Phoenix 1.8.2 |
| **Database** | TimescaleDB (PostgreSQL 17 + time-series) |
| **Vector DB** | Qdrant (semantic memory) + pgvector |
| **Caching** | Redis + Cachex |
| **AI/LLM** | NVIDIA NIM Cloud (14 models) |
| **Background Jobs** | Oban 2.18 |
| **Real-time** | Phoenix LiveView 1.1 + Channels |
| **Frontend** | LiveView + Tailwind CSS v4 |
| **HTTP Client** | Req |

## Architecture

### Life Simulation Engine

Each avatar runs as a `Viva.Sessions.LifeProcess` GenServer:

```
LifeProcess (GenServer per avatar)
├── Tick every 60s (10x time scale)
├── Decay needs (energy, social, stimulation, comfort)
├── Process emotions → calculate mood
├── Generate thoughts via LLM
├── Take autonomous actions
└── Persist state every 5 minutes
```

### Domain Contexts (`lib/viva/`)

| Context | Purpose | Key Files |
|---------|---------|-----------|
| **Avatars** | Personality, emotions, memories | `avatar.ex`, `personality.ex`, `internal_state.ex` |
| **Sessions** | Life processes, registry | `life_process.ex`, `supervisor.ex` |
| **Relationships** | Trust, affection, status | `relationship.ex`, `feelings.ex` |
| **Matchmaker** | Compatibility scoring (cached) | `engine.ex` |
| **Conversations** | Messages between avatars | `conversation.ex`, `message.ex` |
| **Nim** | NVIDIA NIM client with resilience | `nim.ex`, `llm_client.ex` |
| **World** | Simulation clock (1 min = 10 sim min) | `clock.ex` |

### Supervision Tree

```
Viva.Application
├── Viva.Repo (Ecto)
├── Oban (Background Jobs)
├── Cachex (In-Memory Cache)
├── Phoenix.PubSub
├── Viva.Nim.CircuitBreaker
├── Viva.Nim.RateLimiter
├── Viva.Sessions.Supervisor
│   ├── Registry (AvatarRegistry)
│   ├── Task.Supervisor
│   ├── DynamicSupervisor (AvatarSupervisor)
│   ├── Viva.World.Clock
│   └── Viva.Matching.Engine
└── VivaWeb.Endpoint
```

## Development Commands

```bash
# Setup
mix deps.get                    # Install dependencies
docker compose up -d            # Start infra (TimescaleDB, Redis, Qdrant)
mix ecto.setup                  # Create DB, migrate, seed

# Development
iex -S mix phx.server           # Start server with IEx (recommended)
mix phx.server                  # Start server

# Testing
mix test                        # Run all tests
mix test test/path/file.exs     # Run specific file
mix test --failed               # Re-run failed tests

# Code Quality
mix precommit                   # ALWAYS run before commits
mix format                      # Format code
mix credo --strict              # Static analysis
mix dialyzer                    # Type checking

# Database
mix ecto.gen.migration name     # Generate migration
mix ecto.migrate                # Run migrations
mix ecto.reset                  # Drop, create, migrate, seed
```

## Coding Conventions

### Elixir Patterns

```elixir
# List access - use Enum.at/2, never list[index]
Enum.at(list, 0)

# Struct access - direct field access, never struct[:field]
avatar.personality.openness

# Changeset field access
Ecto.Changeset.get_field(changeset, :name)

# Rebind control structure results
socket = if connected?(socket), do: assign(socket, :user, user), else: socket

# HTTP client - always use Req
Req.post(url, json: body, headers: headers)
```

### Phoenix 1.8 & LiveView 1.1

```elixir
# Layouts - templates must start with
<Layouts.app flash={@flash}>
  ...
</Layouts.app>

# Icons - use core component
<.icon name="hero-heart" class="w-5 h-5" />

# Form inputs - use core component
<.input field={@form[:name]} type="text" label="Name" />

# Streams for collections
socket = stream(socket, :avatars, avatars)
# In template:
<div id="avatars" phx-update="stream">
  <div :for={{dom_id, avatar} <- @streams.avatars} id={dom_id}>
    ...
  </div>
</div>

# Navigation
push_navigate(socket, to: ~p"/avatars")
push_patch(socket, to: ~p"/avatars/#{id}")

# Colocated JS hooks must start with "."
<div phx-hook=".AvatarMood">
```

### Tailwind v4

- No `tailwind.config.js` file
- Use `@import "tailwindcss"` in `app.css`
- CSS-first configuration

### Testing

```elixir
# Process cleanup
start_supervised!(MyGenServer)

# Async assertions - avoid Process.sleep/1
ref = Process.monitor(pid)
assert_receive {:DOWN, ^ref, :process, ^pid, _}, 1000

# Synchronize before assertions
:sys.get_state(pid)
```

## Database Schema

### Key Tables

| Table | Purpose |
|-------|---------|
| `users` | Account management |
| `avatars` | Entities with JSONB personality/internal_state |
| `relationships` | Bidirectional social graph |
| `memories` | Vector embeddings (1024-dim HNSW) |
| `conversations` | Interaction sessions |
| `messages` | Conversation history |

### Important Patterns

- All primary keys are binary UUIDs
- `ON DELETE CASCADE` for referential integrity
- 16 composite indexes for query optimization
- pgvector HNSW index for semantic search

## NIM Client Patterns

The `Viva.Nim` module provides resilient HTTP access:

```elixir
# Direct request with retry/circuit breaker
Nim.request("/chat/completions", body)

# Streaming
Nim.stream_request("/chat/completions", body, fn chunk ->
  IO.write(chunk)
end)

# LLM Client convenience functions
Nim.LlmClient.generate(prompt, max_tokens: 500)
Nim.LlmClient.chat(messages, system: "You are helpful")
```

### Resilience Features

- **Circuit Breaker**: Opens after consecutive failures
- **Rate Limiter**: Token bucket (60 req/min default)
- **Retry with Backoff**: Exponential backoff for transient errors

## Common Tasks

### Add a New Avatar Capability

1. Add function to `Viva.Avatars` context
2. Update `LifeProcess` if autonomous behavior needed
3. Add Oban job if async processing required
4. Update `Nim.LlmClient` if new LLM interaction

### Add a New Relationship Feature

1. Add field to `Viva.Relationships.Relationship` schema
2. Create migration: `mix ecto.gen.migration add_feature_to_relationships`
3. Update `Viva.Relationships` context functions
4. Update `Matching.Engine` if affects compatibility

### Create a New LiveView

1. Create `lib/viva_web/live/feature_live.ex`
2. Create template in same directory or use `~H` sigil
3. Add route in `lib/viva_web/router.ex`
4. Use streams for collections, PubSub for real-time

## Environment Variables

Required in `.env`:

```bash
# NVIDIA NIM (Required)
NIM_API_KEY=nvapi-xxx
NIM_BASE_URL=https://integrate.api.nvidia.com/v1

# Database
DATABASE_URL=ecto://postgres:postgres@localhost:5432/viva_dev

# Redis & Qdrant
REDIS_URL=redis://localhost:6379
QDRANT_URL=http://localhost:6333
```

## Known Issues & Workarounds

1. **Oban migrations**: Run `mix ecto.migrate` if `oban_peers` table missing
2. **Test coverage**: Currently at 4.42% - focus on `LifeProcess` and `Nim` clients
3. **Streams not enumerable**: Always refetch and use `stream(socket, :items, items, reset: true)` when filtering

## File Structure Reference

```
lib/
├── viva/
│   ├── accounts/          # User management
│   ├── avatars/           # Avatar domain
│   │   ├── avatar.ex
│   │   ├── personality.ex
│   │   ├── internal_state.ex
│   │   ├── enneagram.ex
│   │   └── memory.ex
│   ├── conversations/     # Messaging
│   ├── matchmaker/        # Compatibility engine
│   ├── nim/               # NVIDIA NIM clients
│   │   ├── llm_client.ex
│   │   ├── circuit_breaker.ex
│   │   └── rate_limiter.ex
│   ├── relationships/     # Social graph
│   ├── sessions/          # Life processes
│   │   ├── life_process.ex
│   │   └── supervisor.ex
│   └── world/             # Simulation clock
├── viva_web/
│   ├── channels/
│   ├── components/
│   ├── controllers/
│   └── live/
└── viva.ex
```

## Tips for Gemini/Antigravity

1. **Always run `mix precommit`** before suggesting commits
2. **Read files before editing** - understand context first
3. **Use Elixir conventions** - pattern matching, pipelines, with statements
4. **Check existing patterns** - follow established code style in the project
5. **Test GenServers properly** - use `start_supervised!/1` and avoid `Process.sleep/1`
6. **Portuguese for avatar content** - all avatar-facing text should be in pt-BR
