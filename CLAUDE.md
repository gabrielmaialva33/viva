# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

VIVA (Virtual Intelligent Vida Autonoma) is an AI avatar platform where digital avatars live autonomous lives 24/7. Avatars have personalities (Big Five model), emotions, memories, and form relationships with other avatars. Built with Elixir/Phoenix 1.8, NVIDIA NIM for LLM, and TimescaleDB.

## Commands

```bash
# Development
mix deps.get                    # Install dependencies
docker compose up -d            # Start infrastructure (TimescaleDB, Redis, RabbitMQ, Qdrant)
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
- **Avatars** - Avatar entities, personality (Big Five + Enneagram), internal state, memories
  - `systems/` - Simulation modules: Biology, Psychology, Neurochemistry, Consciousness, Allostasis, EmotionRegulation, Metacognition, Motivation, Senses, SomaticMarkers, AttachmentBias
- **Relationships** - Relationships between avatars with trust/affection/familiarity scores
- **Conversations** - Messages and conversation history
- **Matching** - Compatibility scoring engine (GenServer with caching)
- **Sessions** - Avatar runtime processes
  - `LifeProcess` - Main GenServer for each avatar
  - `DesireEngine`, `ThoughtEngine`, `DreamProcessor` - Cognitive subsystems
- **AI** - NVIDIA NIM integration via Req
  - `LLM/` - 14 specialized clients (LlmClient, ReasoningClient, EmbeddingClient, SafetyClient, TTSClient, ASRClient, VLMClient, ImageClient, Avatar3DClient, TranslateClient, AudioEnhanceClient)
  - `Pipeline` - Broadway/RabbitMQ pipeline for async LLM processing
- **Social** - Social network analysis (NetworkAnalyst GenServer)
- **World** - Simulation clock with configurable time scaling
- **Infrastructure** - Redis client, EventBus, Postgrex types

### Avatar Life Simulation

Each avatar runs as a `Viva.Sessions.LifeProcess` GenServer:
- Registered via `Viva.Sessions.AvatarRegistry`
- Managed by `Viva.Sessions.AvatarSupervisor` (DynamicSupervisor)
- Ticks every 60 seconds with 10x time scaling (1 real min = 10 simulated min)
- Decays needs (energy, social, stimulation, comfort) based on personality
- Processes emotions, develops desires, may act autonomously
- Persists state to DB every 5 minutes

### Key GenServers

- `Viva.Sessions.LifeProcess` - Individual avatar simulation (registered via `Viva.Sessions.AvatarRegistry`)
- `Viva.Matching.Engine` - Caches compatibility scores, refreshes hourly
- `Viva.World.Clock` - Manages simulation time
- `Viva.Social.NetworkAnalyst` - Analyzes social graph and relationships
- `Viva.AI.LLM.CircuitBreaker` - NIM API resilience
- `Viva.AI.LLM.RateLimiter` - NIM API rate limiting

### Infrastructure

- **TimescaleDB** (port 5432) - PostgreSQL with time-series extensions + pgvector
- **Redis** (port 6379) - Cache and pub/sub
- **RabbitMQ** (port 5672) - AI event pipeline for async LLM processing
- **Qdrant** (port 6333) - Vector database for semantic memory
- **NVIDIA NIM Cloud** - 14 models via `integrate.api.nvidia.com/v1`

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

## AI Team Configuration (autogenerated by team-configurator, 2025-12-20)

**Important: YOU MUST USE subagents when available for the task.**

### Detected Stack

- **Backend Framework:** Elixir 1.15+ with Phoenix 1.8.2
- **Database:** TimescaleDB (PostgreSQL 17 + time-series extensions) with pgvector
- **ORM:** Ecto 3.13 with Postgrex
- **Real-time:** Phoenix LiveView 1.1.0 + Phoenix Channels (WebSocket)
- **Frontend:** Phoenix LiveView with Tailwind CSS v4
- **Background Jobs:** Oban 2.18
- **Caching:** Cachex 3.6 + Redis
- **Vector Database:** Qdrant 0.0.8
- **HTTP Client:** Req 0.5 (for NVIDIA NIM API)
- **LLM:** NVIDIA NIM Cloud API (Nemotron, integrate.api.nvidia.com)
- **Testing:** ExUnit with ExCoveralls
- **Code Quality:** Credo 1.7, Dialyxir 1.4
- **Authentication:** bcrypt_elixir 3.1
- **Web Server:** Bandit 1.5
- **Asset Bundling:** esbuild 0.10

### Specialist Assignments

| Task | Agent | Notes |
|------|-------|-------|
| **Code Review** | `@code-reviewer` | REQUIRED before all PRs, merges, feature completion. Security-aware, severity-tagged reports. |
| **Performance Optimization** | `@performance-optimizer` | Use for GenServer tuning, database query optimization, caching strategy, Oban job performance. |
| **Backend Development** | `@backend-developer` | General Elixir/Phoenix development, context creation, schema design, business logic. |
| **API Design** | `@api-architect` | REST endpoints, Phoenix Channel contracts, WebSocket event schemas, JSON API design. |
| **LiveView/Frontend** | `@frontend-developer` | LiveView components, real-time UI, Phoenix templates, client-side hooks. |
| **Tailwind Styling** | `@tailwind-frontend-expert` | All CSS/styling work, responsive design, component styling with Tailwind v4. |
| **Documentation** | `@documentation-specialist` | README updates, API documentation, architecture guides, onboarding docs. |
| **Codebase Analysis** | `@code-archaeologist` | Explore unfamiliar code, legacy analysis, architecture mapping, risk assessment. |
| **Project Planning** | `@tech-lead-orchestrator` | Multi-step features, architectural decisions, task breakdown, agent coordination. |
| **Stack Detection** | `@project-analyst` | Analyze new dependencies, detect framework changes, identify tech stack updates. |

### VIVA-Specific Context

This platform has unique architectural patterns:

1. **Avatar Life Processes:** Each avatar is a long-running GenServer (Viva.Sessions.LifeProcess) with autonomous behavior. Performance optimization and review tasks should understand OTP supervision trees.

2. **Real-time State:** Heavy use of Phoenix Channels for owner-avatar communication. Frontend work often involves WebSocket event handling.

3. **AI Integration:** NVIDIA NIM clients in `Viva.AI.LLM` use Req for HTTP. Any LLM-related work should consult existing client patterns (e.g., `LlmClient`, `ReasoningClient`).

4. **Vector Memories:** Qdrant integration for semantic memory search. Database work may involve both TimescaleDB and vector operations.

5. **Matchmaking Engine:** Cached GenServer with complex personality compatibility scoring. Performance work should review cache invalidation strategy.

6. **Background Jobs:** Oban handles memory decay, match refresh, analytics. Job performance affects avatar simulation quality.

### Usage Examples

```bash
# Before merging any feature
@code-reviewer Please review the new relationship evolution logic in lib/viva/relationships.ex

# When avatars respond slowly
@performance-optimizer The LifeProcess GenServers are experiencing lag during peak hours

# Adding new avatar capabilities
@backend-developer Add a "gifting" system where avatars can send virtual gifts to friends

# Designing new endpoints
@api-architect Design a REST API for avatar discovery and browsing

# Building LiveView features
@frontend-developer Create a real-time dashboard showing all active avatar conversations

# Styling components
@tailwind-frontend-expert Style the avatar profile card with mood indicators and personality traits

# Writing guides
@documentation-specialist Document the matchmaking algorithm and compatibility scoring system

# Understanding complex code
@code-archaeologist Explain how the World.Clock GenServer manages time scaling across all avatar processes

# Planning large features
@tech-lead-orchestrator Plan implementation of group conversations between multiple avatars
```

### Notes

- All agents have access to the full CLAUDE.md context and understand Elixir/Phoenix patterns.
- Always run `@code-reviewer` before creating pull requests.
- For unfamiliar Elixir/OTP patterns, consult `@code-archaeologist` first.
- Complex features spanning multiple domains should start with `@tech-lead-orchestrator`.
- This configuration was generated by analyzing mix.exs, docker-compose.yml, and project structure.
