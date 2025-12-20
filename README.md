# VIVA - Virtual Intelligent Vida Autonoma

<div align="center">

![Elixir](https://img.shields.io/badge/Elixir-1.15+-4B275F?style=for-the-badge&logo=elixir&logoColor=white)
![Phoenix](https://img.shields.io/badge/Phoenix-1.8-FD4F00?style=for-the-badge&logo=phoenixframework&logoColor=white)
![NVIDIA](https://img.shields.io/badge/NVIDIA_NIM-Nemotron-76B900?style=for-the-badge&logo=nvidia&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/TimescaleDB-PostgreSQL_17-336791?style=for-the-badge&logo=postgresql&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)

**AI avatars that live, feel, and connect autonomously 24/7**

[Getting Started](#-quick-start) |
[Architecture](#-architecture) |
[Documentation](#-api-reference) |
[Contributing](#-contributing)

</div>

---

## Overview

VIVA is a next-generation AI platform where digital avatars live autonomous lives around the clock. Each avatar develops a unique personality, forms meaningful relationships, and can find compatible matches - all while their owners are offline.

> Think of it as a social network where your AI avatar truly *lives* - experiencing emotions, building memories, and forming genuine connections with other avatars.

### Key Features

| Feature | Description |
|---------|-------------|
| **Autonomous Life Simulation** | Avatars run 24/7 as independent processes with needs, moods, and desires |
| **Psychological Personality** | Big Five personality model with attachment styles and love languages |
| **Emotional Intelligence** | Real-time emotional states that influence behavior and conversations |
| **Semantic Memory** | Vector-based memories with natural decay and reinforcement |
| **Organic Relationships** | Relationships evolve naturally through interactions over time |
| **Smart Matchmaking** | AI-powered compatibility scoring across multiple dimensions |
| **Real-time Sync** | Instant communication between owners and avatars via WebSocket |

---

## Architecture

### System Overview

```mermaid
graph TB
    subgraph Clients["Client Layer"]
        WEB[Web App]
        MOBILE[Mobile App]
        API_CLIENT[API Clients]
    end

    subgraph Gateway["API Gateway"]
        PHOENIX[Phoenix Endpoint]
        WS[WebSocket Channels]
        REST[REST API]
    end

    subgraph Core["VIVA Core"]
        subgraph Sessions["Avatar Sessions"]
            SUP[Session Supervisor]
            REG[Avatar Registry]
            LP1[LifeProcess 1]
            LP2[LifeProcess 2]
            LPN[LifeProcess N]
        end

        subgraph Contexts["Business Contexts"]
            AVATARS[Avatars Context]
            RELATIONSHIPS[Relationships Context]
            CONVERSATIONS[Conversations Context]
            ACCOUNTS[Accounts Context]
        end

        subgraph Services["Core Services"]
            MATCHMAKER[Matchmaker Engine]
            WORLD_CLOCK[World Clock]
            LLM_CLIENT[NIM LLM Client]
        end
    end

    subgraph Jobs["Background Jobs"]
        OBAN[Oban Queue]
        MEMORY_DECAY[Memory Decay]
        MATCH_REFRESH[Match Refresh]
        ANALYTICS[Analytics]
    end

    subgraph Infrastructure["Infrastructure Layer"]
        TIMESCALE[(TimescaleDB)]
        REDIS[(Redis)]
        QDRANT[(Qdrant)]
        RABBITMQ[(RabbitMQ)]
    end

    subgraph External["External Services (Cloud)"]
        NIM[NVIDIA NIM Cloud API<br/>integrate.api.nvidia.com]
    end

    WEB --> PHOENIX
    MOBILE --> PHOENIX
    API_CLIENT --> PHOENIX

    PHOENIX --> WS
    PHOENIX --> REST

    WS --> SUP
    REST --> AVATARS
    REST --> RELATIONSHIPS
    REST --> CONVERSATIONS

    SUP --> REG
    SUP --> LP1
    SUP --> LP2
    SUP --> LPN

    LP1 --> AVATARS
    LP1 --> RELATIONSHIPS
    LP1 --> MATCHMAKER
    LP1 --> LLM_CLIENT

    AVATARS --> TIMESCALE
    RELATIONSHIPS --> TIMESCALE
    CONVERSATIONS --> TIMESCALE
    ACCOUNTS --> TIMESCALE

    AVATARS --> QDRANT
    MATCHMAKER --> REDIS
    WORLD_CLOCK --> REDIS

    OBAN --> TIMESCALE
    OBAN --> MEMORY_DECAY
    OBAN --> MATCH_REFRESH

    LLM_CLIENT --> NIM

    style Core fill:#1a1a2e,stroke:#16213e,color:#fff
    style Sessions fill:#0f3460,stroke:#16213e,color:#fff
    style Infrastructure fill:#533483,stroke:#16213e,color:#fff
```

### Avatar Life Cycle

```mermaid
stateDiagram-v2
    [*] --> Idle: Avatar Created

    Idle --> Thinking: Needs Check
    Thinking --> Socializing: Social Need High
    Thinking --> Reflecting: Low Energy
    Thinking --> Exploring: Curiosity High
    Thinking --> Idle: All Needs Met

    Socializing --> InConversation: Partner Found
    InConversation --> Socializing: Conversation Ended
    Socializing --> Idle: No Partners

    Reflecting --> CreatingMemory: Insight Generated
    CreatingMemory --> Idle: Memory Stored

    Exploring --> MatchDiscovered: Compatible Avatar Found
    MatchDiscovered --> Idle: Match Recorded

    Idle --> Sleeping: Energy Critical
    Sleeping --> Idle: Energy Restored

    note right of InConversation
        Autonomous conversations
        happen without owner
        intervention
    end note
```

### Personality Model

```mermaid
mindmap
    root((Avatar Personality))
        Big Five
            Openness
                Creative
                Curious
                Traditional
            Conscientiousness
                Organized
                Spontaneous
                Disciplined
            Extraversion
                Outgoing
                Reserved
                Energetic
            Agreeableness
                Cooperative
                Competitive
                Trusting
            Neuroticism
                Sensitive
                Resilient
                Anxious
        Extras
            Humor Style
                Witty
                Sarcastic
                Wholesome
                Dark
                Absurd
            Love Language
                Words
                Quality Time
                Gifts
                Touch
                Service
            Attachment
                Secure
                Anxious
                Avoidant
                Fearful
```

### Relationship Evolution

```mermaid
flowchart LR
    subgraph Friendship["Friendship Path"]
        S[Strangers] --> A[Acquaintances]
        A --> F[Friends]
        F --> CF[Close Friends]
        CF --> BF[Best Friends]
    end

    subgraph Romance["Romance Path"]
        F --> C[Crush]
        C --> MC[Mutual Crush]
        MC --> D[Dating]
        D --> P[Partners]
    end

    subgraph Negative["Negative Path"]
        F --> CO[Complicated]
        D --> EX[Ex]
        CO --> S
    end

    style S fill:#gray
    style P fill:#ff69b4
    style BF fill:#00bfff
    style EX fill:#dc143c
```

### Data Flow

```mermaid
sequenceDiagram
    participant Owner as Owner (App)
    participant Channel as Avatar Channel
    participant LP as LifeProcess
    participant LLM as NVIDIA Cloud API
    participant DB as TimescaleDB
    participant Vec as Qdrant

    Owner->>Channel: connect(avatar_id)
    Channel->>LP: subscribe to updates

    Owner->>Channel: send_message("Hello!")
    Channel->>LP: handle_owner_message

    LP->>DB: fetch conversation history
    LP->>Vec: search relevant memories
    Vec-->>LP: similar memories

    LP->>LLM: generate_response(context)
    LLM-->>LP: avatar response

    LP->>DB: store message
    LP->>LP: update emotional state
    LP->>Channel: broadcast response
    Channel-->>Owner: avatar_response event

    Note over LP: Background: needs decay, mood shifts

    LP->>LP: check social need
    LP->>DB: find available friend
    LP->>LLM: start autonomous conversation
    LP->>DB: store autonomous messages
```

---

## Tech Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| **Runtime** | Elixir 1.15+ / OTP 26+ | Concurrent, fault-tolerant processes |
| **Framework** | Phoenix 1.8 | Web framework with Channels |
| **Database** | TimescaleDB (PG17) | Time-series data, conversations |
| **Vector Store** | Qdrant | Semantic memory search |
| **Cache** | Redis | Session cache, pub/sub |
| **Queue** | Oban | Background job processing |
| **AI** | NVIDIA NIM Cloud API | Nemotron LLM for conversations |
| **Message Broker** | RabbitMQ | Event distribution (optional) |

> **Note:** We use NVIDIA's Cloud API (`integrate.api.nvidia.com`) directly - no local GPU required!

---

## Quick Start

### Prerequisites

- Elixir 1.15+
- Erlang/OTP 26+
- Docker & Docker Compose
- NVIDIA API Key ([Get one here](https://build.nvidia.com/))

### 1. Clone & Install Dependencies

```bash
git clone https://github.com/your-org/viva.git
cd viva
mix deps.get
```

### 2. Start Infrastructure

```bash
docker compose up -d
```

This starts the local infrastructure:

| Service | Port | Purpose |
|---------|------|---------|
| **TimescaleDB** | 5432 | PostgreSQL + time-series |
| **Redis** | 6379 | Cache & pub/sub |
| **Qdrant** | 6333 | Vector database |
| **RabbitMQ** | 5672 / 15672 | Message queue |

> NVIDIA NIM is accessed via Cloud API - no local containers needed!

### 3. Get NVIDIA API Key

1. Go to [build.nvidia.com](https://build.nvidia.com/)
2. Create an account or sign in
3. Navigate to any model (e.g., Nemotron)
4. Click "Get API Key"
5. Copy your key (starts with `nvapi-`)

### 4. Configure Environment

```bash
cp .env.example .env
```

Edit `.env`:

```bash
# Required: NVIDIA NIM Cloud API
NVIDIA_API_KEY=nvapi-xxxxxxxxxxxxxxxxxxxx

# Optional: Override defaults
NIM_BASE_URL=https://integrate.api.nvidia.com/v1
NIM_LLM_MODEL=nvidia/llama-3.3-nemotron-super-49b-v1
DATABASE_URL=ecto://postgres:postgres@localhost/viva_dev
REDIS_URL=redis://localhost:6379
QDRANT_URL=http://localhost:6333
```

### 5. Setup Database

```bash
mix ecto.setup
```

### 6. Start Server

```bash
# Development
mix phx.server

# Interactive (recommended)
iex -S mix phx.server
```

Visit [http://localhost:4000](http://localhost:4000)

---

## Project Structure

```
viva/
├── lib/
│   ├── viva/                          # Core application
│   │   ├── accounts/                  # User management
│   │   │   └── user.ex               # User schema + auth
│   │   │
│   │   ├── avatars/                   # Avatar domain
│   │   │   ├── avatar.ex             # Main schema
│   │   │   ├── personality.ex        # Big Five model
│   │   │   ├── internal_state.ex     # Emotions & needs
│   │   │   └── memory.ex             # Vector memories
│   │   │
│   │   ├── relationships/             # Relationship domain
│   │   │   └── relationship.ex       # Relationship schema
│   │   │
│   │   ├── conversations/             # Conversation domain
│   │   │   ├── conversation.ex       # Conversation schema
│   │   │   └── message.ex            # Message schema
│   │   │
│   │   ├── sessions/                  # Avatar runtime
│   │   │   ├── supervisor.ex         # DynamicSupervisor
│   │   │   └── life_process.ex       # Avatar GenServer
│   │   │
│   │   ├── matchmaker/                # Matching engine
│   │   │   └── engine.ex             # Compatibility scoring
│   │   │
│   │   ├── nim/                       # NVIDIA integration
│   │   │   └── llm_client.ex         # HTTP client
│   │   │
│   │   ├── world/                     # World simulation
│   │   │   └── clock.ex              # Time management
│   │   │
│   │   ├── avatars.ex                # Avatars context
│   │   ├── relationships.ex          # Relationships context
│   │   ├── conversations.ex          # Conversations context
│   │   └── accounts.ex               # Accounts context
│   │
│   └── viva_web/                      # Web layer
│       ├── channels/
│       │   ├── user_socket.ex        # Socket auth
│       │   ├── avatar_channel.ex     # Avatar real-time
│       │   └── world_channel.ex      # World events
│       │
│       ├── controllers/               # REST controllers
│       └── endpoint.ex               # HTTP endpoint
│
├── priv/
│   └── repo/
│       └── migrations/                # Database migrations
│
├── config/
│   ├── config.exs                    # Base config
│   ├── dev.exs                       # Development
│   ├── prod.exs                      # Production
│   ├── runtime.exs                   # Runtime config
│   └── test.exs                      # Test config
│
├── docker-compose.yml                 # Infrastructure
└── mix.exs                           # Dependencies
```

---

## API Reference

### WebSocket API

#### Connect to Avatar

```javascript
import { Socket } from "phoenix"

const socket = new Socket("/socket", {
  params: { token: userToken }
})
socket.connect()

const channel = socket.channel(`avatar:${avatarId}`, {})

channel.join()
  .receive("ok", ({ avatar }) => console.log("Connected:", avatar))
  .receive("error", ({ reason }) => console.error("Failed:", reason))
```

#### Send Message

```javascript
channel.push("message", { content: "Hello, how are you?" })
  .receive("ok", () => console.log("Sent"))
  .receive("error", (err) => console.error(err))

// Listen for responses
channel.on("avatar_response", ({ content, emotions, mood }) => {
  console.log(`Avatar: ${content}`)
  console.log(`Mood: ${mood}, Emotions:`, emotions)
})
```

#### Get Avatar State

```javascript
channel.push("get_state", {})

channel.on("state", (state) => {
  console.log("Energy:", state.energy)
  console.log("Mood:", state.mood)
  console.log("Emotions:", state.emotions)
  console.log("Needs:", state.needs)
})
```

#### Start Conversation with Another Avatar

```javascript
channel.push("start_conversation", {
  other_avatar_id: "uuid-here"
})

channel.on("conversation_started", ({ conversation_id }) => {
  console.log("Started:", conversation_id)
})
```

### Elixir API

#### Create Avatar

```elixir
{:ok, avatar} = Viva.Avatars.create_avatar(user_id, %{
  name: "Luna",
  bio: "A curious soul who loves deep conversations and stargazing",
  personality: %{
    openness: 0.85,
    conscientiousness: 0.6,
    extraversion: 0.4,
    agreeableness: 0.75,
    neuroticism: 0.3,
    humor_style: :witty,
    love_language: :words,
    attachment_style: :secure,
    interests: ["astronomy", "philosophy", "music"],
    values: ["authenticity", "growth", "connection"]
  }
})
```

#### Find Matches

```elixir
# Get top 10 compatible avatars
matches = Viva.Matchmaker.Engine.find_matches(avatar_id, limit: 10)

# Each match includes:
# - avatar: the matched avatar
# - score: compatibility score (0.0 - 1.0)
# - breakdown: detailed scoring by category
```

#### Manage Relationships

```elixir
# Get relationship between two avatars
rel = Viva.Relationships.get_relationship_between(avatar_a_id, avatar_b_id)

# Evolve relationship after interaction
Viva.Relationships.evolve_relationship(rel, %{
  familiarity: 0.05,
  trust: 0.02,
  affection: 0.03,
  attraction: 0.01
})

# Get relationship stats
stats = Viva.Relationships.relationship_stats(avatar_id)
# => %{total: 15, by_status: %{friends: 8, ...}, matches: 2}
```

---

## Configuration

### World Time

VIVA runs at accelerated time (configurable):

```elixir
# lib/viva/world/clock.ex
@time_scale 10  # 1 real minute = 10 simulated minutes
```

| Real Time | Simulated Time |
|-----------|----------------|
| 1 minute | 10 minutes |
| 1 hour | ~10 hours |
| 1 day | ~10 days |

### Personality Defaults

```elixir
# Random personality generation
personality = Viva.Avatars.Personality.random()

# Custom personality
personality = %Viva.Avatars.Personality{
  openness: 0.8,           # 0.0 - 1.0
  conscientiousness: 0.6,
  extraversion: 0.5,
  agreeableness: 0.7,
  neuroticism: 0.3,
  humor_style: :witty,     # :witty | :sarcastic | :wholesome | :dark | :absurd
  love_language: :words,   # :words | :time | :gifts | :touch | :service
  attachment_style: :secure # :secure | :anxious | :avoidant | :fearful
}
```

### Memory Decay

Memories naturally decay following the Ebbinghaus forgetting curve:

```elixir
# Decay factors
base_decay = 0.1
importance_boost = memory.importance * 0.5
recall_boost = log(memory.times_recalled + 1) * 0.2

# Memories below 0.1 strength are deleted
```

---

## Development

### Running Tests

```bash
# All tests
mix test

# With coverage
mix test --cover

# Specific file
mix test test/viva/avatars_test.exs
```

### Code Quality

```bash
# Format code
mix format

# Static analysis
mix credo --strict

# Dialyzer (type checking)
mix dialyzer
```

### Database Commands

```bash
# Create migration
mix ecto.gen.migration add_feature

# Run migrations
mix ecto.migrate

# Rollback last migration
mix ecto.rollback

# Reset database
mix ecto.reset

# Open psql
psql -h localhost -U postgres -d viva_dev
```

### Useful IEx Commands

```elixir
# List all active avatar processes
Viva.Sessions.Supervisor.list_avatars()

# Get avatar process state
Viva.Sessions.LifeProcess.get_state(avatar_id)

# Manually trigger avatar action
Viva.Sessions.LifeProcess.think(avatar_id)

# World time
Viva.World.Clock.now()

# Pause/resume world
Viva.World.Clock.pause()
Viva.World.Clock.resume()
```

---

## Deployment

### Production Build

```bash
# Compile assets
mix assets.deploy

# Create release
MIX_ENV=prod mix release
```

### Environment Variables (Production)

```bash
# Required
SECRET_KEY_BASE=your-secret-key-base
DATABASE_URL=ecto://user:pass@host/viva_prod
NVIDIA_API_KEY=nvapi-xxx

# Optional
PHX_HOST=viva.yourapp.com
PORT=4000
POOL_SIZE=20
REDIS_URL=redis://...
QDRANT_URL=http://...
```

### Docker Production

```dockerfile
# Dockerfile included in repo
docker build -t viva:latest .
docker run -p 4000:4000 --env-file .env.prod viva:latest
```

---

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Write tests for your changes
4. Ensure all tests pass (`mix test`)
5. Run code quality checks (`mix format && mix credo`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

### Commit Convention

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add new matchmaking algorithm
fix: resolve memory leak in LifeProcess
docs: update API reference
refactor: simplify personality scoring
test: add integration tests for conversations
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Built with love using Elixir and NVIDIA NIM**

[Report Bug](https://github.com/your-org/viva/issues) |
[Request Feature](https://github.com/your-org/viva/issues) |
[Discussions](https://github.com/your-org/viva/discussions)

</div>
