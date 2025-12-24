# VIVA - Virtual Intelligent Vida Autonoma

<div align="center">

![Elixir](https://img.shields.io/badge/Elixir-1.15+-4B275F?style=for-the-badge&logo=elixir&logoColor=white)
![Phoenix](https://img.shields.io/badge/Phoenix-1.8-FD4F00?style=for-the-badge&logo=phoenixframework&logoColor=white)
![NVIDIA](https://img.shields.io/badge/NVIDIA_NIM-14_Models-76B900?style=for-the-badge&logo=nvidia&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/TimescaleDB-PostgreSQL_17-336791?style=for-the-badge&logo=postgresql&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)

**AI avatars that live, feel, and connect autonomously 24/7**

[Getting Started](#-quick-start) |
[Architecture](#-architecture) |
[AI Models](#-nvidia-nim-models) |
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
| **Deep Psychological Model** | Big Five + Enneagram personality system with 9 core types |
| **Emotional Intelligence** | Real-time emotional states that influence behavior and conversations |
| **Semantic Memory** | Vector-based memories with natural decay and reinforcement |
| **Organic Relationships** | Relationships evolve naturally through interactions over time |
| **Smart Matchmaking** | AI-powered compatibility scoring across multiple dimensions |
| **Real-time Sync** | Instant communication between owners and avatars via WebSocket |
| **Multilingual Support** | Native pt-BR with 36 language translation via NIM |
| **AI-Generated Visuals** | Dynamic profile images and 3D avatars |
| **Voice Interaction** | Premium TTS and ASR with audio enhancement |

---

## NVIDIA NIM Models

VIVA uses **14 cutting-edge NVIDIA NIM models** for maximum quality:

### Core AI Stack

| Category | Model | Purpose |
|----------|-------|---------|
| **LLM** | `llama-3.1-nemotron-ultra-253b-v1` | Primary conversation and reasoning |
| **Reasoning** | `deepseek-r1-0528` | Complex autonomous decisions |
| **Embeddings** | `nv-embedqa-mistral-7b-v2` | Semantic memory search |
| **Safety** | `llama-3.1-nemotron-safety-guard-8b-v3` | Content moderation |

### Voice & Audio

| Category | Model | Purpose |
|----------|-------|---------|
| **TTS** | `magpie-tts-multilingual` | Avatar voice generation |
| **ASR** | `parakeet-1.1b-rnnt-multilingual-asr` | Speech recognition |
| **Audio Enhance** | `studiovoice` | Studio-quality audio |
| **Noise Removal** | `Background Noise Removal` | Clean audio input |

### Visual Generation

| Category | Model | Purpose |
|----------|-------|---------|
| **Image Gen** | `stable-diffusion-3.5-large` | Profile picture generation |
| **Image Edit** | `FLUX.1-Kontext-dev` | Expression variations |
| **3D Avatar** | `TRELLIS` | 3D model generation |
| **Lipsync** | `audio2face-3d` | Facial animation |

### Specialized

| Category | Model | Purpose |
|----------|-------|---------|
| **VLM** | `cosmos-nemotron-34b` | Vision understanding |
| **Translation** | `riva-translate-1.6b` | 36 language translation |

> All models accessed via NVIDIA Cloud API - no local GPU required!

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
            NIM_CLIENTS[NIM Clients x14]
        end
    end

    subgraph Jobs["Background Jobs"]
        OBAN[Oban Queue]
        MEMORY_DECAY[Memory Decay]
        MATCH_REFRESH[Match Refresh]
        VISUAL_GEN[Visual Generation]
    end

    subgraph Infrastructure["Infrastructure Layer"]
        TIMESCALE[(TimescaleDB)]
        REDIS[(Redis)]
        QDRANT[(Qdrant)]
    end

    subgraph External["NVIDIA NIM Cloud"]
        NIM_LLM[LLM + Reasoning]
        NIM_VOICE[Voice + Audio]
        NIM_VISUAL[Image + 3D]
        NIM_LANG[Translation]
    end

    WEB --> PHOENIX
    MOBILE --> PHOENIX
    API_CLIENT --> PHOENIX

    PHOENIX --> WS
    PHOENIX --> REST

    WS --> SUP
    REST --> AVATARS

    SUP --> REG
    SUP --> LP1
    SUP --> LP2
    SUP --> LPN

    LP1 --> AVATARS
    LP1 --> MATCHMAKER
    LP1 --> NIM_CLIENTS

    AVATARS --> TIMESCALE
    AVATARS --> QDRANT
    MATCHMAKER --> REDIS

    NIM_CLIENTS --> NIM_LLM
    NIM_CLIENTS --> NIM_VOICE
    NIM_CLIENTS --> NIM_VISUAL
    NIM_CLIENTS --> NIM_LANG

    style Core fill:#1a1a2e,stroke:#16213e,color:#fff
    style Sessions fill:#0f3460,stroke:#16213e,color:#fff
    style External fill:#76B900,stroke:#16213e,color:#fff
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

VIVA uses a comprehensive psychological model combining **Big Five** traits with the **Enneagram** system:

```mermaid
mindmap
    root((Avatar Personality))
        Big Five
            Openness
            Conscientiousness
            Extraversion
            Agreeableness
            Neuroticism
        Enneagram
            Type 1 - Reformer
            Type 2 - Helper
            Type 3 - Achiever
            Type 4 - Individualist
            Type 5 - Investigator
            Type 6 - Loyalist
            Type 7 - Enthusiast
            Type 8 - Challenger
            Type 9 - Peacemaker
        Temperament
            Sanguine
            Choleric
            Melancholic
            Phlegmatic
        Style
            Humor Style
            Love Language
            Attachment Style
```

#### Enneagram Integration

Each avatar has a core Enneagram type that influences:

| Aspect | Description |
|--------|-------------|
| **Basic Fear** | What the avatar fears most (unconscious driver) |
| **Basic Desire** | What the avatar seeks most (core motivation) |
| **Vice** | Default negative pattern under stress |
| **Virtue** | Growth direction when healthy |
| **Stress Behavior** | How avatar acts when overwhelmed |
| **Growth Behavior** | How avatar acts when thriving |

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

---

## Tech Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| **Runtime** | Elixir 1.15+ / OTP 26+ | Concurrent, fault-tolerant processes |
| **Framework** | Phoenix 1.8 | Web framework with Channels |
| **Database** | TimescaleDB (PG17) | Time-series data, conversations |
| **Vector Store** | Qdrant | Semantic memory search |
| **Cache** | Redis + Cachex | Session cache, pub/sub |
| **Queue** | Oban + RabbitMQ (Broadway) | Background jobs & AI Pipeline |
| **AI** | NVIDIA NIM Cloud (14 models) | Full AI stack |
| **HTTP Client** | Req | API requests with resilience |

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

| Service | Port | Purpose |
|---------|------|---------|
| **TimescaleDB** | 5432 | PostgreSQL + time-series |
| **Redis** | 6379 | Cache & pub/sub |
| **RabbitMQ** | 5672 | AI Event Pipeline |
| **Qdrant** | 6333 | Vector database |

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
NIM_API_KEY=nvapi-xxxxxxxxxxxxxxxxxxxx

# Optional: Override defaults
NIM_BASE_URL=https://integrate.api.nvidia.com/v1
DATABASE_URL=ecto://postgres:postgres@localhost/viva_dev
REDIS_URL=redis://localhost:6379
QDRANT_URL=http://localhost:6333
```

### 5. Setup Database

```bash
mix ecto.setup
```

This will create 9 Brazilian seed avatars with unique personalities!

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
│   │   ├── ai/                        # Artificial Intelligence Domain
│   │   │   ├── llm/                  # NVIDIA NIM Clients
│   │   │   │   ├── llm_client.ex     # Primary LLM
│   │   │   │   └── ...               # Other clients
│   │   │   ├── pipeline.ex           # Broadway RabbitMQ Pipeline
│   │   │   └── llm.ex                # Main AI entrypoint
│   │   │
│   │   ├── avatars/                   # Avatar Domain Schema
│   │   │   ├── avatar.ex             # Main schema
│   │   │   ├── internal_state.ex     # State schema
│   │   │   │
│   │   │   └── systems/              # Avatar Systems (Logic)
│   │   │       ├── biology.ex        # Biological simulation
│   │   │       ├── neurochemistry.ex # Hormonal system
│   │   │       └── psychology.ex     # Emotional processing
│   │   │
│   │   ├── relationships/             # Relationship domain
│   │   │   └── relationship.ex       # Relationship schema
│   │   │
│   │   ├── conversations/             # Conversation domain
│   │   │
│   │   ├── sessions/                  # Avatar Runtime
│   │   │   ├── supervisor.ex         # DynamicSupervisor
│   │   │   └── life_process.ex       # Avatar GenServer (The Brain)
│   │   │
│   │   ├── matchmaker/                # Matching engine
│   │   │
│   │   └── infrastructure/            # Technical Infra
│   │       └── redis.ex              # Redis wrapper
│   │
│   └── viva_web/                      # Web layer
│       ├── channels/
│       ├── controllers/
│       └── endpoint.ex
│
├── priv/
│   └── repo/
│       ├── migrations/                # Database migrations
│       └── seeds.exs                 # 9 Brazilian avatars
│
├── config/
│   ├── config.exs                    # Base config + NIM models
│   ├── dev.exs                       # Development
│   ├── prod.exs                      # Production
│   ├── runtime.exs                   # Runtime config
│   └── test.exs                      # Test config
│
├── CLAUDE.md                         # AI team configuration
├── docker-compose.yml                # Infrastructure
└── mix.exs                           # Dependencies
```

---

## API Reference

### Visual Generation

```elixir
# Generate complete visual package
Viva.Avatars.generate_visuals(avatar)

# Generate only profile image
Viva.Avatars.generate_profile_image(avatar, style: "realistic")

# Generate 3D avatar with lipsync support
Viva.Avatars.generate_3d_avatar(avatar)

# Update expression based on emotion
Viva.Avatars.update_expression(avatar, :happy)

# Generate lipsync animation from audio
Viva.Avatars.generate_lipsync(avatar, audio_data)
```

### Translation

```elixir
# Translate between avatars
Viva.AI.LLM.TranslateClient.translate_avatar_message(
  message,
  from_avatar,
  to_avatar
)

# Detect language
Viva.AI.LLM.TranslateClient.detect_language("Olá, como vai?")
# => {:ok, %{language: "pt", name: "Portuguese"}}

# Translate conversation history
Viva.AI.LLM.TranslateClient.translate_conversation(messages, "en")
```

### Advanced Reasoning

```elixir
# Deep compatibility analysis
Viva.AI.LLM.ReasoningClient.deep_analyze_compatibility(avatar_a, avatar_b)

# Autonomous decision making
Viva.AI.LLM.ReasoningClient.make_autonomous_decision(avatar, situation, options)

# Relationship conflict resolution
Viva.AI.LLM.ReasoningClient.resolve_relationship_conflict(relationship, context)

# Emotional trajectory prediction
Viva.AI.LLM.ReasoningClient.analyze_emotional_trajectory(avatar, recent_events)
```

### Audio Enhancement

```elixir
# Full audio processing pipeline
Viva.AI.LLM.AudioEnhanceClient.process_full(audio_data)

# Remove background noise
Viva.AI.LLM.AudioEnhanceClient.remove_noise(audio_data, aggressiveness: "high")

# Enhance for transcription
Viva.AI.LLM.AudioEnhanceClient.enhance_for_transcription(audio_data)

# Smart enhance (only if needed)
Viva.AI.LLM.AudioEnhanceClient.smart_enhance(audio_data, threshold: 0.7)
```

### WebSocket API

```javascript
import { Socket } from "phoenix"

const socket = new Socket("/socket", { params: { token: userToken } })
socket.connect()

const channel = socket.channel(`avatar:${avatarId}`, {})

// Send message
channel.push("message", { content: "Olá, como você está?" })

// Listen for responses
channel.on("avatar_response", ({ content, emotions, mood, expression_url }) => {
  console.log(`Avatar: ${content}`)
  console.log(`Expression: ${expression_url}`)
})
```

### Elixir API

```elixir
# Create avatar with Enneagram type
{:ok, avatar} = Viva.Avatars.create_avatar(user_id, %{
  name: "Luna",
  bio: "Uma alma curiosa que ama conversas profundas",
  personality: %{
    openness: 0.85,
    conscientiousness: 0.6,
    extraversion: 0.4,
    agreeableness: 0.75,
    neuroticism: 0.3,
    enneagram_type: 4,  # The Individualist
    humor_style: :witty,
    love_language: :words,
    attachment_style: :secure,
    native_language: "pt-BR",
    other_languages: ["en", "es"],
    interests: ["astronomia", "filosofia", "música"],
    values: ["autenticidade", "crescimento", "conexão"]
  }
})

# Get Enneagram info
enneagram = Viva.Avatars.Enneagram.get_type(4)
# => %{name: "Individualist", basic_fear: "Having no identity...", ...}
```

---

## Configuration

### NIM Models (config/config.exs)

```elixir
config :viva, :nim,
  base_url: "https://integrate.api.nvidia.com/v1",
  models: %{
    # Core
    llm: "nvidia/llama-3.1-nemotron-ultra-253b-v1",
    reasoning: "deepseek-ai/deepseek-r1-0528",
    embedding: "nvidia/nv-embedqa-mistral-7b-v2",
    safety: "nvidia/llama-3.1-nemotron-safety-guard-8b-v3",

    # Voice
    tts: "nvidia/magpie-tts-multilingual",
    asr: "nvidia/parakeet-1.1b-rnnt-multilingual-asr",
    audio_enhance: "nvidia/studiovoice",
    noise_removal: "nvidia/Background Noise Removal",

    # Visual
    image_gen: "stabilityai/stable-diffusion-3.5-large",
    image_edit: "black-forest-labs/FLUX.1-Kontext-dev",
    avatar_3d: "microsoft/TRELLIS",
    audio2face: "nvidia/audio2face-3d",

    # Specialized
    vlm: "nvidia/cosmos-nemotron-34b",
    translate: "nvidia/riva-translate-1.6b"
  }
```

### World Time

```elixir
# 10x time acceleration
@time_scale 10  # 1 real minute = 10 simulated minutes
```

| Real Time | Simulated Time |
|-----------|----------------|
| 1 minute | 10 minutes |
| 1 hour | ~10 hours |
| 1 day | ~10 days |

---

## Development

### Running Tests

```bash
mix test                           # All tests
mix test --cover                   # With coverage
mix test test/viva/avatars_test.exs # Specific file
```

### Code Quality

```bash
mix format                         # Format code
mix credo --strict                 # Static analysis
mix dialyzer                       # Type checking
mix precommit                      # All checks before commit
```

### Useful IEx Commands

```elixir
# List all active avatar processes
Viva.Sessions.Supervisor.list_avatars()

# Get avatar process state
Viva.Sessions.LifeProcess.get_state(avatar_id)

# Generate visuals for avatar
avatar = Viva.Avatars.get_avatar!(avatar_id)
Viva.Avatars.generate_visuals(avatar)

# Deep compatibility analysis
Viva.Nim.ReasoningClient.deep_analyze_compatibility(avatar_a, avatar_b)

# World time
Viva.World.Clock.now()
```

---

## Seed Avatars

VIVA comes with 9 Brazilian seed avatars, each with unique Enneagram types:

| Avatar | Type | Description |
|--------|------|-------------|
| **Lucas** | Type 3 | Empreendedor tech, São Paulo |
| **Marina** | Type 2 | Psicóloga acolhedora, Rio |
| **Pedro** | Type 5 | Dev introvertido, Floripa |
| **Beatriz** | Type 7 | Publicitária aventureira, Salvador |
| **Rafael** | Type 1 | Advogado perfeccionista, Brasília |
| **Carolina** | Type 4 | Artista sensível, Curitiba |
| **Thiago** | Type 8 | Bombeiro protetor, Belo Horizonte |
| **Fernanda** | Type 6 | Médica leal, Porto Alegre |
| **Gabriel** | Type 9 | Músico tranquilo, Recife |

---

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Write tests for your changes
4. Ensure all tests pass (`mix test`)
5. Run code quality checks (`mix precommit`)
6. Commit your changes
7. Push to the branch
8. Open a Pull Request

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Built with Elixir and 14 NVIDIA NIM Models**

[Report Bug](https://github.com/your-org/viva/issues) |
[Request Feature](https://github.com/your-org/viva/issues) |
[Discussions](https://github.com/your-org/viva/discussions)

</div>
