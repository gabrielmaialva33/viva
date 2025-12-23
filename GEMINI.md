# GEMINI.md

This file provides context and guidance for Gemini when working with the VIVA project.

## Project Overview

**VIVA (Virtual Intelligent Vida Autonoma)** is an AI platform where digital avatars live autonomous lives 24/7.
*   **Core Concept:** Avatars have unique personalities (Big Five + Enneagram), emotions, and memories. They form relationships and interact even when their owners are offline.
*   **Tech Stack:**
    *   **Backend:** Elixir 1.15+ / Phoenix 1.8.2
    *   **Database:** TimescaleDB (PostgreSQL 17 + time-series extensions)
    *   **Vector DB:** Qdrant (for semantic memory)
    *   **Caching:** Redis + Cachex
    *   **AI:** NVIDIA NIM Cloud (14 models for LLM, reasoning, voice, and visuals)
    *   **Background Jobs:** Oban

## Architecture

The system is built around the concept of "Life Processes":

*   **Sessions (`lib/viva/sessions/`):**
    *   `Viva.Sessions.LifeProcess`: A GenServer that runs for each active avatar. It ticks every 60 seconds (scaled time), decaying needs and triggering autonomous actions.
    *   `Viva.Sessions.Supervisor`: DynamicSupervisor managing avatar processes.
    *   `Viva.World.Clock`: Manages simulation time (1 real minute = 10 simulated minutes).

*   **Domain Contexts (`lib/viva/`):**
    *   **Avatars:** Personality, internal state (emotions), memory management.
    *   **Relationships:** Tracks affection, trust, and relationship status.
    *   **Matchmaker:** Engine for calculating and caching compatibility scores.
    *   **Nim:** Client modules for interacting with NVIDIA NIM APIs (LLM, TTS, ASR, etc.) using `Req`.

## Development & Usage

### Prerequisites
*   Elixir 1.15+ & Erlang/OTP 26+
*   Docker & Docker Compose
*   NVIDIA NIM API Key (in `.env`)

### Key Commands

| Action | Command |
| :--- | :--- |
| **Install Deps** | `mix deps.get` |
| **Start Infra** | `docker compose up -d` |
| **Setup DB** | `mix ecto.setup` (Creates, migrates, and seeds) |
| **Start Server** | `mix phx.server` or `iex -S mix phx.server` |
| **Run Tests** | `mix test` |
| **Lint/Format** | `mix precommit` (Runs format, credo, and tests) |
| **Migrations** | `mix ecto.gen.migration <name>` |

### Testing
*   **Process Cleanup:** Always use `start_supervised!/1` in tests.
*   **Async Assertions:** Avoid `Process.sleep/1`. Use `Process.monitor/1` and `assert_receive` or `:sys.get_state/1` for synchronization.

## Coding Conventions

### Phoenix 1.8 & LiveView
*   **Layouts:** LiveView templates must start with `<Layouts.app flash={@flash} ...>`.
*   **Components:** Use `<.icon>` (not Heroicons modules) and `<.input>` (from core_components).
*   **Streams:** Use `stream(socket, :items, items)` for collections.
    *   Template: `<div id="items" phx-update="stream">`.
    *   **Reset:** Streams are not enumerable. To filter, refetch and use `reset: true`.
*   **Hooks:** Colocated JS hooks must start with `.` (e.g., `.PhoneNumber`).
*   **Tailwind:** V4 syntax (`@import "tailwindcss"` in `app.css`). No `tailwind.config.js`.

### Elixir Patterns
*   **List Access:** Use `Enum.at/2`, never `list[index]`.
*   **Struct Access:** Access fields directly (`struct.field`), never `struct[:field]`.
*   **Changesets:** Use `Ecto.Changeset.get_field/2`.
*   **HTTP Client:** Use `Req` (already included). Avoid adding `HTTPoison` or `Tesla`.
*   **Rebinding:** Rebind results of control structures: `socket = if connected?(socket), do: ...`.

## Sub-Agents

When faced with complex tasks requiring deep analysis or architectural understanding, delegate to the `codebase_investigator`.

*   **Codebase Investigator:** Use for "Code Archaeology" (understanding legacy/complex code), architectural mapping, and root cause analysis of bugs.
