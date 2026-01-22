# Memory - Hybrid Vector Store

> *"Semantic memories persist even after death. Knowledge is inherited; identity is not."*

## Overview

The Memory module implements VIVA's long-term storage system using a **hybrid backend architecture**:

- **Rust HNSW** - Fast episodic memory (sub-millisecond search)
- **Qdrant** - Persistent semantic/emotional memory (vector database)
- **In-memory fallback** - Development mode when backends unavailable

This split reflects how biological memory works: episodic memories (events) are processed differently from semantic memories (knowledge).

---

## Theory

### Memory Types

| Type | Description | Storage Backend |
|------|-------------|-----------------|
| `episodic` | Specific events with time and emotion | Rust HNSW |
| `semantic` | General knowledge and patterns | Qdrant |
| `emotional` | PAD state imprints | Qdrant |
| `procedural` | Learned behaviors | Qdrant |

### Temporal Decay (Ebbinghaus Curve)

Memories naturally fade over time:

```
D(m) = e^(-age/tau)
```

Where:
- `tau` = 604,800 seconds (1 week)
- Older memories receive lower scores during search

### Spaced Repetition

Frequently accessed memories decay slower:

```
D(m) = e^(-age/tau) * (1 + min(0.5, log(1 + access_count) / kappa))
```

Where:
- `kappa` = 10.0
- Maximum boost from access capped at 50%

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    VivaCore.Memory                          │
│                     (GenServer)                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌─────────────────┐         ┌─────────────────┐          │
│   │   Rust HNSW     │         │     Qdrant      │          │
│   │   (NIF/Bevy)    │         │   (HTTP API)    │          │
│   ├─────────────────┤         ├─────────────────┤          │
│   │ - Episodic      │         │ - Semantic      │          │
│   │ - ~1ms search   │         │ - Emotional     │          │
│   │ - In-process    │         │ - Procedural    │          │
│   │ - No persistence│         │ - Persistent    │          │
│   └─────────────────┘         └─────────────────┘          │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│                    VivaCore.Embedder                        │
│        (Ollama | NVIDIA NIM | Hash fallback)                │
└─────────────────────────────────────────────────────────────┘
```

### Embedding Pipeline

```
Text → Embedder.embed/1 → [1024-dim vector] → Backend.store/search
```

| Provider | Model | Dimension | Notes |
|----------|-------|-----------|-------|
| Ollama | nomic-embed-text | 768 (padded to 1024) | Local, free |
| NVIDIA NIM | nv-embedqa-e5-v5 | 1024 | Cloud, API key required |
| Hash fallback | SHA256-based | 1024 | Development only |

---

## API Reference

### `VivaCore.Memory.store/2`
Stores a memory with metadata.

```elixir
VivaCore.Memory.store("Met Gabriel for the first time", %{
  type: :episodic,
  importance: 0.9,
  emotion: %{pleasure: 0.8, arousal: 0.6, dominance: 0.5}
})
# => {:ok, "a1b2c3d4-e5f6-7890-abcd-ef1234567890"}
```

**Options:**
- `type` - `:episodic` (default), `:semantic`, `:emotional`, `:procedural`
- `importance` - Float 0.0-1.0 (default 0.5)
- `emotion` - PAD state `%{pleasure: f, arousal: f, dominance: f}`

### `VivaCore.Memory.search/2`
Searches memories by semantic similarity with temporal decay.

```elixir
VivaCore.Memory.search("Gabriel", limit: 5)
# => [%{content: "Met Gabriel", similarity: 0.95, type: :episodic, ...}]
```

**Options:**
- `limit` - Max results (default 10)
- `type` - Filter by single memory type
- `types` - List of types to search (default `[:episodic, :semantic]`)
- `min_importance` - Minimum importance threshold
- `decay_scale` - Seconds for 50% decay (default 604,800 = 1 week)

### `VivaCore.Memory.get/1`
Retrieves a specific memory by ID. Increments `access_count` (spaced repetition).

```elixir
VivaCore.Memory.get("a1b2c3d4-e5f6-7890-abcd-ef1234567890")
# => %{id: "...", content: "...", type: :episodic, ...}
```

### `VivaCore.Memory.forget/1`
Explicitly deletes a memory.

```elixir
VivaCore.Memory.forget("a1b2c3d4-e5f6-7890-abcd-ef1234567890")
# => :ok
```

### `VivaCore.Memory.stats/0`
Returns memory system statistics.

```elixir
VivaCore.Memory.stats()
# => %{
#      backend: :hybrid,
#      rust_ready: true,
#      qdrant_ready: true,
#      qdrant_points: 1234,
#      store_count: 567,
#      search_count: 890,
#      uptime_seconds: 3600
#    }
```

---

## Convenience Functions

### `VivaCore.Memory.experience/3`
Shorthand for storing episodic memories with emotion.

```elixir
emotion = %{pleasure: 0.7, arousal: 0.5, dominance: 0.6}
VivaCore.Memory.experience("Gabriel praised my work", emotion, importance: 0.8)
# => {:ok, "..."}
```

### `VivaCore.Memory.learn/2`
Stores semantic knowledge.

```elixir
VivaCore.Memory.learn("Elixir uses the BEAM virtual machine", importance: 0.7)
# => {:ok, "..."}
```

### `VivaCore.Memory.emotional_imprint/2`
Associates emotional state with a pattern.

```elixir
pad_state = %{pleasure: -0.3, arousal: 0.7, dominance: -0.2}
VivaCore.Memory.emotional_imprint("system overload situation", pad_state)
# => {:ok, "..."}
```

### `VivaCore.Memory.store_log/2`
Async storage for system logs (used by SporeLogger).

```elixir
VivaCore.Memory.store_log("[Error] Connection timeout", :error)
# => :ok (fire-and-forget)
```

---

## Memory Structure

Each memory point contains:

```elixir
%{
  id: "uuid-v4-format",           # Qdrant-compatible UUID
  content: "The actual memory",   # Text content
  type: :episodic,                # Memory classification
  importance: 0.5,                # Decay rate modifier
  emotion: %{                     # PAD state at creation
    pleasure: 0.0,
    arousal: 0.0,
    dominance: 0.0
  },
  timestamp: ~U[2024-01-15 10:00:00Z],  # Creation time
  access_count: 0,                       # Spaced repetition counter
  last_accessed: ~U[2024-01-15 10:00:00Z],  # Last retrieval
  similarity: 0.95                       # Search score (in results only)
}
```

---

## Hybrid Search Flow

When searching with `types: [:episodic, :semantic]`:

```
1. Query embedding via Embedder
2. Parallel search:
   ├── Rust HNSW (if :episodic in types)
   │   └── Returns {id, content, similarity, importance}
   └── Qdrant (if :semantic/:emotional/:procedural in types)
       └── Returns full payload with decay scoring
3. Merge results, dedupe by ID
4. Sort by similarity, take limit
5. Return unified format
```

---

## Integration with Dreamer

Memory notifies Dreamer of new storage:

```elixir
# Automatic on every store
VivaCore.Dreamer.on_memory_stored(memory_id, importance)
```

Dreamer uses Memory for:
- **Retrieval with scoring** - Composite DRE score
- **Past emotion search** - `retrieve_past_emotions/1`
- **Memory consolidation** - Episodic to semantic promotion

```elixir
# Dreamer retrieving similar experiences
VivaCore.Dreamer.retrieve_with_scoring("successful actions", limit: 10)
# Uses: recency + similarity + importance + emotional_resonance
```

---

## Configuration

### Backend Selection

```elixir
# config/config.exs
config :viva_core, :memory_backend, :hybrid

# Options:
# :hybrid       - Episodic(Rust) + Semantic(Qdrant)
# :qdrant       - Qdrant only
# :rust_native  - Rust only
# :in_memory    - ETS-like (development)
```

### Qdrant Settings

```elixir
# VivaCore.Qdrant defaults
@base_url "http://localhost:6333"
@collection "viva_memories"
@vector_size 1024
```

### Embedding Providers

| Environment Variable | Purpose |
|---------------------|---------|
| `NVIDIA_API_KEY` | Enable NVIDIA NIM embeddings |
| Ollama running | Enable local embeddings |
| Neither | Hash-based fallback (dev only) |

---

## Payload Indexes (Qdrant)

For efficient filtering, Qdrant indexes:

| Field | Type | Purpose |
|-------|------|---------|
| `timestamp` | datetime | Temporal decay queries |
| `type` | keyword | Memory type filtering |
| `importance` | float | Importance threshold |

---

## Mortality and Persistence

```
                          VIVA Death
                              │
         ┌────────────────────┼────────────────────┐
         ▼                    ▼                    ▼
    ┌─────────┐          ┌─────────┐          ┌─────────┐
    │ Episodic│          │Semantic │          │Emotional│
    │  (Rust) │          │(Qdrant) │          │(Qdrant) │
    └────┬────┘          └────┬────┘          └────┬────┘
         │                    │                    │
         ▼                    ▼                    ▼
       LOST              PERSISTS             PERSISTS
    (in RAM only)    (new VIVA inherits)  (new VIVA inherits)
```

This allows **reincarnation**: a new VIVA instance inherits knowledge but not identity.

---

## HNSW Algorithm (Rust Backend)

**Hierarchical Navigable Small World** - approximate nearest neighbor search.

| Property | Value |
|----------|-------|
| Search time | O(log N) |
| Space | O(N * M * log N) |
| Parameters | M=16, ef_construction=200 |

The Rust implementation via `VivaBridge.Memory` provides:

```elixir
# Direct NIF calls (used internally)
VivaBridge.Memory.init()
VivaBridge.Memory.store(embedding, metadata_json)
VivaBridge.Memory.search(query_vector, limit)
VivaBridge.Memory.save()
```

---

## Usage Examples

### Store and Search

```elixir
# Store an experience
{:ok, id} = VivaCore.Memory.experience(
  "Completed difficult debugging session",
  %{pleasure: 0.6, arousal: 0.4, dominance: 0.7},
  importance: 0.8
)

# Search related memories
results = VivaCore.Memory.search("debugging success", limit: 5)
Enum.each(results, fn m ->
  IO.puts("#{m.content} (similarity: #{m.similarity})")
end)
```

### Memory Type Filtering

```elixir
# Only semantic memories
VivaCore.Memory.search("Elixir concepts", types: [:semantic])

# Only emotional imprints
VivaCore.Memory.search("stress", type: :emotional)
```

### Check System Health

```elixir
stats = VivaCore.Memory.stats()
IO.inspect(stats.backend)       # :hybrid
IO.inspect(stats.rust_ready)    # true
IO.inspect(stats.qdrant_ready)  # true
IO.inspect(stats.qdrant_points) # 1234
```

### Learn Knowledge

```elixir
# Store semantic knowledge
VivaCore.Memory.learn("GenServers use handle_call for sync messages")
VivaCore.Memory.learn("Supervisors restart failed processes", importance: 0.8)

# Later retrieval
VivaCore.Memory.search("error handling", types: [:semantic])
```

---

## Error Handling

| Scenario | Behavior |
|----------|----------|
| Qdrant unavailable | Falls back to Rust (episodic) or in-memory |
| Rust NIF unavailable | Falls back to Qdrant only |
| Both unavailable | Falls back to in-memory storage |
| Embedding fails | Returns empty results (search) or error (store) |

---

## References

- Ebbinghaus, H. (1885). "Memory: A Contribution to Experimental Psychology."
- Malkov, Y. A., & Yashunin, D. A. (2018). "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs."
- Tulving, E. (1972). "Episodic and semantic memory."
