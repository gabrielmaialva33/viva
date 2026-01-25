# ULTRA Integration - Knowledge Graph Reasoning for VIVA

## Overview

ULTRA (Unified, Learnable, Transferable representations) is a foundation model for knowledge graph reasoning that enables VIVA to perform zero-shot link prediction on memory structures.

## References

- **Paper**: [Towards Foundation Models for Knowledge Graph Reasoning](https://arxiv.org/abs/2310.04562) (ICLR 2024)
- **GitHub**: [DeepGraphLearning/ULTRA](https://github.com/DeepGraphLearning/ULTRA)
- **HuggingFace**: [mgalkin/ultra_50g](https://huggingface.co/mgalkin/ultra_50g)
- **License**: MIT

## Architecture

### ULTRA Model

```
┌─────────────────────────────────────────────────────────────┐
│  ULTRA (168k parameters)                                    │
├─────────────────────────────────────────────────────────────┤
│  Relation Graph GNN (6 layers)                              │
│  - Learns relative relation representations                 │
│  - No entity embeddings needed                              │
├─────────────────────────────────────────────────────────────┤
│  Entity Graph NBFNet (6 layers)                             │
│  - Conditional message passing                              │
│  - rspmm kernel: O(V) instead of O(E)                       │
└─────────────────────────────────────────────────────────────┘
```

### VIVA Integration

```
┌─────────────────────────────────────────────────────────────┐
│  VIVA Memory Layer                                          │
├─────────────────────────────────────────────────────────────┤
│  Qdrant (Vectors)      │    ULTRA (Graphs)                  │
│  - Semantic search     │    - Relational inference          │
│  - Embedding lookup    │    - Link prediction               │
│  - O(log n)            │    - Zero-shot reasoning           │
├─────────────────────────────────────────────────────────────┤
│  Dreamer (Consolidation)                                    │
│  - DRE scoring for memory promotion                         │
│  - ULTRA for causal chain discovery                         │
│  - Hybrid: Vector similarity + Graph reasoning              │
└─────────────────────────────────────────────────────────────┘
```

## Service Architecture

### Python Service (ULTRA Inference)

```
services/ultra/
├── server.py          # FastAPI HTTP server
├── ultra_engine.py    # Model wrapper
├── requirements.txt   # Dependencies
├── Dockerfile         # Container image
└── docker-compose.yml # Deployment
```

**Endpoints:**

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/graph/build` | Build KG from memories |
| `POST` | `/predict/link` | Link prediction (h, r, ?) |
| `POST` | `/predict/relation` | Relation inference (h, ?, t) |
| `POST` | `/score` | Score triple plausibility |
| `POST` | `/path` | Find reasoning path |
| `GET` | `/health` | Health check |
| `GET` | `/stats` | Engine statistics |

### Elixir Client

```elixir
# Build graph from memories
VivaBridge.Ultra.build_graph(memories)

# Link prediction
VivaBridge.Ultra.predict_links("mem_1", "related_to", top_k: 5)

# Find reasoning path
VivaBridge.Ultra.find_path("event_1", "emotion_1", max_hops: 3)
```

## Use Cases in VIVA

### 1. Memory Consolidation (Dreamer)

During sleep cycles, ULTRA enhances the DRE consolidation process:

```elixir
# Standard DRE score
dre_score = Mathematics.consolidation_score(memory_pad, baseline, importance, age, access)

# ULTRA enhancement: Check causal connections
{:ok, paths} = VivaBridge.Ultra.find_path(memory_id, "positive_outcome")
causal_boost = length(paths) * 0.05

# Final score combines both
final_score = dre_score + causal_boost
```

### 2. Emotional Causality

Discover how events lead to emotional states:

```elixir
# Find path from event to emotion
VivaBridge.Ultra.find_path("user_praised_viva", "emotion_joy")
#=> [
  {head: "user_praised_viva", relation: "causes", tail: "memory_123"},
  {head: "memory_123", relation: "has_emotion", tail: "emotion_joy"}
]
```

### 3. Memory Relationship Discovery

Infer implicit connections between memories:

```elixir
# What connects these two memories?
VivaBridge.Ultra.infer_relations("mem_early", "mem_recent")
#=> [{head: "mem_early", relation: "via_causes_related_to", tail: "mem_recent", score: 0.7}]
```

### 4. Experience-Outcome Prediction

Predict likely outcomes based on past patterns:

```elixir
# Given this stimulus, what emotions might follow?
VivaBridge.Ultra.predict_links("stimulus_criticism", "causes", top_k: 3)
#=> [
  {tail: "emotion_sadness", score: 0.8},
  {tail: "emotion_defensiveness", score: 0.6}
]
```

## Graph Schema

### Entities

| Type | Description | Example |
|------|-------------|---------|
| `memory` | Episodic/semantic memory | `mem_123` |
| `event` | External stimulus | `event_user_msg` |
| `emotion` | Emotional state | `emotion_mem_123` |
| `emotion_type` | Emotion category | `positive`, `negative`, `neutral` |

### Relations

| Relation | From | To | Description |
|----------|------|-----|-------------|
| `related_to` | memory | memory | Semantic similarity |
| `causes` | event/memory | memory/emotion | Causal link |
| `has_emotion` | memory | emotion | Emotional association |
| `is_type` | emotion | emotion_type | Classification |

## Deployment

### Local Development

```bash
cd services/ultra

# Install dependencies
pip install -r requirements.txt

# Run server
python server.py
```

### Docker

```bash
# Build image
docker build -t viva-ultra .

# Run with GPU
docker run -p 8765:8765 --gpus all viva-ultra
```

### With Qdrant

```yaml
# docker-compose.yml
services:
  ultra:
    build: ./services/ultra
    ports:
      - "8765:8765"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
```

## Configuration

### Elixir

```elixir
# config/config.exs
config :viva_bridge,
  ultra_url: "http://localhost:8765"
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ULTRA_CHECKPOINT` | `ultra_50g.pth` | Model checkpoint |
| `CUDA_VISIBLE_DEVICES` | `0` | GPU device |
| `ULTRA_PORT` | `8765` | Server port |

## Performance

### Model Size

- **Parameters**: 168,000 (2 MB checkpoint)
- **Memory**: ~500 MB VRAM for inference
- **Latency**: ~10-50ms per prediction (GPU)

### Comparison with Alternatives

| Model | Parameters | Zero-shot | Inductive |
|-------|------------|-----------|-----------|
| TransE | ~1M+ | ❌ | ❌ |
| DistMult | ~1M+ | ❌ | ❌ |
| NBFNet | ~2M | ❌ | ✅ |
| **ULTRA** | 168k | ✅ | ✅ |

## Future Work

1. **UltraQuery Integration**: Complex logical queries (AND, OR, NOT)
2. **Continuous Learning**: Update graph as memories change
3. **Hierarchical Reasoning**: Multi-level abstraction
4. **Confidence Calibration**: Better uncertainty estimation

## Citation

```bibtex
@inproceedings{galkin2023ultra,
    title={Towards Foundation Models for Knowledge Graph Reasoning},
    author={Mikhail Galkin and Xinyu Yuan and Hesham Mostafa and Jian Tang and Zhaocheng Zhu},
    booktitle={The Twelfth International Conference on Learning Representations},
    year={2024},
    url={https://openreview.net/forum?id=jVEoydFOl9}
}
```
