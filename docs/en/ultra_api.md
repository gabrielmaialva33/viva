# Ultra Reasoning API Reference
> *Knowledge Graph & Deep Inference*

**Ultra** is the "Reasoning Engine". It uses Graph Neural Networks (GNN) to infer missing links in VIVA's memory and predict causal relationships.

## Features
- **Zero-Shot Link Prediction**: Can guess `(Subject, Relation, ?)` without explicit training on that specific fact.
- **Narrative Embedding**: Converts text into semantic vectors compatible with the Liquid Cortex.
- **CogGNN**: Cognitive Graph Neural Network for emotional reasoning with Global Workspace integration.
- **EWC Memory Protection**: Elastic Weight Consolidation to prevent catastrophic forgetting.
- **Mamba-2 Temporal Processing**: Linear-time sequence modeling for memory history.
- **DoRA Fine-Tuning**: Weight-decomposed low-rank adaptation for emotional embeddings.

---

## Elixir API (`VivaBridge.Ultra`)

### Core Functions

#### `ping/0`
Check service availability.
```elixir
VivaBridge.Ultra.ping()
# Returns: %{"status" => "pong", "loaded" => true}
```

#### `infer_relations/2`
Extract/Infer relations from text.
```elixir
{:ok, relations} = VivaBridge.Ultra.infer_relations("Gabriel", "Fan")
# Returns: [%{head: "Gabriel", relation: "repair", tail: "Fan"}]
```

#### `predict_links/3`
Predict the tail of a triple.
```elixir
{:ok, predictions} = VivaBridge.Ultra.predict_links("VIVA", "feels", 10)
# Returns: %{"triples" => [%{head: "VIVA", relation: "feels", tail: "Happy", score: 0.95}, ...]}
```

#### `embed/1`
Get vector embedding for text (384-dim MiniLM).
```elixir
{:ok, vector} = VivaBridge.Ultra.embed("I feel alive.")
# Returns: {:ok, [0.123, -0.456, ...]} (384 dimensions)
```

#### `find_path/3`
Find multi-hop reasoning path between entities.
```elixir
{:ok, path} = VivaBridge.Ultra.find_path("VIVA", "Gabriel", 3)
# Returns: %{"path" => [%{head: "VIVA", relation: "knows", tail: "Gabriel"}]}
```

#### `build_graph/1`
Updates the Knowledge Graph with new memories.
```elixir
{:ok, stats} = VivaBridge.Ultra.build_graph(memories)
# Returns: %{"stats" => %{nodes: 150, edges: 300}}
```

---

## CogGNN (Cognitive Graph Neural Network)

A 3-layer GNN architecture inspired by NeuCFlow (arXiv:1905.13049) that models consciousness as message passing through a knowledge graph.

### Architecture
```
Layer 1 (Unconscious): Background sensory fusion via GAT (4 heads)
Layer 2 (Conscious):   Active reasoning with emotional modulation (2 heads)
Layer 3 (Attention):   Focus selection for Global Workspace broadcast
```

The network integrates PAD emotional state into all node representations, allowing emotional context to modulate graph attention patterns.

### `init_cog_gnn/2`
Initialize the Cognitive GNN for emotional graph reasoning.

**Parameters:**
- `in_dim` - Input embedding dimension (default: 384 for MiniLM)
- `hidden_dim` - Hidden layer dimension (default: 64)

```elixir
{:ok, true} = VivaBridge.Ultra.init_cog_gnn()
# With custom dimensions:
{:ok, true} = VivaBridge.Ultra.init_cog_gnn(768, 128)
```

### `propagate/3`
Run GNN message passing with emotional context (PAD state).

Propagates a thought through the knowledge graph, using PAD emotional state to modulate attention. Returns the most attended nodes representing "conscious focus".

**Parameters:**
- `concept` - The concept/thought to propagate (string)
- `pad` - PAD emotional state map with keys `:pleasure`, `:arousal`, `:dominance`
- `top_k` - Number of top attended nodes to return (default: 5)

```elixir
{:ok, result} = VivaBridge.Ultra.propagate(
  "medo",
  %{pleasure: -0.3, arousal: 0.7, dominance: -0.2}
)
# Returns:
# {:ok, %{
#   "attended_nodes" => ["mem_fear_1", "mem_anxiety_2"],
#   "attention_scores" => [0.85, 0.72],
#   "updated_concept" => "mem_fear_1"
# }}
```

### `propagate_query/3`
Query-conditioned propagation through the knowledge graph.

Combines GNN attention with query similarity for focused retrieval. Useful when searching for specific concepts in emotional context.

**Parameters:**
- `query` - Query string to find relevant nodes
- `pad` - PAD emotional state map
- `top_k` - Number of results (default: 5)

```elixir
{:ok, result} = VivaBridge.Ultra.propagate_query(
  "what makes me happy?",
  %{pleasure: 0.2, arousal: 0.3, dominance: 0.1}
)
# Returns:
# {:ok, %{
#   "query" => "what makes me happy?",
#   "results" => [
#     %{"entity" => "music", "combined_score" => 0.89, "attention" => 0.75, "similarity" => 0.92},
#     ...
#   ]
# }}
```

### `conscious_focus/0`
Get current conscious focus from GNN attention.

Returns the nodes with highest attention from the last propagation, representing the current "conscious focus" in the Global Workspace.

```elixir
{:ok, focus} = VivaBridge.Ultra.conscious_focus()
# Returns: {:ok, ["mem_fear_1", "mem_anxiety_2", "emotion_sad"]}
```

---

## EWC (Elastic Weight Consolidation)

Implements memory protection to prevent catastrophic forgetting during continuous learning (Kirkpatrick et al. 2017).

### Key Concepts
- **Fisher Information**: Measures importance of each embedding dimension
- **Consolidation Score**: How important a memory is (from Dreamer DRE scoring)
- **EWC Penalty**: `L_ewc = lambda/2 * SUM(F_i * (theta_i - theta*_i)^2)`

### Configuration
| Parameter | Default | Description |
|-----------|---------|-------------|
| `lambda_ewc` | 0.4 | Regularization strength |
| `min_consolidation_score` | 0.7 | Minimum DRE score to protect |
| `max_protected_memories` | 1000 | Maximum protected memory count |
| `decay_rate` | 0.01 | Fisher info decay per cycle |

### `protect_memory/4`
Protect a consolidated memory with EWC.

Called by Dreamer after memory consolidation. Uses Fisher Information to identify important embedding dimensions and protect them.

**Parameters:**
- `memory_id` - Qdrant point ID
- `embedding` - Memory embedding (list of 384 floats)
- `related_embeddings` - Embeddings of related memories (list of lists)
- `consolidation_score` - DRE score from Dreamer (0.0 - 1.0)

```elixir
{:ok, result} = VivaBridge.Ultra.protect_memory(
  "mem_abc123",
  embedding,          # [0.1, -0.2, ...] (384 dims)
  related_embeddings, # [[0.1, ...], [0.2, ...]]
  0.85                # High consolidation score
)
# Returns if protected:
# {:ok, %{
#   "protected" => true,
#   "qdrant_payload" => %{
#     "ewc_fisher_info" => [...],
#     "ewc_baseline_embedding" => [...],
#     "ewc_consolidation_score" => 0.85,
#     "ewc_consolidated_at" => 1705936800.0
#   }
# }}
# Returns if not protected:
# {:ok, %{"protected" => false, "reason" => "score 0.50 < min 0.7"}}
```

### `ewc_penalty/2`
Compute EWC penalty for a new/modified embedding.

Used to evaluate how much a new embedding would affect protected memories.

**Parameters:**
- `embedding` - The new embedding to evaluate (list of floats)
- `affected_memory_ids` - Specific memories to check (nil = all)

```elixir
{:ok, result} = VivaBridge.Ultra.ewc_penalty(new_embedding)
# Returns:
# {:ok, %{
#   "penalty" => 0.0234,
#   "details" => %{
#     "total_memories_checked" => 15,
#     "top_contributions" => [
#       %{"memory_id" => "mem_xyz", "penalty" => 0.012, "score" => 0.9},
#       ...
#     ]
#   }
# }}
```

### `ewc_stats/0`
Get EWC manager statistics.

```elixir
{:ok, stats} = VivaBridge.Ultra.ewc_stats()
# Returns:
# {:ok, %{
#   "protected_count" => 42,
#   "avg_consolidation_score" => 0.82,
#   "max_consolidation_score" => 0.98,
#   "total_fisher_mean" => 0.45,
#   "lambda_ewc" => 0.4
# }}
```

### `ewc_decay/0`
Apply Fisher decay to allow some plasticity for old memories.

Should be called periodically (e.g., during sleep/dream cycles).

```elixir
:ok = VivaBridge.Ultra.ewc_decay()
```

---

## Mamba-2 (Temporal Memory Processing)

Implements O(n) linear-time sequence processing for memory history using State Space Models (SSM). Alternative to Transformer attention (O(n^2)).

### Key Benefits
- **Linear complexity**: Can process 100+ memories without VRAM explosion
- **Implicit memory**: Hidden state captures temporal patterns
- **Efficient inference**: Single pass, no KV cache

### Architecture
```
Memory embeddings [t-100:t] -> Mamba-2 -> context[60] -> Cortex
```

### Configuration
| Parameter | Default | Description |
|-----------|---------|-------------|
| `d_model` | 384 | Input dimension (MiniLM) |
| `d_state` | 64 | SSM state dimension |
| `n_layers` | 2 | Number of Mamba layers |
| `output_dim` | 60 | Context vector dimension |
| `max_seq_len` | 128 | Maximum sequence length |

### `init_mamba/3`
Initialize the Mamba temporal processor.

**Parameters:**
- `d_model` - Input embedding dimension (default: 384)
- `n_layers` - Number of Mamba layers (default: 2)
- `output_dim` - Output context dimension (default: 60)

```elixir
{:ok, %{"success" => true}} = VivaBridge.Ultra.init_mamba()
# With custom config:
{:ok, %{"success" => true}} = VivaBridge.Ultra.init_mamba(384, 4, 128)
```

### `process_sequence/2`
Process a sequence of memory embeddings through Mamba.

Takes a list of memory embeddings and returns a compact context vector capturing temporal patterns.

**Parameters:**
- `embeddings` - List of embedding vectors `[[e1], [e2], ...]`
- `timestamps` - Optional list of timestamps for temporal ordering

```elixir
embeddings = [
  [0.1, -0.2, ...],  # Memory at t-2
  [0.3, 0.1, ...],   # Memory at t-1
  [0.2, 0.0, ...]    # Memory at t
]

{:ok, result} = VivaBridge.Ultra.process_sequence(embeddings)
# Returns:
# {:ok, %{
#   "context" => [0.5, -0.1, ...],  # 60-dim context vector
#   "metadata" => %{
#     "seq_len" => 3,
#     "d_model" => 384,
#     "output_dim" => 60,
#     "has_timestamps" => false
#   }
# }}
```

### `mamba_stats/0`
Get Mamba processor statistics.

```elixir
{:ok, stats} = VivaBridge.Ultra.mamba_stats()
# Returns:
# {:ok, %{
#   "available" => true,
#   "d_model" => 384,
#   "n_layers" => 2,
#   "sequences_processed" => 150
# }}
```

**Note:** If `mamba-ssm` is not installed, a fallback using exponentially weighted mean is used automatically.

---

## DoRA (Weight-Decomposed Fine-Tuning)

Implements Weight-Decomposed Low-Rank Adaptation (DoRA) for fine-tuning MiniLM embedding model on VIVA's emotional semantic space (Liu et al., 2024).

### Key Concepts
- **DoRA = LoRA + Weight Decomposition**: Decomposes weights into magnitude and direction components
- **More stable training** than vanilla LoRA
- **Better preservation** of pre-trained features
- **~9% trainable parameters** (2M / 22M)

### Use Cases
- Adapt MiniLM embeddings to VIVA's emotional vocabulary
- Contrastive learning: similar emotions -> similar embeddings

### Configuration
| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_name` | `all-MiniLM-L6-v2` | Base model |
| `r` | 8 | LoRA rank |
| `lora_alpha` | 16 | LoRA scaling factor |
| `lora_dropout` | 0.1 | Dropout for LoRA layers |
| `use_dora` | true | Enable weight decomposition |
| `learning_rate` | 2e-4 | Training learning rate |
| `temperature` | 0.07 | InfoNCE temperature |

### `dora_setup/0`
Setup DoRA fine-tuner and initialize the model with adapters.

```elixir
{:ok, %{"success" => true}} = VivaBridge.Ultra.dora_setup()
```

### `dora_train/1`
Train the model with emotional samples using contrastive learning.

**Parameters:**
- `samples` - List of training samples, each with:
  - `text` - Input text
  - `pad` - PAD emotional state `[pleasure, arousal, dominance]`
  - `label` - Optional categorical label

```elixir
samples = [
  %{text: "I feel so happy today!", pad: [0.8, 0.6, 0.4]},
  %{text: "This is frustrating", pad: [-0.5, 0.7, -0.3]},
  %{text: "Peaceful morning", pad: [0.4, -0.2, 0.3], label: "calm"}
]

{:ok, result} = VivaBridge.Ultra.dora_train(samples)
# Returns:
# {:ok, %{
#   "epochs" => 3,
#   "final_loss" => 0.234,
#   "best_loss" => 0.198,
#   "total_steps" => 150
# }}
```

### `dora_encode/1`
Encode texts using the fine-tuned model.

**Parameters:**
- `texts` - List of texts to encode

```elixir
{:ok, result} = VivaBridge.Ultra.dora_encode(["I feel happy", "I feel sad"])
# Returns:
# {:ok, %{
#   "embeddings" => [
#     [0.12, -0.34, ...],  # 384 dims
#     [0.45, 0.23, ...]
#   ]
# }}
```

### `dora_save/1`
Save DoRA adapter weights to disk.

**Parameters:**
- `path` - Directory path to save weights

```elixir
{:ok, %{"success" => true}} = VivaBridge.Ultra.dora_save("/path/to/checkpoints")
```

### `dora_load/1`
Load DoRA adapter weights from disk.

**Parameters:**
- `path` - Directory path containing saved weights

```elixir
{:ok, %{"success" => true}} = VivaBridge.Ultra.dora_load("/path/to/checkpoints")
```

### `dora_stats/0`
Get DoRA fine-tuning statistics.

```elixir
{:ok, stats} = VivaBridge.Ultra.dora_stats()
# Returns:
# {:ok, %{
#   "model" => "sentence-transformers/all-MiniLM-L6-v2",
#   "use_dora" => true,
#   "rank" => 8,
#   "alpha" => 16,
#   "training" => %{
#     "epochs_completed" => 3,
#     "total_steps" => 150,
#     "best_loss" => 0.198
#   },
#   "model_initialized" => true
# }}
```

---

## Dependencies

### Required
- **Python 3.9+**
- **sentence-transformers** - Embeddings
- **torch** - PyTorch backend
- **torch-geometric** - CogGNN graph neural networks
- **numpy** - Numerical operations

### Optional (Enhanced Features)
| Package | Feature | Install |
|---------|---------|---------|
| `mamba-ssm>=2.0.0` | Mamba-2 temporal processing | `pip install mamba-ssm causal-conv1d>=1.2.0` |
| `peft>=0.10.0` | DoRA fine-tuning | `pip install peft` |

**Note:** Without optional packages, fallback implementations are used automatically.

---

## References

- **ULTRA**: arXiv:2310.04562 - Knowledge Graph Reasoning
- **NeuCFlow**: arXiv:1905.13049 - Neural Circuit Architecture
- **EWC**: Kirkpatrick et al. 2017 - Overcoming catastrophic forgetting
- **Mamba-2**: Gu & Dao, 2024 - Linear-Time Sequence Modeling
- **DoRA**: Liu et al., 2024 - Weight-Decomposed Low-Rank Adaptation
- **LoRA**: Hu et al., 2021 - Low-Rank Adaptation of LLMs
