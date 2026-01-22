defmodule VivaBridge.Ultra do
  @moduledoc """
  ULTRA Knowledge Graph Reasoning Client.

  Foundation model for zero-shot link prediction on any knowledge graph.

  ## References

  - Paper: [Towards Foundation Models for Knowledge Graph Reasoning](https://arxiv.org/abs/2310.04562) (ICLR 2024)
  - GitHub: [DeepGraphLearning/ULTRA](https://github.com/DeepGraphLearning/ULTRA)
  - HuggingFace: [mgalkin/ultra_50g](https://huggingface.co/mgalkin/ultra_50g)
  - License: MIT

  ## Architecture

  ULTRA uses:
  - 6-layer GNN for relation graph (relative relation representations)
  - 6-layer NBFNet for entity graph (conditional message passing)
  - 168k parameters total
  - Zero-shot inference on any KG structure

  ## Integration with VIVA

  ```
  ┌─────────────────────────────────────────────────────────┐
  │  VIVA Memory Layer                                      │
  ├─────────────────────────────────────────────────────────┤
  │  Qdrant (Vetores)    │    ULTRA (Grafos)               │
  │  - Busca semântica   │    - Inferência relacional      │
  │  - Embedding lookup  │    - Link prediction            │
  │  - O(log n)          │    - Zero-shot reasoning        │
  └─────────────────────────────────────────────────────────┘
  ```

  ## Usage

      # Build graph from memories
      memories = [%{id: "mem_1", related_to: ["mem_2"]}]
      VivaBridge.Ultra.build_graph(memories)

      # Link prediction
      VivaBridge.Ultra.predict_links("mem_1", "related_to")

      # Find reasoning path
      VivaBridge.Ultra.find_path("event_1", "emotion_1")
  """

  require Logger

  @default_url "http://localhost:8765"
  @timeout 30_000

  # ============================================================================
  # Types
  # ============================================================================

  @type triple :: %{
          head: String.t(),
          relation: String.t(),
          tail: String.t(),
          score: float()
        }

  @type graph_stats :: %{
          entities: non_neg_integer(),
          relations: non_neg_integer(),
          triples: non_neg_integer()
        }

  @type memory :: %{
          id: String.t(),
          content: String.t() | nil,
          emotional_state: map() | nil,
          related_to: [String.t()],
          caused_by: String.t() | nil,
          timestamp: String.t() | nil
        }

  # ============================================================================
  # Public API
  # ============================================================================

  @doc """
  Check if ULTRA service is healthy.

  ## Example

      iex> VivaBridge.Ultra.healthy?()
      true
  """
  @spec healthy?() :: boolean()
  def healthy? do
    case get("/health") do
      {:ok, %{"status" => "healthy"}} -> true
      _ -> false
    end
  end

  @doc """
  Get ULTRA engine statistics.

  ## Example

      iex> VivaBridge.Ultra.stats()
      {:ok, %{
        "loaded" => true,
        "device" => "cuda",
        "checkpoint" => "ultra_50g",
        "graph" => %{"entities" => 10, "relations" => 3, "triples" => 15}
      }}
  """
  @spec stats() :: {:ok, map()} | {:error, term()}
  def stats do
    get("/stats")
  end

  @doc """
  Build knowledge graph from VIVA memory entries.

  This creates the graph structure that ULTRA will reason over.
  Call this when memories are updated in the Dreamer.

  ## Parameters

  - `memories` - List of memory maps with:
    - `:id` - Unique memory identifier
    - `:content` - Memory content (optional)
    - `:emotional_state` - PAD state map (optional)
    - `:related_to` - List of related memory IDs
    - `:caused_by` - Causal memory/event ID (optional)

  ## Example

      memories = [
        %{id: "mem_1", related_to: ["mem_2"], emotional_state: %{pleasure: 0.5}},
        %{id: "mem_2", caused_by: "event_1"}
      ]
      VivaBridge.Ultra.build_graph(memories)
      #=> {:ok, %{entities: 4, relations: 3, triples: 5}}
  """
  @spec build_graph([memory()]) :: {:ok, graph_stats()} | {:error, term()}
  def build_graph(memories) when is_list(memories) do
    formatted =
      Enum.map(memories, fn mem ->
        %{
          id: Map.get(mem, :id) || Map.get(mem, "id"),
          content: Map.get(mem, :content) || Map.get(mem, "content"),
          emotional_state: Map.get(mem, :emotional_state) || Map.get(mem, "emotional_state"),
          related_to: Map.get(mem, :related_to) || Map.get(mem, "related_to") || [],
          caused_by: Map.get(mem, :caused_by) || Map.get(mem, "caused_by"),
          timestamp: Map.get(mem, :timestamp) || Map.get(mem, "timestamp")
        }
      end)

    post("/graph/build", %{memories: formatted})
  end

  @doc """
  Link prediction: Given (head, relation, ?), predict tail entities.

  Use case: "What memories might be related to this one?"

  ## Parameters

  - `head` - Head entity ID
  - `relation` - Relation type (e.g., "related_to", "causes", "has_emotion")
  - `opts` - Options:
    - `:top_k` - Number of predictions (default: 10)

  ## Example

      VivaBridge.Ultra.predict_links("mem_1", "related_to", top_k: 5)
      #=> {:ok, [
        %{head: "mem_1", relation: "related_to", tail: "mem_2", score: 0.85},
        %{head: "mem_1", relation: "related_to", tail: "mem_3", score: 0.72}
      ]}
  """
  @spec predict_links(String.t(), String.t(), keyword()) ::
          {:ok, [triple()]} | {:error, term()}
  def predict_links(head, relation, opts \\ []) do
    top_k = Keyword.get(opts, :top_k, 10)

    post("/predict/link", %{
      head: head,
      relation: relation,
      top_k: top_k
    })
  end

  @doc """
  Relation inference: Given (head, ?, tail), predict relations.

  Use case: "How are these two memories connected?"

  ## Parameters

  - `head` - Head entity ID
  - `tail` - Tail entity ID
  - `opts` - Options:
    - `:top_k` - Number of predictions (default: 5)

  ## Example

      VivaBridge.Ultra.infer_relations("event_1", "emotion_1")
      #=> {:ok, [
        %{head: "event_1", relation: "causes", tail: "emotion_1", score: 0.9}
      ]}
  """
  @spec infer_relations(String.t(), String.t(), keyword()) ::
          {:ok, [triple()]} | {:error, term()}
  def infer_relations(head, tail, opts \\ []) do
    top_k = Keyword.get(opts, :top_k, 5)

    post("/predict/relation", %{
      head: head,
      tail: tail,
      top_k: top_k
    })
  end

  @doc """
  Score a triple for plausibility.

  Returns score in [0, 1] where higher = more plausible.

  Use case: "Is this memory connection valid?"

  ## Example

      VivaBridge.Ultra.score_triple("mem_1", "related_to", "mem_2")
      #=> {:ok, 0.85}
  """
  @spec score_triple(String.t(), String.t(), String.t()) ::
          {:ok, float()} | {:error, term()}
  def score_triple(head, relation, tail) do
    case post("/score", %{head: head, relation: relation, tail: tail}) do
      {:ok, %{"score" => score}} -> {:ok, score}
      error -> error
    end
  end

  @doc """
  Find reasoning path between two entities.

  Multi-hop reasoning to discover how entities connect.

  Use case: "How did this event lead to this emotion?"

  ## Parameters

  - `start` - Starting entity ID
  - `target` - Target entity ID
  - `opts` - Options:
    - `:max_hops` - Maximum path length (default: 3)

  ## Example

      VivaBridge.Ultra.find_path("event_1", "emotion_1", max_hops: 4)
      #=> {:ok, [
        %{head: "event_1", relation: "causes", tail: "mem_1", score: 1.0},
        %{head: "mem_1", relation: "has_emotion", tail: "emotion_1", score: 1.0}
      ]}
  """
  @spec find_path(String.t(), String.t(), keyword()) ::
          {:ok, [triple()]} | {:error, term()}
  def find_path(start, target, opts \\ []) do
    max_hops = Keyword.get(opts, :max_hops, 3)

    post("/path", %{
      "start" => start,
      "end" => target,
      "max_hops" => max_hops
    })
  end

  @doc """
  Add a single triple to the knowledge graph.

  Useful for incremental updates without rebuilding the entire graph.

  ## Example

      VivaBridge.Ultra.add_triple("mem_1", "related_to", "mem_3")
      #=> {:ok, %{entities: 5, relations: 3, triples: 16}}
  """
  @spec add_triple(String.t(), String.t(), String.t()) ::
          {:ok, map()} | {:error, term()}
  def add_triple(head, relation, tail) do
    post("/add_triple?head=#{head}&relation=#{relation}&tail=#{tail}", %{})
  end

  # ============================================================================
  # Integration Helpers
  # ============================================================================

  @doc """
  Convert VIVA memories to ULTRA-compatible format.

  Extracts entities and relations from memory structures.
  """
  @spec memories_to_graph_format([map()]) :: [memory()]
  def memories_to_graph_format(memories) do
    Enum.map(memories, fn mem ->
      %{
        id: extract_id(mem),
        content: Map.get(mem, :content) || Map.get(mem, "content"),
        emotional_state: extract_emotional_state(mem),
        related_to: extract_related_to(mem),
        caused_by: Map.get(mem, :caused_by) || Map.get(mem, "caused_by"),
        timestamp: extract_timestamp(mem)
      }
    end)
  end

  @doc """
  Build graph from Qdrant search results.

  Convenience function to integrate with VivaCore.Qdrant.
  """
  @spec build_graph_from_qdrant_results([map()]) :: {:ok, graph_stats()} | {:error, term()}
  def build_graph_from_qdrant_results(results) do
    memories = memories_to_graph_format(results)
    build_graph(memories)
  end

  # ============================================================================
  # Private Functions
  # ============================================================================

  defp get(path) do
    url = base_url() <> path

    case :httpc.request(:get, {to_charlist(url), []}, [{:timeout, @timeout}], []) do
      {:ok, {{_, 200, _}, _headers, body}} ->
        {:ok, Jason.decode!(to_string(body))}

      {:ok, {{_, status, _}, _headers, body}} ->
        {:error, {:http_error, status, to_string(body)}}

      {:error, reason} ->
        {:error, reason}
    end
  end

  defp post(path, body) do
    url = base_url() <> path
    json_body = Jason.encode!(body)

    headers = [
      {~c"Content-Type", ~c"application/json"}
    ]

    case :httpc.request(
           :post,
           {to_charlist(url), headers, ~c"application/json", to_charlist(json_body)},
           [{:timeout, @timeout}],
           []
         ) do
      {:ok, {{_, 200, _}, _headers, resp_body}} ->
        {:ok, Jason.decode!(to_string(resp_body))}

      {:ok, {{_, status, _}, _headers, resp_body}} ->
        {:error, {:http_error, status, to_string(resp_body)}}

      {:error, reason} ->
        {:error, reason}
    end
  end

  defp base_url do
    Application.get_env(:viva_bridge, :ultra_url, @default_url)
  end

  defp extract_id(mem) do
    Map.get(mem, :id) ||
      Map.get(mem, "id") ||
      Map.get(mem, :memory_id) ||
      Map.get(mem, "memory_id") ||
      :erlang.phash2(mem) |> Integer.to_string()
  end

  defp extract_emotional_state(mem) do
    case Map.get(mem, :emotional_state) || Map.get(mem, "emotional_state") do
      nil ->
        # Try to extract from PAD fields
        p = Map.get(mem, :pleasure) || Map.get(mem, "pleasure")
        a = Map.get(mem, :arousal) || Map.get(mem, "arousal")
        d = Map.get(mem, :dominance) || Map.get(mem, "dominance")

        if p || a || d do
          %{pleasure: p || 0.0, arousal: a || 0.0, dominance: d || 0.0}
        else
          nil
        end

      state ->
        state
    end
  end

  defp extract_related_to(mem) do
    related =
      Map.get(mem, :related_to) ||
        Map.get(mem, "related_to") ||
        Map.get(mem, :associations) ||
        Map.get(mem, "associations") ||
        []

    case related do
      list when is_list(list) -> list
      _ -> []
    end
  end

  defp extract_timestamp(mem) do
    ts =
      Map.get(mem, :timestamp) ||
        Map.get(mem, "timestamp") ||
        Map.get(mem, :created_at) ||
        Map.get(mem, "created_at")

    case ts do
      %DateTime{} = dt -> DateTime.to_iso8601(dt)
      str when is_binary(str) -> str
      _ -> nil
    end
  end
end
