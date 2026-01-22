defmodule VivaBridge.Ultra do
  @moduledoc """
  Elixir Bridge for the ULTRA Knowledge Graph Reasoning Engine.

  Manages the Python process `ultra_service.py` via Port.
  Provides API for zero-shot link prediction and reasoning.

  Reference: arXiv:2310.04562
  """

  use GenServer
  require VivaLog

  # ============================================================================
  # Public API
  # ============================================================================

  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @doc """
  Updates the Knowledge Graph with new memories.

  ## Parameters
  - `memories`: List of memory maps/structs
  """
  def build_graph(memories) do
    GenServer.call(__MODULE__, {:build_graph, memories}, 30_000)
  end

  @doc """
  Predict likely tail entities for a given head and relation.
  (Head, Relation, ?) -> [Tail, Score]
  """
  def predict_links(head, relation, top_k \\ 10) do
    GenServer.call(__MODULE__, {:predict_links, head, relation, top_k}, 15_000)
  end

  @doc """
  Infer relation between two entities.
  (Head, ?, Tail) -> [Relation, Score]
  """
  def infer_relations(head, tail, top_k \\ 5) do
    GenServer.call(__MODULE__, {:infer_relations, head, tail, top_k}, 15_000)
  end

  @doc """
  Find multi-hop reasoning path between entities.
  """
  def find_path(start_node, end_node, max_hops \\ 3) do
    GenServer.call(__MODULE__, {:find_path, start_node, end_node, max_hops}, 20_000)
  end

  @doc """
  Converts text or concept into a semantic vector embedding.
  Returns {:ok, [float]} or {:error, reason}
  """
  def embed(text) do
    case GenServer.call(__MODULE__, {:embed, text}, 10_000) do
      %{"embedding" => list} -> {:ok, list}
      error -> {:error, error}
    end
  end

  def ping do
    GenServer.call(__MODULE__, :ping, 30_000)
  end

  # ============================================================================
  # CogGNN API (Cognitive Graph Neural Network)
  # ============================================================================

  @doc """
  Initialize the Cognitive GNN for emotional graph reasoning.

  ## Parameters
  - `in_dim`: Input embedding dimension (default: 384 for MiniLM)
  - `hidden_dim`: Hidden layer dimension (default: 64)

  ## Returns
  - `{:ok, true}` on success
  - `{:error, reason}` on failure
  """
  def init_cog_gnn(in_dim \\ 384, hidden_dim \\ 64) do
    case GenServer.call(__MODULE__, {:init_cog_gnn, in_dim, hidden_dim}, 30_000) do
      %{"success" => true} -> {:ok, true}
      %{"success" => false} -> {:error, :initialization_failed}
      error -> {:error, error}
    end
  end

  @doc """
  Run GNN message passing with emotional context (PAD state).

  Propagates a thought through the knowledge graph, using PAD emotional
  state to modulate attention. Implements NeuCFlow-inspired 3-layer
  architecture: Unconscious → Conscious → Attention.

  ## Parameters
  - `concept`: The concept/thought to propagate (string)
  - `pad`: PAD emotional state map with keys :pleasure, :arousal, :dominance
  - `top_k`: Number of top attended nodes to return (default: 5)

  ## Returns
  - `{:ok, result}` with attended_nodes, attention_scores, updated_concept
  - `{:error, reason}` on failure

  ## Example
      iex> VivaBridge.Ultra.propagate("medo", %{pleasure: -0.3, arousal: 0.7, dominance: -0.2})
      {:ok, %{
        "attended_nodes" => ["mem_fear_1", "mem_anxiety_2"],
        "attention_scores" => [0.85, 0.72],
        "updated_concept" => "mem_fear_1"
      }}
  """
  def propagate(concept, pad, top_k \\ 5) when is_map(pad) do
    pad_list = [
      Map.get(pad, :pleasure, 0.0),
      Map.get(pad, :arousal, 0.0),
      Map.get(pad, :dominance, 0.0)
    ]

    case GenServer.call(__MODULE__, {:propagate, concept, pad_list, top_k}, 15_000) do
      %{"error" => reason} -> {:error, reason}
      result when is_map(result) -> {:ok, result}
      error -> {:error, error}
    end
  end

  @doc """
  Get current conscious focus from GNN attention.

  Returns the nodes with highest attention from the last propagation,
  representing the current "conscious focus" in the Global Workspace.

  ## Returns
  - `{:ok, [entity_names]}` list of focused entities
  - `{:error, reason}` on failure
  """
  def conscious_focus do
    case GenServer.call(__MODULE__, :conscious_focus, 5_000) do
      %{"focus" => focus} when is_list(focus) -> {:ok, focus}
      error -> {:error, error}
    end
  end

  @doc """
  Query-conditioned propagation through the knowledge graph.

  Combines GNN attention with query similarity for focused retrieval.

  ## Parameters
  - `query`: Query string to find relevant nodes
  - `pad`: PAD emotional state map
  - `top_k`: Number of results (default: 5)

  ## Returns
  - `{:ok, result}` with query, results (entity, combined_score, attention, similarity)
  - `{:error, reason}` on failure
  """
  def propagate_query(query, pad, top_k \\ 5) when is_map(pad) do
    pad_list = [
      Map.get(pad, :pleasure, 0.0),
      Map.get(pad, :arousal, 0.0),
      Map.get(pad, :dominance, 0.0)
    ]

    case GenServer.call(__MODULE__, {:propagate_query, query, pad_list, top_k}, 15_000) do
      %{"error" => reason} -> {:error, reason}
      result when is_map(result) -> {:ok, result}
      error -> {:error, error}
    end
  end

  # ============================================================================
  # EWC API (Elastic Weight Consolidation)
  # ============================================================================

  @doc """
  Protect a consolidated memory with EWC.

  Uses Fisher Information to identify important embedding dimensions
  and protect them from catastrophic forgetting during learning.

  ## Parameters
  - `memory_id`: Qdrant point ID
  - `embedding`: Memory embedding (list of floats)
  - `related_embeddings`: Embeddings of related memories (list of lists)
  - `consolidation_score`: DRE score from Dreamer (0.0 - 1.0)

  ## Returns
  - `{:ok, %{protected: true, qdrant_payload: ...}}` if protected
  - `{:ok, %{protected: false, reason: ...}}` if not protected
  - `{:error, reason}` on failure
  """
  def protect_memory(memory_id, embedding, related_embeddings, consolidation_score) do
    case GenServer.call(
           __MODULE__,
           {:protect_memory, memory_id, embedding, related_embeddings, consolidation_score},
           15_000
         ) do
      %{"error" => reason} -> {:error, reason}
      result when is_map(result) -> {:ok, result}
      error -> {:error, error}
    end
  end

  @doc """
  Compute EWC penalty for a new/modified embedding.

  Used to evaluate how much a new embedding would affect protected memories.

  ## Parameters
  - `embedding`: The new embedding to evaluate
  - `affected_memory_ids`: Specific memories to check (nil = all)

  ## Returns
  - `{:ok, %{penalty: float, details: map}}` with penalty value
  - `{:error, reason}` on failure
  """
  def ewc_penalty(embedding, affected_memory_ids \\ nil) do
    case GenServer.call(__MODULE__, {:ewc_penalty, embedding, affected_memory_ids}, 10_000) do
      %{"error" => reason} -> {:error, reason}
      result when is_map(result) -> {:ok, result}
      error -> {:error, error}
    end
  end

  @doc """
  Get EWC manager statistics.

  ## Returns
  - `{:ok, stats}` with protected_count, avg_consolidation_score, etc.
  - `{:error, reason}` on failure
  """
  def ewc_stats do
    case GenServer.call(__MODULE__, :ewc_stats, 5_000) do
      %{"error" => reason} -> {:error, reason}
      result when is_map(result) -> {:ok, result}
      error -> {:error, error}
    end
  end

  @doc """
  Apply Fisher decay to allow some plasticity for old memories.

  Should be called periodically (e.g., during sleep cycles).
  """
  def ewc_decay do
    case GenServer.call(__MODULE__, :ewc_decay, 5_000) do
      %{"status" => "decay_applied"} -> :ok
      error -> {:error, error}
    end
  end

  # ============================================================================
  # GenServer Callbacks
  # ============================================================================

  @impl true
  def init(_opts) do
    # Locate the python script
    script_path = Path.join([File.cwd!(), "services", "ultra", "ultra_service.py"])

    if File.exists?(script_path) do
      VivaLog.info(:ultra, :starting_service, path: script_path)

      port = Port.open({:spawn, "python3 -u #{script_path}"}, [:binary, :line])

      {:ok, %{port: port, requests: %{}, buffer: ""}}
    else
      VivaLog.error(:ultra, :script_not_found, path: script_path)
      {:stop, :enoent}
    end
  end

  @impl true
  def handle_call(request, from, state) do
    {command, args} =
      case request do
        {:build_graph, mems} ->
          {"build_graph", %{memories: mems}}

        {:predict_links, h, r, k} ->
          {"predict_links", %{head: h, relation: r, top_k: k}}

        {:infer_relations, h, t, k} ->
          {"infer_relations", %{head: h, tail: t, top_k: k}}

        {:find_path, s, e, hops} ->
          {"find_path", %{start: s, end: e, max_hops: hops}}

        {:embed, t} ->
          {"embed", %{text: t}}

        :ping ->
          {"ping", %{}}

        # CogGNN commands
        {:init_cog_gnn, in_dim, hidden_dim} ->
          {"init_cog_gnn", %{in_dim: in_dim, hidden_dim: hidden_dim}}

        {:propagate, concept, pad, top_k} ->
          {"propagate", %{concept: concept, pad: pad, top_k: top_k}}

        :conscious_focus ->
          {"conscious_focus", %{}}

        {:propagate_query, query, pad, top_k} ->
          {"propagate_query", %{query: query, pad: pad, top_k: top_k}}

        # EWC commands
        {:protect_memory, mem_id, emb, related, score} ->
          {"protect_memory",
           %{
             memory_id: mem_id,
             embedding: emb,
             related_embeddings: related,
             consolidation_score: score
           }}

        {:ewc_penalty, emb, affected_ids} ->
          {"ewc_penalty", %{embedding: emb, affected_memory_ids: affected_ids}}

        :ewc_stats ->
          {"ewc_stats", %{}}

        :ewc_decay ->
          {"ewc_decay", %{}}
      end

    # Generate request ID
    req_id = make_ref() |> :erlang.ref_to_list() |> List.to_string()

    payload = %{
      command: command,
      args: args,
      id: req_id
    }

    # Send to Python
    json_data = Jason.encode!(payload)
    Port.command(state.port, "#{json_data}\n")

    # Store caller to reply later
    new_requests = Map.put(state.requests, req_id, from)

    {:noreply, %{state | requests: new_requests}}
  end

  @impl true
  def handle_info({port, {:data, {:noeol, chunk}}}, state) when port == state.port do
    {:noreply, %{state | buffer: state.buffer <> chunk}}
  end

  @impl true
  def handle_info({port, {:data, {:eol, chunk}}}, state) when port == state.port do
    full_line = state.buffer <> chunk

    case Jason.decode(full_line) do
      {:ok, response} ->
        handle_response(response, state)

      {:error, _} ->
        VivaLog.warning(:ultra, :invalid_json, snippet: String.slice(full_line, 0, 100))

        {:noreply, %{state | buffer: ""}}
    end
    # Reset buffer after processing line
    |> case do
      {:noreply, new_state} -> {:noreply, %{new_state | buffer: ""}}
      other -> other
    end
  end

  @impl true
  def handle_info({:EXIT, _port, reason}, state) do
    VivaLog.error(:ultra, :port_crashed, reason: reason)
    {:stop, reason, state}
  end

  defp handle_response(response, state) do
    req_id = response["id"]
    result = response["result"] || response["error"]

    {from, new_requests} = Map.pop(state.requests, req_id)

    if from do
      GenServer.reply(from, result)
    else
      VivaLog.warning(:ultra, :unknown_response_id, id: req_id)
    end

    {:noreply, %{state | requests: new_requests}}
  end
end
