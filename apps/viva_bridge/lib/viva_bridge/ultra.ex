defmodule VivaBridge.Ultra do
  @moduledoc """
  Elixir Bridge for the ULTRA Knowledge Graph Reasoning Engine.

  Manages the Python process `ultra_service.py` via Port.
  Provides API for zero-shot link prediction and reasoning.

  Reference: arXiv:2310.04562
  """

  use GenServer
  require Logger

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

  def ping do
    GenServer.call(__MODULE__, :ping)
  end

  # ============================================================================
  # GenServer Callbacks
  # ============================================================================

  @impl true
  def init(_opts) do
    # Locate the python script
    script_path = Path.join([File.cwd!(), "services", "ultra", "ultra_service.py"])

    if File.exists?(script_path) do
      Logger.info("[VivaBridge.Ultra] Starting ULTRA Service: #{script_path}")

      port = Port.open({:spawn, "python3 -u #{script_path}"}, [:binary, :line])

      {:ok, %{port: port, requests: %{}}}
    else
      Logger.error("[VivaBridge.Ultra] Service script not found at #{script_path}")
      {:stop, :enoent}
    end
  end

  @impl true
  def handle_call(request, from, state) do
    {command, args} =
      case request do
        {:build_graph, mems} -> {"build_graph", %{memories: mems}}
        {:predict_links, h, r, k} -> {"predict_links", %{head: h, relation: r, top_k: k}}
        {:infer_relations, h, t, k} -> {"infer_relations", %{head: h, tail: t, top_k: k}}
        {:find_path, s, e, hops} -> {"find_path", %{start: s, end: e, max_hops: hops}}
        :ping -> {"ping", %{}}
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
  def handle_info({port, {:data, {:eol, line}}}, state) when port == state.port do
    case Jason.decode(line) do
      {:ok, response} ->
        handle_response(response, state)

      {:error, _} ->
        Logger.warning("[VivaBridge.Ultra] Invalid JSON from Python: #{line}")
        {:noreply, state}
    end
  end

  @impl true
  def handle_info({:EXIT, _port, reason}, state) do
    Logger.error("[VivaBridge.Ultra] Port crashed: #{inspect(reason)}")
    {:stop, reason, state}
  end

  defp handle_response(response, state) do
    req_id = response["id"]
    result = response["result"] || response["error"]

    {from, new_requests} = Map.pop(state.requests, req_id)

    if from do
      GenServer.reply(from, result)
    else
      Logger.warning("[VivaBridge.Ultra] Received response for unknown ID: #{req_id}")
    end

    {:noreply, %{state | requests: new_requests}}
  end
end
