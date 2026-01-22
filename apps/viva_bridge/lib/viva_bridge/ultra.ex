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
        {:build_graph, mems} -> {"build_graph", %{memories: mems}}
        {:predict_links, h, r, k} -> {"predict_links", %{head: h, relation: r, top_k: k}}
        {:infer_relations, h, t, k} -> {"infer_relations", %{head: h, tail: t, top_k: k}}
        {:find_path, s, e, hops} -> {"find_path", %{start: s, end: e, max_hops: hops}}
        {:embed, t} -> {"embed", %{text: t}}
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
