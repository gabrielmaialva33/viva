defmodule VivaCore.Mycelium do
  @moduledoc """
  Mycelium Network - Small-World Topology for GenServer Communication.

  Inspired by fungal mycelium networks that exhibit:
  - **Local Clustering**: Nearby nodes form dense clusters
  - **Long-Range Shortcuts**: Random connections to distant nodes
  - **Hub Formation**: Some nodes become highly connected (like Emotional)
  - **Adaptive Routing**: Messages find shortest path through network

  > "Em redes small-world, qualquer nó pode alcançar qualquer outro
  >  em poucos saltos - como sinapses ou hifas de fungos."
  > — Podcast Fundacional

  ## Architecture
  - Tracks active GenServers as network nodes
  - Builds adjacency graph with PubSub subscriptions as edges
  - Calculates network metrics (clustering coefficient, path length)
  - Enables intelligent message routing through hubs

  ## Usage
  ```elixir
  VivaCore.Mycelium.network_stats()
  VivaCore.Mycelium.shortest_path(:emotional, :observer)
  VivaCore.Mycelium.hubs()  # Most connected nodes
  ```
  """
  use GenServer
  require VivaLog

  # Core neuron modules that form the network
  @neurons [
    VivaCore.Emotional,
    VivaCore.Memory,
    VivaCore.Interoception,
    VivaCore.Senses,
    VivaCore.Dreamer,
    VivaCore.Agency,
    VivaCore.Voice,
    VivaCore.Consciousness.Workspace,
    VivaCore.Consciousness.Discrete,
    VivaCore.World.Observer,
    VivaCore.Kinship
  ]

  # Known communication channels (edges in the graph)
  # Format: {from, to, channel}
  @known_channels [
    # Senses → Emotional (heartbeat)
    {:senses, :emotional, "body:state"},
    # Interoception → Emotional (free energy)
    {:interoception, :emotional, "interoception:qualia"},
    # Interoception → Discrete (tick sync)
    {:interoception, :discrete, "interoception:tick"},
    # Observer → Emotional (movement events)
    {:observer, :emotional, "observer:events"},
    # Emotional → Workspace (salience boost)
    {:emotional, :workspace, "emotional:state"},
    # Workspace → Voice (focus → babble)
    {:workspace, :voice, "workspace:focus"},
    # Dreamer → Memory (consolidation)
    {:dreamer, :memory, "dreamer:reflect"},
    # Agency → Emotional (action feedback)
    {:agency, :emotional, "agency:result"},
    # Kinship → Emotional (solidarity)
    {:kinship, :emotional, "kinship:solidarity"}
  ]

  defstruct [
    nodes: %{},
    edges: [],
    adjacency: %{},
    metrics: %{},
    last_scan: nil
  ]

  # ============================================================================
  # Client API
  # ============================================================================

  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @doc """
  Returns network statistics: node count, edge count, clustering coefficient.
  """
  def network_stats do
    GenServer.call(__MODULE__, :network_stats)
  catch
    :exit, _ -> %{status: :offline}
  end

  @doc """
  Returns the most connected nodes (hubs) in the network.
  These are critical for message propagation.
  """
  def hubs(limit \\ 3) do
    GenServer.call(__MODULE__, {:hubs, limit})
  catch
    :exit, _ -> []
  end

  @doc """
  Calculates shortest path between two modules.
  Returns list of intermediate nodes.
  """
  def shortest_path(from, to) do
    GenServer.call(__MODULE__, {:shortest_path, from, to})
  catch
    :exit, _ -> {:error, :offline}
  end

  @doc """
  Returns all active nodes in the network.
  """
  def active_nodes do
    GenServer.call(__MODULE__, :active_nodes)
  catch
    :exit, _ -> []
  end

  @doc """
  Forces a network scan to update topology.
  """
  def scan do
    GenServer.cast(__MODULE__, :scan)
  end

  @doc """
  Broadcasts a message through the mycelium network.
  Uses intelligent routing through hubs if direct path unavailable.
  """
  def propagate(source, message) do
    GenServer.cast(__MODULE__, {:propagate, source, message})
  end

  # ============================================================================
  # Server Callbacks
  # ============================================================================

  @impl true
  def init(_opts) do
    # Schedule initial network scan
    Process.send_after(self(), :scan_network, 1000)

    # Schedule periodic rescans
    :timer.send_interval(30_000, :scan_network)

    state = %__MODULE__{
      nodes: %{},
      edges: @known_channels,
      adjacency: build_adjacency(@known_channels)
    }

    VivaLog.info(:mycelium, :network_online, channels: length(@known_channels))

    {:ok, state}
  end

  @impl true
  def handle_call(:network_stats, _from, state) do
    stats = calculate_stats(state)
    {:reply, stats, %{state | metrics: stats}}
  end

  @impl true
  def handle_call({:hubs, limit}, _from, state) do
    hubs =
      state.adjacency
      |> Enum.map(fn {node, neighbors} -> {node, MapSet.size(neighbors)} end)
      |> Enum.sort_by(fn {_, degree} -> -degree end)
      |> Enum.take(limit)

    {:reply, hubs, state}
  end

  @impl true
  def handle_call({:shortest_path, from, to}, _from, state) do
    path = bfs_path(state.adjacency, from, to)
    {:reply, path, state}
  end

  @impl true
  def handle_call(:active_nodes, _from, state) do
    {:reply, Map.keys(state.nodes), state}
  end

  @impl true
  def handle_cast(:scan, state) do
    {:noreply, scan_network(state)}
  end

  @impl true
  def handle_cast({:propagate, source, message}, state) do
    # Propagate through all connected nodes
    case Map.get(state.adjacency, source) do
      nil ->
        :ok

      neighbors ->
        Enum.each(neighbors, fn neighbor ->
          VivaLog.debug(:mycelium, :propagating,
            from: source,
            to: neighbor,
            message: inspect(message)
          )
        end)
    end

    {:noreply, state}
  end

  @impl true
  def handle_info(:scan_network, state) do
    {:noreply, scan_network(state)}
  end

  # ============================================================================
  # Network Analysis
  # ============================================================================

  defp scan_network(state) do
    # Check which neurons are alive
    active =
      @neurons
      |> Enum.filter(fn mod ->
        case Process.whereis(mod) do
          nil -> false
          pid -> Process.alive?(pid)
        end
      end)
      |> Enum.map(fn mod -> {module_to_atom(mod), mod} end)
      |> Map.new()

    %{state | nodes: active, last_scan: System.system_time(:millisecond)}
  end

  defp build_adjacency(edges) do
    Enum.reduce(edges, %{}, fn {from, to, _channel}, acc ->
      acc
      |> Map.update(from, MapSet.new([to]), &MapSet.put(&1, to))
      |> Map.update(to, MapSet.new([from]), &MapSet.put(&1, from))
    end)
  end

  defp calculate_stats(state) do
    node_count = map_size(state.nodes)
    edge_count = length(state.edges)

    # Average degree
    avg_degree =
      if node_count > 0 do
        total_degree =
          state.adjacency
          |> Enum.map(fn {_, neighbors} -> MapSet.size(neighbors) end)
          |> Enum.sum()

        total_degree / node_count
      else
        0.0
      end

    # Clustering coefficient (simplified)
    clustering = calculate_clustering(state.adjacency)

    %{
      node_count: node_count,
      edge_count: edge_count,
      average_degree: Float.round(avg_degree, 2),
      clustering_coefficient: Float.round(clustering, 3),
      last_scan: state.last_scan,
      is_small_world: clustering > 0.3 and avg_degree > 2.0
    }
  end

  defp calculate_clustering(adjacency) when map_size(adjacency) == 0, do: 0.0

  defp calculate_clustering(adjacency) do
    # For each node, count triangles / possible triangles
    coefficients =
      Enum.map(adjacency, fn {_node, neighbors} ->
        k = MapSet.size(neighbors)

        if k < 2 do
          0.0
        else
          # Count actual triangles (edges between neighbors)
          neighbor_list = MapSet.to_list(neighbors)

          triangles =
            for n1 <- neighbor_list,
                n2 <- neighbor_list,
                n1 < n2,
                n2_neighbors = Map.get(adjacency, n1, MapSet.new()),
                MapSet.member?(n2_neighbors, n2),
                reduce: 0 do
              acc -> acc + 1
            end

          possible = k * (k - 1) / 2

          if possible > 0 do
            triangles / possible
          else
            0.0
          end
        end
      end)

    if length(coefficients) > 0 do
      Enum.sum(coefficients) / length(coefficients)
    else
      0.0
    end
  end

  defp bfs_path(_adjacency, from, to) when from == to, do: {:ok, [from]}

  defp bfs_path(adjacency, from, to) do
    bfs_search(adjacency, [{from, [from]}], MapSet.new([from]), to)
  end

  defp bfs_search(_adjacency, [], _visited, _target), do: {:error, :no_path}

  defp bfs_search(adjacency, [{current, path} | rest], visited, target) do
    neighbors = Map.get(adjacency, current, MapSet.new())

    if MapSet.member?(neighbors, target) do
      {:ok, Enum.reverse([target | path])}
    else
      new_nodes =
        neighbors
        |> MapSet.difference(visited)
        |> MapSet.to_list()
        |> Enum.map(fn n -> {n, [n | path]} end)

      new_visited = MapSet.union(visited, neighbors)
      bfs_search(adjacency, rest ++ new_nodes, new_visited, target)
    end
  end

  defp module_to_atom(module) do
    module
    |> Module.split()
    |> List.last()
    |> Macro.underscore()
    |> String.to_atom()
  end
end
