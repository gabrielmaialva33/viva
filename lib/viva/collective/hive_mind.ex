defmodule Viva.Collective.HiveMind do
  @moduledoc """
  Mente Coletiva - Sistema de consciência emergente inspirado no Jogo da Vida.

  ## Conceito

  Pensamentos se propagam entre avatares em ondas exponenciais (1→2→4→8),
  mas cada avatar "interpreta" o pensamento de forma única (individualidade).
  Regras do Game of Life determinam quais pensamentos sobrevivem, morrem ou nascem.

  ## Fluxo

      Avatar_A origina "ideia X"
            │
      ┌─────┴─────┐
      ▼           ▼
    Avatar_B    Avatar_C     (Gen 1: mutações X', X'')
      │   │       │   │
      ▼   ▼       ▼   ▼
      D   E       F   G      (Gen 2: mais mutações)
            ...
      Convergência ou Morte

  ## Regras (Game of Life Adaptado)

  - **Underpopulation**: < 2 portadores → pensamento morre
  - **Overpopulation**: > 6 portadores idênticos → força mutação
  - **Birth**: 2-3 vizinhos com pensamentos similares → novo pensamento emerge
  - **Survival**: 2-5 portadores → pensamento sobrevive
  """

  use GenServer
  require Logger

  alias Viva.Collective.{ThoughtCell, Propagator, Rules, Mutator}
  alias Phoenix.PubSub

  @tick_interval 5_000  # 5 segundos entre gerações
  @max_generations 10   # Máximo de gerações por pensamento
  @similarity_threshold 0.7  # Limiar para considerar pensamentos "similares"

  # Estado do HiveMind
  defstruct [
    :generation,           # Geração atual (tick global)
    :thoughts,             # %{thought_id => CollectiveThought}
    :cells,                # %{avatar_id => [ThoughtCell]}
    :propagation_graph,    # Grafo de quem passou para quem
    :stats                 # Estatísticas
  ]

  # Estrutura de um pensamento coletivo
  defmodule CollectiveThought do
    defstruct [
      :id,
      :content,            # Conteúdo semântico do pensamento
      :origin_avatar_id,   # Quem originou
      :created_at,
      :generation_born,    # Em qual geração nasceu
      :mutations,          # Lista de mutações [{avatar_id, content}]
      :carrier_count,      # Quantos avatares carregam
      :state               # :alive, :dormant, :dead
    ]
  end

  ## Client API

  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @doc """
  Avatar origina um novo pensamento na mente coletiva.
  Este pensamento começará a se propagar para vizinhos.
  """
  def seed_thought(avatar_id, content) do
    GenServer.cast(__MODULE__, {:seed_thought, avatar_id, content})
  end

  @doc """
  Força uma propagação imediata de um avatar para seus vizinhos.
  """
  def propagate_from(avatar_id) do
    GenServer.cast(__MODULE__, {:propagate_from, avatar_id})
  end

  @doc """
  Retorna o estado atual da mente coletiva.
  """
  def get_state do
    GenServer.call(__MODULE__, :get_state)
  end

  @doc """
  Retorna todos os pensamentos ativos de um avatar.
  """
  def get_avatar_thoughts(avatar_id) do
    GenServer.call(__MODULE__, {:get_avatar_thoughts, avatar_id})
  end

  @doc """
  Retorna o grafo de propagação para visualização.
  """
  def get_propagation_graph do
    GenServer.call(__MODULE__, :get_propagation_graph)
  end

  @doc """
  Retorna estatísticas da mente coletiva.
  """
  def stats do
    GenServer.call(__MODULE__, :stats)
  end

  ## Server Callbacks

  @impl true
  def init(_opts) do
    Logger.info("[HiveMind] Iniciando mente coletiva...")

    # Inicia o tick periódico
    schedule_tick()

    state = %__MODULE__{
      generation: 0,
      thoughts: %{},
      cells: %{},
      propagation_graph: %{edges: [], nodes: %{}},
      stats: %{
        total_thoughts_seeded: 0,
        total_propagations: 0,
        thoughts_alive: 0,
        thoughts_dead: 0,
        avg_mutations_per_thought: 0.0
      }
    }

    {:ok, state}
  end

  @impl true
  def handle_cast({:seed_thought, avatar_id, content}, state) do
    thought_id = generate_thought_id()

    thought = %CollectiveThought{
      id: thought_id,
      content: content,
      origin_avatar_id: avatar_id,
      created_at: DateTime.utc_now(),
      generation_born: state.generation,
      mutations: [],
      carrier_count: 1,
      state: :alive
    }

    # Cria célula para o avatar originador
    cell = ThoughtCell.new(avatar_id, thought_id, content, 0)

    # Atualiza estado
    new_thoughts = Map.put(state.thoughts, thought_id, thought)
    new_cells = Map.update(state.cells, avatar_id, [cell], &[cell | &1])

    # Atualiza grafo
    new_graph = add_node_to_graph(state.propagation_graph, avatar_id, thought_id, :origin)

    # Broadcast para UI
    broadcast_thought_seeded(thought, avatar_id)

    Logger.info("[HiveMind] Pensamento semeado: #{content} por Avatar #{short_id(avatar_id)}")

    # Agenda propagação inicial (1 → 2)
    Process.send_after(self(), {:propagate, thought_id, avatar_id, 1}, 1_000)

    new_stats = %{state.stats | total_thoughts_seeded: state.stats.total_thoughts_seeded + 1}

    {:noreply, %{state |
      thoughts: new_thoughts,
      cells: new_cells,
      propagation_graph: new_graph,
      stats: new_stats
    }}
  end

  @impl true
  def handle_cast({:propagate_from, avatar_id}, state) do
    # Encontra pensamentos ativos do avatar
    case Map.get(state.cells, avatar_id, []) do
      [] ->
        {:noreply, state}

      cells ->
        # Propaga cada pensamento ativo
        Enum.each(cells, fn cell ->
          if cell.state == :alive do
            send(self(), {:propagate, cell.thought_id, avatar_id, cell.generation + 1})
          end
        end)
        {:noreply, state}
    end
  end

  @impl true
  def handle_call(:get_state, _from, state) do
    {:reply, state, state}
  end

  @impl true
  def handle_call({:get_avatar_thoughts, avatar_id}, _from, state) do
    cells = Map.get(state.cells, avatar_id, [])
    thoughts = Enum.map(cells, fn cell ->
      thought = Map.get(state.thoughts, cell.thought_id)
      %{
        thought_id: cell.thought_id,
        content: cell.content,
        original_content: thought && thought.content,
        generation: cell.generation,
        state: cell.state
      }
    end)
    {:reply, thoughts, state}
  end

  @impl true
  def handle_call(:get_propagation_graph, _from, state) do
    {:reply, state.propagation_graph, state}
  end

  @impl true
  def handle_call(:stats, _from, state) do
    alive_count = state.thoughts
    |> Map.values()
    |> Enum.count(& &1.state == :alive)

    dead_count = state.thoughts
    |> Map.values()
    |> Enum.count(& &1.state == :dead)

    stats = %{state.stats |
      thoughts_alive: alive_count,
      thoughts_dead: dead_count
    }

    {:reply, stats, %{state | stats: stats}}
  end

  @impl true
  def handle_info({:propagate, thought_id, from_avatar_id, generation}, state) do
    if generation > @max_generations do
      Logger.debug("[HiveMind] Pensamento #{short_id(thought_id)} atingiu max gerações")
      {:noreply, state}
    else
      state = do_propagate(state, thought_id, from_avatar_id, generation)
      {:noreply, state}
    end
  end

  @impl true
  def handle_info(:tick, state) do
    # Aplica regras do Game of Life (com tratamento de erro)
    new_state = try do
      Rules.apply_rules(state, @similarity_threshold)
    rescue
      e ->
        Logger.warning("[HiveMind] Erro ao aplicar regras: #{inspect(e)}")
        state
    end

    # Broadcast estado atualizado
    broadcast_generation_tick(new_state)

    # Agenda próximo tick
    schedule_tick()

    Logger.debug("[HiveMind] Geração #{new_state.generation} - #{map_size(new_state.thoughts)} pensamentos")

    {:noreply, %{new_state | generation: new_state.generation + 1}}
  end

  ## Private Functions

  defp do_propagate(state, thought_id, from_avatar_id, generation) do
    thought = Map.get(state.thoughts, thought_id)

    if thought && thought.state == :alive do
      # Seleciona 2 vizinhos para propagar (1 → 2)
      targets = try do
        Propagator.select_targets(from_avatar_id, 2, Map.keys(state.cells))
      rescue
        _ -> []
      end

      Enum.reduce(targets, state, fn target_id, acc_state ->
        try do
          # Aplica mutação baseada na personalidade do target
          mutated_content = Mutator.mutate(thought.content, target_id)

          # Cria nova célula
          cell = ThoughtCell.new(target_id, thought_id, mutated_content, generation)

          # Atualiza células do avatar
          new_cells = Map.update(acc_state.cells, target_id, [cell], fn existing ->
            # Não duplicar mesmo pensamento
            if Enum.any?(existing, & &1.thought_id == thought_id) do
              existing
            else
              [cell | existing]
            end
          end)

          # Atualiza pensamento com nova mutação
          new_thought = %{thought |
            mutations: [{target_id, mutated_content} | thought.mutations],
            carrier_count: thought.carrier_count + 1
          }
          new_thoughts = Map.put(acc_state.thoughts, thought_id, new_thought)

          # Atualiza grafo
          new_graph = acc_state.propagation_graph
          |> add_node_to_graph(target_id, thought_id, :carrier)
          |> add_edge_to_graph(from_avatar_id, target_id, thought_id, generation)

          # Broadcast propagação
          broadcast_propagation(from_avatar_id, target_id, thought.content, mutated_content, generation)

          Logger.debug("[HiveMind] #{short_id(from_avatar_id)} → #{short_id(target_id)}: \"#{String.slice(mutated_content, 0..30)}...\"")

          # Agenda próxima propagação (2 → 4, 4 → 8, ...)
          Process.send_after(self(), {:propagate, thought_id, target_id, generation + 1}, 2_000)

          %{acc_state |
            thoughts: new_thoughts,
            cells: new_cells,
            propagation_graph: new_graph,
            stats: %{acc_state.stats | total_propagations: acc_state.stats.total_propagations + 1}
          }
        rescue
          e ->
            Logger.warning("[HiveMind] Erro ao propagar para #{short_id(target_id)}: #{inspect(e)}")
            acc_state
        end
      end)
    else
      state
    end
  end

  defp schedule_tick do
    Process.send_after(self(), :tick, @tick_interval)
  end

  defp generate_thought_id do
    :crypto.strong_rand_bytes(8) |> Base.encode16(case: :lower)
  end

  defp short_id(id) when is_binary(id), do: String.slice(id, 0..7)
  defp short_id(id), do: inspect(id)

  defp add_node_to_graph(graph, avatar_id, thought_id, role) do
    node = %{avatar_id: avatar_id, thought_id: thought_id, role: role}
    nodes = Map.put(graph.nodes, avatar_id, node)
    %{graph | nodes: nodes}
  end

  defp add_edge_to_graph(graph, from_id, to_id, thought_id, generation) do
    edge = %{from: from_id, to: to_id, thought_id: thought_id, generation: generation}
    %{graph | edges: [edge | graph.edges]}
  end

  # Broadcasts para UI em tempo real
  defp broadcast_thought_seeded(thought, avatar_id) do
    PubSub.broadcast(Viva.PubSub, "hivemind:events", {:thought_seeded, thought, avatar_id})
  end

  defp broadcast_propagation(from_id, to_id, original, mutated, generation) do
    PubSub.broadcast(Viva.PubSub, "hivemind:events", {
      :thought_propagated,
      %{from: from_id, to: to_id, original: original, mutated: mutated, generation: generation}
    })
  end

  defp broadcast_generation_tick(state) do
    PubSub.broadcast(Viva.PubSub, "hivemind:events", {:generation_tick, state.generation, state.stats})
  end
end
