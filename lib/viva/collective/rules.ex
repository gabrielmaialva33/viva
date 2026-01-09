defmodule Viva.Collective.Rules do
  @moduledoc """
  Regras do Jogo da Vida adaptadas para Pensamentos Coletivos.

  ## Regras Clássicas de Conway (adaptadas)

  1. **UNDERPOPULATION (Solidão)**:
     Pensamento com < 2 portadores → MORRE
     _Ideias precisam de massa crítica para sobreviver_

  2. **OVERPOPULATION (Saturação)**:
     Pensamento com > 6 portadores IDÊNTICOS → FORÇA MUTAÇÃO
     _Evita homogeneização total da mente coletiva_

  3. **SURVIVAL (Sobrevivência)**:
     Pensamento com 2-5 portadores → SOBREVIVE
     _Zona Goldilocks de propagação_

  4. **BIRTH (Nascimento)**:
     Se 2-3 avatares vizinhos têm pensamentos SIMILARES
     → Novo pensamento EMERGE (fusão)
     _Convergência gera novas ideias_

  5. **DECAY (Decaimento)**:
     Pensamentos perdem força a cada geração
     _Ideias antigas enfraquecem sem reforço_

  ## Dinâmica Emergente

      ┌─────────────────────────────────────────────┐
      │  Diversidade ◄──────────────► Convergência  │
      │       ▲                            ▲        │
      │       │     EQUILÍBRIO DINÂMICO    │        │
      │       ▼                            ▼        │
      │   Mutação  ◄──────────────►  Fusão          │
      └─────────────────────────────────────────────┘
  """

  require Logger

  alias Viva.Collective.{HiveMind, ThoughtCell, Mutator}

  # Thresholds das regras
  @underpopulation_threshold 2
  @overpopulation_threshold 6
  @survival_min 2
  @survival_max 5
  @birth_threshold_min 2
  @birth_threshold_max 3
  @decay_rate 0.1

  @doc """
  Aplica todas as regras do Game of Life ao estado da HiveMind.
  Retorna o novo estado após uma "geração".
  """
  def apply_rules(state, similarity_threshold \\ 0.7) do
    state
    |> apply_decay()
    |> apply_underpopulation()
    |> apply_overpopulation(similarity_threshold)
    |> apply_survival()
    |> check_for_births(similarity_threshold)
    |> cleanup_dead()
  end

  @doc """
  Aplica decaimento a todas as células.
  """
  def apply_decay(state) do
    new_cells = state.cells
    |> Enum.map(fn {avatar_id, cells} ->
      updated_cells = Enum.map(cells, &ThoughtCell.decay(&1, @decay_rate))
      {avatar_id, updated_cells}
    end)
    |> Map.new()

    %{state | cells: new_cells}
  end

  @doc """
  Regra de Underpopulation: < 2 portadores → morre.
  """
  def apply_underpopulation(state) do
    # Conta portadores por pensamento
    carrier_counts = count_carriers(state)

    # Identifica pensamentos que devem morrer
    dying_thoughts = carrier_counts
    |> Enum.filter(fn {_thought_id, count} -> count < @underpopulation_threshold end)
    |> Enum.map(&elem(&1, 0))
    |> MapSet.new()

    if MapSet.size(dying_thoughts) > 0 do
      Logger.debug("[Rules] Underpopulation: #{MapSet.size(dying_thoughts)} pensamentos morrendo")
    end

    # Mata pensamentos nos thoughts e cells
    new_thoughts = Enum.reduce(dying_thoughts, state.thoughts, fn thought_id, acc ->
      case Map.get(acc, thought_id) do
        nil -> acc
        thought -> Map.put(acc, thought_id, %{thought | state: :dead})
      end
    end)

    new_cells = state.cells
    |> Enum.map(fn {avatar_id, cells} ->
      updated_cells = Enum.map(cells, fn cell ->
        if cell.thought_id in dying_thoughts do
          ThoughtCell.kill(cell)
        else
          cell
        end
      end)
      {avatar_id, updated_cells}
    end)
    |> Map.new()

    %{state | thoughts: new_thoughts, cells: new_cells}
  end

  @doc """
  Regra de Overpopulation: > 6 portadores idênticos → força mutação.
  """
  def apply_overpopulation(state, similarity_threshold) do
    # Agrupa pensamentos por similaridade de conteúdo
    thought_groups = group_similar_thoughts(state, similarity_threshold)

    # Encontra grupos superpopulados
    overpopulated = thought_groups
    |> Enum.filter(fn {_content, group} -> length(group) > @overpopulation_threshold end)

    if length(overpopulated) > 0 do
      Logger.debug("[Rules] Overpopulation: #{length(overpopulated)} grupos forçando mutação")
    end

    # Para cada grupo superpopulado, força mutação em alguns membros
    Enum.reduce(overpopulated, state, fn {_content, group}, acc_state ->
      # Seleciona metade para mutar
      to_mutate = group
      |> Enum.shuffle()
      |> Enum.take(div(length(group), 2))

      # Aplica mutação forçada
      Enum.reduce(to_mutate, acc_state, fn {avatar_id, cell}, inner_state ->
        mutated_content = force_mutation(cell.content)

        new_cells = Map.update!(inner_state.cells, avatar_id, fn cells ->
          Enum.map(cells, fn c ->
            if c.thought_id == cell.thought_id do
              %{c | content: mutated_content}
            else
              c
            end
          end)
        end)

        %{inner_state | cells: new_cells}
      end)
    end)
  end

  @doc """
  Regra de Survival: 2-5 portadores → fortalece.
  """
  def apply_survival(state) do
    carrier_counts = count_carriers(state)

    # Pensamentos que sobrevivem ficam mais fortes
    surviving = carrier_counts
    |> Enum.filter(fn {_thought_id, count} ->
      count >= @survival_min and count <= @survival_max
    end)
    |> Enum.map(&elem(&1, 0))
    |> MapSet.new()

    new_cells = state.cells
    |> Enum.map(fn {avatar_id, cells} ->
      updated_cells = Enum.map(cells, fn cell ->
        if cell.thought_id in surviving and cell.state == :alive do
          ThoughtCell.strengthen(cell, 0.1)
        else
          cell
        end
      end)
      {avatar_id, updated_cells}
    end)
    |> Map.new()

    %{state | cells: new_cells}
  end

  @doc """
  Regra de Birth: 2-3 vizinhos com pensamentos similares → novo pensamento emerge.
  """
  def check_for_births(state, similarity_threshold) do
    # Para cada avatar sem pensamentos ativos
    avatars_without_thoughts = find_avatars_without_active_thoughts(state)

    Enum.reduce(avatars_without_thoughts, state, fn avatar_id, acc_state ->
      # Busca vizinhos com pensamentos
      neighbors = get_neighbor_thoughts(avatar_id, acc_state)

      # Verifica se há 2-3 pensamentos similares
      similar_groups = group_by_similarity(neighbors, similarity_threshold)

      case find_birth_candidate(similar_groups) do
        nil ->
          acc_state

        {contents, count} when count >= @birth_threshold_min and count <= @birth_threshold_max ->
          # Nasce novo pensamento por fusão!
          Logger.info("[Rules] Birth! Avatar #{short_id(avatar_id)} recebe pensamento emergente")
          create_emergent_thought(acc_state, avatar_id, contents)

        _ ->
          acc_state
      end
    end)
  end

  @doc """
  Remove células e pensamentos mortos.
  """
  def cleanup_dead(state) do
    # Remove células mortas
    new_cells = state.cells
    |> Enum.map(fn {avatar_id, cells} ->
      {avatar_id, Enum.reject(cells, & &1.state == :dead)}
    end)
    |> Enum.reject(fn {_id, cells} -> Enum.empty?(cells) end)
    |> Map.new()

    # Marca pensamentos sem portadores como mortos
    active_thought_ids = new_cells
    |> Enum.flat_map(fn {_id, cells} -> Enum.map(cells, & &1.thought_id) end)
    |> MapSet.new()

    new_thoughts = state.thoughts
    |> Enum.map(fn {thought_id, thought} ->
      if thought_id not in active_thought_ids do
        {thought_id, %{thought | state: :dead}}
      else
        {thought_id, thought}
      end
    end)
    |> Map.new()

    %{state | cells: new_cells, thoughts: new_thoughts}
  end

  ## Private Functions

  defp count_carriers(state) do
    state.cells
    |> Enum.flat_map(fn {_avatar_id, cells} ->
      cells
      |> Enum.filter(& &1.state == :alive)
      |> Enum.map(& &1.thought_id)
    end)
    |> Enum.frequencies()
  end

  defp group_similar_thoughts(state, threshold) do
    # Coleta todos os pensamentos ativos com seus avatares
    all_cells = state.cells
    |> Enum.flat_map(fn {avatar_id, cells} ->
      cells
      |> Enum.filter(& &1.state == :alive)
      |> Enum.map(&{avatar_id, &1})
    end)

    # Agrupa por similaridade de conteúdo
    Enum.reduce(all_cells, %{}, fn {avatar_id, cell}, groups ->
      content = cell.content

      # Encontra grupo similar existente
      matching_group = Enum.find(groups, fn {group_content, _} ->
        Mutator.similar?(content, group_content, threshold)
      end)

      case matching_group do
        {group_content, members} ->
          Map.put(groups, group_content, [{avatar_id, cell} | members])

        nil ->
          Map.put(groups, content, [{avatar_id, cell}])
      end
    end)
  end

  defp force_mutation(content) do
    # Mutação forçada mais agressiva
    mutations = [
      "Repensando: #{content}",
      "Na verdade, #{String.downcase(content)} de outro ângulo",
      "#{content}... mas será mesmo?",
      "Contraditoriamente, #{content}",
      "Uma nova perspectiva: #{content}"
    ]
    Enum.random(mutations)
  end

  defp find_avatars_without_active_thoughts(state) do
    # IDs de avatares com pensamentos ativos
    with_thoughts = state.cells
    |> Enum.filter(fn {_id, cells} ->
      Enum.any?(cells, & &1.state == :alive)
    end)
    |> Enum.map(&elem(&1, 0))
    |> MapSet.new()

    # Todos os avatares conhecidos (dos pensamentos existentes)
    all_known = state.propagation_graph.nodes
    |> Map.keys()
    |> MapSet.new()

    MapSet.difference(all_known, with_thoughts)
    |> MapSet.to_list()
  end

  defp get_neighbor_thoughts(avatar_id, state) do
    # Busca vizinhos no grafo de propagação
    neighbors = state.propagation_graph.edges
    |> Enum.filter(fn edge ->
      edge.from == avatar_id or edge.to == avatar_id
    end)
    |> Enum.map(fn edge ->
      if edge.from == avatar_id, do: edge.to, else: edge.from
    end)
    |> Enum.uniq()

    # Coleta pensamentos dos vizinhos
    neighbors
    |> Enum.flat_map(fn neighbor_id ->
      cells = Map.get(state.cells, neighbor_id, [])
      Enum.filter(cells, & &1.state == :alive)
    end)
    |> Enum.map(& &1.content)
  end

  defp group_by_similarity(contents, threshold) do
    Enum.reduce(contents, %{}, fn content, groups ->
      matching = Enum.find(groups, fn {group_content, _} ->
        Mutator.similar?(content, group_content, threshold)
      end)

      case matching do
        {group_content, count} ->
          Map.put(groups, group_content, count + 1)

        nil ->
          Map.put(groups, content, 1)
      end
    end)
  end

  defp find_birth_candidate(groups) do
    groups
    |> Enum.max_by(fn {_content, count} -> count end, fn -> nil end)
    |> case do
      nil -> nil
      {content, count} -> {[content], count}
    end
  end

  defp create_emergent_thought(state, avatar_id, contents) do
    # Fusão dos conteúdos similares
    base_content = List.first(contents)
    emergent_content = "Emergente: #{base_content}"

    thought_id = :crypto.strong_rand_bytes(8) |> Base.encode16(case: :lower)

    thought = %HiveMind.CollectiveThought{
      id: thought_id,
      content: emergent_content,
      origin_avatar_id: avatar_id,
      created_at: DateTime.utc_now(),
      generation_born: state.generation,
      mutations: [],
      carrier_count: 1,
      state: :alive
    }

    cell = ThoughtCell.new(avatar_id, thought_id, emergent_content, state.generation)

    new_thoughts = Map.put(state.thoughts, thought_id, thought)
    new_cells = Map.update(state.cells, avatar_id, [cell], &[cell | &1])

    %{state | thoughts: new_thoughts, cells: new_cells}
  end

  defp short_id(id) when is_binary(id), do: String.slice(id, 0..7)
  defp short_id(id), do: inspect(id)
end
