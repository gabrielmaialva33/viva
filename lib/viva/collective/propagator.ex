defmodule Viva.Collective.Propagator do
  @moduledoc """
  Engine de propagação de pensamentos: 1 → 2 → 4 → 8 → ...

  Seleciona alvos para propagação baseado em:
  - Proximidade social (afinidade)
  - Estado emocional compatível
  - Receptividade (energia, abertura)

  ## Estratégia de Seleção

  1. Busca vizinhos no grafo social
  2. Filtra por receptividade
  3. Ordena por afinidade
  4. Seleciona top N (geralmente 2)
  """

  require Logger

  alias Viva.Social.SocialGraph

  @doc """
  Seleciona N avatares para receber propagação.

  ## Parâmetros
  - `from_avatar_id`: Avatar que está propagando
  - `count`: Quantidade de alvos (default: 2)
  - `exclude`: Lista de IDs a excluir (já possuem o pensamento)

  ## Retorno
  Lista de avatar_ids selecionados
  """
  def select_targets(from_avatar_id, count \\ 2, exclude \\ []) do
    # Tenta buscar do grafo social real
    case get_social_neighbors(from_avatar_id) do
      neighbors when is_list(neighbors) and length(neighbors) > 0 ->
        neighbors
        |> Enum.reject(& &1 in exclude)
        |> Enum.reject(& &1 == from_avatar_id)
        |> score_and_rank(from_avatar_id)
        |> Enum.take(count)
        |> Enum.map(& &1.avatar_id)

      _ ->
        # Fallback: seleciona avatares aleatórios ativos
        select_random_targets(from_avatar_id, count, exclude)
    end
  end

  @doc """
  Calcula a probabilidade de um avatar aceitar um pensamento.
  Baseado em estado emocional e personalidade.
  """
  def receptivity_score(avatar_id) do
    case get_avatar_state(avatar_id) do
      nil ->
        0.5  # Default médio

      state ->
        # Fatores que aumentam receptividade:
        # - Energia alta (baixo adenosine = mais energia)
        # - Estado emocional positivo ou neutro

        # Access bio struct fields (structs need Map.get or direct access)
        bio = Map.get(state, :bio) || %{}
        bio_map = if is_struct(bio), do: Map.from_struct(bio), else: bio

        # Adenosine alto = cansado, baixo = energizado
        adenosine = Map.get(bio_map, :adenosine, 0.5)
        energy_factor = 1.0 - adenosine

        # Dopamine contribui para abertura a novas ideias
        dopamine = Map.get(bio_map, :dopamine, 0.5)
        openness_factor = dopamine

        # Estado emocional
        emotional_factor = calculate_emotional_receptivity(state)

        # Média ponderada
        (energy_factor * 0.3 + openness_factor * 0.4 + emotional_factor * 0.3)
        |> min(1.0)
        |> max(0.0)
    end
  rescue
    _ -> 0.5
  end

  @doc """
  Calcula afinidade entre dois avatares.
  """
  def affinity_score(avatar_id_1, avatar_id_2) do
    # Tenta buscar do grafo social
    case SocialGraph.get_relationship(avatar_id_1, avatar_id_2) do
      {:ok, relationship} ->
        relationship.affinity || 0.5

      _ ->
        # Calcula baseado em similaridade de estados
        calculate_state_similarity(avatar_id_1, avatar_id_2)
    end
  rescue
    _ -> 0.5
  end

  ## Private Functions

  defp get_social_neighbors(avatar_id) do
    case SocialGraph.get_neighbors(avatar_id) do
      {:ok, neighbors} -> neighbors
      _ -> []
    end
  rescue
    _ -> []
  end

  defp select_random_targets(from_avatar_id, count, exclude) do
    # Busca todos os avatares ativos
    case Viva.Avatars.list_active_avatar_ids() do
      ids when is_list(ids) ->
        ids
        |> Enum.reject(& &1 in exclude)
        |> Enum.reject(& &1 == from_avatar_id)
        |> Enum.shuffle()
        |> Enum.take(count)

      _ ->
        []
    end
  rescue
    _ -> []
  end

  defp score_and_rank(avatar_ids, from_avatar_id) do
    avatar_ids
    |> Enum.map(fn target_id ->
      receptivity = receptivity_score(target_id)
      affinity = affinity_score(from_avatar_id, target_id)

      # Score combinado: afinidade pesa mais
      score = affinity * 0.6 + receptivity * 0.4

      %{avatar_id: target_id, score: score, receptivity: receptivity, affinity: affinity}
    end)
    |> Enum.sort_by(& &1.score, :desc)
  end

  defp get_avatar_state(avatar_id) do
    case Viva.Sessions.LifeProcess.get_internal_state(avatar_id) do
      {:ok, state} -> state
      _ -> nil
    end
  rescue
    _ -> nil
  end

  defp calculate_emotional_receptivity(state) do
    # Estados que aumentam receptividade
    # Access nested emotional state (may be a struct)
    emotional = Map.get(state, :emotional) || %{}
    emotional_map = if is_struct(emotional), do: Map.from_struct(emotional), else: emotional

    pleasure = Map.get(emotional_map, :pleasure) || Map.get(emotional_map, :valence, 0)
    arousal = Map.get(emotional_map, :arousal) || Map.get(emotional_map, :activation, 0)

    # Alta ativação + valência positiva = mais receptivo
    # Baixa ativação + valência negativa = menos receptivo
    base = (pleasure + 1) / 2  # Normaliza -1..1 para 0..1

    # Ativação moderada é melhor para receptividade
    arousal_factor = 1.0 - abs(arousal) * 0.3

    base * arousal_factor
  rescue
    _ -> 0.5
  end

  defp calculate_state_similarity(id1, id2) do
    state1 = get_avatar_state(id1)
    state2 = get_avatar_state(id2)

    case {state1, state2} do
      {nil, _} -> 0.5
      {_, nil} -> 0.5
      {s1, s2} ->
        # Compara estados emocionais (may be structs)
        e1 = Map.get(s1, :emotional) || %{}
        e2 = Map.get(s2, :emotional) || %{}
        e1_map = if is_struct(e1), do: Map.from_struct(e1), else: e1
        e2_map = if is_struct(e2), do: Map.from_struct(e2), else: e2

        p1 = Map.get(e1_map, :pleasure, 0)
        p2 = Map.get(e2_map, :pleasure, 0)
        a1 = Map.get(e1_map, :arousal, 0)
        a2 = Map.get(e2_map, :arousal, 0)

        # Distância euclidiana normalizada
        dist = :math.sqrt(:math.pow(p1 - p2, 2) + :math.pow(a1 - a2, 2))
        max_dist = :math.sqrt(8)  # Max possível com range -1..1

        # Converte distância em similaridade
        1.0 - (dist / max_dist)
    end
  rescue
    _ -> 0.5
  end
end
