defmodule Viva.Social.NetworkAnalyst do
  @moduledoc """
  Background process that analyzes the social graph to calculate metrics.
  Updates Avatar.social_persona.public_reputation based on graph theory.

  Metrics:
  - Degree Centrality (Popularity)
  - PageRank (Status Influence - being liked by popular people matters more)
  """
  use GenServer

  require Logger

  alias Viva.Avatars.Avatar
  alias Viva.Relationships.Relationship
  alias Viva.Repo

  @interval :timer.minutes(5)

  @spec start_link(any()) :: GenServer.on_start()
  def start_link(_) do
    GenServer.start_link(__MODULE__, %{}, name: __MODULE__)
  end

  @spec init(any()) :: {:ok, map()}
  def init(_) do
    Logger.info("Social Network Analyst started.")
    schedule_analysis()
    {:ok, %{}}
  end

  @spec handle_info(:analyze, map()) :: {:noreply, map()}
  def handle_info(:analyze, state) do
    Logger.info("Running social graph analysis...")
    analyze_network()
    schedule_analysis()
    {:noreply, state}
  end

  defp schedule_analysis do
    Process.send_after(self(), :analyze, @interval)
  end

  defp analyze_network do
    # 1. Fetch all active relationships
    relationships = Repo.all(Relationship.active())

    # 2. Build Adjacency List (Directed Graph for Status)
    # A -> B weight = A's admiration/trust for B
    graph = build_graph(relationships)

    # 3. Calculate PageRank (Simulated)
    # Since we don't have a graph lib, we use a simplified iterative approach
    scores = calculate_pagerank(graph)

    # 4. Update Avatars
    update_avatars(scores)
  end

  defp build_graph(relationships) do
    Enum.reduce(relationships, %{}, fn rel, acc ->
      # Add edge A -> B (A likes B)
      acc =
        Map.update(acc, rel.avatar_a_id, [rel.avatar_b_id], fn list -> [rel.avatar_b_id | list] end)

      # Add edge B -> A (B likes A)
      Map.update(acc, rel.avatar_b_id, [rel.avatar_a_id], fn list -> [rel.avatar_a_id | list] end)
    end)
  end

  defp calculate_pagerank(graph, iterations \\ 5, damping \\ 0.85) do
    # Simplified PageRank implementation
    nodes = Map.keys(graph)
    total_nodes = length(nodes)

    if total_nodes == 0, do: %{}, else: do_calculate(nodes, graph, iterations, damping, total_nodes)
  end

  defp do_calculate(nodes, graph, iterations, damping, total_nodes) do
    initial_score = 1.0 / total_nodes
    scores = Map.new(nodes, fn n -> {n, initial_score} end)

    Enum.reduce(1..iterations, scores, fn _, current_scores ->
      update_scores(nodes, graph, current_scores, damping, total_nodes)
    end)
  end

  defp update_scores(nodes, graph, current_scores, damping, total_nodes) do
    Enum.reduce(nodes, %{}, fn node, acc ->
      incoming_score = calculate_incoming_score(node, nodes, graph, current_scores)
      new_rank = (1 - damping) / total_nodes + damping * incoming_score
      Map.put(acc, node, new_rank)
    end)
  end

  defp calculate_incoming_score(node, nodes, graph, current_scores) do
    Enum.reduce(nodes, 0.0, fn potential_source, sum ->
      out_links = Map.get(graph, potential_source, [])

      if node in out_links do
        sum + Map.get(current_scores, potential_source, 0.0) / length(out_links)
      else
        sum
      end
    end)
  end

  defp update_avatars(scores) do
    # Normalize scores 0.0 to 1.0
    {min, max} =
      if map_size(scores) > 0 do
        scores
        |> Map.values()
        |> Enum.min_max()
      else
        {0.0, 1.0}
      end

    diff = max - min
    scale = if diff == 0, do: 1.0, else: diff

    Enum.each(scores, fn {avatar_id, raw_score} ->
      normalized_score = (raw_score - min) / scale

      # Update DB
      # Use raw SQL or changeset for efficiency
      update_avatar_reputation(avatar_id, normalized_score)
    end)
  end

  defp update_avatar_reputation(avatar_id, score) do
    # We need to update the embedded schema field
    # Ecto makes deep updates tricky without fetching.
    # For now, we fetch and update.
    case Repo.get(Avatar, avatar_id) do
      nil ->
        :ok

      avatar ->
        new_persona = Ecto.Changeset.change(avatar.social_persona, public_reputation: score)

        avatar
        |> Ecto.Changeset.change()
        |> Ecto.Changeset.put_embed(:social_persona, new_persona)
        |> Repo.update()
    end
  end
end
