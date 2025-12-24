defmodule Viva.Avatars.Memory.Engine do
  @moduledoc """
  The Cognitive Engine responsible for Memory Retrieval and Consolidation.
  Implements the Recency + Importance + Relevance scoring algorithm.
  """

  import Ecto.Query
  import Pgvector.Ecto.Query
  alias Viva.Avatars.Memory
  alias Viva.AI.LLM.EmbeddingClient
  alias Viva.Repo

  # Weights for the Retrieval Function (Tuned for balanced recall)
  @weight_recency 0.5
  @weight_importance 0.3
  # Relevance is king in vector databases
  @weight_relevance 1.5

  # Decay factor per hour (Exponential decay)
  @decay_factor 0.99

  @doc """
  Retrieves the most relevant memories for a given context query.
  Does NOT just rely on Vector Search; it re-ranks based on biological recency.
  """
  @spec retrieve_relevant(Ecto.UUID.t(), String.t(), integer()) :: [map()]
  def retrieve_relevant(avatar_id, query_text, limit \\ 5) do
    # 1. Get Query Embedding
    {:ok, query_vector} = EmbeddingClient.embed(query_text)

    # 2. Vector Search via Qdrant (Simulated via Postgres/pgvector for now or direct Qdrant call)
    # Ideally, we fetch top 50 candidates by semantic similarity first
    candidates = search_vectors(avatar_id, query_vector, 50)

    # 3. Re-rank based on Cognitive Score
    current_time = DateTime.utc_now()

    candidates
    |> Enum.map(fn mem ->
      score = calculate_score(mem, query_vector, current_time)
      Map.put(mem, :retrieval_score, score)
    end)
    |> Enum.sort_by(& &1.retrieval_score, :desc)
    |> Enum.take(limit)
  end

  @doc """
  The Master Formula: Score = (Recency * w) + (Importance * w) + (Relevance * w)
  """
  @spec calculate_score(map(), list(), DateTime.t()) :: float()
  def calculate_score(memory, _, current_time) do
    # 1. Recency: Exponential decay based on hours passed
    hours_passed = DateTime.diff(current_time, memory.inserted_at, :hour)
    recency_score = :math.pow(@decay_factor, hours_passed)

    # 2. Importance: Intrinsic weight (0 to 1) normalized
    importance_score = memory.importance

    # 3. Relevance: In a real Qdrant integration, this is the cosine similarity score.
    # For now, we assume the candidates came pre-filtered or we calc it if needed.
    # We will assume a placeholder 0.8 average if using SQL sort, or use real distance.
    # Placeholder for pure math demonstration
    relevance_score = 0.8

    @weight_recency * recency_score +
      @weight_importance * importance_score +
      @weight_relevance * relevance_score
  end

  defp search_vectors(avatar_id, vector, limit) do
    # Using pgvector for simplicity within the monolith structure
    Memory
    |> where([m], m.avatar_id == ^avatar_id)
    |> order_by([m], l2_distance(m.embedding, ^vector))
    |> limit(^limit)
    |> Repo.all()
  end
end
