defmodule VivaCore.Qdrant do
  @moduledoc """
  Qdrant vector database client for VIVA's memory system.

  Provides:
  - Collection management
  - Point storage with embeddings
  - Semantic search with temporal decay
  """

  require Logger

  @base_url "http://localhost:6333"
  @collection "viva_memories"
  # NVIDIA nv-embedqa-e5-v5 dimension
  @vector_size 1024

  # ============================================================================
  # Collection Management
  # ============================================================================

  @doc """
  Ensures the VIVA memories collection exists.
  Creates it with proper schema if missing.
  """
  def ensure_collection do
    case get_collection() do
      {:ok, _} ->
        Logger.debug("[Qdrant] Collection '#{@collection}' exists")
        :ok

      {:error, :not_found} ->
        create_collection()

      {:error, reason} ->
        {:error, reason}
    end
  end

  def get_collection do
    case Req.get("#{@base_url}/collections/#{@collection}") do
      {:ok, %{status: 200, body: body}} ->
        {:ok, body}

      {:ok, %{status: 404}} ->
        {:error, :not_found}

      {:error, reason} ->
        {:error, reason}
    end
  end

  def create_collection do
    body = %{
      vectors: %{
        size: @vector_size,
        distance: "Cosine"
      }
    }

    case Req.put("#{@base_url}/collections/#{@collection}", json: body) do
      {:ok, %{status: 200}} ->
        Logger.info("[Qdrant] Created collection '#{@collection}'")
        # Create payload indexes for efficient filtering
        create_payload_indexes()
        :ok

      {:ok, %{status: status, body: body}} ->
        Logger.error("[Qdrant] Failed to create collection: #{status} - #{inspect(body)}")
        {:error, body}

      {:error, reason} ->
        {:error, reason}
    end
  end

  defp create_payload_indexes do
    # Index for timestamp (for decay queries)
    Req.put("#{@base_url}/collections/#{@collection}/index",
      json: %{field_name: "timestamp", field_schema: "datetime"}
    )

    # Index for memory type
    Req.put("#{@base_url}/collections/#{@collection}/index",
      json: %{field_name: "type", field_schema: "keyword"}
    )

    # Index for importance
    Req.put("#{@base_url}/collections/#{@collection}/index",
      json: %{field_name: "importance", field_schema: "float"}
    )

    :ok
  end

  # ============================================================================
  # Point Operations
  # ============================================================================

  @doc """
  Stores a memory point with embedding and metadata.

  ## Payload Schema
  - content: string (the actual memory text)
  - type: :episodic | :semantic | :emotional
  - importance: float 0.0-1.0
  - emotion: %{pleasure, arousal, dominance}
  - timestamp: ISO8601 datetime
  - access_count: integer (for spaced repetition)
  - last_accessed: ISO8601 datetime
  """
  def upsert_point(id, embedding, payload) do
    # Ensure timestamp is ISO8601 string
    payload =
      Map.update(payload, :timestamp, DateTime.utc_now() |> DateTime.to_iso8601(), fn
        %DateTime{} = dt -> DateTime.to_iso8601(dt)
        str when is_binary(str) -> str
        _ -> DateTime.utc_now() |> DateTime.to_iso8601()
      end)

    body = %{
      points: [
        %{
          id: id,
          vector: embedding,
          payload: payload
        }
      ]
    }

    # wait: true ensures point is indexed before returning (consistency)
    case Req.put("#{@base_url}/collections/#{@collection}/points?wait=true", json: body) do
      {:ok, %{status: 200}} ->
        {:ok, id}

      {:ok, %{status: status, body: resp}} ->
        Logger.error("[Qdrant] Upsert failed: #{status} - #{inspect(resp)}")
        {:error, resp}

      {:error, reason} ->
        {:error, reason}
    end
  end

  @doc """
  Searches for similar memories with temporal decay.

  Uses Qdrant's formula-based scoring:
  final_score = similarity + exp_decay(timestamp)

  Options:
  - limit: max results (default 10)
  - decay_scale: seconds for decay (default 604800 = 1 week)
  - decay_midpoint: score at scale distance (default 0.5)
  - filter: additional Qdrant filter
  """
  def search_with_decay(query_vector, opts \\ []) do
    limit = Keyword.get(opts, :limit, 10)
    # 1 week
    decay_scale = Keyword.get(opts, :decay_scale, 604_800)
    decay_midpoint = Keyword.get(opts, :decay_midpoint, 0.5)
    filter = Keyword.get(opts, :filter, nil)
    now = DateTime.utc_now() |> DateTime.to_iso8601()

    # Build query with exp_decay formula
    body = %{
      prefetch: %{
        query: query_vector,
        # prefetch more for decay filtering
        limit: limit * 2
      },
      query: %{
        formula: %{
          sum: [
            # original similarity
            "$score",
            %{
              mult: [
                # weight for time decay (30%)
                0.3,
                %{
                  exp_decay: %{
                    x: %{datetime_key: "timestamp"},
                    target: %{datetime: now},
                    scale: decay_scale,
                    midpoint: decay_midpoint
                  }
                }
              ]
            }
          ]
        }
      },
      limit: limit,
      with_payload: true
    }

    # Add filter if provided
    body = if filter, do: Map.put(body, :filter, filter), else: body

    case Req.post("#{@base_url}/collections/#{@collection}/points/query", json: body) do
      {:ok, %{status: 200, body: %{"result" => %{"points" => points}}}} ->
        {:ok, Enum.map(points, &parse_point/1)}

      {:ok, %{status: status, body: resp}} ->
        Logger.warning("[Qdrant] Search failed: #{status} - #{inspect(resp)}")
        # Fallback to simple search
        search_simple(query_vector, limit)

      {:error, reason} ->
        {:error, reason}
    end
  end

  @doc """
  Simple vector search without decay (fallback).
  """
  def search_simple(query_vector, limit \\ 10) do
    body = %{
      vector: query_vector,
      limit: limit,
      with_payload: true
    }

    case Req.post("#{@base_url}/collections/#{@collection}/points/search", json: body) do
      {:ok, %{status: 200, body: %{"result" => points}}} ->
        {:ok, Enum.map(points, &parse_point/1)}

      {:ok, %{status: status, body: resp}} ->
        Logger.error("[Qdrant] Simple search failed: #{status}")
        {:error, resp}

      {:error, reason} ->
        {:error, reason}
    end
  end

  @doc """
  Gets a specific point by ID.
  """
  def get_point(id) do
    case Req.get("#{@base_url}/collections/#{@collection}/points/#{id}") do
      {:ok, %{status: 200, body: %{"result" => point}}} ->
        {:ok, parse_point(point)}

      {:ok, %{status: 404}} ->
        {:error, :not_found}

      {:error, reason} ->
        {:error, reason}
    end
  end

  @doc """
  Deletes a point by ID (for explicit forgetting).
  """
  def delete_point(id) do
    body = %{points: [id]}

    case Req.post("#{@base_url}/collections/#{@collection}/points/delete", json: body) do
      {:ok, %{status: 200}} ->
        :ok

      {:error, reason} ->
        {:error, reason}
    end
  end

  @doc """
  Updates point payload (e.g., increment access_count).
  """
  def update_payload(id, payload_updates) do
    body = %{
      points: [id],
      payload: payload_updates
    }

    case Req.post("#{@base_url}/collections/#{@collection}/points/payload", json: body) do
      {:ok, %{status: 200}} ->
        :ok

      {:error, reason} ->
        {:error, reason}
    end
  end

  @doc """
  Gets collection stats.
  """
  def stats do
    case Req.get("#{@base_url}/collections/#{@collection}") do
      {:ok, %{status: 200, body: %{"result" => result}}} ->
        {:ok,
         %{
           points_count: get_in(result, ["points_count"]) || 0,
           vectors_count: get_in(result, ["vectors_count"]) || 0,
           status: get_in(result, ["status"])
         }}

      {:error, reason} ->
        {:error, reason}
    end
  end

  # ============================================================================
  # Private Helpers
  # ============================================================================

  defp parse_point(%{"id" => id, "payload" => payload, "score" => score}) do
    %{
      id: id,
      content: payload["content"],
      type: parse_memory_type(payload["type"]),
      importance: payload["importance"] || 0.5,
      emotion: payload["emotion"],
      timestamp: payload["timestamp"],
      access_count: payload["access_count"] || 0,
      similarity: score
    }
  end

  defp parse_point(%{"id" => id, "payload" => payload}) do
    parse_point(%{"id" => id, "payload" => payload, "score" => nil})
  end

  defp parse_point(other), do: other

  defp parse_memory_type("episodic"), do: :episodic
  defp parse_memory_type("semantic"), do: :semantic
  defp parse_memory_type("emotional"), do: :emotional
  defp parse_memory_type("procedural"), do: :procedural
  defp parse_memory_type(_), do: :generic
end
