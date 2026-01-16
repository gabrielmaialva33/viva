defmodule VivaCore.Embedder do
  @moduledoc """
  Text embedding service for VIVA's memory system.

  Supports multiple backends:
  1. Ollama (local, if embedding model available)
  2. NVIDIA NIM (cloud, if API key configured)
  3. Hash-based (fallback for development)

  Default dimension: 384 (all-MiniLM-L6-v2 compatible)
  """

  require Logger

  @vector_size 384
  @ollama_url "http://localhost:11434"
  @nvidia_url "https://integrate.api.nvidia.com/v1"

  # ============================================================================
  # Public API
  # ============================================================================

  @doc """
  Generates an embedding vector for the given text.

  Returns {:ok, [float]} or {:error, reason}
  """
  def embed(text) when is_binary(text) do
    # Try backends in order of preference
    with {:error, _} <- embed_ollama(text),
         {:error, _} <- embed_nvidia(text) do
      # Fallback to hash-based embedding (deterministic, for dev)
      {:ok, embed_hash(text)}
    end
  end

  def embed(text) when is_atom(text), do: embed(Atom.to_string(text))
  def embed(text), do: embed(inspect(text))

  @doc """
  Embeds multiple texts in batch.
  """
  def embed_batch(texts) when is_list(texts) do
    results = Enum.map(texts, &embed/1)

    if Enum.all?(results, &match?({:ok, _}, &1)) do
      {:ok, Enum.map(results, fn {:ok, vec} -> vec end)}
    else
      {:error, :partial_failure}
    end
  end

  @doc """
  Returns the embedding dimension.
  """
  def dimension, do: @vector_size

  # ============================================================================
  # Ollama Backend
  # ============================================================================

  defp embed_ollama(text) do
    # Try nomic-embed-text first, then all-minilm
    models = ["nomic-embed-text", "all-minilm", "mxbai-embed-large"]

    Enum.reduce_while(models, {:error, :no_model}, fn model, _acc ->
      case try_ollama_model(model, text) do
        {:ok, embedding} -> {:halt, {:ok, embedding}}
        {:error, _} -> {:cont, {:error, :no_model}}
      end
    end)
  end

  defp try_ollama_model(model, text) do
    body = %{model: model, prompt: text}

    case Req.post("#{@ollama_url}/api/embeddings", json: body, receive_timeout: 30_000) do
      {:ok, %{status: 200, body: %{"embedding" => embedding}}} ->
        # Normalize to @vector_size if needed
        {:ok, normalize_vector(embedding)}

      {:ok, %{status: status}} ->
        {:error, {:ollama_error, status}}

      {:error, reason} ->
        {:error, reason}
    end
  end

  # ============================================================================
  # NVIDIA Backend
  # ============================================================================

  defp embed_nvidia(text) do
    api_key = System.get_env("NVIDIA_API_KEY")

    if api_key do
      body = %{
        model: "nvidia/nv-embedqa-e5-v5",
        input: [text],
        input_type: "query",
        encoding_format: "float"
      }

      headers = [
        {"Authorization", "Bearer #{api_key}"},
        {"Content-Type", "application/json"}
      ]

      case Req.post("#{@nvidia_url}/embeddings", json: body, headers: headers, receive_timeout: 30_000) do
        {:ok, %{status: 200, body: %{"data" => [%{"embedding" => embedding} | _]}}} ->
          {:ok, normalize_vector(embedding)}

        {:ok, %{status: status, body: body}} ->
          Logger.debug("[Embedder] NVIDIA failed: #{status} - #{inspect(body)}")
          {:error, {:nvidia_error, status}}

        {:error, reason} ->
          {:error, reason}
      end
    else
      {:error, :no_api_key}
    end
  end

  # ============================================================================
  # Hash-based Fallback (Development)
  # ============================================================================

  @doc """
  Generates a deterministic pseudo-embedding from text hash.

  NOT for production - only for testing without embedding service.
  Semantic similarity won't work, but storage/retrieval will.
  """
  def embed_hash(text) do
    # Use SHA256 hash as seed for deterministic random vector
    hash = :crypto.hash(:sha256, text)
    <<seed::unsigned-64, _rest::binary>> = hash

    # Generate deterministic "random" vector using local state (avoids global seed pollution)
    initial_state = :rand.seed_s(:exsss, {seed, seed, seed})

    {vec, _final_state} =
      Enum.map_reduce(1..@vector_size, initial_state, fn _, state ->
        {val, new_state} = :rand.uniform_s(state)
        {val * 2 - 1, new_state}  # Range [-1, 1]
      end)

    normalize_l2(vec)
  end

  # ============================================================================
  # Vector Utilities
  # ============================================================================

  defp normalize_vector(vec) when length(vec) == @vector_size do
    normalize_l2(vec)
  end

  defp normalize_vector(vec) when length(vec) > @vector_size do
    Logger.warning("[Embedder] Truncating #{length(vec)}D vector to #{@vector_size}D - consider using matching model")
    vec |> Enum.take(@vector_size) |> normalize_l2()
  end

  defp normalize_vector(vec) when length(vec) < @vector_size do
    Logger.warning("[Embedder] Padding #{length(vec)}D vector to #{@vector_size}D - semantic quality degraded")
    # Pad with zeros (not ideal, but maintains determinism)
    padding = List.duplicate(0.0, @vector_size - length(vec))
    (vec ++ padding) |> normalize_l2()
  end

  defp normalize_l2(vec) do
    magnitude = :math.sqrt(Enum.reduce(vec, 0, fn x, acc -> acc + x * x end))

    if magnitude > 0 do
      Enum.map(vec, &(&1 / magnitude))
    else
      vec
    end
  end
end
