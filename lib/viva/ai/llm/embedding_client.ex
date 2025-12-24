defmodule Viva.AI.LLM.EmbeddingClient do
  @moduledoc """
  Embedding client using NVIDIA NV-EmbedQA.

  Uses `nvidia/nv-embedqa-mistral-7b-v2` for high-quality multilingual
  text embeddings for semantic search and RAG.

  ## Features

  - Multilingual embeddings (26+ languages)
  - Semantic similarity search
  - Document retrieval
  - Memory embedding for avatars
  """
  require Logger

  alias Viva.Nim
  alias Viva.AI.LLM.EmbeddingClient, as: Client

  @behaviour Viva.AI.Pipeline.Stage

  # === Types ===

  @type embedding :: [float()]
  @type similarity_result :: %{text: String.t(), similarity: float()}
  @type memory_result :: %{memory: map(), relevance: float()}

  @doc """
  Generate embeddings for text.

  ## Options

  - `:input_type` - "query" for questions, "passage" for documents (default: "query")
  - `:truncate` - Truncation strategy: "NONE", "START", "END" (default: "END")
  """
  @spec embed(String.t(), keyword()) :: {:ok, embedding()} | {:error, term()}
  def embed(text, opts \\ []) when is_binary(text) do
    case embed_batch([text], opts) do
      {:ok, [embedding]} -> {:ok, embedding}
      {:ok, []} -> {:error, :empty_result}
      error -> error
    end
  end

  @doc """
  Generate embeddings for multiple texts in batch.
  More efficient than calling embed/2 multiple times.
  """
  @spec embed_batch([String.t()], keyword()) :: {:ok, [embedding()]} | {:error, term()}
  def embed_batch(texts, opts \\ []) when is_list(texts) do
    model = Keyword.get(opts, :model, Viva.AI.LLM.model(:embedding))
    input_type = Keyword.get(opts, :input_type, "query")

    body = %{
      model: model,
      input: texts,
      input_type: input_type,
      truncate: Keyword.get(opts, :truncate, "END")
    }

    case Viva.AI.LLM.request("/embeddings", body) do
      {:ok, %{"data" => data}} ->
        embeddings = Enum.map(data, fn %{"embedding" => emb} -> emb end)
        {:ok, embeddings}

      {:error, reason} ->
        Logger.error("Embedding error: #{inspect(reason)}")
        {:error, reason}
    end
  end

  @doc """
  Calculate cosine similarity between two embeddings.
  """
  @spec similarity(embedding(), embedding()) :: float()
  def similarity(embedding_a, embedding_b) do
    dot_product =
      embedding_a
      |> Enum.zip(embedding_b)
      |> Enum.reduce(0.0, fn {a, b}, acc -> acc + a * b end)

    sum_a = Enum.reduce(embedding_a, 0.0, fn x, acc -> acc + x * x end)
    sum_b = Enum.reduce(embedding_b, 0.0, fn x, acc -> acc + x * x end)
    norm_a = :math.sqrt(sum_a)
    norm_b = :math.sqrt(sum_b)

    if norm_a > 0 and norm_b > 0 do
      dot_product / (norm_a * norm_b)
    else
      0.0
    end
  end

  @doc """
  Find the most similar texts from a list.

  ## Options

  - `:top_k` - Number of results to return (default: 5)
  - `:threshold` - Minimum similarity threshold (default: 0.0)
  """
  @spec find_similar(String.t(), [String.t()], keyword()) ::
          {:ok, [similarity_result()]} | {:error, term()}
  def find_similar(query, texts, opts \\ []) do
    top_k = Keyword.get(opts, :top_k, 5)
    threshold = Keyword.get(opts, :threshold, 0.0)

    with {:ok, query_embedding} <- embed(query, input_type: "query"),
         {:ok, text_embeddings} <- embed_batch(texts, input_type: "passage") do
      results =
        texts
        |> Enum.zip(text_embeddings)
        |> Enum.map(fn {text, embedding} ->
          %{text: text, similarity: similarity(query_embedding, embedding)}
        end)
        |> Enum.filter(fn %{similarity: sim} -> sim >= threshold end)
        |> Enum.sort_by(& &1.similarity, :desc)
        |> Enum.take(top_k)

      {:ok, results}
    end
  end

  @doc """
  Embed an avatar memory for storage.
  """
  @spec embed_memory(String.t(), map()) :: {:ok, embedding()} | {:error, term()}
  def embed_memory(memory_content, context \\ %{}) do
    enhanced_content =
      case context do
        %{avatar_name: name, emotion: emotion} ->
          "#{name} (feeling #{emotion}): #{memory_content}"

        %{avatar_name: name} ->
          "#{name}: #{memory_content}"

        _ ->
          memory_content
      end

    embed(enhanced_content, input_type: "passage")
  end

  @doc """
  Search avatar memories by semantic similarity.
  """
  @spec search_memories(String.t(), [map()], keyword()) ::
          {:ok, [memory_result()]} | {:error, term()}
  def search_memories(query, memories, opts \\ []) do
    top_k = Keyword.get(opts, :top_k, 10)
    threshold = Keyword.get(opts, :threshold, 0.5)

    {texts, memory_data} =
      memories
      |> Enum.map(fn memory -> {memory.content, memory} end)
      |> Enum.unzip()

    case find_similar(query, texts, top_k: top_k, threshold: threshold) do
      {:ok, results} ->
        # Map back to memory structs with similarity scores
        matched_memories =
          Enum.map(results, fn %{text: text, similarity: sim} ->
            memory = Enum.find(memory_data, fn m -> m.content == text end)
            %{memory: memory, relevance: sim}
          end)

        {:ok, matched_memories}

      error ->
        error
    end
  end

  @doc """
  Get embedding dimension for the current model.
  """
  @spec dimension() :: pos_integer()
  def dimension do
    # nv-embedqa-mistral-7b-v2 produces 4096-dimensional embeddings
    4096
  end
end
