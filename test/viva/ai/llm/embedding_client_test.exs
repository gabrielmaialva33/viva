defmodule Viva.AI.LLM.EmbeddingClientTest do
  use ExUnit.Case, async: true

  alias Viva.AI.LLM.EmbeddingClient

  describe "similarity/2" do
    test "returns 1.0 for identical embeddings" do
      embedding = [0.5, 0.5, 0.5, 0.5]
      assert_in_delta EmbeddingClient.similarity(embedding, embedding), 1.0, 0.001
    end

    test "returns 0.0 for orthogonal embeddings" do
      embedding_a = [1.0, 0.0, 0.0, 0.0]
      embedding_b = [0.0, 1.0, 0.0, 0.0]
      assert_in_delta EmbeddingClient.similarity(embedding_a, embedding_b), 0.0, 0.001
    end

    test "returns -1.0 for opposite embeddings" do
      embedding_a = [1.0, 0.0, 0.0, 0.0]
      embedding_b = [-1.0, 0.0, 0.0, 0.0]
      assert_in_delta EmbeddingClient.similarity(embedding_a, embedding_b), -1.0, 0.001
    end

    test "returns value between -1 and 1 for arbitrary embeddings" do
      embedding_a = [0.1, 0.2, 0.3, 0.4]
      embedding_b = [0.4, 0.3, 0.2, 0.1]

      similarity = EmbeddingClient.similarity(embedding_a, embedding_b)
      assert similarity >= -1.0
      assert similarity <= 1.0
    end

    test "returns 0.0 for zero vectors" do
      embedding_a = [0.0, 0.0, 0.0]
      embedding_b = [1.0, 2.0, 3.0]

      assert EmbeddingClient.similarity(embedding_a, embedding_b) == 0.0
    end

    test "is commutative" do
      embedding_a = [0.1, 0.5, 0.3]
      embedding_b = [0.4, 0.2, 0.6]

      assert_in_delta(
        EmbeddingClient.similarity(embedding_a, embedding_b),
        EmbeddingClient.similarity(embedding_b, embedding_a),
        0.0001
      )
    end

    test "handles high-dimensional embeddings" do
      embedding_a = Enum.map(1..4096, fn i -> :math.sin(i / 100) end)
      embedding_b = Enum.map(1..4096, fn i -> :math.cos(i / 100) end)

      similarity = EmbeddingClient.similarity(embedding_a, embedding_b)
      assert is_float(similarity)
      assert similarity >= -1.0
      assert similarity <= 1.0
    end
  end

  describe "dimension/0" do
    test "returns the expected embedding dimension" do
      dimension = EmbeddingClient.dimension()

      assert dimension == 4096
      assert is_integer(dimension)
      assert dimension > 0
    end
  end

  describe "module structure" do
    test "exports expected functions" do
      functions = EmbeddingClient.__info__(:functions)

      assert {:embed, 1} in functions or {:embed, 2} in functions
      assert {:embed_batch, 1} in functions or {:embed_batch, 2} in functions
      assert {:similarity, 2} in functions
      assert {:find_similar, 2} in functions or {:find_similar, 3} in functions
      assert {:embed_memory, 1} in functions or {:embed_memory, 2} in functions
      assert {:search_memories, 2} in functions or {:search_memories, 3} in functions
      assert {:dimension, 0} in functions
    end
  end
end
