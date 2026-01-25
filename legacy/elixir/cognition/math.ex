defmodule VivaCore.Cognition.Math do
  @moduledoc """
  The Algebra of Thought.

  This module provides the mathematical primitives for VIVA's cognitive operations.
  It treats thoughts as high-dimensional vectors and relationships as geometric transformations.

  "If words are vectors, thinking is geometry." - Gabriel Maia/Gemini

  ## Backend
  Uses `Nx` (Numerical Elixir) for efficient tensor operations.
  """

  import Nx.Defn

  @doc """
  Calculates the Cosine Similarity between two concepts.
  Returns a float between -1.0 (opposite) and 1.0 (identical).

  ## Examples
      iex> Math.similarity(v1, v2)
      0.85
  """
  def similarity(v1, v2) do
    t1 = ensure_tensor(v1)
    t2 = ensure_tensor(v2)
    cosine_similarity(t1, t2) |> Nx.to_number()
  end

  @doc """
  Synthesizes two concepts (Vector Addition).
  concept(A) + concept(B)
  """
  def add(v1, v2) do
    t1 = ensure_tensor(v1)
    t2 = ensure_tensor(v2)
    Nx.add(t1, t2)
  end

  @doc """
  Removes context from a concept (Vector Subtraction).
  concept(A) - concept(B)
  """
  def sub(v1, v2) do
    t1 = ensure_tensor(v1)
    t2 = ensure_tensor(v2)
    Nx.subtract(t1, t2)
  end

  @doc """
  Solves an analogy: A is to B as C is to ?
  Formula: B - A + C
  Example: King - Man + Woman = Queen
  """
  def analogy(a, b, c) do
    ta = ensure_tensor(a)
    tb = ensure_tensor(b)
    tc = ensure_tensor(c)

    # B - A + C
    tb
    |> Nx.subtract(ta)
    |> Nx.add(tc)
  end

  # --- Internal Nx Defns (Compiled) ---

  defn cosine_similarity(t1, t2) do
    dot_product = Nx.dot(t1, t2)
    # Manual Euclidean Norm: sqrt(sum(x^2))
    norm1 = Nx.sqrt(Nx.sum(Nx.pow(t1, 2)))
    norm2 = Nx.sqrt(Nx.sum(Nx.pow(t2, 2)))
    # epsilon for stability
    dot_product / (norm1 * norm2 + 1.0e-10)
  end

  # --- Helpers ---

  defp ensure_tensor(v) when is_list(v), do: Nx.tensor(v)
  defp ensure_tensor(%Nx.Tensor{} = t), do: t

  defp ensure_tensor(other),
    do: raise(ArgumentError, "Expected list or tensor, got: #{inspect(other)}")
end
