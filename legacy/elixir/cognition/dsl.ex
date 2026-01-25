defmodule VivaCore.Cognition.DSL do
  @moduledoc """
  The Language of Thought.

  Provides the `~d` sigil and operator overloading for cognitive algebra.

  Usage:
      use VivaCore.Cognition.DSL

      ~d"King" - ~d"Man" + ~d"Woman"
      # => ~d"Queen" (approx)
  """

  alias VivaCore.Cognition.{Concept, Math}

  defmacro __using__(_opts) do
    quote do
      import Kernel, except: [+: 2, -: 2]
      import VivaCore.Cognition.DSL
    end
  end

  @doc """
  Concept Sigil.
  Resolves the concept string to a vector embedding using Ultra.
  """
  defmacro sigil_d(content, _opts) do
    # At compile time, we just return the AST if possible, or runtime call.
    # Sigils are expanded at compile time, but we want runtime execution for the embedding logic
    # unless we precompute. We want runtime.
    quote do
      embed = VivaBridge.Ultra.embed(unquote(content))

      vector =
        case embed do
          {:ok, list} -> Nx.tensor(list)
          # Or raise?
          _ -> nil
        end

      Concept.new(unquote(content), vector)
    end
  end

  # --- Operator Overloading ---

  def extract_tensor(%Concept{vector: v}), do: v
  def extract_tensor(%Nx.Tensor{} = t), do: t
  def extract_tensor(_), do: nil

  def a + b do
    ta = extract_tensor(a)
    tb = extract_tensor(b)

    if ta && tb do
      result_tensor = Math.add(ta, tb)
      Concept.new("Synthesized Concept", result_tensor)
    else
      Kernel.+(a, b)
    end
  end

  def a - b do
    ta = extract_tensor(a)
    tb = extract_tensor(b)

    if ta && tb do
      result_tensor = Math.sub(ta, tb)
      Concept.new("Analogy Result", result_tensor)
    else
      Kernel.-(a, b)
    end
  end
end
