defmodule VivaCore.Cognition.Concept do
  @moduledoc """
  Represents a semantic concept in VIVA's mind.
  Wraps the raw vector with its debuggable meaning (string).
  """

  defstruct [:content, :vector]

  @type t :: %__MODULE__{
          content: String.t(),
          vector: Nx.Tensor.t() | nil
        }

  def new(content, vector) do
    %__MODULE__{content: content, vector: vector}
  end

  def inspect(concept) do
    "~d\"#{concept.content}\""
  end
end
