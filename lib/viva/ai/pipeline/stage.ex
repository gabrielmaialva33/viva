defmodule Viva.AI.Pipeline.Stage do
  @moduledoc """
  Behaviour for AI pipeline stages.
  """

  @callback process(input :: any(), opts :: keyword()) :: {:ok, any()} | {:error, term()}

  @optional_callbacks process: 2
end
