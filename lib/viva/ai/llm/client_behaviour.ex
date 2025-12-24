defmodule Viva.AI.LLM.ClientBehaviour do
  @moduledoc """
  Behaviour for LLM Clients to enable mocking.
  """

  @type message :: %{role: String.t(), content: String.t()}
  @type response :: {:ok, String.t()} | {:error, term()}

  @callback generate(String.t(), keyword()) :: response
  @callback chat([message()], keyword()) :: response
end
