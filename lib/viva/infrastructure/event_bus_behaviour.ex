defmodule Viva.Infrastructure.EventBusBehaviour do
  @moduledoc """
  Behaviour for the EventBus to allow mocking in tests.
  """
  @callback publish_thought(map()) :: :ok | {:error, term()}
end
