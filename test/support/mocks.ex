defmodule Viva.TestMocks do
  @moduledoc """
  Defines Mox mocks for testing.
  """
  import Mox

  # Define mocks here
  defmock(Viva.AI.Pipeline.MockStage, for: Viva.AI.Pipeline.Stage)
  defmock(Viva.AI.LLM.MockClient, for: Viva.AI.LLM.ClientBehaviour)
  defmock(Viva.Sessions.MockLifeProcess, for: Viva.Sessions.LifeProcessBehaviour)
end
