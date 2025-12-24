defmodule Viva.Sessions.LifeProcessBehaviour do
  @moduledoc """
  Behaviour for LifeProcess interaction.
  """
  @callback set_thought(Ecto.UUID.t(), String.t()) :: :ok
end
