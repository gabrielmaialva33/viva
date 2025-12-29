defmodule Viva.Infrastructure.Redis do
  @moduledoc """
  Wrapper for Redis operations using Redix.
  Used primarily for caching avatar states (View State) to offload GenServer calls.
  """

  @spec child_spec(term()) :: Supervisor.child_spec()
  def child_spec(_) do
    %{
      id: __MODULE__,
      start: {__MODULE__, :start_link, []},
      type: :worker,
      restart: :permanent,
      shutdown: 500
    }
  end

  @spec start_link() :: GenServer.on_start()
  def start_link do
    Redix.start_link(
      host: System.get_env("REDIS_HOST", "localhost"),
      port: String.to_integer(System.get_env("REDIS_PORT", "6379")),
      name: :redix
    )
  end

  @spec set_avatar_view_state(Ecto.UUID.t(), map()) :: {:ok, binary()} | {:error, term()}
  def set_avatar_view_state(avatar_id, state_map) do
    key = "viva:avatar:#{avatar_id}:view"
    json = Jason.encode!(state_map)
    # Expire in 5 minutes if not updated (avoid stale data ghosts)
    Redix.command(:redix, ["SET", key, json, "EX", "300"])
  end

  @spec get_avatar_view_state(Ecto.UUID.t()) :: map() | nil
  def get_avatar_view_state(avatar_id) do
    key = "viva:avatar:#{avatar_id}:view"

    case Redix.command(:redix, ["GET", key]) do
      {:ok, nil} -> nil
      {:ok, json} -> Jason.decode!(json, keys: :strings)
      _ -> nil
    end
  end
end
