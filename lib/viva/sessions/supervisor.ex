defmodule Viva.Sessions.Supervisor do
  @moduledoc """
  Supervisor for all avatar life processes.
  Uses DynamicSupervisor to spawn LifeProcess for each active avatar.
  """
  use Supervisor

  def start_link(init_arg) do
    Supervisor.start_link(__MODULE__, init_arg, name: __MODULE__)
  end

  @impl true
  def init(_init_arg) do
    children = [
      # Registry for looking up avatar processes by ID
      {Registry, keys: :unique, name: Viva.Sessions.AvatarRegistry},

      # Dynamic supervisor for avatar life processes
      {DynamicSupervisor, strategy: :one_for_one, name: Viva.Sessions.AvatarSupervisor},

      # World clock - manages simulation time
      Viva.World.Clock,

      # Matchmaker process
      Viva.Matchmaker.Engine
    ]

    Supervisor.init(children, strategy: :one_for_one)
  end

  @doc "Start a life process for an avatar"
  def start_avatar(avatar_id) do
    spec = {Viva.Sessions.LifeProcess, avatar_id}

    case DynamicSupervisor.start_child(Viva.Sessions.AvatarSupervisor, spec) do
      {:ok, pid} -> {:ok, pid}
      {:error, {:already_started, pid}} -> {:ok, pid}
      error -> error
    end
  end

  @doc "Stop a life process for an avatar"
  def stop_avatar(avatar_id) do
    case Registry.lookup(Viva.Sessions.AvatarRegistry, avatar_id) do
      [{pid, _}] ->
        DynamicSupervisor.terminate_child(Viva.Sessions.AvatarSupervisor, pid)

      [] ->
        {:error, :not_found}
    end
  end

  @doc "Check if an avatar's life process is running"
  def avatar_alive?(avatar_id) do
    case Registry.lookup(Viva.Sessions.AvatarRegistry, avatar_id) do
      [{_pid, _}] -> true
      [] -> false
    end
  end

  @doc "Get the PID of an avatar's life process"
  def get_avatar_pid(avatar_id) do
    case Registry.lookup(Viva.Sessions.AvatarRegistry, avatar_id) do
      [{pid, _}] -> {:ok, pid}
      [] -> {:error, :not_found}
    end
  end

  @doc "List all running avatar IDs"
  def list_running_avatars do
    Registry.select(Viva.Sessions.AvatarRegistry, [{{:"$1", :_, :_}, [], [:"$1"]}])
  end

  @doc "Count running avatars"
  def count_running_avatars do
    Registry.count(Viva.Sessions.AvatarRegistry)
  end

  @doc "Start all active avatars from database"
  def start_all_active_avatars do
    Viva.Avatars.list_active_avatar_ids()
    |> Enum.each(&start_avatar/1)
  end
end
