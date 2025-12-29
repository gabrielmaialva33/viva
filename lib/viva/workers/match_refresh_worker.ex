defmodule Viva.Workers.MatchRefreshWorker do
  @moduledoc """
  Oban worker to refresh matchmaking scores.
  Recalculates compatibility between avatars periodically.
  """
  use Oban.Worker, queue: :matchmaking, max_attempts: 3

  require Logger

  @doc """
  Schedule match refresh for a specific avatar.
  """
  @spec schedule(Ecto.UUID.t()) :: {:ok, Oban.Job.t()} | {:error, term()}
  def schedule(avatar_id) do
    %{avatar_id: avatar_id}
    |> __MODULE__.new()
    |> Oban.insert()
  end

  @doc """
  Schedule refresh for all matches.
  """
  @spec schedule_all() :: {:ok, Oban.Job.t()} | {:error, term()}
  def schedule_all do
    %{}
    |> __MODULE__.new()
    |> Oban.insert()
  end

  @impl Oban.Worker
  def perform(%Oban.Job{args: %{"avatar_id" => avatar_id}}) do
    Logger.debug("[MatchRefresh] Refreshing matches for avatar #{avatar_id}")
    Viva.Matching.Engine.refresh_matches(avatar_id)
    :ok
  end

  def perform(%Oban.Job{args: %{}}) do
    Logger.info("[MatchRefresh] Refreshing matches for all active avatars")

    # Refresh matches for all active avatars with staggered scheduling
    Viva.Sessions.Supervisor.list_running_avatars()
    |> Enum.with_index()
    |> Enum.each(fn {avatar_id, index} ->
      # Stagger jobs by 100ms each to avoid thundering herd
      delay = index * 100

      %{avatar_id: avatar_id}
      |> __MODULE__.new(scheduled_at: DateTime.add(DateTime.utc_now(), delay, :millisecond))
      |> Oban.insert()
    end)

    :ok
  end
end
