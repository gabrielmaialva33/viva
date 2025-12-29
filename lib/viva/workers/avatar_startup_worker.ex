defmodule Viva.Workers.AvatarStartupWorker do
  @moduledoc """
  Oban worker to start all active avatars after application startup.
  Replaces the anti-pattern of using Process.sleep in Application.start/2.
  """
  use Oban.Worker, queue: :avatar_simulation, max_attempts: 3

  require Logger

  @impl Oban.Worker
  def perform(%Oban.Job{}) do
    Logger.info("[AvatarStartup] Starting all active avatars...")

    count =
      Viva.Avatars.list_active_avatar_ids()
      |> Enum.map(&start_avatar_safely/1)
      |> Enum.count(fn result -> result == :ok end)

    Logger.info("[AvatarStartup] Started #{count} avatars successfully")
    :ok
  end

  defp start_avatar_safely(avatar_id) do
    case Viva.Sessions.Supervisor.start_avatar(avatar_id) do
      {:ok, _} ->
        :ok

      {:error, reason} ->
        Logger.warning("[AvatarStartup] Failed to start avatar #{avatar_id}: #{inspect(reason)}")
        :error
    end
  end
end
