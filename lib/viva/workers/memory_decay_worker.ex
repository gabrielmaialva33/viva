defmodule Viva.Workers.MemoryDecayWorker do
  @moduledoc """
  Oban worker to process memory decay for avatars.
  Runs periodically to fade old memories based on recency and importance.
  """
  use Oban.Worker, queue: :memory_processing, max_attempts: 3

  require Logger

  @doc """
  Schedule memory decay for a specific avatar.
  """
  @spec schedule(Ecto.UUID.t()) :: {:ok, Oban.Job.t()} | {:error, term()}
  def schedule(avatar_id) do
    %{avatar_id: avatar_id}
    |> __MODULE__.new()
    |> Oban.insert()
  end

  @doc """
  Schedule memory decay for all active avatars.
  """
  @spec schedule_all() :: {:ok, Oban.Job.t()} | {:error, term()}
  def schedule_all do
    %{}
    |> __MODULE__.new()
    |> Oban.insert()
  end

  @impl Oban.Worker
  def perform(%Oban.Job{args: %{"avatar_id" => avatar_id}}) do
    Logger.debug("[MemoryDecay] Processing memories for avatar #{avatar_id}")
    Viva.Avatars.decay_old_memories(avatar_id)
    :ok
  end

  def perform(%Oban.Job{args: %{}}) do
    # Process all avatars if no specific avatar_id provided
    Logger.info("[MemoryDecay] Processing memories for all active avatars")

    Enum.each(Viva.Avatars.list_active_avatar_ids(), fn avatar_id ->
      %{avatar_id: avatar_id}
      |> __MODULE__.new()
      |> Oban.insert()
    end)

    :ok
  end
end
