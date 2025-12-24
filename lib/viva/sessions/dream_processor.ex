defmodule Viva.Sessions.DreamProcessor do
  @moduledoc """
  Handles dream cycle processing for avatars during sleep.
  Extracted from LifeProcess to reduce module dependencies.
  """

  require Logger

  alias Viva.Avatars.Systems.Dreams

  @type process_state :: map()

  @doc """
  Trigger the dream cycle asynchronously.
  Called when avatar falls asleep.
  """
  @spec trigger_dream_cycle(process_state()) :: :ok
  def trigger_dream_cycle(process_state) do
    avatar_id = process_state.avatar_id
    consciousness = process_state.state.consciousness
    experience_stream = consciousness.experience_stream

    # Async task to process dreams without blocking the heartbeat
    Task.Supervisor.start_child(Viva.Sessions.TaskSupervisor, fn ->
      process_dream_async(avatar_id, consciousness, experience_stream)
    end)

    :ok
  end

  # === Private Functions ===

  defp process_dream_async(avatar_id, consciousness, experience_stream) do
    # Check if avatar should dream (probabilistic based on emotional intensity)
    if Dreams.should_dream?(experience_stream) do
      Logger.info("Avatar #{avatar_id} is dreaming...")

      case Dreams.process_dream_cycle(avatar_id, consciousness, experience_stream) do
        {:ok, _, dream_memory} ->
          maybe_save_dream_memory(avatar_id, dream_memory)

        {:error, reason} ->
          Logger.warning("Dream cycle failed for avatar #{avatar_id}: #{inspect(reason)}")
      end
    else
      Logger.debug("Avatar #{avatar_id} sleeping lightly (no dreams)")
      Dreams.light_sleep_processing(consciousness)
    end
  end

  defp maybe_save_dream_memory(_, nil), do: :ok

  defp maybe_save_dream_memory(avatar_id, dream_memory) do
    changeset =
      Viva.Avatars.Memory.changeset(
        %Viva.Avatars.Memory{},
        Map.from_struct(dream_memory)
      )

    case Viva.Repo.insert(changeset) do
      {:ok, _} ->
        Logger.debug("Dream memory saved for avatar #{avatar_id}")

      {:error, reason} ->
        Logger.warning("Failed to save dream memory: #{inspect(reason)}")
    end
  end
end
