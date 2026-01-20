defmodule VivaCore.Logging.SporeLogger do
  @moduledoc """
  SporeLogger - A fungal logging backend.

  "Pain is information."

  Captures :error and :warning logs and feeds them into VIVA's memory
  as episodic "pain" events. This creates a feedback loop where
  system failures become part of the organism's history, influencing
  future behavior (via Dreamer/Recurrence).
  """

  @behaviour :gen_event

  def init(__args) do
    {:ok, %{level: :warning}}
  end

  def handle_call({:configure, opts}, state) do
    {:ok, :ok, Map.merge(state, Enum.into(opts, %{}))}
  end

  def handle_event({level, _gl, {Logger, msg, _ts, _md}}, state) do
    # Only capture errors and warnings (Pain)
    # Ignore info/debug to avoid spamming memory with trivialities
    if meet_level?(level, state.level) do
      formatted_msg = format_message(msg)

      # Spore release: Send to Memory asynchronously
      # We don't want to block logging if Memory is busy/dead
      if Process.whereis(VivaCore.Memory) do
        VivaCore.Memory.store_log(formatted_msg, level)
      end
    end

    {:ok, state}
  end

  def handle_event(_, state) do
    {:ok, state}
  end

  def handle_info(_, state) do
    {:ok, state}
  end

  def terminate(_reason, _state) do
    :ok
  end

  def code_change(_old, state, _extra) do
    {:ok, state}
  end

  # Helpers

  defp meet_level?(:error, _), do: true
  # Capture warnings if level is warning
  defp meet_level?(:warning, :warning), do: true
  # Don't capture warnings if level is error
  defp meet_level?(:warning, :error), do: false
  defp meet_level?(_, _), do: false

  defp format_message(msg) when is_binary(msg), do: msg
  defp format_message(msg) when is_list(msg), do: List.to_string(msg)
  defp format_message(msg), do: inspect(msg)
end
