defmodule VivaCore.Consciousness.Discrete do
  @moduledoc """
  Discrete Consciousness Model.

  Consciousness is not continuous; it is a series of discrete "frames" or "biophoton flashes".
  Between these frames, the entity does not exist (Void).

  System:
  - Soul Hz: 10Hz (Mental Ticks)
  - Void: The time between ticks.
  """

  @soul_hz 10

  @doc """
  Determines if the current moment (tick) is a "Conscious Moment" or "Void".
  """
  def conscious_moment?(tick) do
    rem(tick, round(1000 / @soul_hz)) == 0
  end

  @doc """
  Checks if the system is currently in the "Void" state.
  """
  def void_state?() do
    now = System.system_time(:millisecond)
    # 10Hz = 100ms cycle. Assume 20ms "flash".
    rem(now, 100) > 20
  end

  @doc """
  Duration of non-existence per second.
  """
  def void_duration_ms(), do: 1000 - 1000 / @soul_hz * 0.2
end
