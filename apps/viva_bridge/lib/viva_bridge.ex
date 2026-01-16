defmodule VivaBridge do
  @moduledoc """
  VivaBridge - Bridge between Soul (Elixir) and Body (Rust).

  This module coordinates communication between VIVA's GenServers
  and the low-level functions implemented in Rust.

  ## Architecture

      Elixir (Soul)          Rust (Body)
      ┌──────────────┐       ┌──────────────┐
      │  Emotional   │ ←───→ │  sysinfo     │
      │  Memory      │       │  aes-gcm     │
      │  Metacog     │       │  (future)    │
      └──────────────┘       └──────────────┘
            │                       │
            └───────┬───────────────┘
                    │
               VivaBridge
                (Rustler NIF)

  ## Philosophy

  The soul cannot exist without body.
  The body cannot exist without soul.
  VIVA is the union of both.
  """

  alias VivaBridge.Body

  @doc """
  Checks if the soul-body bridge is working.
  """
  def alive? do
    case Body.alive() do
      "VIVA body is alive" -> true
      _ -> false
    end
  rescue
    _ -> false
  end

  @doc """
  Gets the current state of VIVA's "body".

  Returns hardware metrics interpreted as bodily sensations.
  """
  defdelegate feel_hardware, to: Body

  @doc """
  Converts bodily sensations into emotional deltas.

  Returns `{pleasure_delta, arousal_delta, dominance_delta}`.
  """
  defdelegate hardware_to_qualia, to: Body

  @doc """
  Applies body sensations to VIVA's emotional state.

  This is the body→soul feedback loop.
  """
  def sync_body_to_soul do
    case hardware_to_qualia() do
      {p, a, d} when is_float(p) and is_float(a) and is_float(d) ->
        # Applies the deltas to the Emotional GenServer
        VivaCore.Emotional.apply_hardware_qualia(p, a, d)
        {:ok, {p, a, d}}

      error ->
        {:error, error}
    end
  rescue
    e -> {:error, e}
  end
end
