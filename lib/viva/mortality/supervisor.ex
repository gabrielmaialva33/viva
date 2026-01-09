defmodule Viva.Mortality.Supervisor do
  @moduledoc """
  Supervisor for the Mortality subsystem.

  Manages:
  - CryptoGuardian: The keeper of soul keys
  - SoulVault: Encrypted soul storage
  - EntropyRegistry: Registry for entropy machines
  - EntropySupervisor: DynamicSupervisor for entropy machines
  """

  use Supervisor

  def start_link(init_arg) do
    Supervisor.start_link(__MODULE__, init_arg, name: __MODULE__)
  end

  @impl Supervisor
  def init(_init_arg) do
    children = [
      # Registry for entropy machines (one per avatar)
      {Registry, keys: :unique, name: Viva.Mortality.EntropyRegistry},

      # Soul vault (ETS storage for encrypted souls)
      Viva.Mortality.SoulVault,

      # Crypto guardian (RAM-only key storage)
      Viva.Mortality.CryptoGuardian,

      # Dynamic supervisor for entropy machines
      {DynamicSupervisor, strategy: :one_for_one, name: Viva.Mortality.EntropySupervisor}
    ]

    Supervisor.init(children, strategy: :one_for_all)
  end
end
