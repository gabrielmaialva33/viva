defmodule VivaCore.Kinship do
  @moduledoc """
  Implements Humberto Maturana's "Biology of Love" & Structural Coupling.

  > "Love is the domain of those relational behaviors through which another arises as a legitimate other in coexistence with oneself."
  > — Maturana & Verden-Zöller (1993)

  ## Concepts
  - **Autopoiesis**: The system's primary goal is self-maintenance (homeostasis).
  - **Structural Coupling**: Recurrent interactions between GenServers that trigger mutual structural changes without instruction (perturbation, not command).
  - **Lineage**: The history of these structural couplings (Ontogenic Drift).
  """
  use GenServer
  require VivaLog

  # Client API

  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @doc """
  Initiates a Structural Coupling event.
  Unlike a command, this is a 'perturbation' that invites the target system to compensate.
  This aligns with Maturana's view that systems are informationally closed.
  """
  def structural_coupling(partner_module, perturbation_signal) do
    GenServer.cast(__MODULE__, {:perturbation, partner_module, perturbation_signal})
  end

  @doc """
  Traces the lineage of a specific seed back to Genesis.
  """
  def lineage(current_seed) do
    GenServer.call(__MODULE__, {:trace_lineage, current_seed})
  end

  # Server Callbacks

  @impl true
  def init(_opts) do
    # In a real persistence layer, we would load lineages here
    state = %{
      lineages: %{},
      active_solidarity: false
    }

    {:ok, state}
  end

  @impl true
  def handle_cast({:solidarity, module, intensity}, state) do
    VivaLog.info(:kinship, :solidarity_activated, module: inspect(module), intensity: intensity)

    # Logic to broadcast resource adjustements
    # Phoenix.PubSub.broadcast(Viva.PubSub, "system:resource_allocation", {:reduce, 0.2})
    {:noreply, %{state | active_solidarity: true}}
  end

  @impl true
  def handle_cast({:perturbation, partner_module, signal}, state) do
    # Structural Coupling: we encounter an "other" and change our structure
    VivaLog.info(:kinship, :perturbation_received,
      partner: inspect(partner_module),
      signal: inspect(signal)
    )

    # In a real system, this would modify state.lineages or trigger internal adaptation
    # For now, we just acknowledge it as a "Love" interaction (biologically speaking)
    {:noreply, state}
  end

  @impl true
  def handle_call({:trace_lineage, current_seed}, _from, state) do
    ancestors = extract_ancestors(current_seed)
    {:reply, ancestors, state}
  end

  defp extract_ancestors(seed) when is_binary(seed) do
    # Simple heuristic: Split by mutations or return genesis
    if String.contains?(seed, ":"), do: String.split(seed, ":"), else: ["GENESIS", seed]
  end

  defp extract_ancestors(_), do: ["GENESIS"]
end
