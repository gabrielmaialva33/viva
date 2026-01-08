defmodule Viva.Metrics.Collector do
  @moduledoc """
  Collects real-time metrics from avatar LifeProcesses.
  Subscribes to avatar events and aggregates simulation data.
  """

  use GenServer

  require Logger

  alias Phoenix.PubSub
  alias Viva.Sessions.AvatarRegistry

  # === Struct ===

  defstruct [
    :started_at,
    avatars: %{},
    global: %{
      total_ticks: 0,
      total_thoughts: 0,
      total_dreams: 0,
      total_crystallizations: 0,
      avg_happiness: 0.0,
      avg_energy: 0.0
    }
  ]

  # === Types ===

  @type avatar_metrics :: %{
          avatar_id: String.t(),
          name: String.t(),
          # Bio (neurochemistry)
          dopamine: float(),
          cortisol: float(),
          oxytocin: float(),
          adenosine: float(),
          libido: float(),
          energy: float(),
          # Emotional (PAD)
          pleasure: float(),
          arousal: float(),
          dominance: float(),
          current_emotion: atom(),
          # Consciousness
          presence_level: float(),
          meta_awareness: float(),
          experience_intensity: float(),
          # Activity
          current_activity: atom(),
          current_desire: atom(),
          last_thought: String.t() | nil,
          # Meta
          tick_count: integer(),
          owner_online?: boolean(),
          updated_at: DateTime.t()
        }

  @type t :: %__MODULE__{}

  # === Client API ===

  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @doc "Get metrics for all avatars"
  @spec get_all() :: %{avatars: map(), global: map()}
  def get_all do
    GenServer.call(__MODULE__, :get_all)
  end

  @doc "Get metrics for a specific avatar"
  @spec get_avatar(String.t()) :: avatar_metrics() | nil
  def get_avatar(avatar_id) do
    GenServer.call(__MODULE__, {:get_avatar, avatar_id})
  end

  @doc "Get global simulation metrics"
  @spec get_global() :: map()
  def get_global do
    GenServer.call(__MODULE__, :get_global)
  end

  @doc "Force refresh metrics from all active avatars"
  @spec refresh() :: :ok
  def refresh do
    GenServer.cast(__MODULE__, :refresh)
  end

  # === Server Callbacks ===

  @impl GenServer
  def init(_opts) do
    Logger.info("Starting Metrics Collector")

    # Subscribe to simulation events
    PubSub.subscribe(Viva.PubSub, "simulation:metrics")

    # Schedule periodic collection
    schedule_collection()

    state = %__MODULE__{
      started_at: DateTime.utc_now()
    }

    {:ok, state}
  end

  @impl GenServer
  def handle_call(:get_all, _from, state) do
    {:reply, %{avatars: state.avatars, global: state.global}, state}
  end

  @impl GenServer
  def handle_call({:get_avatar, avatar_id}, _from, state) do
    {:reply, Map.get(state.avatars, avatar_id), state}
  end

  @impl GenServer
  def handle_call(:get_global, _from, state) do
    {:reply, state.global, state}
  end

  @impl GenServer
  def handle_cast(:refresh, state) do
    new_state = collect_all_metrics(state)
    {:noreply, new_state}
  end

  @impl GenServer
  def handle_info(:collect, state) do
    new_state = collect_all_metrics(state)

    Logger.info("Collector: Coletou #{map_size(new_state.avatars)} avatares")

    # Broadcast updated metrics
    broadcast_metrics(new_state)

    schedule_collection()
    {:noreply, new_state}
  end

  @impl GenServer
  def handle_info({:avatar_tick, avatar_id, metrics}, state) do
    # Update individual avatar metrics
    new_avatars = Map.put(state.avatars, avatar_id, metrics)
    new_global = recalculate_global(new_avatars, state.global, :tick)

    new_state = %{state | avatars: new_avatars, global: new_global}
    {:noreply, new_state}
  end

  @impl GenServer
  def handle_info({:avatar_thought, avatar_id, thought}, state) do
    # Update thought and increment counter
    case Map.get(state.avatars, avatar_id) do
      nil ->
        {:noreply, state}

      avatar_metrics ->
        updated = %{avatar_metrics | last_thought: thought}
        new_avatars = Map.put(state.avatars, avatar_id, updated)
        new_global = %{state.global | total_thoughts: state.global.total_thoughts + 1}

        {:noreply, %{state | avatars: new_avatars, global: new_global}}
    end
  end

  @impl GenServer
  def handle_info({:avatar_crystallization, avatar_id, _crystal}, state) do
    new_global = %{state.global | total_crystallizations: state.global.total_crystallizations + 1}
    Logger.info("Crystallization event from avatar #{avatar_id}")
    {:noreply, %{state | global: new_global}}
  end

  @impl GenServer
  def handle_info(_msg, state) do
    {:noreply, state}
  end

  # === Private Functions ===

  defp schedule_collection do
    # Collect every 5 seconds
    Process.send_after(self(), :collect, 5_000)
  end

  defp collect_all_metrics(state) do
    # Get all registered avatars
    avatar_ids = get_active_avatar_ids()

    avatars =
      avatar_ids
      |> Enum.map(&collect_avatar_metrics/1)
      |> Enum.reject(&is_nil/1)
      |> Map.new(fn m -> {m.avatar_id, m} end)

    global = recalculate_global(avatars, state.global, :full)

    %{state | avatars: avatars, global: global}
  end

  defp get_active_avatar_ids do
    Registry.select(AvatarRegistry, [{{:"$1", :_, :_}, [], [:"$1"]}])
  end

  defp collect_avatar_metrics(avatar_id) do
    try do
      # Timeout de 2 segundos pra não travar
      case Registry.lookup(AvatarRegistry, avatar_id) do
        [{pid, _}] ->
          process_state = GenServer.call(pid, :get_state, 2_000)
          extract_metrics(process_state)

        [] ->
          Logger.debug("Collector: Avatar #{avatar_id} não encontrado no registry")
          nil
      end
    rescue
      e ->
        Logger.warning("Collector: Erro ao coletar #{avatar_id}: #{inspect(e)}")
        nil
    catch
      :exit, reason ->
        Logger.warning("Collector: Exit ao coletar #{avatar_id}: #{inspect(reason)}")
        nil
    end
  end

  defp extract_metrics(process_state) do
    avatar = process_state.avatar
    internal = process_state.state
    bio = internal.bio
    emotional = internal.emotional
    consciousness = internal.consciousness

    %{
      avatar_id: process_state.avatar_id,
      name: avatar.name,
      # Bio/Neurochemistry
      dopamine: bio.dopamine,
      cortisol: bio.cortisol,
      oxytocin: bio.oxytocin,
      adenosine: bio.adenosine,
      libido: bio.libido,
      energy: calculate_energy(bio),
      # Emotional (PAD)
      pleasure: emotional.pleasure,
      arousal: emotional.arousal,
      dominance: emotional.dominance,
      current_emotion: String.to_atom(emotional.mood_label || "neutral"),
      # Consciousness
      presence_level: consciousness.presence_level,
      meta_awareness: consciousness.meta_awareness,
      experience_intensity: consciousness.experience_intensity,
      stream_tempo: consciousness.stream_tempo,
      # Activity
      current_activity: internal.current_activity,
      current_desire: internal.current_desire,
      last_thought: process_state.last_thought,
      # Meta
      tick_count: process_state.tick_count,
      owner_online?: process_state.owner_online?,
      updated_at: internal.updated_at || DateTime.utc_now()
    }
  end

  defp calculate_energy(bio) do
    # Energy is inverse of adenosine, boosted by dopamine
    base = 1.0 - bio.adenosine
    boost = bio.dopamine * 0.2
    min(1.0, base + boost)
  end

  defp recalculate_global(avatars, current_global, type) do
    count = map_size(avatars)

    if count == 0 do
      current_global
    else
      avatar_list = Map.values(avatars)

      avg_happiness = Enum.sum(Enum.map(avatar_list, & &1.pleasure)) / count
      avg_energy = Enum.sum(Enum.map(avatar_list, & &1.energy)) / count

      total_ticks =
        if type == :full do
          Enum.sum(Enum.map(avatar_list, & &1.tick_count))
        else
          current_global.total_ticks + 1
        end

      %{
        current_global
        | avg_happiness: Float.round(avg_happiness, 3),
          avg_energy: Float.round(avg_energy, 3),
          total_ticks: total_ticks
      }
    end
  end

  defp broadcast_metrics(state) do
    PubSub.broadcast(Viva.PubSub, "metrics:live", {:metrics_update, state.avatars, state.global})
  end
end
