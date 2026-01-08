defmodule Viva.Quantum.WorldEngine do
  @moduledoc """
  GenServer that runs the quantum-inspired world simulation.

  Orchestrates:
  - Batch processing of avatar states on GPU
  - Policy network inference for stimulus selection
  - Experience collection for RL training
  - Wave function evolution (decoherence, entanglement)

  The world "retrofeeds" - avatar outcomes inform the policy network,
  which learns to select stimuli that maximize collective wellbeing.
  """

  use GenServer
  require Logger

  alias Viva.Quantum.{WaveFunction, PolicyNetwork, ExperienceBuffer}

  @tick_interval 5_000
  @train_interval 30_000
  @min_experiences 100
  @epsilon_start 1.0
  @epsilon_end 0.05
  @epsilon_decay 0.995

  defstruct [
    :actor_critic_model,
    :params,
    :optimizer_state,
    :wave_functions,
    :epsilon,
    :total_steps,
    :training_enabled
  ]

  # Client API

  def start_link(opts \\ []) do
    name = Keyword.get(opts, :name, __MODULE__)
    GenServer.start_link(__MODULE__, opts, name: name)
  end

  @doc """
  Registers an avatar with the quantum world.
  Creates a wave function for it.
  """
  def register_avatar(server \\ __MODULE__, avatar_id, avatar) do
    GenServer.call(server, {:register_avatar, avatar_id, avatar})
  end

  @doc """
  Unregisters an avatar from the quantum world.
  """
  def unregister_avatar(server \\ __MODULE__, avatar_id) do
    GenServer.cast(server, {:unregister_avatar, avatar_id})
  end

  @doc """
  Gets the wave function for an avatar.
  """
  def get_wave_function(server \\ __MODULE__, avatar_id) do
    GenServer.call(server, {:get_wave_function, avatar_id})
  end

  @doc """
  Collapses an avatar's wave function (observation/measurement).
  Returns the collapsed state.
  """
  def observe_avatar(server \\ __MODULE__, avatar_id) do
    GenServer.call(server, {:observe_avatar, avatar_id})
  end

  @doc """
  Requests a stimulus recommendation for an avatar.
  """
  def recommend_stimulus(server \\ __MODULE__, avatar_id) do
    GenServer.call(server, {:recommend_stimulus, avatar_id})
  end

  @doc """
  Records an experience (for RL learning).
  """
  def record_experience(server \\ __MODULE__, experience) do
    GenServer.cast(server, {:record_experience, experience})
  end

  @doc """
  Triggers a manual training step.
  """
  def train_step(server \\ __MODULE__) do
    GenServer.call(server, :train_step)
  end

  @doc """
  Gets current stats.
  """
  def get_stats(server \\ __MODULE__) do
    GenServer.call(server, :get_stats)
  end

  @doc """
  Enables/disables training.
  """
  def set_training(server \\ __MODULE__, enabled) do
    GenServer.cast(server, {:set_training, enabled})
  end

  # Server Callbacks

  @impl true
  def init(opts) do
    Logger.info("[QuantumWorld] Initializing quantum world engine...")

    # Build the actor-critic model
    model = PolicyNetwork.build_actor_critic()

    # Initialize parameters
    params = PolicyNetwork.init_params(model)

    # Initialize optimizer (Adam)
    {optimizer_init, _optimizer_update} = Polaris.Optimizers.adam(learning_rate: 1.0e-3)
    optimizer_state = optimizer_init.(params)

    # Start experience buffer (ignore if already started)
    case ExperienceBuffer.start_link(name: Viva.Quantum.ExperienceBuffer) do
      {:ok, _pid} -> :ok
      {:error, {:already_started, _pid}} -> :ok
      {:error, reason} -> Logger.warning("[QuantumWorld] Failed to start ExperienceBuffer: #{inspect(reason)}")
    end

    state = %__MODULE__{
      actor_critic_model: model,
      params: params,
      optimizer_state: optimizer_state,
      wave_functions: %{},
      epsilon: @epsilon_start,
      total_steps: 0,
      training_enabled: Keyword.get(opts, :training_enabled, true)
    }

    # Schedule periodic ticks
    if Keyword.get(opts, :auto_tick, true) do
      schedule_tick()
      schedule_train()
    end

    Logger.info("[QuantumWorld] Quantum world engine initialized")
    {:ok, state}
  end

  @impl true
  def handle_call({:register_avatar, avatar_id, avatar}, _from, state) do
    wf = WaveFunction.from_avatar(avatar)
    new_wfs = Map.put(state.wave_functions, avatar_id, wf)
    Logger.debug("[QuantumWorld] Registered avatar #{avatar_id}")
    {:reply, :ok, %{state | wave_functions: new_wfs}}
  end

  @impl true
  def handle_call({:get_wave_function, avatar_id}, _from, state) do
    wf = Map.get(state.wave_functions, avatar_id)
    {:reply, wf, state}
  end

  @impl true
  def handle_call({:observe_avatar, avatar_id}, _from, state) do
    case Map.get(state.wave_functions, avatar_id) do
      nil ->
        {:reply, {:error, :not_found}, state}

      wf ->
        {collapsed_state, updated_wf} = WaveFunction.collapse(wf)
        new_wfs = Map.put(state.wave_functions, avatar_id, updated_wf)
        {:reply, {:ok, collapsed_state}, %{state | wave_functions: new_wfs}}
    end
  end

  @impl true
  def handle_call({:recommend_stimulus, avatar_id}, _from, state) do
    case Map.get(state.wave_functions, avatar_id) do
      nil ->
        {:reply, {:error, :not_found}, state}

      wf ->
        # Use the mean of the wave function as the state for policy
        {action_idx, action_name} =
          PolicyNetwork.select_action(
            state.actor_critic_model,
            state.params,
            wf.mean,
            epsilon: state.epsilon
          )

        stimulus = PolicyNetwork.action_to_stimulus(action_name)
        {:reply, {:ok, action_idx, stimulus}, state}
    end
  end

  @impl true
  def handle_call(:train_step, _from, state) do
    case do_train_step(state) do
      {:ok, new_state, loss} ->
        {:reply, {:ok, loss}, new_state}

      {:error, reason} ->
        {:reply, {:error, reason}, state}
    end
  end

  @impl true
  def handle_call(:get_stats, _from, state) do
    buffer_size = ExperienceBuffer.size()

    avg_coherence =
      if map_size(state.wave_functions) > 0 do
        state.wave_functions
        |> Map.values()
        |> Enum.map(& &1.coherence)
        |> Enum.sum()
        |> Kernel./(map_size(state.wave_functions))
      else
        0.0
      end

    stats = %{
      registered_avatars: map_size(state.wave_functions),
      experience_buffer_size: buffer_size,
      epsilon: state.epsilon,
      total_steps: state.total_steps,
      training_enabled: state.training_enabled,
      average_coherence: avg_coherence
    }

    {:reply, stats, state}
  end

  @impl true
  def handle_cast({:unregister_avatar, avatar_id}, state) do
    new_wfs = Map.delete(state.wave_functions, avatar_id)
    {:noreply, %{state | wave_functions: new_wfs}}
  end

  @impl true
  def handle_cast({:record_experience, experience}, state) do
    ExperienceBuffer.push(experience)
    {:noreply, state}
  end

  @impl true
  def handle_cast({:set_training, enabled}, state) do
    Logger.info("[QuantumWorld] Training #{if enabled, do: "enabled", else: "disabled"}")
    {:noreply, %{state | training_enabled: enabled}}
  end

  @impl true
  def handle_info(:tick, state) do
    # Apply decoherence to all wave functions
    new_wfs =
      state.wave_functions
      |> Enum.map(fn {id, wf} -> {id, WaveFunction.decohere(wf)} end)
      |> Map.new()

    # Decay epsilon
    new_epsilon = max(@epsilon_end, state.epsilon * @epsilon_decay)

    schedule_tick()
    {:noreply, %{state | wave_functions: new_wfs, epsilon: new_epsilon, total_steps: state.total_steps + 1}}
  end

  @impl true
  def handle_info(:train, state) do
    new_state =
      if state.training_enabled do
        case do_train_step(state) do
          {:ok, updated_state, loss} ->
            Logger.debug("[QuantumWorld] Training step completed, loss: #{Float.round(loss, 4)}")
            updated_state

          {:error, _reason} ->
            state
        end
      else
        state
      end

    schedule_train()
    {:noreply, new_state}
  end

  # Private helpers

  defp schedule_tick do
    Process.send_after(self(), :tick, @tick_interval)
  end

  defp schedule_train do
    Process.send_after(self(), :train, @train_interval)
  end

  defp do_train_step(state) do
    buffer_size = ExperienceBuffer.size()

    if buffer_size < @min_experiences do
      {:error, :insufficient_experiences}
    else
      case ExperienceBuffer.sample() do
        nil ->
          {:error, :sample_failed}

        batch ->
          # Forward pass and loss calculation
          {loss, _actor_loss, _critic_loss} =
            PolicyNetwork.train_step(
              state.actor_critic_model,
              state.optimizer_state,
              state.params,
              batch
            )

          loss_value = Nx.to_number(loss)

          # TODO: Implement actual gradient update with Polaris
          # For now, this is a simplified version
          # In production, use Axon.Loop for proper training

          {:ok, state, loss_value}
      end
    end
  end
end
