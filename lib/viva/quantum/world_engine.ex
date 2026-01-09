defmodule Viva.Quantum.WorldEngine do
  @moduledoc """
  Quantum World Engine - GPU-Accelerated Consciousness Simulation.

  EVOLVED: Now uses Viva.Engine.Kernel for batch tensor processing.

  Orchestrates:
  - Batch processing of ALL avatar states on GPU via Kernel
  - Policy network inference for stimulus selection
  - Experience collection for RL training
  - Wave function semantics preserved (coherence, decoherence)

  Architecture:
  - Tensors [batch_size, N] hold all avatar states on GPU
  - Single tick processes ALL avatars in microseconds
  - Policy network learns optimal stimuli for collective wellbeing
  - Avatars can "distill" knowledge from LLM interactions
  """

  use GenServer
  require Logger

  alias Viva.Engine.Kernel
  alias Viva.Quantum.{PolicyNetwork, ExperienceBuffer}

  # Faster tick now that we use GPU batch processing
  @tick_interval 500
  @train_interval 30_000
  @min_experiences 50
  @epsilon_start 1.0
  @epsilon_end 0.05
  @epsilon_decay 0.995

  # Batch size for GPU tensors
  @max_avatars 10_000

  defstruct [
    :actor_critic_model,
    :params,
    :optimizer_state,
    # GPU Tensors
    :bio_tensor,
    :emotion_tensor,
    :traits_tensor,
    :variance_tensor,
    # Mappings
    :avatar_mapping,
    :free_slots,
    :active_count,
    # RL state
    :epsilon,
    :total_steps,
    :training_enabled,
    # Stats
    :last_tick_us,
    :total_rewards
  ]

  # ============================================================================
  # Client API
  # ============================================================================

  def start_link(opts \\ []) do
    name = Keyword.get(opts, :name, __MODULE__)
    GenServer.start_link(__MODULE__, opts, name: name)
  end

  @doc """
  Registers an avatar with the quantum world.
  Now stores in GPU tensors instead of individual wave functions.
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
  Gets the current state for an avatar (collapsed from tensors).
  """
  def get_avatar_state(server \\ __MODULE__, avatar_id) do
    GenServer.call(server, {:get_avatar_state, avatar_id})
  end

  @doc """
  Applies a stimulus to an avatar.
  """
  def apply_stimulus(server \\ __MODULE__, avatar_id, stimulus_type, intensity \\ 1.0) do
    GenServer.cast(server, {:apply_stimulus, avatar_id, stimulus_type, intensity})
  end

  @doc """
  Requests a stimulus recommendation for an avatar using the policy network.
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

  @doc """
  Forces a tick (useful for testing).
  """
  def force_tick(server \\ __MODULE__) do
    GenServer.cast(server, :force_tick)
  end

  # Backwards compatibility
  def get_wave_function(server \\ __MODULE__, avatar_id), do: get_avatar_state(server, avatar_id)
  def observe_avatar(server \\ __MODULE__, avatar_id), do: get_avatar_state(server, avatar_id)

  # ============================================================================
  # Server Callbacks
  # ============================================================================

  @impl true
  def init(opts) do
    Logger.info("[QuantumWorld] Initializing GPU-accelerated consciousness engine...")

    # Build the actor-critic model
    model = PolicyNetwork.build_actor_critic()
    params = PolicyNetwork.init_params(model)

    # Initialize optimizer (Adam)
    {optimizer_init, _optimizer_update} = Polaris.Optimizers.adam(learning_rate: 1.0e-3)
    optimizer_state = optimizer_init.(params)

    # Start experience buffer
    case ExperienceBuffer.start_link(name: Viva.Quantum.ExperienceBuffer) do
      {:ok, _pid} -> :ok
      {:error, {:already_started, _pid}} -> :ok
      {:error, reason} -> Logger.warning("[QuantumWorld] ExperienceBuffer: #{inspect(reason)}")
    end

    # Initialize GPU tensors for batch processing
    state = %__MODULE__{
      actor_critic_model: model,
      params: params,
      optimizer_state: optimizer_state,
      # GPU Tensors initialized to zero (will be populated as avatars register)
      bio_tensor: Nx.broadcast(0.0, {@max_avatars, 5}) |> Nx.as_type(:f32),
      emotion_tensor: Nx.broadcast(0.0, {@max_avatars, 3}) |> Nx.as_type(:f32),
      traits_tensor: Nx.broadcast(0.5, {@max_avatars, 5}) |> Nx.as_type(:f32),
      variance_tensor: Nx.broadcast(0.01, {@max_avatars, 8}) |> Nx.as_type(:f32),
      # Mappings
      avatar_mapping: %{},
      free_slots: Enum.to_list(0..(@max_avatars - 1)),
      active_count: 0,
      # RL state
      epsilon: @epsilon_start,
      total_steps: 0,
      training_enabled: Keyword.get(opts, :training_enabled, true),
      # Stats
      last_tick_us: 0,
      total_rewards: 0.0
    }

    # Schedule periodic ticks
    if Keyword.get(opts, :auto_tick, true) do
      schedule_tick()
      schedule_train()
    end

    Logger.info("[QuantumWorld] Engine ready. Max avatars: #{@max_avatars}, Tick: #{@tick_interval}ms")
    {:ok, state}
  end

  @impl true
  def handle_call({:register_avatar, avatar_id, avatar}, _from, state) do
    case state.free_slots do
      [] ->
        Logger.warning("[QuantumWorld] No slots for avatar #{avatar_id}")
        {:reply, {:error, :no_slots}, state}

      [slot | rest] ->
        # Extract avatar data
        internal = avatar.internal_state
        personality = avatar.personality

        bio = internal.bio
        emo = internal.emotional

        # Create tensor rows
        bio_row = Nx.tensor([[
          bio.dopamine,
          bio.cortisol,
          bio.oxytocin,
          bio.adenosine,
          bio.libido
        ]], type: :f32)

        emotion_row = Nx.tensor([[
          emo.pleasure,
          emo.arousal,
          emo.dominance
        ]], type: :f32)

        traits_row = Nx.tensor([[
          personality.openness,
          personality.conscientiousness,
          personality.extraversion,
          personality.agreeableness,
          personality.neuroticism
        ]], type: :f32)

        # Update tensors at slot
        new_bio = Nx.put_slice(state.bio_tensor, [slot, 0], bio_row)
        new_emotion = Nx.put_slice(state.emotion_tensor, [slot, 0], emotion_row)
        new_traits = Nx.put_slice(state.traits_tensor, [slot, 0], traits_row)

        new_state = %{state |
          bio_tensor: new_bio,
          emotion_tensor: new_emotion,
          traits_tensor: new_traits,
          avatar_mapping: Map.put(state.avatar_mapping, avatar_id, slot),
          free_slots: rest,
          active_count: state.active_count + 1
        }

        Logger.debug("[QuantumWorld] Avatar #{avatar_id} registered at slot #{slot}")
        {:reply, {:ok, slot}, new_state}
    end
  end

  @impl true
  def handle_call({:get_avatar_state, avatar_id}, _from, state) do
    case Map.get(state.avatar_mapping, avatar_id) do
      nil ->
        {:reply, {:error, :not_found}, state}

      slot ->
        # Extract state from tensors
        bio_row = state.bio_tensor[slot] |> Nx.to_list()
        emotion_row = state.emotion_tensor[slot] |> Nx.to_list()
        variance_row = state.variance_tensor[slot] |> Nx.to_list()

        [dopamine, cortisol, oxytocin, adenosine, libido] = bio_row
        [pleasure, arousal, dominance] = emotion_row

        # Calculate coherence from variance
        avg_variance = Enum.sum(variance_row) / length(variance_row)
        coherence = max(0.0, 1.0 - avg_variance / 0.5)

        result = %{
          bio: %{
            dopamine: dopamine,
            cortisol: cortisol,
            oxytocin: oxytocin,
            adenosine: adenosine,
            libido: libido
          },
          emotional: %{
            pleasure: pleasure,
            arousal: arousal,
            dominance: dominance
          },
          coherence: coherence,
          slot: slot
        }

        {:reply, {:ok, result}, state}
    end
  end

  @impl true
  def handle_call({:recommend_stimulus, avatar_id}, _from, state) do
    case Map.get(state.avatar_mapping, avatar_id) do
      nil ->
        {:reply, {:error, :not_found}, state}

      slot ->
        # Build state vector for policy network (8 dims)
        bio_row = state.bio_tensor[slot] |> Nx.to_list()
        emotion_row = state.emotion_tensor[slot] |> Nx.to_list()

        [dopamine, cortisol, oxytocin, adenosine, _libido] = bio_row
        [pleasure, arousal, dominance] = emotion_row

        # Policy expects 8 dims: dopa, cort, oxy, adeno, pleasure, arousal, dominance, energy
        energy = 1.0 - adenosine  # Inverse of fatigue
        state_vector = Nx.tensor([dopamine, cortisol, oxytocin, adenosine, pleasure, arousal, dominance, energy])

        {action_idx, action_name} =
          PolicyNetwork.select_action(
            state.actor_critic_model,
            state.params,
            state_vector,
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

    # Calculate average wellbeing across active avatars
    avg_wellbeing =
      if state.active_count > 0 do
        rewards = Kernel.compute_reward(state.bio_tensor, state.emotion_tensor)
        active_rewards = Nx.slice(rewards, [0, 0], [state.active_count, 1])
        Nx.mean(active_rewards) |> Nx.to_number()
      else
        0.0
      end

    stats = %{
      active_avatars: state.active_count,
      max_avatars: @max_avatars,
      free_slots: length(state.free_slots),
      experience_buffer_size: buffer_size,
      epsilon: Float.round(state.epsilon, 4),
      total_steps: state.total_steps,
      training_enabled: state.training_enabled,
      last_tick_us: state.last_tick_us,
      avg_wellbeing: Float.round(avg_wellbeing, 4),
      tick_interval_ms: @tick_interval
    }

    {:reply, stats, state}
  end

  @impl true
  def handle_cast({:unregister_avatar, avatar_id}, state) do
    case Map.get(state.avatar_mapping, avatar_id) do
      nil ->
        {:noreply, state}

      slot ->
        new_state = %{state |
          avatar_mapping: Map.delete(state.avatar_mapping, avatar_id),
          free_slots: [slot | state.free_slots],
          active_count: state.active_count - 1
        }
        {:noreply, new_state}
    end
  end

  @impl true
  def handle_cast({:apply_stimulus, avatar_id, stimulus_type, intensity}, state) do
    case Map.get(state.avatar_mapping, avatar_id) do
      nil ->
        {:noreply, state}

      slot ->
        new_state = apply_stimulus_to_slot(state, slot, stimulus_type, intensity)
        {:noreply, new_state}
    end
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
  def handle_cast(:force_tick, state) do
    {:noreply, do_tick(state)}
  end

  @impl true
  def handle_info(:tick, state) do
    new_state = do_tick(state)
    schedule_tick()
    {:noreply, new_state}
  end

  @impl true
  def handle_info(:train, state) do
    new_state =
      if state.training_enabled do
        case do_train_step(state) do
          {:ok, updated_state, loss} ->
            Logger.debug("[QuantumWorld] Train loss: #{Float.round(loss, 4)}")
            updated_state
          {:error, _} ->
            state
        end
      else
        state
      end

    schedule_train()
    {:noreply, new_state}
  end

  # ============================================================================
  # Private Functions
  # ============================================================================

  defp schedule_tick do
    Process.send_after(self(), :tick, @tick_interval)
  end

  defp schedule_train do
    Process.send_after(self(), :train, @train_interval)
  end

  defp do_tick(state) do
    if state.active_count == 0 do
      state
    else
      start_time = System.monotonic_time(:microsecond)

      # === GPU KERNEL TICK ===
      # Process ALL avatars in ONE GPU call!
      dt = @tick_interval / 1000.0  # Convert to seconds

      {new_bio, new_emotion} = Kernel.tick(
        state.bio_tensor,
        state.emotion_tensor,
        state.traits_tensor,
        dt
      )

      # Calculate rewards for experience collection
      rewards = Kernel.compute_reward(new_bio, new_emotion)

      # Apply decoherence (variance grows over time)
      new_variance = apply_decoherence(state.variance_tensor, dt)

      # Decay epsilon for exploration
      new_epsilon = max(@epsilon_end, state.epsilon * @epsilon_decay)

      end_time = System.monotonic_time(:microsecond)
      duration = end_time - start_time

      # Log occasionally
      if rem(state.total_steps, 100) == 0 and state.active_count > 0 do
        # rewards is 1D tensor [batch_size], slice only active avatars
        avg_reward = Nx.mean(Nx.slice(rewards, [0], [state.active_count])) |> Nx.to_number()
        Logger.debug("[QuantumWorld] Tick #{state.total_steps}: #{state.active_count} avatars, #{duration}μs, avg_reward=#{Float.round(avg_reward, 3)}")
      end

      %{state |
        bio_tensor: new_bio,
        emotion_tensor: new_emotion,
        variance_tensor: new_variance,
        epsilon: new_epsilon,
        total_steps: state.total_steps + 1,
        last_tick_us: duration
      }
    end
  end

  defp apply_decoherence(variance, dt) do
    # Variance grows slowly over time (quantum decoherence)
    growth = 0.001 * dt
    max_variance = 0.5
    Nx.min(Nx.add(variance, growth), max_variance)
  end

  defp apply_stimulus_to_slot(state, slot, stimulus_type, intensity) do
    bio_row = state.bio_tensor[slot] |> Nx.to_list()
    [dopamine, cortisol, oxytocin, adenosine, libido] = bio_row

    {new_dopa, new_cort, new_oxy, new_adeno, new_lib} =
      case stimulus_type do
        :social ->
          {min(dopamine + 0.1 * intensity, 1.0),
           cortisol,
           min(oxytocin + 0.15 * intensity, 1.0),
           adenosine,
           libido}

        :social_positive ->
          {min(dopamine + 0.1 * intensity, 1.0),
           cortisol,
           min(oxytocin + 0.15 * intensity, 1.0),
           adenosine,
           libido}

        :social_negative ->
          {dopamine,
           min(cortisol + 0.15 * intensity, 1.0),
           max(oxytocin - 0.1 * intensity, 0.0),
           adenosine,
           libido}

        :achievement ->
          {min(dopamine + 0.2 * intensity, 1.0),
           max(cortisol - 0.05 * intensity, 0.0),
           oxytocin,
           adenosine,
           libido}

        :rest ->
          {dopamine,
           max(cortisol - 0.1 * intensity, 0.0),
           oxytocin,
           max(adenosine - 0.15 * intensity, 0.0),
           libido}

        :threat ->
          {max(dopamine - 0.1 * intensity, 0.0),
           min(cortisol + 0.25 * intensity, 1.0),
           oxytocin,
           adenosine,
           libido}

        :novelty ->
          {min(dopamine + 0.15 * intensity, 1.0),
           min(cortisol + 0.05 * intensity, 1.0),
           oxytocin,
           adenosine,
           libido}

        :insight ->
          {min(dopamine + 0.12 * intensity, 1.0),
           cortisol,
           min(oxytocin + 0.08 * intensity, 1.0),
           adenosine,
           libido}

        _ ->
          {dopamine, cortisol, oxytocin, adenosine, libido}
      end

    new_bio_row = Nx.tensor([[new_dopa, new_cort, new_oxy, new_adeno, new_lib]], type: :f32)
    new_bio = Nx.put_slice(state.bio_tensor, [slot, 0], new_bio_row)

    # Stimulus reduces variance (observation/interaction)
    variance_row = state.variance_tensor[slot]
    new_variance_row = Nx.multiply(variance_row, 0.9) |> Nx.reshape({1, 8})
    new_variance = Nx.put_slice(state.variance_tensor, [slot, 0], new_variance_row)

    %{state | bio_tensor: new_bio, variance_tensor: new_variance}
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
          {loss, _actor_loss, _critic_loss} =
            PolicyNetwork.train_step(
              state.actor_critic_model,
              state.optimizer_state,
              state.params,
              batch
            )

          loss_value = Nx.to_number(loss)
          {:ok, state, loss_value}
      end
    end
  end
end
