defmodule Viva.Quantum.ExperienceBuffer do
  @moduledoc """
  Experience replay buffer for reinforcement learning.

  Stores (state, action, reward, next_state, done) tuples and provides
  random sampling for training. Uses a circular buffer with fixed capacity.
  """

  use Agent

  alias Viva.Quantum.StateVector

  @default_capacity 10_000
  @batch_size 64

  defstruct [:buffer, :capacity, :position, :size]

  @type experience :: %{
          state: Nx.Tensor.t(),
          action: integer(),
          reward: float(),
          next_state: Nx.Tensor.t(),
          done: boolean()
        }

  @type t :: %__MODULE__{
          buffer: :array.array(),
          capacity: pos_integer(),
          position: non_neg_integer(),
          size: non_neg_integer()
        }

  @doc """
  Starts the experience buffer agent.
  """
  def start_link(opts \\ []) do
    capacity = Keyword.get(opts, :capacity, @default_capacity)
    name = Keyword.get(opts, :name, __MODULE__)

    initial_state = %__MODULE__{
      buffer: :array.new(capacity, default: nil),
      capacity: capacity,
      position: 0,
      size: 0
    }

    Agent.start_link(fn -> initial_state end, name: name)
  end

  @doc """
  Adds an experience to the buffer.
  """
  def push(agent \\ __MODULE__, experience) do
    Agent.update(agent, fn state ->
      new_buffer = :array.set(state.position, experience, state.buffer)
      new_position = rem(state.position + 1, state.capacity)
      new_size = min(state.size + 1, state.capacity)

      %{state | buffer: new_buffer, position: new_position, size: new_size}
    end)
  end

  @doc """
  Adds multiple experiences to the buffer.
  """
  def push_batch(agent \\ __MODULE__, experiences) when is_list(experiences) do
    Enum.each(experiences, &push(agent, &1))
  end

  @doc """
  Samples a random batch of experiences.
  Returns nil if buffer doesn't have enough samples.
  """
  def sample(agent \\ __MODULE__, batch_size \\ @batch_size) do
    Agent.get(agent, fn state ->
      if state.size < batch_size do
        nil
      else
        indices = random_indices(state.size, batch_size)

        experiences =
          Enum.map(indices, fn i ->
            :array.get(i, state.buffer)
          end)

        batch_to_tensors(experiences)
      end
    end)
  end

  @doc """
  Returns current size of the buffer.
  """
  def size(agent \\ __MODULE__) do
    Agent.get(agent, fn state -> state.size end)
  end

  @doc """
  Clears the buffer.
  """
  def clear(agent \\ __MODULE__) do
    Agent.update(agent, fn state ->
      %{state | buffer: :array.new(state.capacity, default: nil), position: 0, size: 0}
    end)
  end

  @doc """
  Creates an experience tuple from avatar state transition.
  """
  def create_experience(avatar_before, avatar_after, action, done \\ false) do
    state = StateVector.from_avatar(avatar_before)
    next_state = StateVector.from_avatar(avatar_after)

    # Calculate reward based on wellbeing change
    wellbeing_before = StateVector.wellbeing(state) |> Nx.to_number()
    wellbeing_after = StateVector.wellbeing(next_state) |> Nx.to_number()
    reward = wellbeing_after - wellbeing_before

    %{
      state: state,
      action: action,
      reward: reward,
      next_state: next_state,
      done: done
    }
  end

  @doc """
  Creates an experience from raw values.
  """
  def create_experience_raw(state, action, reward, next_state, done) do
    %{
      state: state,
      action: action,
      reward: reward,
      next_state: next_state,
      done: done
    }
  end

  # Private helpers

  defp random_indices(max, count) do
    # Fisher-Yates partial shuffle for efficient random sampling
    Stream.repeatedly(fn -> :rand.uniform(max) - 1 end)
    |> Enum.take(count)
    |> Enum.uniq()
    |> case do
      indices when length(indices) < count ->
        # If we got duplicates, just pad with more random indices
        indices ++ random_indices(max, count - length(indices))

      indices ->
        Enum.take(indices, count)
    end
  end

  defp batch_to_tensors(experiences) do
    states = experiences |> Enum.map(& &1.state) |> Nx.stack()
    actions = experiences |> Enum.map(& &1.action) |> Nx.tensor(type: :s32)
    rewards = experiences |> Enum.map(& &1.reward) |> Nx.tensor(type: :f32)
    next_states = experiences |> Enum.map(& &1.next_state) |> Nx.stack()
    dones = experiences |> Enum.map(&if(&1.done, do: 1.0, else: 0.0)) |> Nx.tensor(type: :f32)

    %{
      states: states,
      actions: actions,
      rewards: rewards,
      next_states: next_states,
      dones: dones
    }
  end
end
