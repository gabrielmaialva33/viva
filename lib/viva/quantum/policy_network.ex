defmodule Viva.Quantum.PolicyNetwork do
  @moduledoc """
  Axon neural network for RL policy that learns optimal stimulus selection.

  Input: Avatar state vector (8 dimensions)
  Output: Probability distribution over possible stimuli (6 actions)

  Uses Actor-Critic architecture:
  - Actor: outputs action probabilities
  - Critic: estimates state value for advantage calculation

  Actions (stimuli):
  0 - :social     (boost oxytocin, potential dopamine)
  1 - :novelty    (boost dopamine, slight cortisol)
  2 - :threat     (spike cortisol, suppress dopamine)
  3 - :rest       (reduce cortisol, slight dopamine recovery)
  4 - :achievement (big dopamine boost)
  5 - :insight    (dopamine + oxytocin)
  """

  alias Viva.Quantum.StateVector

  @state_dims StateVector.dims()
  @num_actions 6
  @action_names [:social, :novelty, :threat, :rest, :achievement, :insight]

  @doc """
  Returns the action names in order.
  """
  def action_names, do: @action_names

  @doc """
  Returns the number of possible actions.
  """
  def num_actions, do: @num_actions

  @doc """
  Builds the Actor network (policy).
  Returns action probabilities given a state.
  """
  def build_actor do
    Axon.input("state", shape: {nil, @state_dims})
    |> Axon.dense(64, activation: :relu, name: "actor_dense1")
    |> Axon.dropout(rate: 0.1, name: "actor_dropout1")
    |> Axon.dense(32, activation: :relu, name: "actor_dense2")
    |> Axon.dense(@num_actions, activation: :softmax, name: "actor_output")
  end

  @doc """
  Builds the Critic network (value function).
  Returns state value estimation.
  """
  def build_critic do
    Axon.input("state", shape: {nil, @state_dims})
    |> Axon.dense(64, activation: :relu, name: "critic_dense1")
    |> Axon.dropout(rate: 0.1, name: "critic_dropout1")
    |> Axon.dense(32, activation: :relu, name: "critic_dense2")
    |> Axon.dense(1, name: "critic_output")
  end

  @doc """
  Builds the combined Actor-Critic model.
  Returns both action probabilities and value estimate.
  """
  def build_actor_critic do
    input = Axon.input("state", shape: {nil, @state_dims})

    # Shared layers
    shared =
      input
      |> Axon.dense(64, activation: :relu, name: "shared_dense1")
      |> Axon.dropout(rate: 0.1, name: "shared_dropout")
      |> Axon.dense(32, activation: :relu, name: "shared_dense2")

    # Actor head
    actor_output =
      shared
      |> Axon.dense(@num_actions, activation: :softmax, name: "actor_head")

    # Critic head
    critic_output =
      shared
      |> Axon.dense(1, name: "critic_head")

    # Container output
    Axon.container({actor_output, critic_output})
  end

  @doc """
  Initializes model parameters randomly.
  Returns Axon.ModelState struct (Axon 0.8+ API).
  """
  def init_params(model, _opts \\ []) do
    template = %{"state" => Nx.template({1, @state_dims}, :f32)}
    {init_fn, _predict_fn} = Axon.build(model)
    # Axon 0.8: init_fn takes (template, params) - empty map for fresh initialization
    init_fn.(template, %{})
  end

  @doc """
  Selects an action given a state and model parameters.
  Uses epsilon-greedy exploration.
  """
  def select_action(model, params, state, opts \\ []) do
    epsilon = Keyword.get(opts, :epsilon, 0.1)
    key = Keyword.get(opts, :key, Nx.Random.key(System.system_time()))

    # Get action probabilities from model (Axon 0.8+ API)
    state_batch = Nx.reshape(state, {1, @state_dims})
    probs = Axon.predict(model, params, %{"state" => state_batch})

    # Epsilon-greedy: explore with probability epsilon
    {rand_val, key} = Nx.Random.uniform(key, type: :f32)

    action =
      if Nx.to_number(rand_val) < epsilon do
        # Random exploration
        {action_idx, _} = Nx.Random.randint(key, 0, @num_actions)
        Nx.to_number(action_idx)
      else
        # Exploit: sample from probability distribution
        probs_flat = Nx.squeeze(probs)
        sample_from_probs(probs_flat, key)
      end

    {action, Enum.at(@action_names, action)}
  end

  @doc """
  Computes the policy loss for training.
  Uses policy gradient with baseline (advantage).
  """
  def policy_loss(actor_probs, actions, advantages) do
    # Get probabilities of taken actions
    batch_size = Nx.axis_size(actions, 0)
    indices = Nx.stack([Nx.iota({batch_size}), actions], axis: 1)
    action_probs = Nx.gather(actor_probs, indices)

    # Policy gradient loss: -log(pi(a|s)) * A(s,a)
    log_probs = Nx.log(Nx.add(action_probs, 1.0e-8))
    Nx.mean(Nx.negate(Nx.multiply(log_probs, advantages)))
  end

  @doc """
  Computes the value loss for the critic.
  """
  def value_loss(predicted_values, target_values) do
    Nx.mean(Nx.pow(Nx.subtract(predicted_values, target_values), 2))
  end

  @doc """
  Creates the training step function for actor-critic.
  """
  def train_step(model, _optimizer_state, params, batch) do
    %{states: states, actions: actions, rewards: rewards, next_states: next_states, dones: dones} =
      batch

    gamma = 0.99

    # Forward pass (Axon 0.8+ API)
    {actor_probs, values} = Axon.predict(model, params, %{"state" => states})
    {_, next_values} = Axon.predict(model, params, %{"state" => next_states})

    # Calculate TD targets
    next_values_masked = Nx.multiply(next_values, Nx.subtract(1.0, dones))
    td_targets = Nx.add(rewards, Nx.multiply(gamma, next_values_masked))

    # Advantages (detached from gradient computation)
    advantages = Nx.subtract(td_targets, values)

    # Combined loss (no gradient through advantages/targets in actor-critic)
    actor_loss = policy_loss(actor_probs, actions, advantages)
    critic_loss = value_loss(values, td_targets)
    total_loss = Nx.add(actor_loss, Nx.multiply(0.5, critic_loss))

    {total_loss, actor_loss, critic_loss}
  end

  @doc """
  Converts an action index to stimulus map for the Biology system.
  """
  def action_to_stimulus(action_idx) when is_integer(action_idx) do
    action_name = Enum.at(@action_names, action_idx)
    action_to_stimulus(action_name)
  end

  def action_to_stimulus(action_name) when is_atom(action_name) do
    base_intensity = 0.6

    case action_name do
      :social ->
        %{type: :social, intensity: base_intensity, valence: 0.3}

      :novelty ->
        %{type: :novelty, intensity: base_intensity, valence: 0.2}

      :threat ->
        %{type: :threat, intensity: base_intensity * 0.5, valence: -0.5}

      :rest ->
        %{type: :rest, intensity: base_intensity * 0.8, valence: 0.4}

      :achievement ->
        %{type: :achievement, intensity: base_intensity, valence: 0.5}

      :insight ->
        %{type: :insight, intensity: base_intensity * 0.7, valence: 0.4}

      _ ->
        %{type: :ambient, intensity: 0.3, valence: 0.0}
    end
  end

  @doc """
  Calculates entropy bonus for exploration.
  Higher entropy = more exploration.
  """
  def entropy(probs) do
    log_probs = Nx.log(Nx.add(probs, 1.0e-8))
    Nx.negate(Nx.sum(Nx.multiply(probs, log_probs)))
  end

  # Private helpers

  defp sample_from_probs(probs, key) do
    # Categorical sampling from probability distribution
    cumsum = Nx.cumulative_sum(probs)
    {rand, _} = Nx.Random.uniform(key, type: :f32)

    # Find first index where cumsum > rand
    mask = Nx.greater(cumsum, rand)
    # Argmax of boolean tensor gives first true index
    Nx.argmax(mask) |> Nx.to_number()
  end
end
