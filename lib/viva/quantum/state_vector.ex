defmodule Viva.Quantum.StateVector do
  @moduledoc """
  Nx tensor representation of avatar states for GPU-accelerated simulation.

  Maps avatar neurochemistry and emotional state to a numerical vector:
  - Index 0: dopamine (0.0 - 1.0)
  - Index 1: cortisol (0.0 - 1.0)
  - Index 2: oxytocin (0.0 - 1.0)
  - Index 3: adenosine (0.0 - 1.0)
  - Index 4: pleasure (-1.0 - 1.0)
  - Index 5: arousal (-1.0 - 1.0)
  - Index 6: dominance (-1.0 - 1.0)
  - Index 7: energy (0.0 - 1.0)
  """

  import Nx.Defn

  @state_dims 8
  @dim_names [:dopamine, :cortisol, :oxytocin, :adenosine, :pleasure, :arousal, :dominance, :energy]

  @doc """
  Returns the number of dimensions in a state vector.
  """
  def dims, do: @state_dims

  @doc """
  Returns the ordered list of dimension names.
  """
  def dim_names, do: @dim_names

  @doc """
  Creates a state vector from an avatar's current state.
  """
  def from_avatar(avatar) do
    bio = avatar.bio_state || %{}
    emotional = avatar.emotional_state || %{}

    values = [
      Map.get(bio, :dopamine, 0.5),
      Map.get(bio, :cortisol, 0.3),
      Map.get(bio, :oxytocin, 0.4),
      Map.get(bio, :adenosine, 0.2),
      Map.get(emotional, :pleasure, 0.0),
      Map.get(emotional, :arousal, 0.0),
      Map.get(emotional, :dominance, 0.0),
      Map.get(avatar, :energy, 0.7)
    ]

    Nx.tensor(values, type: :f32)
  end

  @doc """
  Creates a batch of state vectors from multiple avatars.
  Returns a {n_avatars, 8} tensor for GPU batch processing.
  """
  def batch_from_avatars(avatars) when is_list(avatars) do
    avatars
    |> Enum.map(&from_avatar/1)
    |> Nx.stack()
  end

  @doc """
  Extracts a map of named values from a state vector.
  """
  def to_map(state_vector) do
    values = Nx.to_flat_list(state_vector)

    @dim_names
    |> Enum.zip(values)
    |> Map.new()
  end

  @doc """
  Creates a random state vector (useful for initialization).
  """
  def random(opts \\ []) do
    key = Keyword.get(opts, :key, Nx.Random.key(System.system_time()))

    # Different ranges for different dimensions
    {dopamine, key} = Nx.Random.uniform(key, 0.3, 0.7, type: :f32)
    {cortisol, key} = Nx.Random.uniform(key, 0.1, 0.5, type: :f32)
    {oxytocin, key} = Nx.Random.uniform(key, 0.2, 0.6, type: :f32)
    {adenosine, key} = Nx.Random.uniform(key, 0.0, 0.4, type: :f32)
    {pleasure, key} = Nx.Random.uniform(key, -0.5, 0.5, type: :f32)
    {arousal, key} = Nx.Random.uniform(key, -0.5, 0.5, type: :f32)
    {dominance, key} = Nx.Random.uniform(key, -0.3, 0.3, type: :f32)
    {energy, _key} = Nx.Random.uniform(key, 0.5, 1.0, type: :f32)

    Nx.stack([dopamine, cortisol, oxytocin, adenosine, pleasure, arousal, dominance, energy])
    |> Nx.reshape({@state_dims})
  end

  @doc """
  Creates a batch of random state vectors.
  """
  def random_batch(n, opts \\ []) do
    key = Keyword.get(opts, :key, Nx.Random.key(System.system_time()))

    1..n
    |> Enum.reduce({[], key}, fn _, {acc, k} ->
      {new_key, k} = Nx.Random.split(k)
      vec = random(key: new_key)
      {[vec | acc], k}
    end)
    |> elem(0)
    |> Enum.reverse()
    |> Nx.stack()
  end

  @doc """
  Clamps all values in a state vector to valid ranges.
  """
  defn clamp(state_vector) do
    # Hormones: 0-1
    # PAD dimensions: -1 to 1
    # Energy: 0-1

    dopamine = Nx.clip(state_vector[0], 0.0, 1.0)
    cortisol = Nx.clip(state_vector[1], 0.0, 1.0)
    oxytocin = Nx.clip(state_vector[2], 0.0, 1.0)
    adenosine = Nx.clip(state_vector[3], 0.0, 1.0)
    pleasure = Nx.clip(state_vector[4], -1.0, 1.0)
    arousal = Nx.clip(state_vector[5], -1.0, 1.0)
    dominance = Nx.clip(state_vector[6], -1.0, 1.0)
    energy = Nx.clip(state_vector[7], 0.0, 1.0)

    Nx.stack([dopamine, cortisol, oxytocin, adenosine, pleasure, arousal, dominance, energy])
  end

  @doc """
  Calculates wellbeing score from state vector.
  Higher dopamine, oxytocin, pleasure and lower cortisol = higher wellbeing.
  """
  defn wellbeing(state_vector) do
    dopamine = state_vector[0]
    cortisol = state_vector[1]
    oxytocin = state_vector[2]
    pleasure = state_vector[4]

    # Wellbeing formula: positive factors minus negative factors
    dopamine * 0.25 + oxytocin * 0.25 + (pleasure + 1.0) / 2.0 * 0.3 - cortisol * 0.2
  end

  @doc """
  Calculates wellbeing for a batch of state vectors.
  Returns a {n} tensor of wellbeing scores.
  """
  defn batch_wellbeing(batch) do
    dopamine = batch[[.., 0]]
    cortisol = batch[[.., 1]]
    oxytocin = batch[[.., 2]]
    pleasure = batch[[.., 4]]

    dopamine * 0.25 + oxytocin * 0.25 + (pleasure + 1.0) / 2.0 * 0.3 - cortisol * 0.2
  end
end
