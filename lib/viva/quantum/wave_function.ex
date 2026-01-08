defmodule Viva.Quantum.WaveFunction do
  @moduledoc """
  Probabilistic avatar state representation.

  Instead of a deterministic state vector, avatars exist in a "superposition"
  of possible states. Each dimension has a mean (expected value) and variance
  (uncertainty). Interactions "collapse" this to specific values.

  Quantum-inspired mechanics:
  - Superposition: Avatar exists in multiple states simultaneously
  - Measurement: Observing state collapses to specific value
  - Decoherence: Uncertainty grows over time without interaction
  - Entanglement: Social bonds create correlated state distributions
  """

  import Nx.Defn

  alias Viva.Quantum.StateVector

  defstruct [:mean, :variance, :last_collapse, :coherence]

  @type t :: %__MODULE__{
          mean: Nx.Tensor.t(),
          variance: Nx.Tensor.t(),
          last_collapse: DateTime.t(),
          coherence: float()
        }

  @dims StateVector.dims()
  @min_variance 0.001
  @max_variance 0.5
  @decoherence_rate 0.001

  @doc """
  Creates a wave function from a deterministic state vector.
  Initial variance is low (high certainty).
  """
  def from_state_vector(state_vector, opts \\ []) do
    initial_variance = Keyword.get(opts, :variance, 0.01)

    %__MODULE__{
      mean: state_vector,
      variance: Nx.broadcast(initial_variance, {@dims}) |> Nx.as_type(:f32),
      last_collapse: DateTime.utc_now(),
      coherence: 1.0
    }
  end

  @doc """
  Creates a wave function from an avatar.
  """
  def from_avatar(avatar, opts \\ []) do
    state_vector = StateVector.from_avatar(avatar)
    from_state_vector(state_vector, opts)
  end

  @doc """
  Creates a wave function with maximum uncertainty (fully mixed state).
  """
  def maximally_uncertain do
    # Means at center of ranges
    mean =
      Nx.tensor([
        0.5,
        0.3,
        0.4,
        0.3,
        0.0,
        0.0,
        0.0,
        0.7
      ])

    %__MODULE__{
      mean: mean,
      variance: Nx.broadcast(@max_variance, {@dims}) |> Nx.as_type(:f32),
      last_collapse: DateTime.utc_now(),
      coherence: 0.0
    }
  end

  @doc """
  Collapses the wave function to a specific state (measurement).
  Samples from the probability distribution.
  """
  def collapse(%__MODULE__{mean: mean, variance: variance} = wf, opts \\ []) do
    key = Keyword.get(opts, :key, Nx.Random.key(System.system_time()))

    # Sample from normal distribution: N(mean, sqrt(variance))
    std = Nx.sqrt(variance)
    {noise, _} = Nx.Random.normal(key, shape: {@dims}, type: :f32)

    collapsed_state = Nx.add(mean, Nx.multiply(noise, std))
    collapsed_state = StateVector.clamp(collapsed_state)

    # After collapse, variance reduces but doesn't go to zero
    new_variance = Nx.max(variance |> Nx.multiply(0.5), @min_variance)

    updated_wf = %{
      wf
      | variance: new_variance,
        last_collapse: DateTime.utc_now(),
        coherence: calculate_coherence(new_variance)
    }

    {collapsed_state, updated_wf}
  end

  @doc """
  Apply decoherence - uncertainty grows over time without interaction.
  Call this on each tick to simulate quantum decoherence.
  """
  def decohere(%__MODULE__{variance: variance, last_collapse: last} = wf) do
    now = DateTime.utc_now()
    elapsed_seconds = DateTime.diff(now, last, :second)

    # Variance grows logarithmically with time
    growth = @decoherence_rate * :math.log(1 + elapsed_seconds)
    new_variance = Nx.min(Nx.add(variance, growth), @max_variance)

    %{
      wf
      | variance: new_variance,
        coherence: calculate_coherence(new_variance)
    }
  end

  @doc """
  Apply a stimulus to the wave function.
  Shifts the mean and can affect variance.
  """
  defn apply_stimulus(mean, variance, stimulus_effect) do
    # Stimulus shifts the mean
    new_mean = Nx.add(mean, stimulus_effect)

    # Stimulus reduces uncertainty slightly (information gained)
    new_variance = Nx.max(Nx.multiply(variance, 0.95), @min_variance)

    {new_mean, new_variance}
  end

  def apply_stimulus_to_wf(%__MODULE__{mean: mean, variance: variance} = wf, stimulus_effect) do
    {new_mean, new_variance} = apply_stimulus(mean, variance, stimulus_effect)
    new_mean = StateVector.clamp(new_mean)

    %{
      wf
      | mean: new_mean,
        variance: new_variance,
        coherence: calculate_coherence(new_variance)
    }
  end

  @doc """
  Entangle two wave functions (social bond creates correlation).
  Both wave functions shift toward each other.
  """
  def entangle(%__MODULE__{} = wf1, %__MODULE__{} = wf2, strength \\ 0.1) do
    # Calculate midpoint (they pull toward each other)
    midpoint = Nx.add(wf1.mean, wf2.mean) |> Nx.divide(2)

    # Shift each toward midpoint based on strength
    new_mean1 = Nx.add(wf1.mean, Nx.multiply(Nx.subtract(midpoint, wf1.mean), strength))
    new_mean2 = Nx.add(wf2.mean, Nx.multiply(Nx.subtract(midpoint, wf2.mean), strength))

    # Entanglement reduces individual variance (correlated systems have shared information)
    correlation_factor = 1.0 - strength * 0.5
    new_var1 = Nx.multiply(wf1.variance, correlation_factor)
    new_var2 = Nx.multiply(wf2.variance, correlation_factor)

    {
      %{wf1 | mean: new_mean1, variance: new_var1, coherence: calculate_coherence(new_var1)},
      %{wf2 | mean: new_mean2, variance: new_var2, coherence: calculate_coherence(new_var2)}
    }
  end

  @doc """
  Calculate the expected wellbeing from the wave function.
  """
  def expected_wellbeing(%__MODULE__{mean: mean}) do
    StateVector.wellbeing(mean) |> Nx.to_number()
  end

  @doc """
  Calculate uncertainty in wellbeing (based on variance in contributing dimensions).
  """
  def wellbeing_uncertainty(%__MODULE__{variance: variance}) do
    # Wellbeing depends on dopamine, cortisol, oxytocin, pleasure
    # Sum their variances weighted by contribution to wellbeing
    weights = Nx.tensor([0.25, 0.2, 0.25, 0.0, 0.3, 0.0, 0.0, 0.0])
    weighted_var = Nx.dot(variance, weights)
    Nx.sqrt(weighted_var) |> Nx.to_number()
  end

  @doc """
  Get a human-readable summary of the wave function state.
  """
  def describe(%__MODULE__{mean: mean, variance: variance, coherence: coherence}) do
    mean_map = StateVector.to_map(mean)
    var_list = Nx.to_flat_list(variance)

    dim_descriptions =
      StateVector.dim_names()
      |> Enum.zip(var_list)
      |> Enum.map(fn {dim, var} ->
        certainty =
          cond do
            var < 0.05 -> "certain"
            var < 0.15 -> "probable"
            var < 0.3 -> "uncertain"
            true -> "indeterminate"
          end

        {dim, Map.get(mean_map, dim), certainty}
      end)

    %{
      dimensions: dim_descriptions,
      coherence: coherence,
      state_quality:
        cond do
          coherence > 0.8 -> :collapsed
          coherence > 0.5 -> :semi_coherent
          coherence > 0.2 -> :superposition
          true -> :maximally_mixed
        end
    }
  end

  # Private helpers

  defp calculate_coherence(variance) do
    # Coherence is inverse of average variance (normalized)
    avg_var = Nx.mean(variance) |> Nx.to_number()
    max(0.0, 1.0 - avg_var / @max_variance)
  end
end
