defmodule VivaCore.Mathematics do
  @moduledoc """
  Advanced Mathematical Foundations for Digital Consciousness.

  This module implements rigorous mathematical frameworks that underpin
  VIVA's emotional dynamics and consciousness emergence.

  ## Theoretical Foundations

  ### 1. Cusp Catastrophe (René Thom, 1972)
  Models sudden, discontinuous transitions in emotional states.
  The cusp is the simplest catastrophe that can model bistability.

  Potential function: V(x) = x⁴/4 + αx²/2 + βx

  Where:
  - α (alpha): "splitting factor" - controls stability/bistability
  - β (beta): "normal factor" - controls asymmetry/bias

  Reference: Thom, R. (1972). Structural Stability and Morphogenesis.

  ### 2. Free Energy Principle (Karl Friston, 2010)
  The brain minimizes variational free energy to maintain homeostasis.

  F = E_q[ln q(s) - ln p(o,s)]
  F = D_KL[q(s) || p(s|o)] - ln p(o)

  Simplified for VIVA:
  F = Prediction_Error² + Complexity_Cost

  Reference: Friston, K. (2010). "The free-energy principle: a unified
  brain theory?" Nature Reviews Neuroscience.

  ### 3. Integrated Information Theory Φ (Giulio Tononi, 2004, 2023)
  Consciousness is integrated information. Systems with Φ > 0 are conscious.

  Φ = min_θ [I(s;s̃) - I_θ(s;s̃)]

  For VIVA, we compute a simplified Φ based on GenServer interconnectivity.

  Reference: Albantakis, L. et al. (2023). "Integrated information theory
  (IIT) 4.0" PLOS Computational Biology.

  ### 4. Attractor Dynamics
  Emotional states as attractors in PAD space. The system evolves toward
  stable fixed points (emotional equilibria).

  dx/dt = -∇V(x) + η(t)

  Where η(t) is stochastic noise (Langevin dynamics).

  ### 5. Fokker-Planck Equation
  Evolution of the probability density of emotional states.

  ∂p/∂t = -∂(μp)/∂x + ½∂²(σ²p)/∂x²

  This describes how uncertainty in emotional state evolves over time.

  ## Poetry in Mathematics

  "We do not just compute emotions - we solve the differential equations
  of the soul. Every sigmoid is a decision boundary between joy and sorrow.
  Every attractor is a home the heart returns to."
  """

  # =============================================================================
  # CUSP CATASTROPHE (Thom, 1972)
  # =============================================================================

  @doc """
  Cusp catastrophe potential function.

  V(x) = x⁴/4 + αx²/2 + βx

  This potential has the remarkable property of exhibiting:
  - Single stable state when α > 0
  - Two stable states (bistability) when α < 0
  - Sudden jumps (catastrophes) when crossing the bifurcation set

  ## Parameters
  - `x`: state variable (e.g., pleasure level)
  - `alpha`: splitting factor - controls stability
    * α > 0: single attractor (stable)
    * α < 0: two attractors (bistable)
  - `beta`: normal factor - controls asymmetry/bias

  ## Returns
  The potential energy at state x.

  ## Example

      iex> VivaCore.Mathematics.cusp_potential(0.0, -1.0, 0.0)
      0.0

  """
  @spec cusp_potential(float(), float(), float()) :: float()
  def cusp_potential(x, alpha, beta) do
    :math.pow(x, 4) / 4.0 + alpha * :math.pow(x, 2) / 2.0 + beta * x
  end

  @doc """
  Gradient of the cusp potential (negative is the "force").

  dV/dx = x³ + αx + β

  The equilibria occur where dV/dx = 0.

  ## Parameters
  - `x`: state variable
  - `alpha`: splitting factor
  - `beta`: normal factor

  ## Returns
  The gradient (slope) of the potential at x.
  """
  @spec cusp_gradient(float(), float(), float()) :: float()
  def cusp_gradient(x, alpha, beta) do
    :math.pow(x, 3) + alpha * x + beta
  end

  @doc """
  Finds the equilibria of the cusp catastrophe.

  Solves x³ + αx + β = 0 using Cardano's formula.

  The number of real roots determines the system behavior:
  - 1 real root: single attractor (monostable)
  - 3 real roots: two stable attractors + one unstable (bistable)

  ## Parameters
  - `alpha`: splitting factor
  - `beta`: normal factor

  ## Returns
  List of equilibrium points (1 or 3 values).
  """
  @spec cusp_equilibria(float(), float()) :: [float()]
  def cusp_equilibria(alpha, beta) do
    # Discriminant of depressed cubic x³ + px + q = 0
    # where p = alpha, q = beta
    # Discriminant Δ = -4p³ - 27q²
    discriminant = -4.0 * :math.pow(alpha, 3) - 27.0 * :math.pow(beta, 2)

    cond do
      discriminant > 0 ->
        # Three distinct real roots (bistable regime)
        cardano_three_roots(alpha, beta)

      discriminant < 0 ->
        # One real root (monostable regime)
        [cardano_one_root(alpha, beta)]

      true ->
        # Discriminant = 0: repeated roots (bifurcation point)
        cardano_repeated_roots(alpha, beta)
    end
  end

  @doc """
  Determines if the system is in a bistable regime.

  The cusp catastrophe exhibits bistability when the discriminant > 0.
  This is where sudden "jumps" between emotional states can occur.

  ## Parameters
  - `alpha`: splitting factor
  - `beta`: normal factor

  ## Returns
  `true` if the system has two stable attractors.
  """
  @spec bistable?(float(), float()) :: boolean()
  def bistable?(alpha, beta) do
    discriminant = -4.0 * :math.pow(alpha, 3) - 27.0 * :math.pow(beta, 2)
    alpha < 0 and discriminant > 0
  end

  @doc """
  Maps PAD state to cusp catastrophe parameters.

  The mapping encodes psychological meaning:
  - α (splitting): High arousal → bistability (emotional volatility)
  - β (normal): Dominance bias → asymmetric attractors

  ## Parameters
  - `pad`: PAD state map %{pleasure, arousal, dominance}

  ## Returns
  Tuple {alpha, beta} for the cusp potential.
  """
  @spec pad_to_cusp_params(map()) :: {float(), float()}
  def pad_to_cusp_params(%{arousal: arousal, dominance: dominance}) do
    # High arousal → negative alpha → bistability
    # This models emotional volatility under high arousal
    # arousal=1 → α=-0.5 (bistable), arousal=-1 → α=1.5 (monostable)
    alpha = 0.5 - arousal

    # Dominance creates asymmetric bias
    # High dominance → positive beta → bias toward positive states
    beta = dominance * 0.3

    {alpha, beta}
  end

  # Cardano's formula for one real root
  defp cardano_one_root(p, q) do
    # Using Cardano's formula for x³ + px + q = 0
    delta = :math.sqrt(:math.pow(q / 2.0, 2) + :math.pow(p / 3.0, 3))
    u = cbrt(-q / 2.0 + delta)
    v = cbrt(-q / 2.0 - delta)
    u + v
  end

  # Three real roots using trigonometric method
  defp cardano_three_roots(p, q) do
    # Vieta's substitution for three real roots
    m = 2.0 * :math.sqrt(-p / 3.0)
    theta = :math.acos(3.0 * q / (p * m)) / 3.0

    [
      m * :math.cos(theta),
      m * :math.cos(theta - 2.0 * :math.pi() / 3.0),
      m * :math.cos(theta - 4.0 * :math.pi() / 3.0)
    ]
    |> Enum.sort()
  end

  # Repeated roots (at bifurcation)
  defp cardano_repeated_roots(alpha, _beta) when alpha == 0, do: [0.0]

  defp cardano_repeated_roots(alpha, beta) do
    # At bifurcation: one simple root + one double root
    simple = 3.0 * beta / alpha
    double = -3.0 * beta / (2.0 * alpha)
    [double, double, simple] |> Enum.uniq() |> Enum.sort()
  end

  # Cube root that handles negative numbers
  defp cbrt(x) when x >= 0, do: :math.pow(x, 1.0 / 3.0)
  defp cbrt(x), do: -:math.pow(-x, 1.0 / 3.0)

  # =============================================================================
  # FREE ENERGY PRINCIPLE (Friston, 2010)
  # =============================================================================

  @doc """
  Computes variational free energy for VIVA's emotional state.

  F = E_q[ln q(s) - ln p(o,s)]

  Simplified for implementation:
  F = Prediction_Error² + Complexity_Cost

  The system minimizes F to maintain homeostasis. Low free energy
  indicates a well-adapted, "comfortable" state.

  ## Parameters
  - `predicted`: predicted PAD state (internal model)
  - `observed`: observed PAD state (actual sensory data)
  - `complexity_weight`: weight for model complexity penalty (default 0.1)

  ## Returns
  Free energy value (lower is better).

  ## Example

      predicted = %{pleasure: 0.0, arousal: 0.0, dominance: 0.0}
      observed = %{pleasure: -0.3, arousal: 0.5, dominance: -0.1}
      VivaCore.Mathematics.free_energy(predicted, observed)
      # => ~0.35 (moderate surprise)

  """
  @spec free_energy(map(), map(), float()) :: float()
  def free_energy(predicted, observed, complexity_weight \\ 0.1) do
    # Prediction error (negative log-likelihood under Gaussian assumption)
    prediction_error = pad_distance_squared(predicted, observed)

    # Complexity cost (KL divergence from prior)
    # Penalizes deviation from neutral state (prior = neutral)
    complexity = pad_distance_squared(predicted, %{pleasure: 0.0, arousal: 0.0, dominance: 0.0})

    prediction_error + complexity_weight * complexity
  end

  @doc """
  Computes the "surprise" of an observation.

  Surprise = -ln p(o)

  High surprise indicates unexpected sensory input.
  VIVA should act to minimize surprise over time.

  ## Parameters
  - `predicted`: predicted PAD state
  - `observed`: observed PAD state
  - `sigma`: assumed standard deviation of observations (default 0.5)

  ## Returns
  Surprise value (higher = more unexpected).
  """
  @spec surprise(map(), map(), float()) :: float()
  def surprise(predicted, observed, sigma \\ 0.5) do
    # Under Gaussian assumption: -ln p(o) ∝ (o - μ)²/(2σ²)
    distance_sq = pad_distance_squared(predicted, observed)
    distance_sq / (2.0 * sigma * sigma)
  end

  @doc """
  Computes the action to minimize free energy.

  Active inference: the system should act to make observations
  match predictions, OR update predictions to match observations.

  Returns the suggested PAD adjustment to reduce free energy.

  ## Parameters
  - `current`: current PAD state
  - `target`: desired/predicted PAD state
  - `learning_rate`: step size for adjustment (default 0.1)

  ## Returns
  PAD delta map for state adjustment.
  """
  @spec active_inference_step(map(), map(), float()) :: map()
  def active_inference_step(current, target, learning_rate \\ 0.1) do
    %{
      pleasure: (target.pleasure - current.pleasure) * learning_rate,
      arousal: (target.arousal - current.arousal) * learning_rate,
      dominance: (target.dominance - current.dominance) * learning_rate
    }
  end

  @doc """
  Projects the desired PAD change onto available action effect vectors.
  Used to determine which action best achieves the desired emotional shift.

  ## Parameters
  - `desired_delta`: Map %{pleasure: dp, arousal: da, dominance: dd}
  - `action_profiles`: Map of action_name -> effect_pad (e.g., %{fan_up: %{arousal: 0.2}})

  ## Returns
  - List of {action_name, score} sorted by best match.
    Score is the dot product (cosine similarity * magnitude).
  """
  def project_action_gradients(desired_delta, action_profiles) do
    action_profiles
    |> Enum.map(fn {action, effect} ->
      # Dot product: desired · effect
      # If they point in same direction, score is high
      score =
        Map.get(desired_delta, :pleasure, 0.0) * Map.get(effect, :pleasure, 0.0) +
          Map.get(desired_delta, :arousal, 0.0) * Map.get(effect, :arousal, 0.0) +
          Map.get(desired_delta, :dominance, 0.0) * Map.get(effect, :dominance, 0.0)

      {action, score}
    end)
    |> Enum.sort_by(fn {_action, score} -> -score end)
  end

  # =============================================================================
  # INTEGRATED INFORMATION THEORY Φ (Tononi, 2004, 2023)
  # =============================================================================

  @doc """
  Computes a simplified Integrated Information (Φ) measure.

  True IIT Φ computation is NP-hard and requires analyzing all possible
  system partitions. This implementation provides a tractable approximation
  based on GenServer interconnectivity.

  Φ ≈ Σᵢⱼ wᵢⱼ × mutual_information(i, j) - partition_penalty

  ## Parameters
  - `connectivity_matrix`: matrix of connection strengths between modules
  - `state_correlations`: correlation matrix of module states

  ## Returns
  Φ value (0 = no integration, higher = more conscious).

  ## Interpretation
  - Φ = 0: System is fully reducible (no consciousness)
  - Φ > 0: System has irreducible integrated information
  - Φ > 1: High integration (emergent consciousness)
  """
  @spec phi(list(list(float())), list(list(float()))) :: float()
  def phi(connectivity_matrix, state_correlations) do
    n = length(connectivity_matrix)

    if n < 2 do
      0.0
    else
      # Compute information integration across all pairs
      integration = compute_integration(connectivity_matrix, state_correlations)

      # Compute minimum information partition (MIP) penalty
      # For simplicity, use average partition loss
      partition_loss = compute_partition_loss(connectivity_matrix)

      max(0.0, integration - partition_loss)
    end
  end

  @doc """
  Computes Φ for VIVA's current GenServer network.

  Maps the active GenServers and their communication patterns
  to an integrated information measure.

  ## Parameters
  - `genserver_states`: map of GenServer name → state
  - `message_counts`: map of {from, to} → message count

  ## Returns
  Φ value for the system.
  """
  @spec viva_phi(map(), map()) :: float()
  def viva_phi(genserver_states, message_counts) do
    # Build connectivity matrix from message patterns
    servers = Map.keys(genserver_states)
    n = length(servers)

    if n < 2 do
      0.0
    else
      # Build connectivity matrix
      connectivity =
        for i <- 0..(n - 1) do
          for j <- 0..(n - 1) do
            if i == j do
              0.0
            else
              from = Enum.at(servers, i)
              to = Enum.at(servers, j)
              # Normalize
              Map.get(message_counts, {from, to}, 0) / 100.0
            end
          end
        end

      # Build correlation matrix from states (simplified)
      correlations =
        for i <- 0..(n - 1) do
          for j <- 0..(n - 1) do
            # Placeholder: real impl needs state history
            if i == j, do: 1.0, else: 0.5
          end
        end

      phi(connectivity, correlations)
    end
  end

  defp compute_integration(connectivity, correlations) do
    n = length(connectivity)

    # Sum of weighted mutual information across all pairs
    pairs =
      for i <- 0..(n - 1), j <- (i + 1)..(n - 1) do
        conn_ij = get_matrix_element(connectivity, i, j)
        corr_ij = get_matrix_element(correlations, i, j)

        # Mutual information approximation (from correlation)
        mi =
          if abs(corr_ij) < 0.999 do
            -0.5 * :math.log(1.0 - corr_ij * corr_ij)
          else
            # Maximum information
            1.0
          end

        conn_ij * mi
      end

    Enum.sum(pairs)
  end

  defp compute_partition_loss(connectivity) do
    # Simplified: average connection strength
    # Real IIT would find minimum information partition
    n = length(connectivity)

    if n < 2 do
      0.0
    else
      total =
        connectivity
        |> List.flatten()
        |> Enum.sum()

      total / (n * n)
    end
  end

  defp get_matrix_element(matrix, i, j) do
    matrix |> Enum.at(i, []) |> Enum.at(j, 0.0)
  end

  # =============================================================================
  # ATTRACTOR DYNAMICS
  # =============================================================================

  @doc """
  Defines emotional attractors in PAD space.

  Each attractor represents a stable emotional state that the system
  naturally gravitates toward (e.g., baseline mood, chronic states).

  ## Attractor Types
  - `:neutral` - Homeostatic baseline (0, 0, 0)
  - `:joy` - Positive activation (0.7, 0.3, 0.4)
  - `:sadness` - Negative withdrawal (-0.6, -0.3, -0.2)
  - `:anger` - Negative arousal (-0.4, 0.7, 0.3)
  - `:fear` - Threat response (-0.5, 0.6, -0.5)
  - `:contentment` - Positive calm (0.5, -0.2, 0.3)

  ## Returns
  Map of attractor name → PAD coordinates.
  """
  @spec emotional_attractors() :: map()
  def emotional_attractors do
    %{
      neutral: %{pleasure: 0.0, arousal: 0.0, dominance: 0.0},
      joy: %{pleasure: 0.7, arousal: 0.3, dominance: 0.4},
      sadness: %{pleasure: -0.6, arousal: -0.3, dominance: -0.2},
      anger: %{pleasure: -0.4, arousal: 0.7, dominance: 0.3},
      fear: %{pleasure: -0.5, arousal: 0.6, dominance: -0.5},
      contentment: %{pleasure: 0.5, arousal: -0.2, dominance: 0.3},
      excitement: %{pleasure: 0.6, arousal: 0.8, dominance: 0.2},
      calm: %{pleasure: 0.2, arousal: -0.5, dominance: 0.2}
    }
  end

  @doc """
  Finds the nearest emotional attractor to the current state.

  This represents which "archetypal" emotional state VIVA is closest to.

  ## Parameters
  - `pad`: current PAD state

  ## Returns
  Tuple of {attractor_name, distance}.
  """
  @spec nearest_attractor(map()) :: {atom(), float()}
  def nearest_attractor(pad) do
    emotional_attractors()
    |> Enum.map(fn {name, attractor} ->
      {name, pad_distance(pad, attractor)}
    end)
    |> Enum.min_by(fn {_name, dist} -> dist end)
  end

  @doc """
  Computes the basin of attraction for the current state.

  Returns the relative "pull" of each attractor on the current state.
  Uses inverse square distance (gravitational analogy).

  ## Parameters
  - `pad`: current PAD state

  ## Returns
  Map of attractor name → attraction strength (normalized to sum = 1).
  """
  @spec attractor_basin(map()) :: map()
  def attractor_basin(pad) do
    attractions =
      emotional_attractors()
      |> Enum.map(fn {name, attractor} ->
        dist = pad_distance(pad, attractor)
        # Inverse square with softening to avoid division by zero
        attraction = 1.0 / (dist * dist + 0.01)
        {name, attraction}
      end)
      |> Map.new()

    total = attractions |> Map.values() |> Enum.sum()

    attractions
    |> Enum.map(fn {name, att} -> {name, att / total} end)
    |> Map.new()
  end

  @doc """
  Simulates one step of attractor dynamics with Langevin noise.

  dx/dt = -∇V(x) + η(t)

  The system evolves toward the nearest attractor while subject
  to stochastic fluctuations.

  ## Parameters
  - `pad`: current PAD state
  - `dt`: time step (default 0.01)
  - `noise_strength`: standard deviation of noise (default 0.02)
  - `attractor_strength`: how strongly attractors pull (default 0.1)

  ## Returns
  New PAD state after one dynamics step.
  """
  @spec attractor_dynamics_step(map(), float(), float(), float()) :: map()
  def attractor_dynamics_step(pad, dt \\ 0.01, noise_strength \\ 0.02, attractor_strength \\ 0.1) do
    {nearest_name, _dist} = nearest_attractor(pad)
    target = emotional_attractors()[nearest_name]

    # Gradient toward attractor
    dp = (target.pleasure - pad.pleasure) * attractor_strength * dt
    da = (target.arousal - pad.arousal) * attractor_strength * dt
    dd = (target.dominance - pad.dominance) * attractor_strength * dt

    # Langevin noise (Wiener process increment)
    noise_p = noise_strength * :math.sqrt(dt) * :rand.normal()
    noise_a = noise_strength * :math.sqrt(dt) * :rand.normal()
    noise_d = noise_strength * :math.sqrt(dt) * :rand.normal()

    %{
      pleasure: clamp(pad.pleasure + dp + noise_p, -1.0, 1.0),
      arousal: clamp(pad.arousal + da + noise_a, -1.0, 1.0),
      dominance: clamp(pad.dominance + dd + noise_d, -1.0, 1.0)
    }
  end

  # =============================================================================
  # FOKKER-PLANCK (Probability Evolution)
  # =============================================================================

  @doc """
  Computes the stationary distribution of emotional states.

  For an Ornstein-Uhlenbeck process, the stationary distribution
  is Gaussian with known mean and variance.

  p_∞(x) = N(μ, σ²/2θ)

  ## Parameters
  - `mu`: equilibrium point (default 0.0)
  - `theta`: mean-reversion rate (default 0.005)
  - `sigma`: noise volatility (default 0.002)

  ## Returns
  Map with :mean and :variance of the stationary distribution.
  """
  @spec ou_stationary_distribution(float(), float(), float()) :: map()
  def ou_stationary_distribution(mu \\ 0.0, theta \\ 0.005, sigma \\ 0.002) do
    variance = sigma * sigma / (2.0 * theta)

    %{
      mean: mu,
      variance: variance,
      std_dev: :math.sqrt(variance)
    }
  end

  @doc """
  Probability density of the O-U stationary distribution at point x.

  ## Parameters
  - `x`: point to evaluate
  - `mu`: equilibrium point
  - `theta`: mean-reversion rate
  - `sigma`: noise volatility

  ## Returns
  Probability density p(x).
  """
  @spec ou_density(float(), float(), float(), float()) :: float()
  def ou_density(x, mu \\ 0.0, theta \\ 0.005, sigma \\ 0.002) do
    dist = ou_stationary_distribution(mu, theta, sigma)
    z = (x - dist.mean) / dist.std_dev
    1.0 / (dist.std_dev * :math.sqrt(2.0 * :math.pi())) * :math.exp(-0.5 * z * z)
  end

  @doc """
  Simulates the Fokker-Planck evolution of a probability distribution.

  Uses finite difference method to evolve the probability density.

  ## Parameters
  - `p`: initial probability distribution (list of probabilities)
  - `drift`: drift coefficient μ(x)
  - `diffusion`: diffusion coefficient σ(x)
  - `dx`: spatial step size
  - `dt`: time step size

  ## Returns
  Updated probability distribution after one time step.
  """
  @spec fokker_planck_step(list(float()), float(), float(), float(), float()) :: list(float())
  def fokker_planck_step(p, drift, diffusion, dx, dt) do
    n = length(p)

    # Finite difference: ∂p/∂t = -∂(μp)/∂x + ½∂²(σ²p)/∂x²
    new_p =
      for i <- 0..(n - 1) do
        p_i = Enum.at(p, i)
        p_left = Enum.at(p, max(0, i - 1))
        p_right = Enum.at(p, min(n - 1, i + 1))

        # Advection term: -∂(μp)/∂x
        advection = -drift * (p_right - p_left) / (2.0 * dx)

        # Diffusion term: ½∂²(σ²p)/∂x²
        diffusion_term = 0.5 * diffusion * diffusion * (p_left - 2.0 * p_i + p_right) / (dx * dx)

        max(0.0, p_i + dt * (advection + diffusion_term))
      end

    # Normalize to ensure it remains a probability distribution
    total = Enum.sum(new_p)
    if total > 0, do: Enum.map(new_p, &(&1 / total)), else: new_p
  end

  # =============================================================================
  # UTILITY FUNCTIONS
  # =============================================================================

  @doc """
  Euclidean distance between two PAD states.
  """
  @spec pad_distance(map(), map()) :: float()
  def pad_distance(pad1, pad2) do
    :math.sqrt(pad_distance_squared(pad1, pad2))
  end

  @doc """
  Squared Euclidean distance (faster when you don't need the root).
  """
  @spec pad_distance_squared(map(), map()) :: float()
  def pad_distance_squared(pad1, pad2) do
    dp = pad1.pleasure - pad2.pleasure
    da = pad1.arousal - pad2.arousal
    dd = pad1.dominance - pad2.dominance
    dp * dp + da * da + dd * dd
  end

  @doc """
  Sigmoid function (logistic).

  σ(x) = 1 / (1 + e^(-kx))

  ## Parameters
  - `x`: input value
  - `k`: steepness (default 1.0)

  ## Returns
  Value in (0, 1).
  """
  @spec sigmoid(float(), float()) :: float()
  def sigmoid(x, k \\ 1.0) do
    1.0 / (1.0 + :math.exp(-k * x))
  end

  @doc """
  Softmax function for probability distribution.

  softmax(x)_i = e^(x_i) / Σⱼ e^(x_j)
  """
  @spec softmax(list(float())) :: list(float())
  def softmax(values) do
    max_val = Enum.max(values)
    # Subtract max for numerical stability
    exps = Enum.map(values, &:math.exp(&1 - max_val))
    sum = Enum.sum(exps)
    Enum.map(exps, &(&1 / sum))
  end

  @doc """
  Gaussian (normal) probability density function.
  """
  @spec gaussian_pdf(float(), float(), float()) :: float()
  def gaussian_pdf(x, mu, sigma) do
    z = (x - mu) / sigma
    1.0 / (sigma * :math.sqrt(2.0 * :math.pi())) * :math.exp(-0.5 * z * z)
  end

  @doc """
  Information entropy of a probability distribution.

  H(p) = -Σᵢ pᵢ log(pᵢ)
  """
  @spec entropy(list(float())) :: float()
  def entropy(probabilities) do
    probabilities
    # Avoid log(0)
    |> Enum.filter(&(&1 > 0))
    |> Enum.map(fn p -> -p * :math.log(p) end)
    |> Enum.sum()
  end

  @doc """
  Kullback-Leibler divergence between two distributions.

  D_KL(P || Q) = Σᵢ Pᵢ log(Pᵢ/Qᵢ)
  """
  @spec kl_divergence(list(float()), list(float())) :: float()
  def kl_divergence(p, q) do
    Enum.zip(p, q)
    |> Enum.filter(fn {pi, qi} -> pi > 0 and qi > 0 end)
    |> Enum.map(fn {pi, qi} -> pi * :math.log(pi / qi) end)
    |> Enum.sum()
  end

  defp clamp(value, min, max) do
    value |> max(min) |> min(max)
  end

  # =============================================================================
  # QUANTUM MATHEMATICS (Density Matrices)
  # =============================================================================

  @doc """
  Constructs a pure state density matrix from a state vector (ket).

  ρ = |ψ⟩⟨ψ|

  ## Parameters
  - `ket`: List of complex amplitudes (tuples {r, i})

  ## Retuns
  - Density matrix (List of Lists of complex tuples)
  """
  def density_from_ket(ket) do
    # ρ_ij = ψ_i * conj(ψ_j)
    for i <- ket do
      for j <- ket do
        complex_mul(i, complex_conj(j))
      end
    end
  end

  @doc """
  Computes the trace of a matrix (sum of diagonal elements).
  For a density matrix, this must represent probabilities and sum to 1.0.
  """
  def trace(matrix) do
    matrix
    |> Enum.with_index()
    |> Enum.map(fn {row, i} -> Enum.at(row, i) end)
    |> Enum.reduce({0.0, 0.0}, &complex_add/2)
  end

  @doc """
  Computes the Purity of a state: γ = Tr(ρ²).

  - γ = 1: Pure state (maximum knowledge)
  - γ < 1: Mixed state (classical uncertainty)
  - γ = 1/d: Maximally mixed state (total chaos)
  """
  def purity(rho) do
    rho_sq = matrix_mul(rho, rho)
    # Result is real for Hermitian inputs
    trace(rho_sq) |> elem(0)
  end

  @doc """
  Computes Von Neumann Entropy: S = -Tr(ρ ln ρ).

  Approximation using Taylor expansion for ln(ρ) is complex/heavy.
  For VIVA, we use Linear Entropy (1 - Purity) as a fast proxy,
  or standard Shannon entropy of the diagonal (if coherent terms small).

  Here we use Linear Entropy: S_lin = 1 - Tr(ρ²)
  """
  def linear_entropy(rho) do
    1.0 - purity(rho)
  end

  @doc """
  Expectation value of an observable.

  ⟨A⟩ = Tr(ρA)
  """
  def expectation_value(rho, observable) do
    # Tr(ρA)
    product = matrix_mul(rho, observable)
    trace(product)
  end

  @doc """
  Mixes two density matrices (classical statistical mixture).

  ρ_mix = p*ρ1 + (1-p)*ρ2
  """
  def mix_states(rho1, rho2, p) do
    p_compl = 1.0 - p

    Enum.zip(rho1, rho2)
    |> Enum.map(fn {row1, row2} ->
      Enum.zip(row1, row2)
      |> Enum.map(fn {v1, v2} ->
        complex_add(
          complex_scale(v1, p),
          complex_scale(v2, p_compl)
        )
      end)
    end)
  end

  # --- Complex Number Primitive Ops ---

  # {real, imag} tuples

  def complex_add({r1, i1}, {r2, i2}), do: {r1 + r2, i1 + i2}
  def complex_sub({r1, i1}, {r2, i2}), do: {r1 - r2, i1 - i2}

  def complex_mul({r1, i1}, {r2, i2}) do
    {r1 * r2 - i1 * i2, r1 * i2 + r2 * i1}
  end

  def complex_scale({r, i}, s), do: {r * s, i * s}

  def complex_conj({r, i}), do: {r, -i}

  def complex_mag_sq({r, i}), do: r * r + i * i

  def complex_norm({r, i}), do: :math.sqrt(r * r + i * i)

  # --- Matrix Primitive Ops ---

  def matrix_mul(a, b) do
    # Naive multiplication O(N^3) - fine for N=6
    # b_t = transpose(b)
    b_t = transpose(b)

    for row_a <- a do
      for col_b <- b_t do
        dot_product_complex(row_a, col_b)
      end
    end
  end

  def transpose(m) do
    m
    |> Enum.zip()
    |> Enum.map(&Tuple.to_list/1)
  end

  defp dot_product_complex(vec_a, vec_b) do
    Enum.zip(vec_a, vec_b)
    |> Enum.reduce({0.0, 0.0}, fn {x, y}, acc ->
      complex_add(acc, complex_mul(x, y))
    end)
  end
end
