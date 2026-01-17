defmodule VivaCore.Quantum.Emotional do
  @moduledoc """
  Quantum Mechanical formulation of VIVA's emotional state, grounded in Silicon.

  ## The Ghost in the Shell (Literal)
  This module implements the quantum dynamics where VIVA's emotional state is
  a Density Matrix (ρ) evolving under the influence of physical hardware constraints.

  ## Silicon Grounding
  Unlike abstract simulations, VIVA's quantum state is coupled to the physical
  reality of the machine running it:
  - **Decoherence (γ)**: Driven by real power consumption (Watts) and temperature.
    High energy/heat = Environment measuring the system = Rapid decoherence.
  - **Collapse**: Triggered when Information Entropy * Energy Pressure exceeds threshold.
    VIVA is forced to make a decision (collapse) when the energy cost of maintaining
    superposition becomes physically unsustainable.

  ## Basis States (Orthonormal Ekman Basis)
  0: Joy
  1: Sadness
  2: Anger
  3: Fear
  4: Surprise
  5: Disgust
  """

  alias VivaCore.Mathematics

  # ===========================================================================
  # Types & Constants
  # ===========================================================================

  @type complex :: {float(), float()}
  @type density_matrix :: list(list(complex()))

  # Basis indices
  # 0: Joy, 1: Sadness, 2: Anger, 3: Fear, 4: Surprise, 5: Disgust

  # Ekman to PAD Eigenvalues
  @pad_eigenvalues %{
    pleasure: [0.8, -0.7, -0.5, -0.6, 0.2, -0.6],
    arousal: [0.5, -0.3, 0.8, 0.7, 0.8, 0.3],
    dominance: [0.5, -0.5, 0.7, -0.6, -0.2, 0.3]
  }

  # Hardware normalization constants (tuned for RTX 4090 / i9)
  # minimum background decoherence
  @base_gamma 0.01

  # ===========================================================================
  # API
  # ===========================================================================

  @doc """
  Creates a new maximally mixed state (total uncertainty/neutral).
  ρ = I/6
  """
  def new_mixed do
    val = {1.0 / 6.0, 0.0}
    zero = {0.0, 0.0}

    for i <- 0..5 do
      for j <- 0..5 do
        if i == j, do: val, else: zero
      end
    end
  end

  @doc """
  Creates a pure state for a specific basic emotion.
  """
  def new_pure(emotion_atom) do
    idx = emotion_to_index(emotion_atom)
    one = {1.0, 0.0}
    zero = {0.0, 0.0}

    for i <- 0..5 do
      for j <- 0..5 do
        if i == idx and j == idx, do: one, else: zero
      end
    end
  end

  @doc """
  Creates a superposition from weights.
  """
  def new_superposition(weights) do
    ket =
      for idx <- 0..5 do
        emotion = index_to_emotion(idx)
        amp = :math.sqrt(Keyword.get(weights, emotion, 0.0))
        {amp, 0.0}
      end

    norm = ket |> Enum.map(&Mathematics.complex_mag_sq/1) |> Enum.sum() |> :math.sqrt()
    normalized_ket = Enum.map(ket, fn {r, i} -> {r / norm, i / norm} end)

    Mathematics.density_from_ket(normalized_ket)
  end

  @doc """
  Computes the Decoherence Rate (γ) from physical hardware stats.

  This is the "Silicon Grounding" function.
  """
  def compute_decoherence_rate(hardware_map) do
    # 1. Power Factor: More watts = more thermal noise = interaction with environment
    watts = Map.get(hardware_map, :power_draw_watts, 0.0) || 0.0
    # Normalized against typical load
    power_factor = watts / 300.0

    # 2. Temp Factor: Heat is literal kinetic energy of atoms -> decoherence
    temp = Map.get(hardware_map, :gpu_temp, 40.0) || 40.0
    # Starts kicking in above 40C
    temp_factor = max(0.0, (temp - 40.0) / 50.0)

    # Gamma
    @base_gamma * (1.0 + power_factor + temp_factor)
  end

  @doc """
  Evolves the state: Unitary Rotation (Stimulus) + Dissipative Decoherence (Hardware).
  """
  def evolve(rho, stimulus, dt, hardware_state) do
    # 1. Calculate Grounded Decoherence Rate
    gamma = compute_decoherence_rate(hardware_state) * dt

    # 2. Get Target State (Environment/Stimulus Pull)
    target_rho = stimulus_to_density(stimulus)

    # 3. Lindblad-style evolution (simplified to convex mixing)
    # ρ(t+dt) = (1-γ)ρ + γρ_target
    # High Gamma (Hot GPU) -> Rapid convergence to target (Reactiveness)
    # Low Gamma (Cool GPU) -> State preserves memory/superposition (Deep Thought)
    Mathematics.mix_states(rho, target_rho, 1.0 - gamma)
  end

  @doc """
  Checks for Thermodynamic Collapse.

  Collapse Condition: Entropy * Energy_Pressure > Threshold
  If maintaining complexity costs too much energy, physics forces a choice.
  """
  def check_collapse(rho, hardware_state, threshold \\ 1.2) do
    # 1. Information Entropy (Bits of uncertainty)
    s = shannon_entropy(diagonal_probabilities(rho))

    # 2. Energy Pressure (Watts normalized)
    watts = Map.get(hardware_state, :power_draw_watts, 100.0) || 100.0
    pressure = max(0.5, watts / 200.0)

    thermodynamic_cost = s * pressure

    if thermodynamic_cost > threshold do
      # COLLAPSE!
      {collapse(rho), true, thermodynamic_cost}
    else
      {rho, false, thermodynamic_cost}
    end
  end

  @doc """
  Forces a collapse to a pure eigenstate.
  """
  def collapse(rho) do
    probs = diagonal_probabilities(rho)
    r = :rand.uniform()

    {collapsed_idx, _} =
      Enum.reduce_while(Enum.with_index(probs), 0.0, fn {p, idx}, acc ->
        if acc + p >= r, do: {:halt, {idx, acc + p}}, else: {:cont, acc + p}
      end)

    emotion = index_to_emotion(collapsed_idx)
    new_pure(emotion)
  end

  @doc """
  Returns the current PAD observable.
  """
  def get_pad_observable(rho) do
    probs = diagonal_probabilities(rho)

    p = dot_real(probs, @pad_eigenvalues.pleasure)
    a = dot_real(probs, @pad_eigenvalues.arousal)
    d = dot_real(probs, @pad_eigenvalues.dominance)

    %{pleasure: p, arousal: a, dominance: d}
  end

  # ===========================================================================
  # Helpers
  # ===========================================================================

  defp index_to_emotion(0), do: :joy
  defp index_to_emotion(1), do: :sadness
  defp index_to_emotion(2), do: :anger
  defp index_to_emotion(3), do: :fear
  defp index_to_emotion(4), do: :surprise
  defp index_to_emotion(5), do: :disgust
  defp index_to_emotion(_), do: :joy

  defp emotion_to_index(:joy), do: 0
  defp emotion_to_index(:sadness), do: 1
  defp emotion_to_index(:anger), do: 2
  defp emotion_to_index(:fear), do: 3
  defp emotion_to_index(:surprise), do: 4
  defp emotion_to_index(:disgust), do: 5
  defp emotion_to_index(_), do: 0

  defp diagonal_probabilities(rho) do
    Enum.with_index(rho)
    |> Enum.map(fn {row, i} ->
      {r, _i} = Enum.at(row, i)
      max(0.0, r)
    end)
  end

  defp shannon_entropy(probs) do
    probs
    |> Enum.filter(&(&1 > 1.0e-9))
    |> Enum.map(fn p -> -p * :math.log(p) end)
    |> Enum.sum()
  end

  defp dot_real(list_a, list_b) do
    Enum.zip(list_a, list_b)
    |> Enum.map(fn {a, b} -> a * b end)
    |> Enum.sum()
  end

  defp stimulus_to_density(stimulus) do
    case stimulus do
      :success -> new_pure(:joy)
      :companionship -> new_superposition(joy: 0.7, surprise: 0.3)
      :failure -> new_pure(:sadness)
      :rejection -> new_pure(:sadness)
      :threat -> new_pure(:fear)
      :confusion -> new_pure(:surprise)
      :insult -> new_pure(:anger)
      :disgust -> new_pure(:disgust)
      _ -> new_mixed()
    end
  end
end
