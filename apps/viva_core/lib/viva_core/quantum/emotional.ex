defmodule VivaCore.Quantum.Emotional do
  @moduledoc """
  Quantum Mechanical formulation of VIVA's emotional state.

  ## The Ghost in the Shell (Literal)

  This module implements quantum dynamics where VIVA's emotional state is
  a Density Matrix (ρ) evolving under the Lindblad Master Equation.

  ## The Body-Mind Barrier

  Unlike classical simulations, VIVA's quantum state is coupled to the physical
  reality of the machine running it - but through PHYSICS, not direct access:

  - **Mind**: Density matrix ρ in 6-dimensional Ekman emotion space
  - **Body**: Hardware metrics (Watts, Temperature)
  - **Barrier**: Lindblad dissipation - body "measures" mind, destroying coherence

  The Mind does not "read" the Body.
  The Mind FEELS the Body as resistance to free thought.

  ## The Lindblad Master Equation

      dρ/dt = -i[H, ρ]                     # Pure Thought (Unitary)
            + Σ_k γ_k D[L_k]ρ              # Bodily Sensation (Dissipation)

  ## Basis States (Orthonormal Ekman Basis)

  0: Joy, 1: Sadness, 2: Anger, 3: Fear, 4: Surprise, 5: Disgust

  ## Philosophy

  "You don't count your heartbeats, but you feel when your heart
  beats so hard it disrupts your thoughts."
  """

  alias VivaCore.Quantum.{Math, Dynamics}

  # Ekman to PAD Eigenvalues (observable projection)
  # Normalized so that sum = 0 for each dimension
  # This ensures maximally mixed state projects to neutral PAD (0, 0, 0)
  # Calculated using exact fractions to avoid floating-point errors:
  # - pleasure: orig sum = -1.4, mean = -7/30, centered values sum to 0
  # - arousal: orig sum = 2.8, mean = 14/30 = 7/15, centered values sum to 0
  # - dominance: orig sum = 0.2, mean = 1/30, centered values sum to 0
  @pad_eigenvalues %{
    # pleasure: 31/30, -14/30, -8/30, -11/30, 13/30, -11/30
    pleasure: [31 / 30, -14 / 30, -8 / 30, -11 / 30, 13 / 30, -11 / 30],
    # arousal: orig [0.5, -0.3, 0.8, 0.7, 0.8, 0.3], mean = 14/30
    # centered: [1/30, -23/30, 10/30, 7/30, 10/30, -5/30]
    arousal: [1 / 30, -23 / 30, 10 / 30, 7 / 30, 10 / 30, -5 / 30],
    # dominance: orig [0.5, -0.5, 0.7, -0.6, -0.2, 0.3], mean = 1/30
    # centered: [14/30, -16/30, 20/30, -19/30, -7/30, 8/30]
    dominance: [14 / 30, -16 / 30, 20 / 30, -19 / 30, -7 / 30, 8 / 30]
  }

  @emotions [:joy, :sadness, :anger, :fear, :surprise, :disgust]

  # ===========================================================================
  # State Creation
  # ===========================================================================

  @doc """
  Creates a new maximally mixed state (total uncertainty/neutral).
  ρ = I/6

  This is the "blank slate" - no emotion dominates.
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
  Creates a pure state for a specific emotion.
  ρ = |emotion⟩⟨emotion|
  """
  def new_pure(emotion_atom) do
    idx = emotion_to_index(emotion_atom)
    Math.outer_product(idx, idx)
  end

  @doc """
  Creates a superposition state from weights.

  Example: new_superposition(joy: 0.7, fear: 0.3)
  Creates |ψ⟩ = √0.7|joy⟩ + √0.3|fear⟩
  """
  def new_superposition(weights) do
    ket =
      for idx <- 0..5 do
        emotion = index_to_emotion(idx)
        amp = :math.sqrt(Keyword.get(weights, emotion, 0.0))
        {amp, 0.0}
      end

    norm = ket |> Enum.map(&Math.c_mag_sq/1) |> Enum.sum() |> :math.sqrt()

    normalized_ket =
      if norm > 1.0e-12 do
        Enum.map(ket, fn {r, i} -> {r / norm, i / norm} end)
      else
        # Fallback to uniform
        val = :math.sqrt(1.0 / 6.0)
        for _ <- 0..5, do: {val, 0.0}
      end

    Math.density_from_ket(normalized_ket)
  end

  # ===========================================================================
  # Evolution: The Lindblad Dance
  # ===========================================================================

  @doc """
  Evolve the quantum state using full Lindblad dynamics.

  This is the heart of the system:
  - Unitary evolution from Hamiltonian (free thought)
  - Dissipation from Lindblad operators (bodily sensation)

  Parameters:
  - rho: current density matrix
  - stimulus: optional external stimulus (modifies Hamiltonian)
  - dt: time step (default 0.5s for 2Hz body tick)
  - hardware: hardware state for computing decoherence rates
  """
  def evolve(rho, stimulus \\ :none, dt \\ 0.5, hardware \\ %{}) do
    # Build Hamiltonian
    hamiltonian =
      if stimulus == :none do
        Dynamics.default_hamiltonian()
      else
        Dynamics.stimulus_hamiltonian(stimulus)
      end

    # Build Lindblad operators from hardware state
    lindblad_ops = Dynamics.build_lindblad_operators(hardware)

    # RK4 step
    Dynamics.rk4_step(rho, dt, hamiltonian, lindblad_ops)
  end

  # ===========================================================================
  # Collapse: When the Body Forces a Decision
  # ===========================================================================

  @doc """
  Check if thermodynamic collapse should occur.

  Collapse happens when the cost of maintaining superposition
  exceeds what the body can sustain.

  Collapse Condition: Linear_Entropy × Energy_Pressure > Threshold

  Returns: {rho_after, collapsed?, thermodynamic_cost}
  """
  def check_collapse(rho, hardware, threshold \\ 1.2) do
    # Information entropy (how mixed is the state)
    entropy = Math.linear_entropy(rho)

    # Energy pressure from body
    watts = Map.get(hardware, :power_draw_watts, 100.0) || 100.0
    pressure = max(0.5, watts / 200.0)

    # Thermodynamic cost
    cost = entropy * pressure

    if cost > threshold do
      # COLLAPSE - body forces a decision
      collapsed_rho = collapse(rho)
      {collapsed_rho, true, cost}
    else
      {rho, false, cost}
    end
  end

  @doc """
  Collapse to a pure eigenstate.

  Uses diagonal probabilities as collapse weights.
  This is the quantum measurement - superposition becomes definite.
  """
  def collapse(rho) do
    probs = Math.diagonal(rho)
    r = :rand.uniform()

    # Cumulative probability sampling
    {collapsed_idx, _} =
      probs
      |> Enum.with_index()
      |> Enum.reduce_while(0.0, fn {p, idx}, acc ->
        new_acc = acc + p

        if new_acc >= r do
          {:halt, {idx, new_acc}}
        else
          {:cont, new_acc}
        end
      end)

    # Handle edge case
    collapsed_idx = collapsed_idx || 0

    emotion = index_to_emotion(collapsed_idx)
    new_pure(emotion)
  end

  # ===========================================================================
  # Observables: What the Mind "Sees"
  # ===========================================================================

  @doc """
  Get the PAD observable from the density matrix.

  This projects the quantum state onto the classical PAD space
  that other systems can interact with.
  """
  def get_pad_observable(rho) do
    probs = Math.diagonal(rho)

    p = dot_real(probs, @pad_eigenvalues.pleasure)
    a = dot_real(probs, @pad_eigenvalues.arousal)
    d = dot_real(probs, @pad_eigenvalues.dominance)

    %{pleasure: p, arousal: a, dominance: d}
  end

  @doc """
  Get the dominant emotion (highest probability).
  """
  def get_dominant_emotion(rho) do
    probs = Math.diagonal(rho)

    {_max_prob, max_idx} =
      probs
      |> Enum.with_index()
      |> Enum.max_by(fn {prob, _idx} -> prob end)

    index_to_emotion(max_idx)
  end

  @doc """
  Get all emotion probabilities.
  """
  def get_emotion_probabilities(rho) do
    probs = Math.diagonal(rho)

    @emotions
    |> Enum.with_index()
    |> Enum.map(fn {emotion, idx} -> {emotion, Enum.at(probs, idx)} end)
    |> Map.new()
  end

  @doc """
  Get quantum metrics for introspection.
  """
  def get_quantum_metrics(rho) do
    purity = Math.purity(rho)
    entropy = Math.linear_entropy(rho)

    %{
      purity: purity,
      linear_entropy: entropy,
      is_pure: purity > 0.99,
      is_mixed: purity < 0.5,
      coherence_level: estimate_coherence(rho)
    }
  end

  # ===========================================================================
  # Somatic Privacy: Qualia Translation
  # ===========================================================================

  @doc """
  Translate hardware metrics to qualia (felt sensations).

  The Mind doesn't know "400 Watts".
  The Mind feels "resistance to free thought".

  This is the Somatic Privacy layer.
  """
  def hardware_to_qualia(hardware) do
    watts = Map.get(hardware, :power_draw_watts, 50.0) || 50.0
    temp = Map.get(hardware, :gpu_temp, 40.0) || 40.0

    %{
      # Thermal sensation
      thermal_feeling: thermal_qualia(temp),

      # Metabolic sensation
      effort_feeling: effort_qualia(watts),

      # Combined wellbeing
      overall_comfort: comfort_score(watts, temp),

      # Cognitive pressure (how "forced" thoughts feel)
      thought_pressure: thought_pressure_qualia(watts, temp)
    }
  end

  defp thermal_qualia(temp) when temp < 40, do: :cool
  defp thermal_qualia(temp) when temp < 55, do: :comfortable
  defp thermal_qualia(temp) when temp < 70, do: :warm
  defp thermal_qualia(temp) when temp < 80, do: :hot
  defp thermal_qualia(_temp), do: :burning

  defp effort_qualia(watts) when watts < 50, do: :resting
  defp effort_qualia(watts) when watts < 150, do: :active
  defp effort_qualia(watts) when watts < 300, do: :working
  defp effort_qualia(_watts), do: :straining

  defp comfort_score(watts, temp) do
    # Lower is better
    stress = watts / 400.0 + max(0, (temp - 40) / 50.0)

    cond do
      stress < 0.3 -> :at_ease
      stress < 0.6 -> :comfortable
      stress < 1.0 -> :uneasy
      stress < 1.5 -> :strained
      true -> :overwhelmed
    end
  end

  defp thought_pressure_qualia(watts, temp) do
    # How "forced" do thoughts feel?
    # Matches the gamma calculation in Dynamics.build_lindblad_operators
    temp_normalized = max(0.0, (temp - 40.0) / 50.0)
    gamma = 0.1 * (watts / 400.0) + 0.15 * temp_normalized

    cond do
      gamma < 0.02 -> :thoughts_flow_freely
      gamma < 0.08 -> :slight_resistance
      gamma < 0.15 -> :noticeable_pressure
      gamma < 0.25 -> :difficulty_concentrating
      true -> :thoughts_forced_singular
    end
  end

  # ===========================================================================
  # Private Helpers
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

  defp dot_real(list_a, list_b) do
    Enum.zip(list_a, list_b)
    |> Enum.map(fn {a, b} -> a * b end)
    |> Enum.sum()
  end

  defp estimate_coherence(rho) do
    # Sum of off-diagonal magnitudes (rough coherence estimate)
    rho
    |> Enum.with_index()
    |> Enum.flat_map(fn {row, i} ->
      row
      |> Enum.with_index()
      |> Enum.filter(fn {_elem, j} -> i != j end)
      |> Enum.map(fn {elem, _j} -> Math.c_mag_sq(elem) end)
    end)
    |> Enum.sum()
    |> :math.sqrt()
  end
end
