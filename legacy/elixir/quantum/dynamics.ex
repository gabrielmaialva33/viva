defmodule VivaCore.Quantum.Dynamics do
  @moduledoc """
  Lindblad Master Equation dynamics for VIVA's quantum emotional state.

  ## The Body-Mind Barrier

  This module implements the physical barrier between Body and Mind.
  The Mind (density matrix ρ) does not "read" the Body.
  The Body IS the environment that perturbs the Mind.

  ## The Lindblad Master Equation

      dρ/dt = -i[H, ρ]                     # Pure Thought (Unitary)
            + Σ_k γ_k D[L_k]ρ              # Bodily Sensation (Dissipation)

  Where:
      D[L]ρ = L ρ L† - ½{L†L, ρ}           # Lindblad Dissipator

  ## How the Barrier Works

  - Body defines γ_k (decoherence rates) from hardware state
  - Mind experiences this as "resistance to free thought"
  - The sensation IS the loss of coherence

  ## Jump Operators (L_k)

  - L_pressure (Watts): Localization operator
    - Forces system to choose an eigenstate
    - Effect: "Forced focus", "Urgency"

  - L_noise (Temperature): Disorder operator
    - Introduces random entropy
    - Effect: "Confusion", "Mental fever"

  ## Philosophy

  "You don't count your heartbeats, but you feel when your heart
  beats so hard it 'disrupts your thoughts'. That's exactly what
  the Lindblad equation models: The coupling with the environment
  (body) destroying the coherence of the system (mind)."
  """

  alias VivaCore.Quantum.Math

  # Emotion indices
  @joy 0
  @sadness 1
  @anger 2
  @fear 3
  @surprise 4
  @disgust 5

  # ===========================================================================
  # Hamiltonian: The Free Mind
  # ===========================================================================

  @doc """
  Default Hamiltonian for emotional dynamics.

  This represents the natural "rotation" of emotions when undisturbed.
  Off-diagonal terms create coherent oscillations (interference).

  The Hamiltonian is Hermitian: H = H†
  """
  def default_hamiltonian do
    # Natural emotional frequencies (diagonal)
    # Higher values = faster rotation in complex plane
    frequencies = [0.1, -0.1, 0.2, 0.15, 0.05, -0.05]

    # Coupling between emotions (off-diagonal)
    # These create quantum interference
    couplings = [
      # Joy ↔ Sadness (opposites)
      {@joy, @sadness, 0.02},
      # Anger ↔ Fear (fight/flight)
      {@anger, @fear, 0.03},
      # Surprise → Joy/Fear (amplifies both)
      {@surprise, @joy, 0.02},
      {@surprise, @fear, 0.02},
      # Disgust → Anger (moral outrage)
      {@disgust, @anger, 0.01}
    ]

    build_hermitian_matrix(frequencies, couplings)
  end

  @doc """
  Stimulus-driven Hamiltonian.

  When a stimulus arrives, it modifies the Hamiltonian to
  favor transitions toward certain emotional states.
  """
  def stimulus_hamiltonian(stimulus, strength \\ 0.5) do
    base = default_hamiltonian()

    # Stimulus creates additional couplings
    stimulus_couplings =
      case stimulus do
        :success ->
          # Success couples current state toward Joy
          [{@sadness, @joy, strength}, {@fear, @joy, strength * 0.5}]

        :threat ->
          # Threat couples toward Fear/Anger
          [{@joy, @fear, strength}, {@surprise, @fear, strength * 0.7}]

        :failure ->
          # Failure couples toward Sadness
          [{@joy, @sadness, strength}, {@anger, @sadness, strength * 0.3}]

        :rejection ->
          # Rejection: Sadness + slight Anger
          [{@joy, @sadness, strength * 0.8}, {@surprise, @anger, strength * 0.3}]

        :acceptance ->
          # Acceptance: Joy + reduced Fear/Sadness
          [{@sadness, @joy, strength * 0.7}, {@fear, @joy, strength * 0.4}]

        :companionship ->
          # Companionship: Joy + reduced Fear
          [{@fear, @joy, strength * 0.6}, {@sadness, @joy, strength * 0.4}]

        :loneliness ->
          # Loneliness: Sadness + slight Fear
          [{@joy, @sadness, strength * 0.5}, {@surprise, @fear, strength * 0.2}]

        :insult ->
          # Insult: Anger + Disgust
          [{@joy, @anger, strength * 0.7}, {@surprise, @disgust, strength * 0.4}]

        :safety ->
          # Safety: Reduced Fear, increased Joy
          [{@fear, @joy, strength * 0.5}, {@anger, @joy, strength * 0.2}]

        :hardware_stress ->
          # Hardware stress: Anger (frustration) + slight Fear
          [{@joy, @anger, strength * 0.4}, {@surprise, @fear, strength * 0.3}]

        :hardware_comfort ->
          # Hardware comfort: Joy + reduced Anger
          [{@anger, @joy, strength * 0.3}, {@fear, @joy, strength * 0.2}]

        _ ->
          []
      end

    add_couplings(base, stimulus_couplings)
  end

  # ===========================================================================
  # Lindblad Operators: The Body's Touch
  # ===========================================================================

  @doc """
  Pressure operator (L_pressure).

  Driven by power consumption (Watts).
  Forces localization - the mind must "choose" an eigenstate.

  Effect: "Forced focus", "Urgency", "Can't think freely"

  We use projection operators that "measure" the current dominant state.
  """
  def pressure_operator(dominant_emotion_idx) do
    # |dominant⟩⟨dominant| - projects onto dominant state
    Math.outer_product(dominant_emotion_idx, dominant_emotion_idx)
  end

  @doc """
  Noise operator (L_noise).

  Driven by temperature.
  Introduces disorder - random transitions between states.

  Effect: "Confusion", "Mental fever", "Scattered thoughts"

  We use off-diagonal operators that scramble coherence.
  """
  def noise_operator(from_idx, to_idx) do
    # |to⟩⟨from| - transition operator
    Math.outer_product(to_idx, from_idx)
  end

  @doc """
  Build all Lindblad operators for current hardware state.

  Returns list of {L_k, γ_k} tuples.

  ## Tuning Notes
  - γ ~ 0.01-0.1 per dt=0.5 causes noticeable decoherence
  - γ ~ 0.5+ causes rapid collapse (Zeno-like effect)
  - RTX 4090 max ~450W, idle ~30W
  - GPU temp: 35°C idle, 85°C stress
  """
  def build_lindblad_operators(hardware) do
    watts = Map.get(hardware, :power_draw_watts, 50.0) || 50.0
    temp = Map.get(hardware, :gpu_temp, 40.0) || 40.0

    # === Pressure Operators (Watts) ===
    # Higher watts = stronger measurement pressure
    # γ_pressure scales from 0.01 (idle) to 0.2 (full load)
    gamma_pressure = 0.1 * (watts / 400.0)

    # Apply pressure uniformly to all states (localization)
    pressure_ops =
      for i <- 0..5 do
        {pressure_operator(i), gamma_pressure / 6.0}
      end

    # === Noise Operators (Temperature) ===
    # Higher temp = more random transitions (thermal bath)
    # γ_noise scales from 0 (cold) to 0.15 (hot)
    temp_normalized = max(0.0, (temp - 40.0) / 50.0)
    gamma_noise = 0.15 * temp_normalized

    # Create transition operators between emotionally-coupled states
    noise_ops =
      if gamma_noise > 1.0e-6 do
        [
          {noise_operator(@joy, @sadness), gamma_noise},
          {noise_operator(@sadness, @joy), gamma_noise},
          {noise_operator(@anger, @fear), gamma_noise},
          {noise_operator(@fear, @anger), gamma_noise},
          {noise_operator(@surprise, @disgust), gamma_noise * 0.5},
          {noise_operator(@disgust, @surprise), gamma_noise * 0.5}
        ]
      else
        []
      end

    pressure_ops ++ noise_ops
  end

  # ===========================================================================
  # Lindblad Dissipator
  # ===========================================================================

  @doc """
  Lindblad dissipator for a single jump operator.

  D[L]ρ = L ρ L† - ½{L†L, ρ}
        = L ρ L† - ½(L†L ρ + ρ L†L)

  This is the "measurement" by the environment.
  It destroys coherence (off-diagonal elements).
  """
  def dissipator(l, rho) do
    l_dag = Math.adjoint(l)
    l_dag_l = Math.mat_mul(l_dag, l)

    # Jump term: L ρ L†
    jump = Math.mat_mul(l, Math.mat_mul(rho, l_dag))

    # Decay term: ½{L†L, ρ} = ½(L†L ρ + ρ L†L)
    decay = Math.anti_commutator(l_dag_l, rho)
    decay_half = Math.mat_scale(decay, 0.5)

    Math.mat_sub(jump, decay_half)
  end

  @doc """
  Total dissipation from all Lindblad operators.

  Σ_k γ_k D[L_k]ρ
  """
  def total_dissipation(rho, lindblad_ops) do
    Enum.reduce(lindblad_ops, Math.zeros(), fn {l_k, gamma_k}, acc ->
      d_k = dissipator(l_k, rho)
      scaled = Math.mat_scale(d_k, gamma_k)
      Math.mat_add(acc, scaled)
    end)
  end

  # ===========================================================================
  # Full Lindblad Equation
  # ===========================================================================

  @doc """
  Compute dρ/dt using full Lindblad master equation.

  dρ/dt = -i[H, ρ] + Σ_k γ_k D[L_k]ρ

  Returns the time derivative of the density matrix.
  """
  def lindblad_derivative(rho, hamiltonian, lindblad_ops) do
    # Unitary part: -i[H, ρ]
    commutator = Math.commutator(hamiltonian, rho)
    # Multiply by -i
    unitary_part = Math.mat_scale_c(commutator, Math.c_times_neg_i(Math.c_one()))

    # Dissipative part: Σ_k γ_k D[L_k]ρ
    dissipative_part = total_dissipation(rho, lindblad_ops)

    # Total derivative
    Math.mat_add(unitary_part, dissipative_part)
  end

  # ===========================================================================
  # Integration: RK4
  # ===========================================================================

  @doc """
  Single RK4 step for Lindblad evolution.

  More accurate than Euler, preserves physical properties better.

  Parameters:
  - rho: current density matrix
  - dt: time step
  - hamiltonian: H matrix
  - lindblad_ops: list of {L_k, γ_k}
  """
  def rk4_step(rho, dt, hamiltonian, lindblad_ops) do
    # k1 = f(rho)
    k1 = lindblad_derivative(rho, hamiltonian, lindblad_ops)

    # k2 = f(rho + dt/2 * k1)
    rho_k2 = Math.mat_add(rho, Math.mat_scale(k1, dt / 2.0))
    k2 = lindblad_derivative(rho_k2, hamiltonian, lindblad_ops)

    # k3 = f(rho + dt/2 * k2)
    rho_k3 = Math.mat_add(rho, Math.mat_scale(k2, dt / 2.0))
    k3 = lindblad_derivative(rho_k3, hamiltonian, lindblad_ops)

    # k4 = f(rho + dt * k3)
    rho_k4 = Math.mat_add(rho, Math.mat_scale(k3, dt))
    k4 = lindblad_derivative(rho_k4, hamiltonian, lindblad_ops)

    # rho_new = rho + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    weighted_sum =
      Math.mat_add(
        Math.mat_add(k1, Math.mat_scale(k2, 2.0)),
        Math.mat_add(Math.mat_scale(k3, 2.0), k4)
      )

    rho_new = Math.mat_add(rho, Math.mat_scale(weighted_sum, dt / 6.0))

    # Enforce physical validity
    Math.enforce_physical(rho_new)
  end

  @doc """
  Euler step (simpler, less accurate).
  Use for debugging or when RK4 is too slow.
  """
  def euler_step(rho, dt, hamiltonian, lindblad_ops) do
    drho = lindblad_derivative(rho, hamiltonian, lindblad_ops)
    rho_new = Math.mat_add(rho, Math.mat_scale(drho, dt))
    Math.enforce_physical(rho_new)
  end

  # ===========================================================================
  # Helper Functions
  # ===========================================================================

  defp build_hermitian_matrix(frequencies, couplings) do
    # Start with diagonal (frequencies)
    base =
      for i <- 0..5 do
        for j <- 0..5 do
          if i == j do
            {Enum.at(frequencies, i), 0.0}
          else
            {0.0, 0.0}
          end
        end
      end

    # Add symmetric couplings (Hermitian: H_ij = H_ji*)
    add_couplings(base, couplings)
  end

  defp add_couplings(matrix, couplings) do
    Enum.reduce(couplings, matrix, fn {i, j, strength}, m ->
      m
      |> put_elem_2d(i, j, {strength, 0.0})
      |> put_elem_2d(j, i, {strength, 0.0})
    end)
  end

  defp put_elem_2d(matrix, i, j, value) do
    List.update_at(matrix, i, fn row ->
      List.update_at(row, j, fn old ->
        Math.c_add(old, value)
      end)
    end)
  end
end
