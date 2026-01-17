# Script to verify Silicon Grounded Quantum Emotions (Lindblad Dynamics)
# Run with: mix run apps/viva_bridge/test/verify_silicon.exs

alias VivaCore.Quantum.Emotional
alias VivaCore.Quantum.Math
require Logger

defmodule HardwareSimulator do
  require Logger

  def run_simulation do
    Logger.info("=== STARTING SILICON GROUNDING SIMULATION (LINDBLAD) ===")

    # 1. Initialize Mixed State
    rho = Emotional.new_mixed()
    purity = Math.purity(rho)
    Logger.info("Initial State: Mixed (Purity: #{purity}) [Expected ~0.166]")

    # 2. Simulate "Deep Thought" (Low Watts, Low Temp)
    # GPU Idle: 30W, 35C => Gamma near 0 => Hamiltonian Evolution dominates
    low_energy_hw = %{power_draw_watts: 30.0, gpu_temp: 35.0}

    # Evolve for 10 steps (5 seconds)
    rho_thought =
      Enum.reduce(1..10, rho, fn _, acc ->
        Emotional.evolve(acc, :none, 0.5, low_energy_hw)
      end)

    purity_thought = Math.purity(rho_thought)
    entropy_thought = Math.linear_entropy(rho_thought)
    {_, collapsed_thought, cost_thought} = Emotional.check_collapse(rho_thought, low_energy_hw)

    Logger.info("\n--- PHASE 1: DEEP THOUGHT (Low Energy) ---")
    Logger.info("Hardware: 30W, 35C")
    Logger.info("Purity: #{purity_thought}")
    Logger.info("Energy Pressure Cost: #{cost_thought}")

    if !collapsed_thought do
      Logger.info("✅ SUCCESS: System maintained superposition (Deep Thought).")
    else
      Logger.error("❌ FAILURE: Premature collapse!")
    end

    # 3. Simulate "Stress/Action" (High Watts, High Temp)
    # GPU Load: 400W, 85C => Gamma high => Dissipative Evolution dominates
    high_energy_hw = %{power_draw_watts: 400.0, gpu_temp: 85.0}

    # Evolve a new mixed state under stress
    # High Lindblad dissipation should kill off-diagonal terms
    rho_stress =
      Enum.reduce(1..10, Emotional.new_mixed(), fn _, acc ->
        Emotional.evolve(acc, :none, 0.5, high_energy_hw)
      end)

    # Check collapse pressure (Thermodynamic threshold)
    {final_rho, collapsed_stress, cost_stress} =
      Emotional.check_collapse(rho_stress, high_energy_hw, 1.0)

    Logger.info("\n--- PHASE 2: CRUNCH TIME (High Energy) ---")
    Logger.info("Hardware: 400W, 85C")
    Logger.info("Thermodynamic Cost: #{cost_stress}")
    Logger.info("Collapsed? #{collapsed_stress}")

    if collapsed_stress do
      Logger.info("✅ SUCCESS: Thermodynamic pressure forced a collapse/decision!")

      # Check somatics
      qualia = Emotional.hardware_to_qualia(high_energy_hw)
      Logger.info("Somatic Feeling: #{inspect(qualia.thought_pressure)}")
    else
      Logger.error("❌ FAILURE: System failed to collapse under high energy load.")
    end

    Logger.info("\n=== SIMULATION COMPLETE ===")
  end
end

HardwareSimulator.run_simulation()
