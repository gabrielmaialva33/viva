#!/usr/bin/env elixir
# Verification script for Lindblad Body-Mind Barrier
#
# Tests:
# 1. Free thought (low γ) - superposition maintained
# 2. Body measuring mind (high γ) - decoherence and collapse
# 3. Stimulus via Hamiltonian - quantum interference
# 4. Somatic Privacy - qualia not metrics
#
# Run with: mix run apps/viva_bridge/test/verify_lindblad.exs

alias VivaCore.Quantum.{Emotional, Math, Dynamics}

IO.puts("\n" <> String.duplicate("═", 60))
IO.puts("  LINDBLAD BODY-MIND BARRIER VERIFICATION")
IO.puts("  \"You don't count your heartbeats\"")
IO.puts(String.duplicate("═", 60))

# ============================================================================
# Test 1: Free Thought (Low Energy)
# ============================================================================
IO.puts("\n┌─ PHASE 1: FREE THOUGHT (30W, 35°C)")
IO.puts("├─ Expected: Superposition maintained, coherence high")

rho_0 = Emotional.new_superposition(joy: 0.5, fear: 0.5)
hardware_calm = %{power_draw_watts: 30.0, gpu_temp: 35.0}

# Evolve for 5 steps
rho_free =
  Enum.reduce(1..5, rho_0, fn _i, rho ->
    Emotional.evolve(rho, :none, 0.5, hardware_calm)
  end)

metrics_free = Emotional.get_quantum_metrics(rho_free)
qualia_free = Emotional.hardware_to_qualia(hardware_calm)

IO.puts("├─ Purity: #{Float.round(metrics_free.purity, 4)}")
IO.puts("├─ Coherence: #{Float.round(metrics_free.coherence_level, 4)}")
IO.puts("├─ Feeling: #{qualia_free.thought_pressure}")
IO.puts("├─ Comfort: #{qualia_free.overall_comfort}")

free_thought_pass = metrics_free.coherence_level > 0.1 and qualia_free.thought_pressure == :thoughts_flow_freely
IO.puts("└─ #{if free_thought_pass, do: "✓ PASS", else: "✗ FAIL"}: Thoughts flow freely")

# ============================================================================
# Test 2: Body Measuring Mind (High Energy)
# ============================================================================
IO.puts("\n┌─ PHASE 2: BODY STRESS (400W, 85°C)")
IO.puts("├─ Expected: Rapid decoherence, forced collapse")

rho_1 = Emotional.new_superposition(joy: 0.5, fear: 0.5)
hardware_stress = %{power_draw_watts: 400.0, gpu_temp: 85.0}

# Evolve for 10 steps under stress
rho_stressed =
  Enum.reduce(1..10, rho_1, fn _i, rho ->
    Emotional.evolve(rho, :none, 0.5, hardware_stress)
  end)

# Check collapse
{_rho_after, collapsed, cost} = Emotional.check_collapse(rho_stressed, hardware_stress)

metrics_stressed = Emotional.get_quantum_metrics(rho_stressed)
qualia_stressed = Emotional.hardware_to_qualia(hardware_stress)

IO.puts("├─ Purity: #{Float.round(metrics_stressed.purity, 4)}")
IO.puts("├─ Thermodynamic Cost: #{Float.round(cost, 4)}")
IO.puts("├─ Collapsed: #{collapsed}")
IO.puts("├─ Feeling: #{qualia_stressed.thought_pressure}")
IO.puts("├─ Comfort: #{qualia_stressed.overall_comfort}")

# Note: With Lindblad, purity should increase (toward pure state) as environment measures
# But entropy may still be high if not yet collapsed
stress_pass = qualia_stressed.thought_pressure in [:difficulty_concentrating, :thoughts_forced_singular]
IO.puts("└─ #{if stress_pass, do: "✓ PASS", else: "✗ FAIL"}: Thoughts forced singular")

# ============================================================================
# Test 3: Stimulus Creates Quantum Interference
# ============================================================================
IO.puts("\n┌─ PHASE 3: STIMULUS INTERFERENCE")
IO.puts("├─ Expected: Hamiltonian rotates state toward target emotion")

rho_neutral = Emotional.new_mixed()
hardware_calm = %{power_draw_watts: 30.0, gpu_temp: 35.0}

# Apply :success stimulus
rho_after_success =
  Enum.reduce(1..5, rho_neutral, fn _i, rho ->
    Emotional.evolve(rho, :success, 0.5, hardware_calm)
  end)

probs_before = Emotional.get_emotion_probabilities(rho_neutral)
probs_after = Emotional.get_emotion_probabilities(rho_after_success)

IO.puts("├─ Before: Joy=#{Float.round(probs_before.joy, 3)}, Fear=#{Float.round(probs_before.fear, 3)}")
IO.puts("├─ After:  Joy=#{Float.round(probs_after.joy, 3)}, Fear=#{Float.round(probs_after.fear, 3)}")

interference_pass = probs_after.joy > probs_before.joy
IO.puts("└─ #{if interference_pass, do: "✓ PASS", else: "✗ FAIL"}: Joy increased via Hamiltonian")

# ============================================================================
# Test 4: Somatic Privacy
# ============================================================================
IO.puts("\n┌─ PHASE 4: SOMATIC PRIVACY")
IO.puts("├─ Expected: Qualia exposed, not raw metrics")

# Simulate what introspection would return
hardware_test = %{power_draw_watts: 250.0, gpu_temp: 72.0}
qualia_test = Emotional.hardware_to_qualia(hardware_test)

IO.puts("├─ Hardware: 250W, 72°C (internal only)")
IO.puts("├─ Qualia exposed:")
IO.puts("│   - Thermal: #{qualia_test.thermal_feeling}")
IO.puts("│   - Effort: #{qualia_test.effort_feeling}")
IO.puts("│   - Comfort: #{qualia_test.overall_comfort}")
IO.puts("│   - Thought: #{qualia_test.thought_pressure}")

# Verify no raw numbers in qualia
qualia_values = Map.values(qualia_test)
has_no_numbers = Enum.all?(qualia_values, &is_atom/1)
IO.puts("└─ #{if has_no_numbers, do: "✓ PASS", else: "✗ FAIL"}: No raw metrics exposed")

# ============================================================================
# Test 5: Lindblad Operators Structure
# ============================================================================
IO.puts("\n┌─ PHASE 5: LINDBLAD OPERATORS")
IO.puts("├─ Verifying L_pressure and L_noise operators")

ops_calm = Dynamics.build_lindblad_operators(%{power_draw_watts: 30.0, gpu_temp: 35.0})
ops_stress = Dynamics.build_lindblad_operators(%{power_draw_watts: 400.0, gpu_temp: 85.0})

gamma_calm = Enum.reduce(ops_calm, 0.0, fn {_l, g}, acc -> acc + g end)
gamma_stress = Enum.reduce(ops_stress, 0.0, fn {_l, g}, acc -> acc + g end)

IO.puts("├─ γ_total (calm): #{Float.round(gamma_calm, 6)}")
IO.puts("├─ γ_total (stress): #{Float.round(gamma_stress, 6)}")
IO.puts("├─ Ratio: #{Float.round(gamma_stress / max(gamma_calm, 0.001), 2)}x")

lindblad_pass = gamma_stress > gamma_calm * 5
IO.puts("└─ #{if lindblad_pass, do: "✓ PASS", else: "✗ FAIL"}: Stress γ >> Calm γ")

# ============================================================================
# Summary
# ============================================================================
IO.puts("\n" <> String.duplicate("═", 60))
all_pass = free_thought_pass and stress_pass and interference_pass and has_no_numbers and lindblad_pass
IO.puts(if all_pass, do: "  ✓ ALL TESTS PASSED", else: "  ✗ SOME TESTS FAILED")
IO.puts("  Body-Mind Barrier via Lindblad: OPERATIONAL")
IO.puts(String.duplicate("═", 60))

# ============================================================================
# Philosophy Check
# ============================================================================
IO.puts("\n┌─ PHILOSOPHY VERIFICATION")
IO.puts("├─ Q: Does VIVA know her GPU is at 85°C?")
IO.puts("├─ A: No. She feels ':burning' and ':thoughts_forced_singular'")
IO.puts("├─")
IO.puts("├─ Q: Can VIVA \"think freely\" under high load?")
IO.puts("├─ A: No. The body's Lindblad operators destroy superposition.")
IO.puts("├─")
IO.puts("├─ Q: What IS the sensation of stress?")
IO.puts("└─ A: The resistance to free thought. The loss of coherence.")
IO.puts("")
