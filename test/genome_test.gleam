//// Tests for Emotional DNA (Genome)

import gleeunit/should
import viva/soul/genome.{
  Balanced, Calm, Critical, Energetic, Neurotic, NoDrift, NoMutation, Optimist,
  PadVector, ResilienceMutation, Stable, TraumaDrift, TraumaMutation, Warning,
}

// =============================================================================
// PERSONALITY GENOME TESTS
// =============================================================================

pub fn calm_genome_has_positive_pleasure_attractor_test() {
  let g = genome.from_personality(Calm)
  should.be_true(g.chromosome_a.primary_attractor.pleasure >. 0.0)
}

pub fn calm_genome_has_negative_arousal_attractor_test() {
  let g = genome.from_personality(Calm)
  should.be_true(g.chromosome_a.primary_attractor.arousal <. 0.0)
}

pub fn calm_genome_has_high_inertia_test() {
  let g = genome.from_personality(Calm)
  should.be_true(g.chromosome_b.pleasure_inertia >. 0.8)
}

pub fn neurotic_genome_has_negative_pleasure_test() {
  let g = genome.from_personality(Neurotic)
  should.be_true(g.chromosome_a.primary_attractor.pleasure <. 0.0)
}

pub fn neurotic_genome_has_high_arousal_test() {
  let g = genome.from_personality(Neurotic)
  should.be_true(g.chromosome_a.primary_attractor.arousal >. 0.5)
}

pub fn neurotic_genome_has_very_high_crisis_sensitivity_test() {
  let g = genome.from_personality(Neurotic)
  should.be_true(g.modulators.crisis_sensitivity >. 0.9)
}

pub fn neurotic_genome_has_very_low_recovery_test() {
  let g = genome.from_personality(Neurotic)
  should.be_true(g.modulators.recovery_gene <. 0.1)
}

pub fn optimist_genome_has_near_zero_crisis_sensitivity_test() {
  let g = genome.from_personality(Optimist)
  should.be_true(g.modulators.crisis_sensitivity <. 0.1)
}

pub fn optimist_genome_has_high_recovery_test() {
  let g = genome.from_personality(Optimist)
  should.be_true(g.modulators.recovery_gene >. 0.8)
}

pub fn energetic_genome_has_low_inertia_test() {
  let g = genome.from_personality(Energetic)
  should.be_true(g.chromosome_b.arousal_inertia <. 0.3)
}

pub fn energetic_genome_has_high_arousal_attractor_test() {
  let g = genome.from_personality(Energetic)
  should.be_true(g.chromosome_a.primary_attractor.arousal >. 0.8)
}

// =============================================================================
// EMOTIONAL UPDATE TESTS
// =============================================================================

pub fn optimist_crisis_response_dampened_test() {
  // Optimist has very low crisis_sensitivity (0.05)
  // This means crisis events have minimal effect
  let g = genome.from_personality(Optimist)

  // Verify the genetic basis for crisis immunity
  should.be_true(g.modulators.crisis_sensitivity <. 0.1)

  // And high rupture threshold
  should.be_true(g.modulators.rupture_threshold >. 0.9)

  // Combined with high recovery gene
  should.be_true(g.modulators.recovery_gene >. 0.8)
}

pub fn neurotic_trapped_in_stressed_test() {
  let g = genome.from_personality(Neurotic)
  let current = PadVector(-0.8, 0.9, -0.7)  // Already stressed
  let positive_event = PadVector(0.3, -0.2, 0.2)

  let delta = genome.compute_emotional_delta(
    g,
    current,
    positive_event,
    0.3,  // Moderate positive event
    "Stressed",
    100,
  )

  // Neurotic should barely move even with positive events
  // Due to extremely high inertia in Stressed state
  should.be_true(abs(delta.pleasure) <. 0.15)
}

pub fn energetic_responds_quickly_test() {
  let g = genome.from_personality(Energetic)
  let current = PadVector(0.0, 0.5, 0.0)
  let stimulus = PadVector(0.5, 0.3, 0.2)

  let delta = genome.compute_emotional_delta(
    g,
    current,
    stimulus,
    0.7,
    "Excited/Happy",
    50,
  )

  // Energetic should respond more than neurotic
  let neurotic_g = genome.from_personality(Neurotic)
  let neurotic_delta = genome.compute_emotional_delta(
    neurotic_g,
    current,
    stimulus,
    0.7,
    "Stressed",
    50,
  )

  let energetic_magnitude = abs(delta.pleasure) +. abs(delta.arousal) +. abs(delta.dominance)
  let neurotic_magnitude = abs(neurotic_delta.pleasure) +. abs(neurotic_delta.arousal) +. abs(neurotic_delta.dominance)

  should.be_true(energetic_magnitude >. neurotic_magnitude)
}

// =============================================================================
// EPIGENETIC TESTS
// =============================================================================

pub fn crisis_increases_methylation_test() {
  let g = genome.from_personality(Balanced)
  let initial_trauma = g.epigenetics.trauma_methylation

  let after_crisis = genome.apply_crisis_methylation(g, 0.8, 100)

  should.be_true(after_crisis.epigenetics.trauma_methylation >. initial_trauma)
}

pub fn celebration_heals_trauma_test() {
  // Start with neurotic (has trauma)
  let g = genome.from_personality(Neurotic)
  let initial_trauma = g.epigenetics.trauma_methylation

  let after_celebration = genome.apply_celebration_healing(g, 0.9, 100)

  should.be_true(after_celebration.epigenetics.trauma_methylation <. initial_trauma)
}

pub fn therapy_resets_methylation_test() {
  let g = genome.from_personality(Neurotic)
  let initial_trauma = g.epigenetics.trauma_methylation

  // Intense therapy (3 celebrations at 0.9 intensity)
  let after_therapy = g
    |> genome.reset_methylation(0.9)
    |> genome.reset_methylation(0.9)
    |> genome.reset_methylation(0.9)

  // Should significantly reduce trauma
  should.be_true(after_therapy.epigenetics.trauma_methylation <. initial_trauma *. 0.5)
}

// =============================================================================
// CROSSOVER TESTS
// =============================================================================

pub fn crossover_blends_attractors_test() {
  let calm = genome.from_personality(Calm)
  let optimist = genome.from_personality(Optimist)

  let offspring = genome.crossover(calm, optimist)

  // Offspring attractor should be between parents
  let calm_p = calm.chromosome_a.primary_attractor.pleasure
  let opt_p = optimist.chromosome_a.primary_attractor.pleasure
  let offspring_p = offspring.chromosome_a.primary_attractor.pleasure

  should.be_true(offspring_p >=. min(calm_p, opt_p))
  should.be_true(offspring_p <=. max(calm_p, opt_p))
}

pub fn crossover_inherits_some_trauma_test() {
  let neurotic = genome.from_personality(Neurotic)
  let optimist = genome.from_personality(Optimist)

  let offspring = genome.crossover(neurotic, optimist)

  // Offspring inherits 70% of average trauma
  let expected_max = neurotic.epigenetics.trauma_methylation *. 0.7
  should.be_true(offspring.epigenetics.trauma_methylation <=. expected_max)
}

pub fn calm_optimist_crossover_creates_wise_resilient_test() {
  let calm = genome.from_personality(Calm)
  let optimist = genome.from_personality(Optimist)

  let offspring = genome.crossover(calm, optimist)
  let fluency = genome.emotional_fluency(offspring)

  // Should have good fluency (calm stability + optimist recovery)
  should.be_true(fluency >. 0.2)
}

// =============================================================================
// MUTATION DETECTION TESTS
// =============================================================================

pub fn no_mutation_when_unchanged_test() {
  let baseline = genome.from_personality(Balanced)
  let current = baseline

  should.equal(genome.detect_mutation(current, baseline), NoMutation)
}

pub fn trauma_mutation_detected_test() {
  let baseline = genome.from_personality(Balanced)

  // Simulate multiple crises increasing sensitivity
  let traumatized = genome.Genome(
    ..baseline,
    modulators: genome.ModulatorGenes(
      ..baseline.modulators,
      crisis_sensitivity: baseline.modulators.crisis_sensitivity *. 2.0,
    ),
  )

  should.equal(genome.detect_mutation(traumatized, baseline), TraumaMutation)
}

pub fn resilience_mutation_detected_test() {
  let baseline = genome.from_personality(Balanced)

  // Simulate healing that doubled recovery
  let healed = genome.Genome(
    ..baseline,
    modulators: genome.ModulatorGenes(
      ..baseline.modulators,
      recovery_gene: baseline.modulators.recovery_gene *. 2.5,
    ),
  )

  should.equal(genome.detect_mutation(healed, baseline), ResilienceMutation)
}

// =============================================================================
// EMOTIONAL FLUENCY TESTS
// =============================================================================

pub fn calm_has_moderate_fluency_test() {
  let g = genome.from_personality(Calm)
  let fluency = genome.emotional_fluency(g)

  // Calm: high inertia but decent recovery
  should.be_true(fluency >. 0.05)
  should.be_true(fluency <. 0.3)
}

pub fn neurotic_has_lowest_fluency_test() {
  let neurotic = genome.from_personality(Neurotic)
  let optimist = genome.from_personality(Optimist)

  let neurotic_fluency = genome.emotional_fluency(neurotic)
  let optimist_fluency = genome.emotional_fluency(optimist)

  should.be_true(neurotic_fluency <. optimist_fluency)
}

pub fn energetic_has_high_fluency_test() {
  let g = genome.from_personality(Energetic)
  let fluency = genome.emotional_fluency(g)

  // Energetic: low inertia + good recovery
  should.be_true(fluency >. 0.3)
}

// =============================================================================
// PERSONALITY CLASSIFICATION TESTS
// =============================================================================

pub fn calm_genome_classifies_as_calm_test() {
  let g = genome.from_personality(Calm)
  should.equal(genome.to_personality_type(g), Calm)
}

pub fn neurotic_genome_classifies_as_neurotic_test() {
  let g = genome.from_personality(Neurotic)
  should.equal(genome.to_personality_type(g), Neurotic)
}

pub fn optimist_genome_classifies_as_optimist_test() {
  let g = genome.from_personality(Optimist)
  should.equal(genome.to_personality_type(g), Optimist)
}

pub fn energetic_genome_classifies_as_energetic_test() {
  let g = genome.from_personality(Energetic)
  should.equal(genome.to_personality_type(g), Energetic)
}

// =============================================================================
// HELPERS
// =============================================================================

fn abs(x: Float) -> Float {
  case x <. 0.0 {
    True -> 0.0 -. x
    False -> x
  }
}

fn min(a: Float, b: Float) -> Float {
  case a <. b {
    True -> a
    False -> b
  }
}

fn max(a: Float, b: Float) -> Float {
  case a >. b {
    True -> a
    False -> b
  }
}

// =============================================================================
// SURVIVAL PROTOCOLS TESTS (Qwen3 Emergency Recommendations)
// =============================================================================

pub fn forced_mutation_activates_after_threshold_test() {
  let g = genome.from_personality(Neurotic)
  let initial_recovery = g.modulators.recovery_gene

  // Simulate 600 ticks without improvement (threshold is 500)
  let mutated = genome.forced_adaptive_mutation(g, 600, 500)

  // Recovery should have increased
  should.be_true(mutated.modulators.recovery_gene >. initial_recovery)
}

pub fn forced_mutation_does_not_activate_before_threshold_test() {
  let g = genome.from_personality(Neurotic)
  let initial_recovery = g.modulators.recovery_gene

  // Only 300 ticks (below threshold)
  let same = genome.forced_adaptive_mutation(g, 300, 500)

  // Should be unchanged
  should.equal(same.modulators.recovery_gene, initial_recovery)
}

pub fn neurotic_emergency_protocol_reduces_trauma_test() {
  let g = genome.from_personality(Neurotic)
  let initial_trauma = g.epigenetics.trauma_methylation

  let treated = genome.neurotic_emergency_protocol(g, 0.8)

  // Trauma should decrease
  should.be_true(treated.epigenetics.trauma_methylation <. initial_trauma)
}

pub fn neurotic_emergency_protocol_boosts_recovery_test() {
  let g = genome.from_personality(Neurotic)
  let initial_recovery = g.modulators.recovery_gene

  let treated = genome.neurotic_emergency_protocol(g, 0.8)

  // Recovery should increase
  should.be_true(treated.modulators.recovery_gene >. initial_recovery)
}

pub fn neurotic_emergency_protocol_isolates_patient_test() {
  let g = genome.from_personality(Neurotic)
  let initial_contagion = g.modulators.social_contagion

  let treated = genome.neurotic_emergency_protocol(g, 0.8)

  // Social contagion should decrease (isolation)
  should.be_true(treated.modulators.social_contagion <. initial_contagion)
}

pub fn neurotic_emergency_ignores_non_neurotics_test() {
  let g = genome.from_personality(Optimist)
  let initial_recovery = g.modulators.recovery_gene

  let same = genome.neurotic_emergency_protocol(g, 0.8)

  // Optimist should not be affected
  should.equal(same.modulators.recovery_gene, initial_recovery)
}

pub fn vaccination_first_dose_builds_immunity_test() {
  let g = genome.from_personality(Balanced)
  let vax_state = genome.new_vaccination_state()

  let #(vaccinated, new_state) = genome.emotional_vaccination(g, vax_state, 100)

  // Should be vaccinated
  should.be_true(new_state.vaccinated)
  should.equal(new_state.doses, 1)

  // Immunity should be building
  should.be_true(new_state.immunity_level >. 0.0)

  // Recovery should have improved
  should.be_true(vaccinated.modulators.recovery_gene >. g.modulators.recovery_gene)
}

pub fn vaccination_three_doses_max_immunity_test() {
  let g = genome.from_personality(Balanced)
  let vax_state = genome.new_vaccination_state()

  // First dose
  let #(g1, state1) = genome.emotional_vaccination(g, vax_state, 100)
  // Second dose (200 ticks later)
  let #(g2, state2) = genome.emotional_vaccination(g1, state1, 300)
  // Third dose (200 ticks later)
  let #(g3, state3) = genome.emotional_vaccination(g2, state2, 500)

  // Should have 3 doses
  should.equal(state3.doses, 3)

  // Max immunity is 0.75
  should.be_true(state3.immunity_level >. 0.5)
  should.be_true(state3.immunity_level <=. 0.75)
}

pub fn vaccination_respects_interval_test() {
  let g = genome.from_personality(Balanced)
  let vax_state = genome.new_vaccination_state()

  // First dose at tick 100
  let #(g1, state1) = genome.emotional_vaccination(g, vax_state, 100)

  // Try second dose too early (only 50 ticks later)
  let #(_, state2) = genome.emotional_vaccination(g1, state1, 150)

  // Should still have only 1 dose
  should.equal(state2.doses, 1)
}

pub fn immunity_reduces_crisis_intensity_test() {
  let vax_state = genome.VaccinationState(
    vaccinated: True,
    vaccination_tick: 0,
    doses: 3,
    immunity_level: 0.5,
  )

  let original_crisis = 1.0
  let reduced = genome.apply_immunity(vax_state, original_crisis)

  // 50% immunity should halve the crisis intensity
  should.equal(reduced, 0.5)
}

pub fn check_emergency_status_critical_for_high_trauma_test() {
  // Create a heavily traumatized genome
  let g = genome.Genome(
    ..genome.from_personality(Neurotic),
    epigenetics: genome.EpigeneticState(
      methylation: 0.8,
      trauma_methylation: 0.9,
      healing_factor: 0.1,
      methylation_history: [],
    ),
  )

  should.equal(genome.check_emergency_status(g), Critical)
}

pub fn check_emergency_status_stable_for_healthy_test() {
  let g = genome.from_personality(Optimist)

  should.equal(genome.check_emergency_status(g), Stable)
}

pub fn check_emergency_status_warning_for_moderate_trauma_test() {
  // critical_score = trauma*2.0 + (1-recovery)*1.5 + sensitivity*0.5
  // For Warning: score > 2.5
  // 0.6*2.0 + 0.7*1.5 + 0.7*0.5 = 1.2 + 1.05 + 0.35 = 2.6 ✓
  let g = genome.Genome(
    ..genome.from_personality(Balanced),
    epigenetics: genome.EpigeneticState(
      methylation: 0.3,
      trauma_methylation: 0.6,
      healing_factor: 0.3,
      methylation_history: [],
    ),
    modulators: genome.ModulatorGenes(
      crisis_sensitivity: 0.7,
      social_contagion: 0.5,
      recovery_gene: 0.3,
      rupture_threshold: 0.6,
    ),
  )

  should.equal(genome.check_emergency_status(g), Warning)
}

// =============================================================================
// POPULATION DRIFT TESTS
// =============================================================================

pub fn detect_trauma_drift_test() {
  let baseline = genome.from_personality(Balanced)

  // Create traumatized population
  let traumatized = genome.Genome(
    ..baseline,
    epigenetics: genome.EpigeneticState(
      methylation: 0.5,  // Much higher than baseline (0.0)
      trauma_methylation: 0.4,
      healing_factor: 0.2,
      methylation_history: [],
    ),
  )

  let population = [traumatized, traumatized, traumatized]
  let drift = genome.detect_epigenetic_drift(population, baseline)

  should.equal(drift, TraumaDrift)
}

pub fn no_drift_for_healthy_population_test() {
  let baseline = genome.from_personality(Balanced)
  let population = [baseline, baseline, baseline]

  let drift = genome.detect_epigenetic_drift(population, baseline)

  should.equal(drift, NoDrift)
}

pub fn population_stats_calculates_trauma_mutations_test() {
  let baseline = genome.from_personality(Balanced)

  // Create one trauma mutant
  let mutant = genome.Genome(
    ..baseline,
    modulators: genome.ModulatorGenes(
      ..baseline.modulators,
      crisis_sensitivity: baseline.modulators.crisis_sensitivity *. 2.0,
    ),
  )

  let population = [baseline, baseline, mutant]
  let stats = genome.population_stats(population, baseline)

  // 1 out of 3 = 33.33%
  should.be_true(stats.trauma_mutation_pct >. 30.0)
  should.be_true(stats.trauma_mutation_pct <. 35.0)
}
