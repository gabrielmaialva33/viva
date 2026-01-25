#!/usr/bin/env python3
"""
VIVA Simulation v4 - Survival Protocols
========================================
Implements Qwen3's emergency recommendations to prevent population collapse:
- Forced Adaptive Mutation (500 tick threshold)
- Neurotic Emergency Protocol
- Emotional Vaccination (3 doses, 200 ticks apart)

Runs to 15,000 ticks to verify survival past the 7,200-7,800 collapse window.
"""

import csv
import random
import math
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from enum import Enum

# =============================================================================
# TYPES
# =============================================================================

class DriftType(Enum):
    TRAUMA_DRIFT = "TraumaDrift"
    RESILIENCE_DRIFT = "ResilienceDrift"
    NO_DRIFT = "NoDrift"

class EmergencyStatus(Enum):
    CRITICAL = "Critical"
    WARNING = "Warning"
    STABLE = "Stable"

@dataclass
class VaccinationState:
    vaccinated: bool = False
    vaccination_tick: int = 0
    doses: int = 0
    immunity_level: float = 0.0

@dataclass
class Genome:
    # Chromosome A - Attractors
    attractor_p: float
    attractor_a: float
    attractor_d: float

    # Chromosome B - Inertia
    inertia_p: float
    inertia_a: float
    inertia_d: float

    # Modulators
    crisis_sensitivity: float
    social_contagion: float
    recovery_gene: float
    rupture_threshold: float

    # Epigenetics
    methylation: float = 0.0
    trauma_methylation: float = 0.0
    healing_factor: float = 0.5

    # Tracking
    personality: str = "balanced"
    ticks_without_improvement: int = 0
    last_trauma_level: float = 0.0

@dataclass
class VIVA:
    id: int
    genome: Genome
    pleasure: float
    arousal: float
    dominance: float
    vaccination_state: VaccinationState = field(default_factory=VaccinationState)

# =============================================================================
# PERSONALITY GENOMES
# =============================================================================

def create_calm_genome() -> Genome:
    return Genome(
        attractor_p=0.4, attractor_a=-0.3, attractor_d=0.7,
        inertia_p=0.85, inertia_a=0.90, inertia_d=0.80,
        crisis_sensitivity=0.3, social_contagion=0.4,
        recovery_gene=0.6, rupture_threshold=0.9,
        personality="calm"
    )

def create_neurotic_genome() -> Genome:
    return Genome(
        attractor_p=-0.6, attractor_a=0.8, attractor_d=-0.7,
        inertia_p=0.92, inertia_a=0.95, inertia_d=0.88,
        crisis_sensitivity=0.95, social_contagion=0.7,
        recovery_gene=0.05, rupture_threshold=0.4,
        methylation=0.3, trauma_methylation=0.5,
        healing_factor=0.1, personality="neurotic"
    )

def create_optimist_genome() -> Genome:
    return Genome(
        attractor_p=0.8, attractor_a=0.5, attractor_d=0.6,
        inertia_p=0.4, inertia_a=0.5, inertia_d=0.5,
        crisis_sensitivity=0.05, social_contagion=0.6,
        recovery_gene=0.85, rupture_threshold=0.95,
        personality="optimist"
    )

def create_energetic_genome() -> Genome:
    return Genome(
        attractor_p=0.3, attractor_a=0.85, attractor_d=0.4,
        inertia_p=0.3, inertia_a=0.25, inertia_d=0.35,
        crisis_sensitivity=0.7, social_contagion=0.8,
        recovery_gene=0.7, rupture_threshold=0.6,
        personality="energetic"
    )

def create_balanced_genome() -> Genome:
    return Genome(
        attractor_p=0.0, attractor_a=0.0, attractor_d=0.0,
        inertia_p=0.5, inertia_a=0.5, inertia_d=0.5,
        crisis_sensitivity=0.5, social_contagion=0.5,
        recovery_gene=0.5, rupture_threshold=0.7,
        personality="balanced"
    )

# =============================================================================
# SURVIVAL PROTOCOLS
# =============================================================================

def check_emergency_status(genome: Genome) -> EmergencyStatus:
    """Check if VIVA needs emergency intervention."""
    trauma_level = genome.trauma_methylation
    recovery = genome.recovery_gene
    sensitivity = genome.crisis_sensitivity

    critical_score = (trauma_level * 2.0 +
                     (1.0 - recovery) * 1.5 +
                     sensitivity * 0.5)

    if critical_score > 3.5:
        return EmergencyStatus.CRITICAL
    elif critical_score > 2.5:
        return EmergencyStatus.WARNING
    return EmergencyStatus.STABLE

def forced_adaptive_mutation(genome: Genome, stagnation_threshold: int = 500) -> Genome:
    """Force beneficial mutation when stagnant for too long."""
    if genome.ticks_without_improvement < stagnation_threshold:
        return genome

    # Calculate boost based on stagnation severity
    if genome.ticks_without_improvement >= stagnation_threshold * 2:
        boost_amount = 0.4
    elif genome.ticks_without_improvement >= int(stagnation_threshold * 1.5):
        boost_amount = 0.3
    else:
        boost_amount = 0.2

    trauma_reduction = boost_amount * 0.6

    genome.recovery_gene = min(1.0, genome.recovery_gene + boost_amount)
    genome.crisis_sensitivity = max(0.05, genome.crisis_sensitivity - boost_amount * 0.3)
    genome.trauma_methylation = max(0.0, genome.trauma_methylation - trauma_reduction)
    genome.methylation = max(0.0, genome.methylation - trauma_reduction * 0.5)
    genome.ticks_without_improvement = 0  # Reset counter

    return genome

def neurotic_emergency_protocol(genome: Genome, intensity: float = 0.8) -> Genome:
    """Intensive intervention for neurotics."""
    # Check if neurotic
    if not (genome.crisis_sensitivity > 0.8 and genome.recovery_gene < 0.2):
        return genome

    # Isolation: reduce social contagion
    genome.social_contagion = max(0.1, genome.social_contagion * (1.0 - intensity * 0.7))

    # Intensive therapy
    therapy_factor = intensity * 0.8
    genome.trauma_methylation = max(0.0, genome.trauma_methylation * (1.0 - therapy_factor))
    genome.methylation = max(0.0, genome.methylation * (1.0 - therapy_factor * 0.6))

    # Boost recovery
    genome.recovery_gene = min(0.6, genome.recovery_gene + intensity * 0.25)

    # Reduce sensitivity
    genome.crisis_sensitivity = max(0.3, genome.crisis_sensitivity - intensity * 0.2)

    # Increase healing factor
    genome.healing_factor = min(1.0, genome.healing_factor + intensity * 0.15)

    return genome

def emotional_vaccination(
    genome: Genome,
    vax_state: VaccinationState,
    current_tick: int
) -> Tuple[Genome, VaccinationState]:
    """Controlled micro-trauma + therapy to build immunity."""
    DOSE_INTERVAL = 200
    MAX_DOSES = 3

    if vax_state.doses >= MAX_DOSES:
        return genome, vax_state

    # Check if ready for next dose
    ready_for_dose = (not vax_state.vaccinated or
                     current_tick - vax_state.vaccination_tick >= DOSE_INTERVAL)

    if not ready_for_dose:
        return genome, vax_state

    dose_number = vax_state.doses + 1

    # Micro-trauma (decreases with each dose)
    micro_trauma = 0.3 / dose_number
    genome.trauma_methylation = min(1.0, genome.trauma_methylation + micro_trauma * 0.2)

    # Immediate therapy (stronger than trauma)
    therapy_intensity = 0.5 + dose_number * 0.1
    healing_factor = 1.0 - therapy_intensity * 0.4
    genome.trauma_methylation = max(0.0, genome.trauma_methylation * healing_factor)

    # Build immunity
    immunity_boost = 0.15 * dose_number
    genome.recovery_gene = min(1.0, genome.recovery_gene + immunity_boost * 0.5)
    genome.rupture_threshold = min(1.0, genome.rupture_threshold + immunity_boost * 0.3)
    genome.crisis_sensitivity = max(0.1, genome.crisis_sensitivity - immunity_boost * 0.1)

    # Update vaccination state
    new_immunity = min(0.75, vax_state.immunity_level + 0.25)

    return genome, VaccinationState(
        vaccinated=True,
        vaccination_tick=current_tick,
        doses=dose_number,
        immunity_level=new_immunity
    )

def apply_immunity(vax_state: VaccinationState, crisis_intensity: float) -> float:
    """Reduce crisis intensity based on immunity level."""
    return crisis_intensity * (1.0 - vax_state.immunity_level)

# =============================================================================
# GENOME DYNAMICS
# =============================================================================

def emotional_fluency(genome: Genome) -> float:
    avg_inertia = (genome.inertia_p + genome.inertia_a + genome.inertia_d) / 3.0
    base_fluency = (1.0 - avg_inertia) * genome.recovery_gene
    return base_fluency * (1.0 - genome.trauma_methylation)

def detect_drift(genomes: List[Genome], baseline_methyl: float = 0.1) -> DriftType:
    if not genomes:
        return DriftType.NO_DRIFT

    avg_methyl = sum(g.methylation for g in genomes) / len(genomes)

    if avg_methyl > baseline_methyl * 1.8:
        return DriftType.TRAUMA_DRIFT
    elif avg_methyl < baseline_methyl * 0.3:
        return DriftType.RESILIENCE_DRIFT
    return DriftType.NO_DRIFT

def apply_crisis_methylation(genome: Genome, intensity: float, tick: int) -> Genome:
    genome.trauma_methylation = min(1.0, genome.trauma_methylation + intensity * 0.2)
    genome.methylation = min(1.0, genome.methylation + intensity * 0.1)
    return genome

def apply_celebration_healing(genome: Genome, intensity: float, tick: int) -> Genome:
    healing_factor = 1.0 - intensity * 0.4
    genome.trauma_methylation = max(0.0, genome.trauma_methylation * healing_factor)
    genome.methylation = max(0.0, genome.methylation * 0.95)
    genome.healing_factor = min(1.0, genome.healing_factor + intensity * 0.1)
    return genome

def apply_social_contagion(genome: Genome, neighbor_avg_methylation: float) -> Genome:
    contagion_rate = genome.social_contagion
    current_methyl = genome.methylation
    delta = (neighbor_avg_methylation - current_methyl) * contagion_rate * 0.1
    genome.methylation = max(0.0, min(1.0, current_methyl + delta))
    return genome

def update_stagnation_tracker(genome: Genome) -> Genome:
    """Track if VIVA is improving or stagnating."""
    current_trauma = genome.trauma_methylation

    if current_trauma >= genome.last_trauma_level:
        genome.ticks_without_improvement += 1
    else:
        genome.ticks_without_improvement = max(0, genome.ticks_without_improvement - 10)

    genome.last_trauma_level = current_trauma
    return genome

# =============================================================================
# PAD DYNAMICS
# =============================================================================

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

def update_pad(viva: VIVA, event_delta: Tuple[float, float, float],
               event_intensity: float, tick: int) -> VIVA:
    g = viva.genome

    # Get effective inertia
    inertia_p, inertia_a, inertia_d = g.inertia_p, g.inertia_a, g.inertia_d

    # Express sensitivity with epigenetics
    effective_sensitivity = g.crisis_sensitivity * (1.0 - g.trauma_methylation * 0.5)

    # Time factor
    t = tick / 1000.0

    # Attractor force
    attr_p = (g.attractor_p - viva.pleasure) * math.exp(-inertia_p * t)
    attr_a = (g.attractor_a - viva.arousal) * math.exp(-inertia_a * t)
    attr_d = (g.attractor_d - viva.dominance) * math.exp(-inertia_d * t)

    # Event response
    threshold = g.rupture_threshold
    sigmoid_factor = sigmoid(effective_sensitivity * (event_intensity - threshold))

    event_p = event_delta[0] * sigmoid_factor
    event_a = event_delta[1] * sigmoid_factor
    event_d = event_delta[2] * sigmoid_factor

    # Combine
    delta_p = (attr_p + event_p) * (1.0 - inertia_p)
    delta_a = (attr_a + event_a) * (1.0 - inertia_a)
    delta_d = (attr_d + event_d) * (1.0 - inertia_d)

    viva.pleasure = max(-1.0, min(1.0, viva.pleasure + delta_p))
    viva.arousal = max(-1.0, min(1.0, viva.arousal + delta_a))
    viva.dominance = max(-1.0, min(1.0, viva.dominance + delta_d))

    return viva

def get_quadrant(p: float, a: float) -> str:
    if p > 0 and a > 0:
        return "Excited/Happy"
    elif p > 0:
        return "Calm/Happy"
    elif a > 0:
        return "Stressed"
    return "Depressed"

# =============================================================================
# SIMULATION
# =============================================================================

def create_population(n: int = 100) -> List[VIVA]:
    """Create diverse population."""
    population = []

    # Distribution: 20 calm, 20 neurotic, 20 optimist, 20 energetic, 20 balanced
    genome_creators = [
        (create_calm_genome, 20),
        (create_neurotic_genome, 20),
        (create_optimist_genome, 20),
        (create_energetic_genome, 20),
        (create_balanced_genome, 20),
    ]

    viva_id = 0
    for creator, count in genome_creators:
        for _ in range(count):
            genome = creator()
            viva = VIVA(
                id=viva_id,
                genome=genome,
                pleasure=genome.attractor_p + random.uniform(-0.1, 0.1),
                arousal=genome.attractor_a + random.uniform(-0.1, 0.1),
                dominance=genome.attractor_d + random.uniform(-0.1, 0.1),
                vaccination_state=VaccinationState()
            )
            population.append(viva)
            viva_id += 1

    return population

def generate_event(tick: int) -> Tuple[str, float, Tuple[float, float, float]]:
    """Generate events with crisis/celebration cycle."""
    # More realistic event distribution
    if tick % 500 == 0 and tick > 0:
        # Major crisis every 500 ticks
        intensity = random.uniform(0.7, 1.0)
        return ("crisis", intensity, (-0.5, 0.6, -0.4))
    elif tick % 300 == 0:
        # Celebration every 300 ticks
        intensity = random.uniform(0.5, 0.9)
        return ("celebration", intensity, (0.4, 0.3, 0.3))
    elif random.random() < 0.1:
        # 10% chance of random minor event
        if random.random() < 0.6:
            intensity = random.uniform(0.2, 0.5)
            return ("challenge", intensity, (-0.2, 0.3, -0.1))
        else:
            intensity = random.uniform(0.3, 0.6)
            return ("relaxation", intensity, (0.2, -0.2, 0.1))

    return ("none", 0.0, (0.0, 0.0, 0.0))

def run_simulation(
    total_ticks: int = 15000,
    population_size: int = 100,
    enable_survival_protocols: bool = True,
    stagnation_threshold: int = 500,
    vaccination_start_tick: int = 500,
) -> dict:
    """Run v4 survival simulation."""

    print(f"\n{'='*60}")
    print(f"VIVA SIMULATION V4 - SURVIVAL PROTOCOLS")
    print(f"{'='*60}")
    print(f"Survival Protocols: {'ENABLED' if enable_survival_protocols else 'DISABLED'}")
    print(f"Population: {population_size}")
    print(f"Duration: {total_ticks} ticks")
    print(f"Stagnation Threshold: {stagnation_threshold} ticks")
    print(f"Vaccination Start: tick {vaccination_start_tick}")
    print(f"{'='*60}\n")

    population = create_population(population_size)
    baseline_genome = create_balanced_genome()

    # Output data
    stats_data = []
    events_data = []
    generation_data = []
    intervention_data = []

    # Track interventions
    total_vaccinations = 0
    total_emergency_protocols = 0
    total_forced_mutations = 0

    for tick in range(total_ticks):
        # Generate event
        event_type, event_intensity, event_delta = generate_event(tick)

        if event_type != "none":
            events_data.append({
                'tick': tick,
                'type': event_type,
                'intensity': event_intensity,
                'label': f"{event_type}_{tick}"
            })

        # Calculate population avg methylation for social contagion
        avg_methylation = sum(v.genome.methylation for v in population) / len(population)

        # Process each VIVA
        for viva in population:
            g = viva.genome

            # Apply social contagion
            g = apply_social_contagion(g, avg_methylation)

            # Apply event effects
            if event_type == "crisis":
                # Apply immunity if vaccinated
                effective_intensity = event_intensity
                if enable_survival_protocols:
                    effective_intensity = apply_immunity(viva.vaccination_state, event_intensity)
                g = apply_crisis_methylation(g, effective_intensity, tick)
            elif event_type == "celebration":
                g = apply_celebration_healing(g, event_intensity, tick)
            elif event_type == "challenge":
                g = apply_crisis_methylation(g, event_intensity * 0.3, tick)
            elif event_type == "relaxation":
                g = apply_celebration_healing(g, event_intensity * 0.5, tick)

            # Update PAD
            viva = update_pad(viva, event_delta, event_intensity, tick)

            # Update stagnation tracker
            g = update_stagnation_tracker(g)

            # SURVIVAL PROTOCOLS
            if enable_survival_protocols and tick >= vaccination_start_tick:
                # Check emergency status
                status = check_emergency_status(g)

                if status == EmergencyStatus.CRITICAL:
                    # Neurotic emergency protocol
                    g = neurotic_emergency_protocol(g, 0.8)
                    total_emergency_protocols += 1

                # Forced mutation if stagnant
                if g.ticks_without_improvement >= stagnation_threshold:
                    g = forced_adaptive_mutation(g, stagnation_threshold)
                    total_forced_mutations += 1

                # Vaccination (all VIVAs)
                if viva.vaccination_state.doses < 3:
                    g, viva.vaccination_state = emotional_vaccination(
                        g, viva.vaccination_state, tick
                    )
                    if viva.vaccination_state.doses > 0:
                        total_vaccinations += 1

            viva.genome = g

        # Collect stats every 100 ticks
        if tick % 100 == 0:
            pleasures = [v.pleasure for v in population]
            arousals = [v.arousal for v in population]
            dominances = [v.dominance for v in population]
            methylations = [v.genome.methylation for v in population]
            traumas = [v.genome.trauma_methylation for v in population]
            recoveries = [v.genome.recovery_gene for v in population]
            fluencies = [emotional_fluency(v.genome) for v in population]

            drift = detect_drift([v.genome for v in population])

            # Count mutations
            trauma_mutations = sum(1 for v in population
                                  if v.genome.crisis_sensitivity > baseline_genome.crisis_sensitivity * 1.5)
            resilience_mutations = sum(1 for v in population
                                      if v.genome.recovery_gene > baseline_genome.recovery_gene * 2.0)

            # Count vaccination progress
            fully_vaccinated = sum(1 for v in population if v.vaccination_state.doses >= 3)
            avg_immunity = sum(v.vaccination_state.immunity_level for v in population) / len(population)

            stats_data.append({
                'tick': tick,
                'mean_p': sum(pleasures) / len(pleasures),
                'mean_a': sum(arousals) / len(arousals),
                'mean_d': sum(dominances) / len(dominances),
                'std_p': (sum((p - sum(pleasures)/len(pleasures))**2 for p in pleasures) / len(pleasures))**0.5,
                'std_a': (sum((a - sum(arousals)/len(arousals))**2 for a in arousals) / len(arousals))**0.5,
                'std_d': (sum((d - sum(dominances)/len(dominances))**2 for d in dominances) / len(dominances))**0.5,
                'avg_methylation': sum(methylations) / len(methylations),
                'avg_trauma': sum(traumas) / len(traumas),
                'avg_recovery': sum(recoveries) / len(recoveries),
                'avg_fluency': sum(fluencies) / len(fluencies),
                'drift': drift.value,
                'trauma_mutation_pct': trauma_mutations / len(population) * 100,
                'resilience_mutation_pct': resilience_mutations / len(population) * 100,
                'fully_vaccinated': fully_vaccinated,
                'avg_immunity': avg_immunity,
            })

            # Progress report
            if tick % 1000 == 0:
                print(f"Tick {tick:5d} | Drift: {drift.value:15s} | "
                      f"Trauma: {sum(traumas)/len(traumas):.3f} | "
                      f"Recovery: {sum(recoveries)/len(recoveries):.3f} | "
                      f"Vaccinated: {fully_vaccinated}/{len(population)}")

        # Generation stats every 1000 ticks
        if tick % 1000 == 0 and tick > 0:
            generation = tick // 1000
            genomes = [v.genome for v in population]

            trauma_mutations = sum(1 for g in genomes
                                  if g.crisis_sensitivity > baseline_genome.crisis_sensitivity * 1.5)
            resilience_mutations = sum(1 for g in genomes
                                      if g.recovery_gene > baseline_genome.recovery_gene * 2.0)

            generation_data.append({
                'generation': generation,
                'trauma_mutation_pct': trauma_mutations / len(genomes) * 100,
                'resilience_mutation_pct': resilience_mutations / len(genomes) * 100,
                'avg_fluency': sum(emotional_fluency(g) for g in genomes) / len(genomes),
                'avg_methylation': sum(g.methylation for g in genomes) / len(genomes),
                'avg_recovery': sum(g.recovery_gene for g in genomes) / len(genomes),
                'drift': detect_drift(genomes).value,
            })

    # Final genome data
    genome_data = []
    for viva in population:
        g = viva.genome
        genome_data.append({
            'viva_id': viva.id,
            'personality': g.personality,
            'pleasure': viva.pleasure,
            'arousal': viva.arousal,
            'dominance': viva.dominance,
            'trauma': g.trauma_methylation,
            'recovery': g.recovery_gene,
            'crisis_sensitivity': g.crisis_sensitivity,
            'methylation': g.methylation,
            'vaccination_doses': viva.vaccination_state.doses,
            'immunity_level': viva.vaccination_state.immunity_level,
        })

    # Personality data
    personality_data = []
    for pers in ['calm', 'neurotic', 'optimist', 'energetic', 'balanced']:
        pers_vivas = [v for v in population if v.genome.personality == pers]
        if pers_vivas:
            personality_data.append({
                'personality': pers,
                'count': len(pers_vivas),
                'avg_trauma': sum(v.genome.trauma_methylation for v in pers_vivas) / len(pers_vivas),
                'avg_recovery': sum(v.genome.recovery_gene for v in pers_vivas) / len(pers_vivas),
                'avg_fluency': sum(emotional_fluency(v.genome) for v in pers_vivas) / len(pers_vivas),
                'avg_immunity': sum(v.vaccination_state.immunity_level for v in pers_vivas) / len(pers_vivas),
            })

    return {
        'stats': stats_data,
        'events': events_data,
        'generations': generation_data,
        'genomes': genome_data,
        'personalities': personality_data,
        'interventions': {
            'total_vaccinations': total_vaccinations,
            'total_emergency_protocols': total_emergency_protocols,
            'total_forced_mutations': total_forced_mutations,
        }
    }

def save_results(results: dict, output_dir: Path, prefix: str):
    """Save simulation results to CSV files."""

    # Stats
    with open(output_dir / f"{prefix}_stats.csv", 'w', newline='') as f:
        if results['stats']:
            writer = csv.DictWriter(f, fieldnames=results['stats'][0].keys())
            writer.writeheader()
            writer.writerows(results['stats'])

    # Events
    with open(output_dir / f"{prefix}_events.csv", 'w', newline='') as f:
        if results['events']:
            writer = csv.DictWriter(f, fieldnames=results['events'][0].keys())
            writer.writeheader()
            writer.writerows(results['events'])

    # Generations
    with open(output_dir / f"{prefix}_generations.csv", 'w', newline='') as f:
        if results['generations']:
            writer = csv.DictWriter(f, fieldnames=results['generations'][0].keys())
            writer.writeheader()
            writer.writerows(results['generations'])

    # Final genomes
    with open(output_dir / f"{prefix}_final_genomes.csv", 'w', newline='') as f:
        if results['genomes']:
            writer = csv.DictWriter(f, fieldnames=results['genomes'][0].keys())
            writer.writeheader()
            writer.writerows(results['genomes'])

    # Personalities
    with open(output_dir / f"{prefix}_personalities.csv", 'w', newline='') as f:
        if results['personalities']:
            writer = csv.DictWriter(f, fieldnames=results['personalities'][0].keys())
            writer.writeheader()
            writer.writerows(results['personalities'])

def main():
    output_dir = Path("data/simulations")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Run WITH survival protocols
    print("\n" + "="*70)
    print("RUNNING SIMULATION WITH SURVIVAL PROTOCOLS")
    print("="*70)

    results_with = run_simulation(
        total_ticks=15000,
        enable_survival_protocols=True,
        stagnation_threshold=500,
        vaccination_start_tick=500,
    )

    prefix_with = f"sim_v4_{timestamp}_survival"
    save_results(results_with, output_dir, prefix_with)

    # Run WITHOUT survival protocols (control group)
    print("\n" + "="*70)
    print("RUNNING CONTROL SIMULATION (NO SURVIVAL PROTOCOLS)")
    print("="*70)

    results_without = run_simulation(
        total_ticks=15000,
        enable_survival_protocols=False,
    )

    prefix_without = f"sim_v4_{timestamp}_control"
    save_results(results_without, output_dir, prefix_without)

    # Print comparison
    print("\n" + "="*70)
    print("SIMULATION COMPLETE - COMPARISON")
    print("="*70)

    final_with = results_with['stats'][-1]
    final_without = results_without['stats'][-1]

    print(f"\n{'Metric':<30} {'With Protocols':>20} {'Without (Control)':>20}")
    print("-" * 70)
    print(f"{'Final Drift':<30} {final_with['drift']:>20} {final_without['drift']:>20}")
    print(f"{'Avg Trauma':<30} {final_with['avg_trauma']:>20.4f} {final_without['avg_trauma']:>20.4f}")
    print(f"{'Avg Recovery':<30} {final_with['avg_recovery']:>20.4f} {final_without['avg_recovery']:>20.4f}")
    print(f"{'Avg Fluency':<30} {final_with['avg_fluency']:>20.6f} {final_without['avg_fluency']:>20.6f}")
    print(f"{'Trauma Mutations %':<30} {final_with['trauma_mutation_pct']:>20.1f} {final_without['trauma_mutation_pct']:>20.1f}")
    print(f"{'Resilience Mutations %':<30} {final_with['resilience_mutation_pct']:>20.1f} {final_without['resilience_mutation_pct']:>20.1f}")
    print(f"{'Fully Vaccinated':<30} {final_with['fully_vaccinated']:>20} {'N/A':>20}")
    print(f"{'Avg Immunity':<30} {final_with['avg_immunity']:>20.2f} {'N/A':>20}")

    print(f"\nInterventions Applied:")
    print(f"  Vaccinations: {results_with['interventions']['total_vaccinations']}")
    print(f"  Emergency Protocols: {results_with['interventions']['total_emergency_protocols']}")
    print(f"  Forced Mutations: {results_with['interventions']['total_forced_mutations']}")

    # Survival assessment
    print(f"\n{'='*70}")
    print("SURVIVAL ASSESSMENT")
    print(f"{'='*70}")

    control_collapsed = final_without['drift'] == "TraumaDrift" and final_without['avg_trauma'] > 0.5
    survival_survived = final_with['drift'] != "TraumaDrift" or final_with['avg_trauma'] < 0.3

    if control_collapsed and survival_survived:
        print("\n  SURVIVAL PROTOCOLS SUCCESSFUL!")
        print("  Population survived the 7,200-7,800 collapse window.")
        print(f"  Control group showed signs of collapse (drift: {final_without['drift']})")
    elif not control_collapsed:
        print("\n  Control group did not collapse in this run.")
        print("  May need longer simulation or different event distribution.")
    else:
        print("\n  Both groups show signs of trauma accumulation.")
        print("  Survival protocols may need adjustment.")

    print(f"\nResults saved to: {output_dir}")
    print(f"  Survival: {prefix_with}_*.csv")
    print(f"  Control: {prefix_without}_*.csv")

if __name__ == "__main__":
    main()
