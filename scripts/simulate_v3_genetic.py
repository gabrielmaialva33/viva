#!/usr/bin/env python3
"""
VIVA Simulation v3 - Genetic Evolution
Based on Qwen3 recommendations for DNA Emocional Dual

Features:
- Genome-based emotional dynamics
- Epigenetic drift monitoring
- Adaptive mutations under extreme stress
- Social contagion between VIVAs
- Generation tracking and evolution visualization
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import time
import json

# =============================================================================
# GENOME TYPES (Python mirror of Gleam types)
# =============================================================================

@dataclass
class PadVector:
    pleasure: float
    arousal: float
    dominance: float

@dataclass
class ModulatorGenes:
    crisis_sensitivity: float
    social_contagion: float
    recovery_gene: float
    rupture_threshold: float

@dataclass
class EpigeneticState:
    methylation: float = 0.0
    trauma_methylation: float = 0.0
    healing_factor: float = 0.5
    history: List[Tuple[int, float, str]] = field(default_factory=list)

@dataclass
class Genome:
    primary_attractor: PadVector
    inertia_p: float
    inertia_a: float
    inertia_d: float
    modulators: ModulatorGenes
    epigenetics: EpigeneticState
    personality: str

# =============================================================================
# PERSONALITY GENOMES
# =============================================================================

def create_genome(personality: str) -> Genome:
    """Create genome from personality type"""
    configs = {
        'calm': {
            'attractor': PadVector(0.4, -0.3, 0.7),
            'inertia': (0.85, 0.90, 0.80),
            'modulators': ModulatorGenes(0.30, 0.40, 0.60, 0.90),
            'trauma': 0.0,
        },
        'neurotic': {
            'attractor': PadVector(-0.6, 0.8, -0.7),
            'inertia': (0.92, 0.95, 0.88),
            'modulators': ModulatorGenes(0.95, 0.70, 0.05, 0.40),
            'trauma': 0.5,
        },
        'optimist': {
            'attractor': PadVector(0.8, 0.5, 0.6),
            'inertia': (0.40, 0.50, 0.50),
            'modulators': ModulatorGenes(0.05, 0.60, 0.85, 0.95),
            'trauma': 0.0,
        },
        'energetic': {
            'attractor': PadVector(0.3, 0.85, 0.4),
            'inertia': (0.30, 0.25, 0.35),
            'modulators': ModulatorGenes(0.70, 0.80, 0.70, 0.60),
            'trauma': 0.0,
        },
        'balanced': {
            'attractor': PadVector(0.0, 0.0, 0.0),
            'inertia': (0.50, 0.50, 0.50),
            'modulators': ModulatorGenes(0.50, 0.50, 0.50, 0.70),
            'trauma': 0.0,
        },
    }

    cfg = configs[personality]
    return Genome(
        primary_attractor=cfg['attractor'],
        inertia_p=cfg['inertia'][0],
        inertia_a=cfg['inertia'][1],
        inertia_d=cfg['inertia'][2],
        modulators=cfg['modulators'],
        epigenetics=EpigeneticState(trauma_methylation=cfg['trauma']),
        personality=personality,
    )

# =============================================================================
# EMOTIONAL UPDATE FORMULA (Non-Linear)
# =============================================================================

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))

def compute_emotional_delta(
    genome: Genome,
    current: PadVector,
    event_delta: PadVector,
    event_intensity: float,
    tick: int,
) -> PadVector:
    """
    Non-linear emotional update based on genome.
    ΔP = (attractor - current) * e^(-inertia * t) + event_δ * sigmoid(sensitivity * (event - θ))
    """
    attractor = genome.primary_attractor
    t = tick / 1000.0

    # Express crisis_sensitivity with epigenetics
    effective_sensitivity = genome.modulators.crisis_sensitivity * (1.0 - genome.epigenetics.trauma_methylation * 0.5)

    # Attractor force (exponential decay based on inertia)
    attractor_force_p = (attractor.pleasure - current.pleasure) * np.exp(-genome.inertia_p * t)
    attractor_force_a = (attractor.arousal - current.arousal) * np.exp(-genome.inertia_a * t)
    attractor_force_d = (attractor.dominance - current.dominance) * np.exp(-genome.inertia_d * t)

    # Event response (sigmoid for non-linearity)
    threshold = genome.modulators.rupture_threshold
    sigmoid_factor = sigmoid(effective_sensitivity * (event_intensity - threshold))

    event_force_p = event_delta.pleasure * sigmoid_factor
    event_force_a = event_delta.arousal * sigmoid_factor
    event_force_d = event_delta.dominance * sigmoid_factor

    # Combine forces, scaled by (1 - inertia)
    return PadVector(
        pleasure=(attractor_force_p + event_force_p) * (1.0 - genome.inertia_p),
        arousal=(attractor_force_a + event_force_a) * (1.0 - genome.inertia_a),
        dominance=(attractor_force_d + event_force_d) * (1.0 - genome.inertia_d),
    )

# =============================================================================
# EPIGENETIC DYNAMICS
# =============================================================================

def apply_crisis_methylation(genome: Genome, intensity: float, tick: int) -> Genome:
    """Apply crisis trauma to genome"""
    new_trauma = min(1.0, genome.epigenetics.trauma_methylation + intensity * 0.2)
    new_methyl = min(1.0, genome.epigenetics.methylation + intensity * 0.1)

    genome.epigenetics.trauma_methylation = new_trauma
    genome.epigenetics.methylation = new_methyl
    genome.epigenetics.history.append((tick, intensity, 'crisis'))
    return genome

def apply_celebration_healing(genome: Genome, intensity: float, tick: int) -> Genome:
    """Apply celebration healing to genome"""
    healing_factor = 1.0 - intensity * 0.4
    genome.epigenetics.trauma_methylation = max(0.0, genome.epigenetics.trauma_methylation * healing_factor)
    genome.epigenetics.methylation = max(0.0, genome.epigenetics.methylation * 0.95)
    genome.epigenetics.history.append((tick, intensity, 'celebration'))
    return genome

def trigger_adaptive_mutation(genome: Genome, crisis_intensity: float) -> Genome:
    """Trigger adaptive mutation under extreme crisis"""
    if crisis_intensity > 0.9 and genome.modulators.recovery_gene < 0.1:
        # Activate resilience mutation
        genome.modulators.recovery_gene = min(1.0, genome.modulators.recovery_gene + 0.3)
        genome.epigenetics.trauma_methylation *= 0.5  # Partial reset
    return genome

# =============================================================================
# SOCIAL CONTAGION
# =============================================================================

def apply_social_contagion(genome: Genome, neighbor_avg_methylation: float) -> Genome:
    """Apply social contagion from neighbors"""
    contagion_rate = genome.modulators.social_contagion
    current = genome.epigenetics.methylation
    delta = (neighbor_avg_methylation - current) * contagion_rate * 0.1
    genome.epigenetics.methylation = np.clip(current + delta, 0.0, 1.0)
    return genome

# =============================================================================
# POPULATION MONITORING
# =============================================================================

def detect_drift(genomes: List[Genome], baseline_methyl: float = 0.1) -> str:
    """Detect epigenetic drift in population"""
    avg_methyl = np.mean([g.epigenetics.methylation for g in genomes])

    if avg_methyl > baseline_methyl * 1.8:
        return "TraumaDrift"  # Population accumulating trauma
    elif avg_methyl < baseline_methyl * 0.3:
        return "ResilienceDrift"  # Toxic resilience
    return "NoDrift"

def population_stats(genomes: List[Genome]) -> Dict:
    """Calculate population statistics"""
    methylations = [g.epigenetics.methylation for g in genomes]
    traumas = [g.epigenetics.trauma_methylation for g in genomes]
    recoveries = [g.modulators.recovery_gene for g in genomes]
    sensitivities = [g.modulators.crisis_sensitivity for g in genomes]

    return {
        'avg_methylation': np.mean(methylations),
        'std_methylation': np.std(methylations),
        'avg_trauma': np.mean(traumas),
        'avg_recovery': np.mean(recoveries),
        'avg_sensitivity': np.mean(sensitivities),
        'drift': detect_drift(genomes),
    }

# =============================================================================
# VIVA ENTITY
# =============================================================================

@dataclass
class Viva:
    id: int
    genome: Genome
    pad: PadVector

    def update(self, event_delta: PadVector, event_intensity: float, tick: int):
        """Update VIVA state based on genome and event"""
        delta = compute_emotional_delta(
            self.genome, self.pad, event_delta, event_intensity, tick
        )

        # Apply delta to current state
        self.pad = PadVector(
            pleasure=np.clip(self.pad.pleasure + delta.pleasure, -1, 1),
            arousal=np.clip(self.pad.arousal + delta.arousal, -1, 1),
            dominance=np.clip(self.pad.dominance + delta.dominance, -1, 1),
        )

    def get_quadrant(self) -> str:
        """Get emotional quadrant"""
        if self.pad.pleasure > 0 and self.pad.arousal > 0:
            return "Excited/Happy"
        elif self.pad.pleasure > 0:
            return "Calm/Happy"
        elif self.pad.arousal > 0:
            return "Stressed"
        return "Depressed"

# =============================================================================
# SIMULATION
# =============================================================================

class GeneticSimulation:
    def __init__(self, n_vivas: int = 100, ticks: int = 10000):
        self.n_vivas = n_vivas
        self.total_ticks = ticks
        self.vivas: List[Viva] = []
        self.events: List[Dict] = []
        self.stats_history: List[Dict] = []
        self.generation_history: List[Dict] = []

    def initialize(self):
        """Initialize population with random personalities"""
        personalities = ['calm', 'neurotic', 'optimist', 'energetic', 'balanced']
        weights = [0.20, 0.15, 0.25, 0.20, 0.20]  # Distribution

        for i in range(self.n_vivas):
            pers = np.random.choice(personalities, p=weights)
            genome = create_genome(pers)

            # Random starting state
            pad = PadVector(
                pleasure=np.random.uniform(-0.5, 0.5),
                arousal=np.random.uniform(-0.5, 0.5),
                dominance=np.random.uniform(-0.5, 0.5),
            )

            self.vivas.append(Viva(id=i+1, genome=genome, pad=pad))

        # Generate events
        self._generate_events()

    def _generate_events(self):
        """Generate random global events"""
        event_types = [
            ('crisis', -0.15, 0.3, -0.1),
            ('resolution', 0.1, -0.2, 0.1),
            ('challenge', 0.0, 0.2, 0.2),
            ('relaxation', 0.15, -0.1, 0.0),
            ('uncertainty', -0.1, 0.1, -0.2),
            ('celebration', 0.2, 0.2, 0.1),
            ('disappointment', -0.2, -0.1, 0.0),
            ('empowerment', 0.0, -0.2, 0.2),
        ]

        # Events every 500 ticks
        for tick in range(500, self.total_ticks, 500):
            etype, dp, da, dd = event_types[np.random.randint(len(event_types))]
            intensity = np.random.uniform(0.5, 0.9)

            self.events.append({
                'tick': tick,
                'type': etype,
                'label': etype,
                'dp': dp * intensity,
                'da': da * intensity,
                'dd': dd * intensity,
                'intensity': intensity,
            })

    def run(self):
        """Run simulation"""
        print(f"Starting Genetic Simulation v3: {self.n_vivas} VIVAs, {self.total_ticks} ticks")

        event_idx = 0
        current_event = None

        for tick in range(self.total_ticks):
            # Check for events
            if event_idx < len(self.events) and self.events[event_idx]['tick'] == tick:
                current_event = self.events[event_idx]
                event_idx += 1
                print(f"  Tick {tick}: EVENT {current_event['type']} (intensity={current_event['intensity']:.2f})")

            # Update each VIVA
            avg_methylation = np.mean([v.genome.epigenetics.methylation for v in self.vivas])

            for viva in self.vivas:
                if current_event and current_event['tick'] == tick:
                    # Apply event
                    event_delta = PadVector(
                        current_event['dp'],
                        current_event['da'],
                        current_event['dd'],
                    )
                    viva.update(event_delta, current_event['intensity'], tick)

                    # Apply epigenetic changes
                    if current_event['type'] == 'crisis':
                        apply_crisis_methylation(viva.genome, current_event['intensity'], tick)
                        trigger_adaptive_mutation(viva.genome, current_event['intensity'])
                    elif current_event['type'] == 'celebration':
                        apply_celebration_healing(viva.genome, current_event['intensity'], tick)
                    elif current_event['type'] == 'challenge':
                        apply_crisis_methylation(viva.genome, current_event['intensity'] * 0.3, tick)
                    elif current_event['type'] == 'relaxation':
                        apply_celebration_healing(viva.genome, current_event['intensity'] * 0.5, tick)
                else:
                    # Normal tick - just attractor force
                    viva.update(PadVector(0, 0, 0), 0.0, tick)

                # Apply social contagion
                apply_social_contagion(viva.genome, avg_methylation)

            # Record stats every 100 ticks
            if tick % 100 == 0:
                self._record_stats(tick)

            # Record generation stats every 1000 ticks
            if tick % 1000 == 0:
                self._record_generation(tick // 1000)

        print("Simulation complete!")

    def _record_stats(self, tick: int):
        """Record population statistics"""
        pads = [(v.pad.pleasure, v.pad.arousal, v.pad.dominance) for v in self.vivas]
        p_vals = [p[0] for p in pads]
        a_vals = [p[1] for p in pads]
        d_vals = [p[2] for p in pads]

        pop_stats = population_stats([v.genome for v in self.vivas])

        self.stats_history.append({
            'tick': tick,
            'mean_p': np.mean(p_vals),
            'mean_a': np.mean(a_vals),
            'mean_d': np.mean(d_vals),
            'std_p': np.std(p_vals),
            'std_a': np.std(a_vals),
            'std_d': np.std(d_vals),
            **pop_stats,
        })

    def _record_generation(self, generation: int):
        """Record generation statistics"""
        genomes = [v.genome for v in self.vivas]

        # Count mutations
        baseline = create_genome('balanced')
        trauma_count = sum(1 for g in genomes
                          if g.modulators.crisis_sensitivity > baseline.modulators.crisis_sensitivity * 1.5)
        resilience_count = sum(1 for g in genomes
                               if g.modulators.recovery_gene > baseline.modulators.recovery_gene * 2.0)

        avg_fluency = np.mean([
            (1 - (g.inertia_p + g.inertia_a + g.inertia_d) / 3) * g.modulators.recovery_gene * (1 - g.epigenetics.trauma_methylation)
            for g in genomes
        ])

        self.generation_history.append({
            'generation': generation,
            'trauma_mutation_pct': trauma_count / len(genomes) * 100,
            'resilience_mutation_pct': resilience_count / len(genomes) * 100,
            'avg_fluency': avg_fluency,
            'avg_methylation': np.mean([g.epigenetics.methylation for g in genomes]),
            'avg_trauma': np.mean([g.epigenetics.trauma_methylation for g in genomes]),
            'drift': detect_drift(genomes),
        })

    def save_results(self):
        """Save simulation results"""
        timestamp = int(time.time())
        prefix = f"sim_v3_{timestamp}"
        data_dir = Path("data/simulations")
        data_dir.mkdir(parents=True, exist_ok=True)

        # Stats
        stats_df = pd.DataFrame(self.stats_history)
        stats_df.to_csv(data_dir / f"{prefix}_stats.csv", index=False)

        # Events
        events_df = pd.DataFrame(self.events)
        events_df.to_csv(data_dir / f"{prefix}_events.csv", index=False)

        # PAD states
        pads_data = []
        for tick_data in self.stats_history:
            tick = tick_data['tick']
            for viva in self.vivas:
                pads_data.append({
                    'tick': tick,
                    'viva_id': viva.id,
                    'pleasure': viva.pad.pleasure,
                    'arousal': viva.pad.arousal,
                    'dominance': viva.pad.dominance,
                    'personality': viva.genome.personality,
                    'methylation': viva.genome.epigenetics.methylation,
                    'trauma': viva.genome.epigenetics.trauma_methylation,
                    'recovery': viva.genome.modulators.recovery_gene,
                })
        # Only save final state to reduce file size
        final_pads = [{
            'viva_id': v.id,
            'pleasure': v.pad.pleasure,
            'arousal': v.pad.arousal,
            'dominance': v.pad.dominance,
            'personality': v.genome.personality,
            'methylation': v.genome.epigenetics.methylation,
            'trauma': v.genome.epigenetics.trauma_methylation,
            'recovery': v.genome.modulators.recovery_gene,
            'crisis_sensitivity': v.genome.modulators.crisis_sensitivity,
        } for v in self.vivas]
        pd.DataFrame(final_pads).to_csv(data_dir / f"{prefix}_final_genomes.csv", index=False)

        # Personalities
        pers_data = [{'viva_id': v.id, 'personality': v.genome.personality} for v in self.vivas]
        pd.DataFrame(pers_data).to_csv(data_dir / f"{prefix}_personalities.csv", index=False)

        # Generation evolution
        gen_df = pd.DataFrame(self.generation_history)
        gen_df.to_csv(data_dir / f"{prefix}_generations.csv", index=False)

        print(f"\nResults saved with prefix: {prefix}")
        print(f"  - {prefix}_stats.csv")
        print(f"  - {prefix}_events.csv")
        print(f"  - {prefix}_final_genomes.csv")
        print(f"  - {prefix}_personalities.csv")
        print(f"  - {prefix}_generations.csv")

        return prefix

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    sim = GeneticSimulation(n_vivas=100, ticks=10000)
    sim.initialize()
    sim.run()
    prefix = sim.save_results()

    # Print final summary
    print("\n" + "="*60)
    print("GENETIC SIMULATION V3 - FINAL SUMMARY")
    print("="*60)

    pop_stats = population_stats([v.genome for v in sim.vivas])
    print(f"\nPopulation Statistics:")
    print(f"  Avg Methylation: {pop_stats['avg_methylation']:.4f}")
    print(f"  Avg Trauma: {pop_stats['avg_trauma']:.4f}")
    print(f"  Avg Recovery: {pop_stats['avg_recovery']:.4f}")
    print(f"  Drift Type: {pop_stats['drift']}")

    if sim.generation_history:
        last_gen = sim.generation_history[-1]
        print(f"\nGeneration {last_gen['generation']} Evolution:")
        print(f"  Trauma Mutations: {last_gen['trauma_mutation_pct']:.1f}%")
        print(f"  Resilience Mutations: {last_gen['resilience_mutation_pct']:.1f}%")
        print(f"  Avg Fluency: {last_gen['avg_fluency']:.6f}")

    # Quadrant distribution
    quadrants = {}
    for v in sim.vivas:
        q = v.get_quadrant()
        quadrants[q] = quadrants.get(q, 0) + 1

    print(f"\nFinal Quadrant Distribution:")
    for q, count in sorted(quadrants.items(), key=lambda x: -x[1]):
        print(f"  {q}: {count} ({count/len(sim.vivas)*100:.1f}%)")

    # Personality breakdown
    print(f"\nGenome Health by Personality:")
    for pers in ['calm', 'neurotic', 'optimist', 'energetic', 'balanced']:
        vivas_pers = [v for v in sim.vivas if v.genome.personality == pers]
        if vivas_pers:
            avg_trauma = np.mean([v.genome.epigenetics.trauma_methylation for v in vivas_pers])
            avg_recovery = np.mean([v.genome.modulators.recovery_gene for v in vivas_pers])
            print(f"  {pers}: trauma={avg_trauma:.3f}, recovery={avg_recovery:.3f}")
