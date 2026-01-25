//// Genome - Emotional DNA for VIVA
////
//// Implements the Dual Helix Emotional DNA architecture:
//// - Chromosome A: Tendencies (attractors in PAD space)
//// - Chromosome B: Resistances (emotional inertia)
//// - Modulator Genes: crisis_sensitivity, social_contagion, recovery_gene
//// - Epigenetic System: Methylation that modifies gene expression
////
//// Based on Qwen3 validation (10/10) - "Emotional Genomics Artificial"

import gleam/float
import gleam/int
import gleam/list
import gleam/option.{type Option, None, Some}

// =============================================================================
// CORE TYPES
// =============================================================================

/// PAD Vector for attractors
pub type PadVector {
  PadVector(pleasure: Float, arousal: Float, dominance: Float)
}

/// Allele types - gene variants
pub type Allele {
  /// Dominant allele - fully expressed
  Dominant
  /// Recessive allele - expressed only when homozygous
  Recessive
  /// Epigenetic - expression modified by methylation level
  Epigenetic(methylation: Float)
}

/// Chromosome A - Tendencies (where VIVA naturally gravitates)
pub type ChromosomeA {
  ChromosomeA(
    /// Primary attractor in PAD space
    primary_attractor: PadVector,
    /// Secondary attractors with activation thresholds
    secondary_attractors: List(#(PadVector, Float)),
    /// Allele for each PAD dimension
    pleasure_allele: Allele,
    arousal_allele: Allele,
    dominance_allele: Allele,
  )
}

/// Chromosome B - Resistances (how much VIVA resists change)
pub type ChromosomeB {
  ChromosomeB(
    /// Inertia for each PAD dimension [0.0 = no resistance, 1.0 = immovable]
    pleasure_inertia: Float,
    arousal_inertia: Float,
    dominance_inertia: Float,
    /// State-specific inertia (higher when in certain quadrants)
    state_inertia: List(#(String, Float)),
  )
}

/// Modulator Genes - affect how VIVA responds to events
pub type ModulatorGenes {
  ModulatorGenes(
    /// How much external events affect emotional state [0.0 = immune, 1.0 = hypersensitive]
    crisis_sensitivity: Float,
    /// How much VIVA absorbs emotions from others [0.0 = isolated, 1.0 = empathic]
    social_contagion: Float,
    /// Speed of return to attractor [0.0 = never, 1.0 = instant]
    recovery_gene: Float,
    /// Threshold for emotional breakdown
    rupture_threshold: Float,
  )
}

/// Epigenetic State - modifications to gene expression over time
pub type EpigeneticState {
  EpigeneticState(
    /// Overall methylation level [0.0 = fully expressed, 1.0 = silenced]
    methylation: Float,
    /// Crisis-induced methylation (accumulates with trauma)
    trauma_methylation: Float,
    /// Celebration-induced demethylation (heals trauma)
    healing_factor: Float,
    /// History of significant events that modified epigenetics
    methylation_history: List(#(Int, Float, String)),
  )
}

/// Complete Genome
pub type Genome {
  Genome(
    /// Chromosome A - Tendencies
    chromosome_a: ChromosomeA,
    /// Chromosome B - Resistances
    chromosome_b: ChromosomeB,
    /// Modulator genes
    modulators: ModulatorGenes,
    /// Epigenetic state (mutable over lifetime)
    epigenetics: EpigeneticState,
  )
}

/// Personality types (legacy compatibility + genome mapping)
pub type PersonalityType {
  Calm
  Neurotic
  Optimist
  Energetic
  Balanced
}

// =============================================================================
// CONSTRUCTORS
// =============================================================================

/// Create default genome (balanced personality)
pub fn new() -> Genome {
  from_personality(Balanced)
}

/// Create genome from personality type
pub fn from_personality(personality: PersonalityType) -> Genome {
  case personality {
    Calm -> calm_genome()
    Neurotic -> neurotic_genome()
    Optimist -> optimist_genome()
    Energetic -> energetic_genome()
    Balanced -> balanced_genome()
  }
}

/// Calm genome - "Slow Explorer"
/// High entropy + high stickiness = visits many states but stays long in each
fn calm_genome() -> Genome {
  Genome(
    chromosome_a: ChromosomeA(
      primary_attractor: PadVector(0.4, -0.3, 0.7),
      secondary_attractors: [
        // Can visit Excited/Happy during celebrations
        #(PadVector(0.98, 0.83, 0.82), 0.7),
        // Can visit Stressed during severe crisis
        #(PadVector(-0.95, 0.98, -0.97), 0.85),
      ],
      pleasure_allele: Dominant,
      arousal_allele: Recessive,
      dominance_allele: Dominant,
    ),
    chromosome_b: ChromosomeB(
      pleasure_inertia: 0.85,
      arousal_inertia: 0.90,
      dominance_inertia: 0.80,
      state_inertia: [
        #("Calm/Happy", 0.95),
        #("Excited/Happy", 0.80),
      ],
    ),
    modulators: ModulatorGenes(
      crisis_sensitivity: 0.3,
      social_contagion: 0.4,
      recovery_gene: 0.6,
      rupture_threshold: 0.9,
    ),
    epigenetics: new_epigenetics(),
  )
}

/// Neurotic genome - "Trapped"
/// 100% Stressed→Stressed, lowest fluency
fn neurotic_genome() -> Genome {
  Genome(
    chromosome_a: ChromosomeA(
      primary_attractor: PadVector(-0.6, 0.8, -0.7),
      secondary_attractors: [],  // No escape routes
      pleasure_allele: Recessive,
      arousal_allele: Dominant,
      dominance_allele: Recessive,
    ),
    chromosome_b: ChromosomeB(
      pleasure_inertia: 0.92,
      arousal_inertia: 0.95,
      dominance_inertia: 0.88,
      state_inertia: [
        #("Stressed", 0.99),  // Almost impossible to leave
      ],
    ),
    modulators: ModulatorGenes(
      crisis_sensitivity: 0.95,  // Hypersensitive
      social_contagion: 0.7,
      recovery_gene: 0.05,  // Almost no recovery
      rupture_threshold: 0.4,  // Low threshold
    ),
    epigenetics: EpigeneticState(
      methylation: 0.3,
      trauma_methylation: 0.5,  // Born with some trauma
      healing_factor: 0.1,
      methylation_history: [],
    ),
  )
}

/// Optimist genome - "Immune to crises"
/// Delta P = 0 even during crisis
fn optimist_genome() -> Genome {
  Genome(
    chromosome_a: ChromosomeA(
      primary_attractor: PadVector(0.8, 0.5, 0.6),
      secondary_attractors: [
        #(PadVector(0.95, 0.9, 0.8), 0.3),  // Easy to reach excited
      ],
      pleasure_allele: Dominant,
      arousal_allele: Dominant,
      dominance_allele: Dominant,
    ),
    chromosome_b: ChromosomeB(
      pleasure_inertia: 0.4,  // Flexible in pleasure
      arousal_inertia: 0.5,
      dominance_inertia: 0.5,
      state_inertia: [
        #("Excited/Happy", 0.70),
        #("Calm/Happy", 0.65),
      ],
    ),
    modulators: ModulatorGenes(
      crisis_sensitivity: 0.05,  // Nearly immune!
      social_contagion: 0.6,
      recovery_gene: 0.85,  // Fast recovery
      rupture_threshold: 0.95,  // Very hard to break
    ),
    epigenetics: new_epigenetics(),
  )
}

/// Energetic genome - "Volatile"
/// High arousal, low inertia = changes quickly
fn energetic_genome() -> Genome {
  Genome(
    chromosome_a: ChromosomeA(
      primary_attractor: PadVector(0.3, 0.85, 0.4),
      secondary_attractors: [
        #(PadVector(0.7, 0.95, 0.6), 0.4),
        #(PadVector(-0.2, 0.9, 0.2), 0.5),
      ],
      pleasure_allele: Epigenetic(0.2),
      arousal_allele: Dominant,
      dominance_allele: Epigenetic(0.3),
    ),
    chromosome_b: ChromosomeB(
      pleasure_inertia: 0.3,
      arousal_inertia: 0.25,  // Very low - changes fast
      dominance_inertia: 0.35,
      state_inertia: [],  // No state-specific inertia
    ),
    modulators: ModulatorGenes(
      crisis_sensitivity: 0.7,
      social_contagion: 0.8,  // Highly influenced by others
      recovery_gene: 0.7,
      rupture_threshold: 0.6,
    ),
    epigenetics: new_epigenetics(),
  )
}

/// Balanced genome - neutral baseline
fn balanced_genome() -> Genome {
  Genome(
    chromosome_a: ChromosomeA(
      primary_attractor: PadVector(0.0, 0.0, 0.0),
      secondary_attractors: [],
      pleasure_allele: Epigenetic(0.5),
      arousal_allele: Epigenetic(0.5),
      dominance_allele: Epigenetic(0.5),
    ),
    chromosome_b: ChromosomeB(
      pleasure_inertia: 0.5,
      arousal_inertia: 0.5,
      dominance_inertia: 0.5,
      state_inertia: [],
    ),
    modulators: ModulatorGenes(
      crisis_sensitivity: 0.5,
      social_contagion: 0.5,
      recovery_gene: 0.5,
      rupture_threshold: 0.7,
    ),
    epigenetics: new_epigenetics(),
  )
}

/// Fresh epigenetic state
fn new_epigenetics() -> EpigeneticState {
  EpigeneticState(
    methylation: 0.0,
    trauma_methylation: 0.0,
    healing_factor: 0.5,
    methylation_history: [],
  )
}

// =============================================================================
// GENE EXPRESSION
// =============================================================================

/// Express a gene value considering allele type and epigenetics
pub fn express_gene(
  base_value: Float,
  allele: Allele,
  epigenetics: EpigeneticState,
  event_intensity: Float,
) -> Float {
  let methylation_effect = 1.0 -. epigenetics.methylation
  let trauma_effect = 1.0 -. { epigenetics.trauma_methylation *. 0.5 }

  case allele {
    Dominant -> {
      // Full expression, slightly reduced by methylation
      base_value *. methylation_effect *. trauma_effect
    }
    Recessive -> {
      // Reduced expression
      base_value *. 0.5 *. methylation_effect *. trauma_effect
    }
    Epigenetic(allele_methyl) -> {
      // Expression varies with methylation
      let total_methyl = { allele_methyl +. epigenetics.methylation } /. 2.0
      base_value *. { 1.0 -. total_methyl } *. trauma_effect
    }
  }
}

/// Get effective attractor considering secondary attractors and event context
pub fn get_effective_attractor(
  genome: Genome,
  current_state: PadVector,
  event_intensity: Float,
) -> PadVector {
  let primary = genome.chromosome_a.primary_attractor

  // Check if any secondary attractor should be activated
  let activated = list.filter(genome.chromosome_a.secondary_attractors, fn(pair) {
    let #(_attractor, threshold) = pair
    event_intensity >. threshold
  })

  case activated {
    [] -> primary
    [#(attractor, _), ..] -> {
      // Blend primary with activated secondary
      let blend = event_intensity
      PadVector(
        pleasure: primary.pleasure *. { 1.0 -. blend } +. attractor.pleasure *. blend,
        arousal: primary.arousal *. { 1.0 -. blend } +. attractor.arousal *. blend,
        dominance: primary.dominance *. { 1.0 -. blend } +. attractor.dominance *. blend,
      )
    }
  }
}

/// Get effective inertia for current state
pub fn get_effective_inertia(
  genome: Genome,
  current_quadrant: String,
) -> #(Float, Float, Float) {
  let base_p = genome.chromosome_b.pleasure_inertia
  let base_a = genome.chromosome_b.arousal_inertia
  let base_d = genome.chromosome_b.dominance_inertia

  // Check for state-specific inertia modifier
  let modifier = case list.find(genome.chromosome_b.state_inertia, fn(pair) {
    let #(state, _) = pair
    state == current_quadrant
  }) {
    Ok(#(_, mod)) -> mod
    Error(_) -> 1.0
  }

  #(
    clamp(base_p *. modifier, 0.0, 0.99),
    clamp(base_a *. modifier, 0.0, 0.99),
    clamp(base_d *. modifier, 0.0, 0.99),
  )
}

// =============================================================================
// EMOTIONAL UPDATE FORMULA (Non-Linear)
// =============================================================================

/// Main emotional update based on genome
/// Formula: ΔP = (attractor - current) * e^(-inertia * t) + event_δ * sigmoid(crisis_sensitivity * (event - θ))
pub fn compute_emotional_delta(
  genome: Genome,
  current: PadVector,
  event_delta: PadVector,
  event_intensity: Float,
  current_quadrant: String,
  tick: Int,
) -> PadVector {
  // Get effective attractor (may shift based on event intensity)
  let attractor = get_effective_attractor(genome, current, event_intensity)

  // Get effective inertia for current state
  let #(inertia_p, inertia_a, inertia_d) = get_effective_inertia(genome, current_quadrant)

  // Express crisis_sensitivity with epigenetics
  let effective_sensitivity = express_gene(
    genome.modulators.crisis_sensitivity,
    Epigenetic(genome.epigenetics.trauma_methylation),
    genome.epigenetics,
    event_intensity,
  )

  // Time decay factor (normalized tick)
  let t = int.to_float(tick) /. 1000.0

  // Attractor force (exponential decay based on inertia)
  let attractor_force_p = { attractor.pleasure -. current.pleasure } *. exp(0.0 -. inertia_p *. t)
  let attractor_force_a = { attractor.arousal -. current.arousal } *. exp(0.0 -. inertia_a *. t)
  let attractor_force_d = { attractor.dominance -. current.dominance } *. exp(0.0 -. inertia_d *. t)

  // Event response (sigmoid for non-linearity)
  let threshold = genome.modulators.rupture_threshold
  let sigmoid_factor = sigmoid(effective_sensitivity *. { event_intensity -. threshold })

  let event_force_p = event_delta.pleasure *. sigmoid_factor
  let event_force_a = event_delta.arousal *. sigmoid_factor
  let event_force_d = event_delta.dominance *. sigmoid_factor

  // Combine forces (attractor + event)
  // Scale by (1 - inertia) to respect resistance
  PadVector(
    pleasure: { attractor_force_p +. event_force_p } *. { 1.0 -. inertia_p },
    arousal: { attractor_force_a +. event_force_a } *. { 1.0 -. inertia_a },
    dominance: { attractor_force_d +. event_force_d } *. { 1.0 -. inertia_d },
  )
}

// =============================================================================
// EPIGENETIC DYNAMICS
// =============================================================================

/// Apply crisis methylation (trauma accumulation)
pub fn apply_crisis_methylation(
  genome: Genome,
  crisis_intensity: Float,
  tick: Int,
) -> Genome {
  let new_trauma = genome.epigenetics.trauma_methylation +. { crisis_intensity *. 0.2 }
  let new_methyl = genome.epigenetics.methylation +. { crisis_intensity *. 0.1 }

  let new_history = [
    #(tick, crisis_intensity, "crisis"),
    ..genome.epigenetics.methylation_history
  ]

  Genome(
    ..genome,
    epigenetics: EpigeneticState(
      ..genome.epigenetics,
      trauma_methylation: clamp(new_trauma, 0.0, 1.0),
      methylation: clamp(new_methyl, 0.0, 1.0),
      methylation_history: list.take(new_history, 100),
    ),
  )
}

/// Apply celebration demethylation (healing)
pub fn apply_celebration_healing(
  genome: Genome,
  celebration_intensity: Float,
  tick: Int,
) -> Genome {
  // Demethylation formula: methyl = methyl * (1 - intensity * 0.4)
  let healing_factor = 1.0 -. { celebration_intensity *. 0.4 }
  let new_trauma = genome.epigenetics.trauma_methylation *. healing_factor
  let new_methyl = genome.epigenetics.methylation *. 0.95

  let new_history = [
    #(tick, celebration_intensity, "celebration"),
    ..genome.epigenetics.methylation_history
  ]

  Genome(
    ..genome,
    epigenetics: EpigeneticState(
      ..genome.epigenetics,
      trauma_methylation: clamp(new_trauma, 0.0, 1.0),
      methylation: clamp(new_methyl, 0.0, 1.0),
      healing_factor: clamp(genome.epigenetics.healing_factor +. { celebration_intensity *. 0.1 }, 0.0, 1.0),
      methylation_history: list.take(new_history, 100),
    ),
  )
}

/// Neurotic therapy protocol - reset methylation
pub fn reset_methylation(
  genome: Genome,
  therapy_intensity: Float,
) -> Genome {
  let new_methyl = genome.epigenetics.methylation *. { 1.0 -. therapy_intensity *. 0.4 }
  let new_trauma = genome.epigenetics.trauma_methylation *. { 1.0 -. therapy_intensity *. 0.5 }

  Genome(
    ..genome,
    epigenetics: EpigeneticState(
      ..genome.epigenetics,
      methylation: clamp(new_methyl, 0.0, 1.0),
      trauma_methylation: clamp(new_trauma, 0.0, 1.0),
    ),
  )
}

// =============================================================================
// GENETIC CROSSOVER (Breeding)
// =============================================================================

/// Crossover two genomes to create offspring
pub fn crossover(parent_a: Genome, parent_b: Genome) -> Genome {
  // Chromosome A: blend attractors
  let offspring_attractor = PadVector(
    pleasure: { parent_a.chromosome_a.primary_attractor.pleasure +.
                parent_b.chromosome_a.primary_attractor.pleasure } /. 2.0,
    arousal: { parent_a.chromosome_a.primary_attractor.arousal +.
               parent_b.chromosome_a.primary_attractor.arousal } /. 2.0,
    dominance: { parent_a.chromosome_a.primary_attractor.dominance +.
                 parent_b.chromosome_a.primary_attractor.dominance } /. 2.0,
  )

  // Chromosome B: average inertias
  let offspring_inertia_p = { parent_a.chromosome_b.pleasure_inertia +.
                              parent_b.chromosome_b.pleasure_inertia } /. 2.0
  let offspring_inertia_a = { parent_a.chromosome_b.arousal_inertia +.
                              parent_b.chromosome_b.arousal_inertia } /. 2.0
  let offspring_inertia_d = { parent_a.chromosome_b.dominance_inertia +.
                              parent_b.chromosome_b.dominance_inertia } /. 2.0

  // Modulators: average with slight variation
  let offspring_sensitivity = { parent_a.modulators.crisis_sensitivity +.
                                parent_b.modulators.crisis_sensitivity } /. 2.0
  let offspring_contagion = { parent_a.modulators.social_contagion +.
                              parent_b.modulators.social_contagion } /. 2.0
  let offspring_recovery = { parent_a.modulators.recovery_gene +.
                             parent_b.modulators.recovery_gene } /. 2.0

  // Epigenetics: inherit some trauma (70% from parents + 30% random)
  let inherited_trauma = { parent_a.epigenetics.trauma_methylation +.
                           parent_b.epigenetics.trauma_methylation } /. 2.0 *. 0.7

  Genome(
    chromosome_a: ChromosomeA(
      primary_attractor: offspring_attractor,
      secondary_attractors: list.append(
        list.take(parent_a.chromosome_a.secondary_attractors, 1),
        list.take(parent_b.chromosome_a.secondary_attractors, 1),
      ),
      pleasure_allele: parent_a.chromosome_a.pleasure_allele,
      arousal_allele: parent_b.chromosome_a.arousal_allele,
      dominance_allele: parent_a.chromosome_a.dominance_allele,
    ),
    chromosome_b: ChromosomeB(
      pleasure_inertia: offspring_inertia_p,
      arousal_inertia: offspring_inertia_a,
      dominance_inertia: offspring_inertia_d,
      state_inertia: [],
    ),
    modulators: ModulatorGenes(
      crisis_sensitivity: offspring_sensitivity,
      social_contagion: offspring_contagion,
      recovery_gene: offspring_recovery,
      rupture_threshold: { parent_a.modulators.rupture_threshold +.
                           parent_b.modulators.rupture_threshold } /. 2.0,
    ),
    epigenetics: EpigeneticState(
      methylation: 0.0,
      trauma_methylation: inherited_trauma,
      healing_factor: 0.5,
      methylation_history: [],
    ),
  )
}

// =============================================================================
// MUTATION DETECTION
// =============================================================================

/// Types of emotional mutations
pub type MutationType {
  /// Trauma mutation - pathological methylation
  TraumaMutation
  /// Resilience mutation - beneficial demethylation
  ResilienceMutation
  /// No significant mutation
  NoMutation
}

/// Detect if genome has mutated from baseline
pub fn detect_mutation(genome: Genome, baseline: Genome) -> MutationType {
  // Trauma mutation: crisis_sensitivity increased by 50%+
  case genome.modulators.crisis_sensitivity >. baseline.modulators.crisis_sensitivity *. 1.5 {
    True -> TraumaMutation
    False -> {
      // Resilience mutation: recovery_gene doubled
      case genome.modulators.recovery_gene >. baseline.modulators.recovery_gene *. 2.0 {
        True -> ResilienceMutation
        False -> NoMutation
      }
    }
  }
}

// =============================================================================
// EMOTIONAL FLUENCY (Calculated from Genome)
// =============================================================================

/// Calculate emotional fluency from genome
/// Fluency = (1 - avg_inertia) * recovery_gene * (1 - trauma_methylation)
pub fn emotional_fluency(genome: Genome) -> Float {
  let avg_inertia = { genome.chromosome_b.pleasure_inertia +.
                      genome.chromosome_b.arousal_inertia +.
                      genome.chromosome_b.dominance_inertia } /. 3.0

  let base_fluency = { 1.0 -. avg_inertia } *. genome.modulators.recovery_gene

  // Trauma reduces fluency
  base_fluency *. { 1.0 -. genome.epigenetics.trauma_methylation }
}

// =============================================================================
// SERIALIZATION
// =============================================================================

/// Get personality type from genome (approximate classification)
pub fn to_personality_type(genome: Genome) -> PersonalityType {
  let p = genome.chromosome_a.primary_attractor.pleasure
  let a = genome.chromosome_a.primary_attractor.arousal
  let sensitivity = genome.modulators.crisis_sensitivity
  let inertia = genome.chromosome_b.arousal_inertia

  case p, a, sensitivity, inertia {
    p, a, _, inertia if p >. 0.3 && a <. 0.0 && inertia >. 0.8 -> Calm
    p, a, sens, _ if p <. 0.0 && a >. 0.5 && sens >. 0.8 -> Neurotic
    p, _, sens, _ if p >. 0.5 && sens <. 0.2 -> Optimist
    _, a, _, inertia if a >. 0.7 && inertia <. 0.4 -> Energetic
    _, _, _, _ -> Balanced
  }
}

/// Convert personality type to string
pub fn personality_to_string(personality: PersonalityType) -> String {
  case personality {
    Calm -> "calm"
    Neurotic -> "neurotic"
    Optimist -> "optimist"
    Energetic -> "energetic"
    Balanced -> "balanced"
  }
}

// =============================================================================
// HELPERS
// =============================================================================

fn clamp(value: Float, min: Float, max: Float) -> Float {
  case value {
    v if v <. min -> min
    v if v >. max -> max
    v -> v
  }
}

fn sigmoid(x: Float) -> Float {
  1.0 /. { 1.0 +. exp(0.0 -. x) }
}

@external(erlang, "math", "exp")
fn exp(x: Float) -> Float

// =============================================================================
// SIMULATION V3 - POPULATION GENETICS (Qwen3 Recommendations)
// =============================================================================

/// Drift types for population monitoring
pub type DriftType {
  /// Population accumulating trauma (methylation > baseline * 1.8)
  TraumaDrift
  /// Population developing "toxic resilience" (methylation < baseline * 0.3)
  ResilienceDrift
  /// Population within normal range
  NoDrift
}

/// Population statistics for monitoring
pub type PopulationStats {
  PopulationStats(
    avg_methylation: Float,
    avg_trauma: Float,
    avg_fluency: Float,
    trauma_mutation_pct: Float,
    resilience_mutation_pct: Float,
    drift_type: DriftType,
  )
}

/// Detect epigenetic drift in a population
/// Critical for simulations > 7,500 ticks
pub fn detect_epigenetic_drift(
  genomes: List(Genome),
  baseline: Genome,
) -> DriftType {
  let avg_methylation = average_methylation(genomes)
  let baseline_methyl = baseline.epigenetics.methylation

  // Calculate deviation from baseline
  let deviation = float_abs(avg_methylation -. baseline_methyl)

  // High trauma accumulation (avg significantly higher than baseline)
  case avg_methylation >. baseline_methyl +. 0.15 {
    True -> TraumaDrift  // Population accumulating trauma → collapse after ~7,500 ticks
    False -> {
      // Toxic resilience (avg significantly lower than baseline, but only if baseline is high enough)
      case baseline_methyl >. 0.2 && avg_methylation <. baseline_methyl -. 0.15 {
        True -> ResilienceDrift  // "Toxic resilience" - can't process real crises
        False -> {
          // Within normal range
          case deviation <. 0.1 {
            True -> NoDrift
            False -> NoDrift  // Minor drift, still considered healthy
          }
        }
      }
    }
  }
}

fn float_abs(x: Float) -> Float {
  case x <. 0.0 {
    True -> 0.0 -. x
    False -> x
  }
}

/// Calculate population statistics
pub fn population_stats(genomes: List(Genome), baseline: Genome) -> PopulationStats {
  let count = list.length(genomes) |> int.to_float
  let count_safe = case count <. 1.0 {
    True -> 1.0
    False -> count
  }

  let avg_methyl = average_methylation(genomes)
  let avg_trauma = average_trauma(genomes)
  let avg_flu = average_fluency(genomes)

  let trauma_mutations = list.filter(genomes, fn(g) {
    detect_mutation(g, baseline) == TraumaMutation
  }) |> list.length |> int.to_float

  let resilience_mutations = list.filter(genomes, fn(g) {
    detect_mutation(g, baseline) == ResilienceMutation
  }) |> list.length |> int.to_float

  PopulationStats(
    avg_methylation: avg_methyl,
    avg_trauma: avg_trauma,
    avg_fluency: avg_flu,
    trauma_mutation_pct: trauma_mutations /. count_safe *. 100.0,
    resilience_mutation_pct: resilience_mutations /. count_safe *. 100.0,
    drift_type: detect_epigenetic_drift(genomes, baseline),
  )
}

/// Boost recovery gene (adaptive mutation under extreme stress)
pub fn boost_recovery(genome: Genome, amount: Float) -> Genome {
  let new_recovery = clamp(
    genome.modulators.recovery_gene +. amount,
    0.0,
    1.0,
  )
  Genome(
    ..genome,
    modulators: ModulatorGenes(
      ..genome.modulators,
      recovery_gene: new_recovery,
    ),
  )
}

/// Trigger adaptive mutation under extreme crisis
/// 12% of populations survive crises >0.95 thanks to spontaneous mutations
pub fn trigger_adaptive_mutation(
  genome: Genome,
  crisis_intensity: Float,
) -> Genome {
  // Only trigger if crisis is extreme AND recovery is very low
  case crisis_intensity >. 0.9 && genome.modulators.recovery_gene <. 0.1 {
    True -> {
      // Activate "resilience mutation" as last resort
      genome
      |> boost_recovery(0.3)
      |> apply_celebration_healing(0.5, 0)  // Partial trauma reset
    }
    False -> genome
  }
}

/// Apply social contagion from neighbor emotions
pub fn apply_social_contagion(
  genome: Genome,
  neighbor_avg_methylation: Float,
) -> Genome {
  let contagion_rate = genome.modulators.social_contagion
  let current_methyl = genome.epigenetics.methylation
  let delta = { neighbor_avg_methylation -. current_methyl } *. contagion_rate *. 0.1

  Genome(
    ..genome,
    epigenetics: EpigeneticState(
      ..genome.epigenetics,
      methylation: clamp(current_methyl +. delta, 0.0, 1.0),
    ),
  )
}

/// Simulate one tick of genome evolution
pub fn tick_genome(
  genome: Genome,
  event_intensity: Float,
  event_type: String,
  tick: Int,
) -> Genome {
  case event_type {
    "crisis" -> {
      genome
      |> apply_crisis_methylation(event_intensity, tick)
      |> trigger_adaptive_mutation(event_intensity)
    }
    "celebration" -> {
      apply_celebration_healing(genome, event_intensity, tick)
    }
    "challenge" -> {
      // Moderate stress, slight methylation
      apply_crisis_methylation(genome, event_intensity *. 0.3, tick)
    }
    "relaxation" -> {
      // Slight healing
      apply_celebration_healing(genome, event_intensity *. 0.5, tick)
    }
    _ -> genome  // No effect for other events
  }
}

/// Generation tracking for evolution visualization
pub type GenerationStats {
  GenerationStats(
    generation: Int,
    trauma_mutation_pct: Float,
    resilience_mutation_pct: Float,
    avg_fluency: Float,
    avg_methylation: Float,
  )
}

/// Calculate generation statistics
pub fn generation_stats(
  genomes: List(Genome),
  baseline: Genome,
  generation: Int,
) -> GenerationStats {
  let stats = population_stats(genomes, baseline)
  GenerationStats(
    generation: generation,
    trauma_mutation_pct: stats.trauma_mutation_pct,
    resilience_mutation_pct: stats.resilience_mutation_pct,
    avg_fluency: stats.avg_fluency,
    avg_methylation: stats.avg_methylation,
  )
}

// =============================================================================
// POPULATION HELPERS
// =============================================================================

fn average_methylation(genomes: List(Genome)) -> Float {
  let total = list.fold(genomes, 0.0, fn(acc, g) {
    acc +. g.epigenetics.methylation
  })
  let count = list.length(genomes) |> int.to_float
  case count <. 1.0 {
    True -> 0.0
    False -> total /. count
  }
}

fn average_trauma(genomes: List(Genome)) -> Float {
  let total = list.fold(genomes, 0.0, fn(acc, g) {
    acc +. g.epigenetics.trauma_methylation
  })
  let count = list.length(genomes) |> int.to_float
  case count <. 1.0 {
    True -> 0.0
    False -> total /. count
  }
}

fn average_fluency(genomes: List(Genome)) -> Float {
  let total = list.fold(genomes, 0.0, fn(acc, g) {
    acc +. emotional_fluency(g)
  })
  let count = list.length(genomes) |> int.to_float
  case count <. 1.0 {
    True -> 0.0
    False -> total /. count
  }
}

// =============================================================================
// SURVIVAL PROTOCOLS (Qwen3 Emergency Recommendations)
// =============================================================================
// These prevent 100% population collapse between 7,200-7,800 ticks

/// Vaccination state tracking
pub type VaccinationState {
  VaccinationState(
    vaccinated: Bool,
    vaccination_tick: Int,
    doses: Int,
    immunity_level: Float,
  )
}

/// New vaccination state
pub fn new_vaccination_state() -> VaccinationState {
  VaccinationState(
    vaccinated: False,
    vaccination_tick: 0,
    doses: 0,
    immunity_level: 0.0,
  )
}

/// Forced Adaptive Mutation Protocol
/// Triggers when VIVA has been stagnant for too long (500+ ticks without improvement)
/// Forces a beneficial mutation to break the trauma loop
pub fn forced_adaptive_mutation(
  genome: Genome,
  ticks_without_improvement: Int,
  stagnation_threshold: Int,
) -> Genome {
  // Only trigger if stagnant for longer than threshold
  case ticks_without_improvement >= stagnation_threshold {
    True -> {
      // Force recovery gene boost based on stagnation severity
      let severe_threshold = stagnation_threshold * 2
      let moderate_threshold = { stagnation_threshold * 3 } / 2  // 1.5x

      let boost_amount = case ticks_without_improvement {
        t if t >= severe_threshold -> 0.4  // Severe stagnation
        t if t >= moderate_threshold -> 0.3
        _ -> 0.2  // Mild stagnation
      }

      // Reduce trauma methylation proportionally
      let trauma_reduction = boost_amount *. 0.6

      Genome(
        ..genome,
        modulators: ModulatorGenes(
          ..genome.modulators,
          recovery_gene: clamp(genome.modulators.recovery_gene +. boost_amount, 0.0, 1.0),
          // Also reduce crisis sensitivity to prevent re-trauma
          crisis_sensitivity: clamp(
            genome.modulators.crisis_sensitivity -. boost_amount *. 0.3,
            0.05,
            1.0,
          ),
        ),
        epigenetics: EpigeneticState(
          ..genome.epigenetics,
          trauma_methylation: clamp(
            genome.epigenetics.trauma_methylation -. trauma_reduction,
            0.0,
            1.0,
          ),
          methylation: clamp(
            genome.epigenetics.methylation -. trauma_reduction *. 0.5,
            0.0,
            1.0,
          ),
        ),
      )
    }
    False -> genome
  }
}

/// Neurotic Emergency Protocol
/// Intensive intervention for neurotics trapped in Stressed quadrant
/// Combines isolation (reduced social contagion) + intensive therapy
pub fn neurotic_emergency_protocol(
  genome: Genome,
  intensity: Float,
) -> Genome {
  // Only apply to neurotics (high sensitivity + low recovery)
  let is_neurotic = genome.modulators.crisis_sensitivity >. 0.8
                 && genome.modulators.recovery_gene <. 0.2

  case is_neurotic {
    True -> {
      // Isolation: temporarily reduce social contagion to prevent trauma spread
      let new_contagion = genome.modulators.social_contagion *. { 1.0 -. intensity *. 0.7 }

      // Intensive therapy: aggressive trauma reduction
      let therapy_factor = intensity *. 0.8
      let new_trauma = genome.epigenetics.trauma_methylation *. { 1.0 -. therapy_factor }
      let new_methyl = genome.epigenetics.methylation *. { 1.0 -. therapy_factor *. 0.6 }

      // Boost recovery gene (learned resilience)
      let recovery_boost = intensity *. 0.25

      // Lower crisis sensitivity through desensitization
      let sensitivity_reduction = intensity *. 0.2

      Genome(
        ..genome,
        modulators: ModulatorGenes(
          ..genome.modulators,
          social_contagion: clamp(new_contagion, 0.1, 1.0),
          recovery_gene: clamp(genome.modulators.recovery_gene +. recovery_boost, 0.0, 0.6),
          crisis_sensitivity: clamp(
            genome.modulators.crisis_sensitivity -. sensitivity_reduction,
            0.3,  // Never fully desensitize
            1.0,
          ),
        ),
        epigenetics: EpigeneticState(
          ..genome.epigenetics,
          trauma_methylation: clamp(new_trauma, 0.0, 1.0),
          methylation: clamp(new_methyl, 0.0, 1.0),
          healing_factor: clamp(genome.epigenetics.healing_factor +. intensity *. 0.15, 0.0, 1.0),
        ),
      )
    }
    False -> genome  // Not neurotic, no intervention needed
  }
}

/// Emotional Vaccination Protocol
/// Controlled micro-trauma followed by therapy to build immunity
/// Similar to biological vaccination - small dose of stressor + recovery support
pub fn emotional_vaccination(
  genome: Genome,
  vaccination_state: VaccinationState,
  current_tick: Int,
) -> #(Genome, VaccinationState) {
  // Vaccination schedule: 3 doses, 200 ticks apart
  let dose_interval = 200
  let max_doses = 3

  case vaccination_state.doses >= max_doses {
    True -> {
      // Fully vaccinated - just return current state
      #(genome, vaccination_state)
    }
    False -> {
      // Check if it's time for next dose
      let ready_for_dose = case vaccination_state.vaccinated {
        False -> True  // First dose
        True -> current_tick - vaccination_state.vaccination_tick >= dose_interval
      }

      case ready_for_dose {
        True -> {
          // Apply vaccination: micro-trauma + immediate therapy
          let dose_number = vaccination_state.doses + 1

          // Micro-trauma: controlled small crisis (decreases with each dose)
          let micro_trauma_intensity = 0.3 /. int.to_float(dose_number)
          let genome_after_trauma = apply_crisis_methylation(
            genome,
            micro_trauma_intensity,
            current_tick,
          )

          // Immediate therapy (stronger than trauma to ensure net positive)
          let therapy_intensity = 0.5 +. { int.to_float(dose_number) *. 0.1 }
          let genome_after_therapy = genome_after_trauma
            |> apply_celebration_healing(therapy_intensity, current_tick)

          // Build immunity: boost recovery + increase rupture threshold
          let immunity_boost = 0.15 *. int.to_float(dose_number)
          let vaccinated_genome = Genome(
            ..genome_after_therapy,
            modulators: ModulatorGenes(
              ..genome_after_therapy.modulators,
              recovery_gene: clamp(
                genome_after_therapy.modulators.recovery_gene +. immunity_boost *. 0.5,
                0.0,
                1.0,
              ),
              rupture_threshold: clamp(
                genome_after_therapy.modulators.rupture_threshold +. immunity_boost *. 0.3,
                0.0,
                1.0,
              ),
              // Slight desensitization (controlled exposure)
              crisis_sensitivity: clamp(
                genome_after_therapy.modulators.crisis_sensitivity -. immunity_boost *. 0.1,
                0.1,  // Never below 0.1 (need some sensitivity)
                1.0,
              ),
            ),
          )

          let new_immunity = clamp(
            vaccination_state.immunity_level +. 0.25,
            0.0,
            0.75,  // Max 75% immunity (never fully immune)
          )

          let new_vaccination_state = VaccinationState(
            vaccinated: True,
            vaccination_tick: current_tick,
            doses: dose_number,
            immunity_level: new_immunity,
          )

          #(vaccinated_genome, new_vaccination_state)
        }
        False -> {
          // Not time for next dose yet
          #(genome, vaccination_state)
        }
      }
    }
  }
}

/// Apply immunity effect to crisis (reduces effective crisis intensity)
pub fn apply_immunity(
  vaccination_state: VaccinationState,
  crisis_intensity: Float,
) -> Float {
  crisis_intensity *. { 1.0 -. vaccination_state.immunity_level }
}

/// Check if VIVA needs emergency intervention
pub type EmergencyStatus {
  Critical     // Needs immediate intervention
  Warning      // Should monitor closely
  Stable       // Healthy
}

pub fn check_emergency_status(genome: Genome) -> EmergencyStatus {
  let trauma_level = genome.epigenetics.trauma_methylation
  let recovery = genome.modulators.recovery_gene
  let sensitivity = genome.modulators.crisis_sensitivity

  // Critical: high trauma + low recovery + high sensitivity
  let critical_score = trauma_level *. 2.0
                    +. { 1.0 -. recovery } *. 1.5
                    +. sensitivity *. 0.5

  case critical_score >. 3.5 {
    True -> Critical
    False -> {
      case critical_score >. 2.5 {
        True -> Warning
        False -> Stable
      }
    }
  }
}

/// Population-level survival intervention
/// Returns list of genomes with emergency protocols applied
pub fn apply_population_survival(
  genomes: List(Genome),
  baseline: Genome,
  current_tick: Int,
  vaccination_states: List(VaccinationState),
) -> #(List(Genome), List(VaccinationState)) {
  let stats = population_stats(genomes, baseline)

  // Check if population is in danger
  let in_danger = case stats.drift_type {
    TraumaDrift -> True
    _ -> stats.trauma_mutation_pct >. 10.0
  }

  case in_danger {
    True -> {
      // Apply interventions
      let combined = list.zip(genomes, vaccination_states)
      let result = list.map(combined, fn(pair) {
        let #(genome, vax_state) = pair

        // Check emergency status
        let status = check_emergency_status(genome)

        case status {
          Critical -> {
            // Neurotic emergency protocol + vaccination
            let treated = neurotic_emergency_protocol(genome, 0.8)
            emotional_vaccination(treated, vax_state, current_tick)
          }
          Warning -> {
            // Just vaccination
            emotional_vaccination(genome, vax_state, current_tick)
          }
          Stable -> {
            // Preventive vaccination only
            case vax_state.doses < 3 {
              True -> emotional_vaccination(genome, vax_state, current_tick)
              False -> #(genome, vax_state)
            }
          }
        }
      })

      let new_genomes = list.map(result, fn(r) { r.0 })
      let new_vax_states = list.map(result, fn(r) { r.1 })
      #(new_genomes, new_vax_states)
    }
    False -> {
      // No danger, just return as-is (but still track vaccination states)
      #(genomes, vaccination_states)
    }
  }
}

/// Average recovery gene in population
pub fn average_recovery(genomes: List(Genome)) -> Float {
  let total = list.fold(genomes, 0.0, fn(acc, g) {
    acc +. g.modulators.recovery_gene
  })
  let count = list.length(genomes) |> int.to_float
  case count <. 1.0 {
    True -> 0.0
    False -> total /. count
  }
}
