import gleeunit/should
import viva/memory/hrr
import viva/neural/distillation.{
  CompressionStats, DistillConfig, DistillLoss, EWCState, PADState,
}
import viva_tensor/nf4

// =============================================================================
// COMPRESSION TESTS
// =============================================================================

pub fn compress_hrr_achieves_8x_test() {
  // Create a random HRR vector
  let h = hrr.random(8192)

  // Compress with NF4
  let compressed = distillation.compress_hrr(h, nf4.default_config())

  // Check compression ratio
  let stats = distillation.compression_stats(h, compressed)

  // NF4 should achieve ~7-8x compression
  should.be_true(stats.ratio >. 6.0)
  should.be_true(stats.ratio <. 9.0)

  // Error should be small
  should.be_true(stats.reconstruction_error <. 0.1)
}

pub fn compress_decompress_roundtrip_test() {
  let h = hrr.random(4096)
  let config = nf4.default_config()

  // Compress
  let compressed = distillation.compress_hrr(h, config)

  // Decompress
  let decompressed = distillation.decompress_hrr(compressed)

  // Should have same dimension
  should.equal(hrr.dim(decompressed), hrr.dim(h))

  // Should have high similarity (>0.9)
  let sim = hrr.similarity(h, decompressed)
  should.be_true(sim >. 0.9)
}

// =============================================================================
// LOSS FUNCTION TESTS
// =============================================================================

pub fn dhc_loss_components_test() {
  let student = hrr.random(512)
  let teacher = hrr.random(512)

  let ewc =
    EWCState(
      fisher_diag: [0.1, 0.2, 0.3],
      consolidated_params: [1.0, 2.0, 3.0],
    )
  let current_params = [1.1, 2.0, 2.9]

  let config =
    DistillConfig(
      teacher_dim: 4096,
      hrr_dim: 512,
      extraction_layer: 16,
      nf4_config: nf4.default_config(),
      temperature: 2.0,
      alpha_task: 1.0,
      alpha_kd: 0.5,
      alpha_ewc: 0.1,
    )

  let loss =
    distillation.compute_loss(student, teacher, ewc, current_params, config)

  // Task loss should be positive
  should.be_true(loss.task_loss >=. 0.0)

  // KD loss should be between 0 and 2 (1 - cosine, cosine in [-1, 1])
  should.be_true(loss.kd_loss >=. 0.0)
  should.be_true(loss.kd_loss <=. 2.0)

  // EWC loss should be positive
  should.be_true(loss.ewc_loss >=. 0.0)

  // Total should be weighted sum
  let expected_total =
    config.alpha_task
    *. loss.task_loss
    +. config.alpha_kd
    *. loss.kd_loss
    +. config.alpha_ewc
    *. loss.ewc_loss

  // Allow small floating point error
  should.be_true(float_abs(loss.total -. expected_total) <. 0.0001)
}

pub fn identical_hrr_has_zero_kd_loss_test() {
  let h = hrr.random(256)

  let ewc = EWCState(fisher_diag: [], consolidated_params: [])

  let config = distillation.default_config()

  let loss = distillation.compute_loss(h, h, ewc, [], config)

  // Same vector = cosine 1.0, so KD loss = 0
  should.be_true(loss.kd_loss <. 0.0001)

  // Task loss (MSE) should also be ~0
  should.be_true(loss.task_loss <. 0.0001)
}

// =============================================================================
// EWC TESTS
// =============================================================================

pub fn init_ewc_creates_zeros_test() {
  let ewc = distillation.init_ewc(100)

  should.equal(
    ewc.fisher_diag,
    list_repeat(0.0, 100),
  )
  should.equal(
    ewc.consolidated_params,
    list_repeat(0.0, 100),
  )
}

pub fn update_fisher_computes_squared_gradients_test() {
  let ewc = distillation.init_ewc(3)

  let gradients = [[1.0, 2.0, 3.0], [2.0, 0.0, 1.0]]

  let current = [0.5, 0.5, 0.5]

  let updated = distillation.update_fisher(ewc, gradients, current)

  // Fisher = mean of squared gradients
  // [(1^2 + 2^2)/2, (2^2 + 0^2)/2, (3^2 + 1^2)/2]
  // = [2.5, 2.0, 5.0]
  should.equal(updated.fisher_diag, [2.5, 2.0, 5.0])
  should.equal(updated.consolidated_params, current)
}

// =============================================================================
// KNOWLEDGE PACKAGING TESTS
// =============================================================================

pub fn package_knowledge_computes_salience_test() {
  let key = hrr.random(512)
  let value = hrr.random(512)

  // High arousal + pleasure = high salience
  let emotional = PADState(pleasure: 0.8, arousal: 0.9, dominance: 0.5)

  let config = distillation.small_config()

  let knowledge =
    distillation.package_knowledge(key, value, emotional, config, "test-teacher")

  // Salience = (|pleasure| + arousal) / 2 = (0.8 + 0.9) / 2 = 0.85
  should.be_true(knowledge.salience >. 0.8)
  should.be_true(knowledge.salience <. 0.9)

  // Metadata should be set
  should.equal(knowledge.metadata.teacher_id, "test-teacher")
  should.equal(knowledge.metadata.extraction_layer, config.extraction_layer)
}

pub fn low_arousal_gives_low_salience_test() {
  let key = hrr.random(512)
  let value = hrr.random(512)

  // Low arousal = low salience (boring)
  let emotional = PADState(pleasure: 0.1, arousal: 0.1, dominance: 0.5)

  let config = distillation.small_config()

  let knowledge =
    distillation.package_knowledge(key, value, emotional, config, "test")

  // Salience = (|0.1| + 0.1) / 2 = 0.1
  should.be_true(knowledge.salience <. 0.2)
}

// =============================================================================
// HELPERS
// =============================================================================

fn float_abs(x: Float) -> Float {
  case x <. 0.0 {
    True -> 0.0 -. x
    False -> x
  }
}

import gleam/list

fn list_repeat(value: a, n: Int) -> List(a) {
  list.repeat(value, n)
}
