// Test LLM Distillation - Pure Gleam test
// Run with: gleam run -m test_distill_ollama

import gleam/io
import gleam/float
import gleam/int
import gleam/list
import viva/neural/distillation.{
  init_progressive, step_progressive, advanced_config,
  compute_cka, compute_cka_loss,
}
import viva/memory/hrr

pub fn main() {
  io.println("=== VIVA Distillation Test ===")
  io.println("")

  // 1. Simulate teacher embedding (from Ollama nomic-embed-text)
  // Real embedding would be 768 dimensions, we use 256 for speed
  io.println("1. Creating simulated teacher embedding...")
  let dim = 768
  let teacher_hrr = hrr.random(dim)
  io.println("   Teacher HRR dimension: " <> int.to_string(hrr.dim(teacher_hrr)))
  io.println("")

  // 2. Test CKA
  io.println("2. CKA Similarity Tests...")
  let cka_self = compute_cka(teacher_hrr, teacher_hrr)
  io.println("   CKA(teacher, teacher): " <> float.to_string(cka_self))

  let random_hrr = hrr.random(dim)
  let cka_random = compute_cka(teacher_hrr, random_hrr)
  io.println("   CKA(teacher, random): " <> float.to_string(cka_random))

  let cka_loss = compute_cka_loss(teacher_hrr, random_hrr)
  io.println("   CKA Loss: " <> float.to_string(cka_loss))
  io.println("")

  // 3. Progressive Distillation
  io.println("3. Progressive Distillation Schedule...")
  let config = advanced_config()
  let prog0 = init_progressive(10)
  io.println("   Epoch 0: alpha_kd=" <> float.to_string(prog0.alpha_kd)
             <> " temp=" <> float.to_string(prog0.temperature))

  let prog5 = list.fold(list.range(1, 5), prog0, fn(p, _) {
    step_progressive(p, config)
  })
  io.println("   Epoch 5: alpha_kd=" <> float.to_string(prog5.alpha_kd)
             <> " temp=" <> float.to_string(prog5.temperature))

  let prog10 = list.fold(list.range(1, 5), prog5, fn(p, _) {
    step_progressive(p, config)
  })
  io.println("   Epoch 10: alpha_kd=" <> float.to_string(prog10.alpha_kd)
             <> " temp=" <> float.to_string(prog10.temperature))
  io.println("")

  // 4. Multi-prompt Distillation Test
  io.println("4. Student-Teacher Comparison...")
  let student_hrr = hrr.random(dim)
  let initial_loss = compute_cka_loss(student_hrr, teacher_hrr)
  io.println("   Student (random) CKA loss: " <> float.to_string(initial_loss))

  // Compare similarity
  let similarity = hrr.similarity(student_hrr, teacher_hrr)
  io.println("   Student-Teacher similarity: " <> float.to_string(similarity))

  // Test with normalized vectors
  let teacher_norm = hrr.normalize(teacher_hrr)
  let student_norm = hrr.normalize(student_hrr)
  let sim_norm = hrr.similarity(teacher_norm, student_norm)
  io.println("   Normalized similarity: " <> float.to_string(sim_norm))
  io.println("")

  // 5. HRR Binding Test (memory association)
  io.println("5. HRR Memory Association...")
  let concept_a = hrr.random(dim)
  let concept_b = hrr.random(dim)

  case hrr.bind(concept_a, concept_b) {
    Ok(bound) -> {
      io.println("   Bound A * B successfully")

      case hrr.unbind(bound, concept_a) {
        Ok(recovered_b) -> {
          let recovery_sim = hrr.similarity(recovered_b, concept_b)
          io.println("   Recovery similarity: " <> float.to_string(recovery_sim))
        }
        Error(_) -> io.println("   Unbind failed")
      }
    }
    Error(_) -> io.println("   Bind failed")
  }
  io.println("")

  io.println("=== Test Complete ===")
}
