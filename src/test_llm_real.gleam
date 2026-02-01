// Real LLM Distillation Test - loads Qwen3-32B
// Run with: gleam run -m test_llm_real

import gleam/io
import gleam/int
import gleam/float
import gleam/list
import viva/neural/llm
import viva/neural/distillation
import viva/memory/hrr

pub fn main() {
  io.println("=== VIVA LLM Distillation Test ===")
  io.println("")

  // 1. Check system memory
  io.println("1. System Memory Check")
  let #(total, free) = llm.memory_status()
  io.println("   Total: " <> int.to_string(total / 1_000_000_000) <> " GB")
  io.println("   Free:  " <> int.to_string(free / 1_000_000_000) <> " GB")
  io.println("")

  // 2. Check native capabilities
  io.println("2. Native Capabilities")
  let caps = llm.native_check()
  io.println("   CPU: " <> caps)
  io.println("")

  // 3. Load Qwen3-32B teacher
  io.println("3. Loading Qwen3-32B Teacher Model...")
  io.println("   Path: models/Qwen3-32B-Q4_K_M.gguf")
  io.println("   GPU Layers: 65 (full offload)")

  let model = llm.load_model("models/Qwen3-32B-Q4_K_M.gguf", 65)
  let info = llm.model_info(model)

  io.println("   Loaded!")
  io.println("   Embedding dim: " <> int.to_string(info.n_embd))
  io.println("   Vocab size: " <> int.to_string(info.n_vocab))
  io.println("")

  // 4. Test tokenization
  io.println("4. Tokenization Test")
  let test_text = "What is consciousness?"
  let tokens = llm.tokenize(model, test_text)
  io.println("   Input: \"" <> test_text <> "\"")
  io.println("   Tokens: " <> int.to_string(list.length(tokens)))
  io.println("")

  // 5. Extract hidden states
  io.println("5. Hidden States Extraction")
  let states = llm.get_hidden_states(model, test_text, 2048)
  io.println("   Prompt: \"" <> test_text <> "\"")
  io.println("   N tokens: " <> int.to_string(states.n_tokens))
  io.println("   N embedding: " <> int.to_string(states.n_embd))
  io.println("   Total floats: " <> int.to_string(list.length(states.embeddings)))
  io.println("")

  // 6. Mean pooled embedding
  io.println("6. Mean Pooled Embedding")
  let pooled = llm.mean_pooled_embedding(states)
  io.println("   Dimension: " <> int.to_string(list.length(pooled)))

  // Show first 5 values
  let first_5 = list.take(pooled, 5)
  io.println("   First 5 values: " <> format_floats(first_5))
  io.println("")

  // 7. Test distillation CKA
  io.println("7. CKA Similarity Test")
  let hrr1 = hrr.from_list(pooled)
  let hrr2 = hrr.random(list.length(pooled))
  let cka = distillation.compute_cka(hrr1, hrr1)
  io.println("   CKA(self, self): " <> float.to_string(cka))

  let cka2 = distillation.compute_cka(hrr1, hrr2)
  io.println("   CKA(self, random): " <> float.to_string(cka2))
  io.println("")

  // 8. Progressive distillation test
  io.println("8. Progressive Distillation Schedule")
  let config = distillation.advanced_config()
  let prog = distillation.init_progressive(10)
  io.println("   Epoch 0: alpha_kd=" <> float.to_string(prog.alpha_kd) <> " temp=" <> float.to_string(prog.temperature))

  let prog5 = list.fold(list.range(1, 5), prog, fn(p, _) {
    distillation.step_progressive(p, config)
  })
  io.println("   Epoch 5: alpha_kd=" <> float.to_string(prog5.alpha_kd) <> " temp=" <> float.to_string(prog5.temperature))

  let prog10 = list.fold(list.range(1, 5), prog5, fn(p, _) {
    distillation.step_progressive(p, config)
  })
  io.println("   Epoch 10: alpha_kd=" <> float.to_string(prog10.alpha_kd) <> " temp=" <> float.to_string(prog10.temperature))
  io.println("")

  io.println("=== Test Complete ===")
}

fn format_floats(floats: List(Float)) -> String {
  floats
  |> list.map(fn(f) { float.to_string(f) })
  |> list.intersperse(", ")
  |> list.fold("", fn(acc, s) { acc <> s })
}
