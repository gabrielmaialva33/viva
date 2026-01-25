# scripts/test_llm.exs
# Test script for Viva.Llm

IO.puts("\n=== VIVA GLANDS DIGESTION TEST ===\n")

# 1. Check Architecture
IO.puts("Checking Native Architecture...")

try do
  arch = Viva.Llm.native_check()
  IO.puts("  [OK] Architecture: #{arch}")
rescue
  e -> IO.puts("  [FAIL] Native check failed: #{inspect(e)}")
end

# 2. Check Model
model_path = "models/Llama-3-8B-Instruct.Q4_K_M.gguf"

if File.exists?(model_path) do
  IO.puts("\nLoading Model: #{model_path}")
  # 0 GPU layers for safe testing in CLI
  try do
    resource = Viva.Llm.load_model(model_path, 0)
    IO.puts("  [OK] Model Loaded.")

    prompt = "Define 'Consciousness' in one sentence."
    IO.puts("\nDigesting Concept: '#{prompt}'")

    {text, embedding} = Viva.Llm.predict(resource, prompt)
    IO.puts("  [OK] Digestion Successful.")
    IO.puts("  Output Text: \"#{text}\"")
    IO.puts("  Embedding Vector Size: #{length(embedding)}")
  rescue
    e -> IO.puts("  [FAIL] Error: #{inspect(e)}")
  end
else
  IO.puts("\n[WARN] Model file not found at #{model_path}")
  IO.puts("Skipping inference test. Please run ./scripts/download_model.sh")
end
