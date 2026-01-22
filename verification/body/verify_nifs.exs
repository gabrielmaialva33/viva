# ============================================================================
# NIF Verification Script
# ============================================================================
# Tests the Rust NIFs exposed by VivaBridge.Body
#
# Run: mix run verification/body/verify_nifs.exs
# ============================================================================

alias VivaBridge.Body
alias VivaBridge.Memory, as: NativeMemory
alias VivaBridge.Cortex

IO.puts("\n" <> String.duplicate("=", 60))
IO.puts("NIF VERIFICATION")
IO.puts(String.duplicate("=", 60))

# ----------------------------------------------------------------------------
# 1. Body NIFs
# ----------------------------------------------------------------------------
IO.puts("\n[1] Body NIFs")
IO.puts(String.duplicate("-", 40))

# Check alive
alive = Body.alive()
IO.puts("  alive/0: #{inspect(alive)}")

# Hardware feel
try do
  hw = Body.feel_hardware()
  IO.puts("  feel_hardware/0: #{String.slice(inspect(hw), 0, 60)}...")
rescue
  e -> IO.puts("  feel_hardware/0: ERROR - #{Exception.message(e)}")
end

# Qualia
try do
  qualia = Body.hardware_to_qualia()
  IO.puts("  hardware_to_qualia/0: #{inspect(qualia)}")
rescue
  e -> IO.puts("  hardware_to_qualia/0: ERROR - #{Exception.message(e)}")
end

# ----------------------------------------------------------------------------
# 2. Memory NIFs
# ----------------------------------------------------------------------------
IO.puts("\n[2] Memory NIFs")
IO.puts(String.duplicate("-", 40))

# Init
try do
  init_result = Body.memory_init("hnsw")
  IO.puts("  memory_init/1: #{inspect(init_result)}")
rescue
  e -> IO.puts("  memory_init/1: ERROR - #{Exception.message(e)}")
end

# Store
try do
  # Create a simple 1024D vector
  test_vector = List.duplicate(0.1, 1024)
  meta_json = Jason.encode!(%{
    id: "test_nif_#{:rand.uniform(10000)}",
    content: "NIF verification test",
    memory_type: "Episodic",
    importance: 0.5,
    timestamp: System.system_time(:second),
    access_count: 0,
    last_accessed: System.system_time(:second)
  })

  {time, result} = :timer.tc(fn -> Body.memory_store(test_vector, meta_json) end)
  IO.puts("  memory_store/2: #{inspect(result)} (#{time / 1000}ms)")
rescue
  e -> IO.puts("  memory_store/2: ERROR - #{Exception.message(e)}")
end

# Search
try do
  query_vector = List.duplicate(0.1, 1024)
  {time, results} = :timer.tc(fn -> Body.memory_search(query_vector, 5) end)
  IO.puts("  memory_search/2: #{length(results)} results (#{time / 1000}ms)")
rescue
  e -> IO.puts("  memory_search/2: ERROR - #{Exception.message(e)}")
end

# Stats
try do
  stats = Body.memory_stats("hnsw")
  IO.puts("  memory_stats/1: #{inspect(stats)}")
rescue
  e -> IO.puts("  memory_stats/1: ERROR - #{Exception.message(e)}")
end

# ----------------------------------------------------------------------------
# 3. Cortex NIFs
# ----------------------------------------------------------------------------
IO.puts("\n[3] Cortex NIFs (via Brain NIFs)")
IO.puts(String.duplicate("-", 40))

try do
  init_result = Body.brain_init()
  IO.puts("  brain_init/0: #{inspect(init_result)}")
rescue
  e -> IO.puts("  brain_init/0: ERROR - #{Exception.message(e)}")
end

try do
  exp_result = Body.brain_experience("Test input", 0.5, 0.3, 0.2)
  IO.puts("  brain_experience/4: vector of #{length(exp_result)} dims")
rescue
  e -> IO.puts("  brain_experience/4: ERROR - #{Exception.message(e)}")
end

# ----------------------------------------------------------------------------
# 4. Emotional NIFs
# ----------------------------------------------------------------------------
IO.puts("\n[4] Emotional NIFs")
IO.puts(String.duplicate("-", 40))

try do
  stimulus = Body.apply_stimulus(0.3, 0.2, 0.1)
  IO.puts("  apply_stimulus/3: #{inspect(stimulus)}")
rescue
  e -> IO.puts("  apply_stimulus/3: ERROR - #{Exception.message(e)}")
end

# ----------------------------------------------------------------------------
# 5. Metabolism NIFs
# ----------------------------------------------------------------------------
IO.puts("\n[5] Metabolism NIFs")
IO.puts(String.duplicate("-", 40))

try do
  init_result = Body.metabolism_init()
  IO.puts("  metabolism_init/0: #{inspect(init_result)}")
rescue
  e -> IO.puts("  metabolism_init/0: ERROR - #{Exception.message(e)}")
end

try do
  tick_result = Body.metabolism_tick()
  IO.puts("  metabolism_tick/0: #{String.slice(inspect(tick_result), 0, 60)}...")
rescue
  e -> IO.puts("  metabolism_tick/0: ERROR - #{Exception.message(e)}")
end

IO.puts("\n" <> String.duplicate("=", 60))
IO.puts("VERIFICATION COMPLETE")
IO.puts(String.duplicate("=", 60) <> "\n")
