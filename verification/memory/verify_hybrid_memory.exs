# ============================================================================
# Hybrid Memory Verification Script
# ============================================================================
# Tests the hybrid memory architecture:
# - Episodic: Rust HNSW (fast, ~1ms)
# - Semantic/Emotional: Qdrant (persistent)
#
# Run: mix run verification/memory/verify_hybrid_memory.exs
# ============================================================================

alias VivaCore.Memory
alias VivaBridge.Memory, as: NativeMemory

IO.puts("\n" <> String.duplicate("=", 60))
IO.puts("HYBRID MEMORY VERIFICATION")
IO.puts(String.duplicate("=", 60))

# ----------------------------------------------------------------------------
# 1. Backend Status
# ----------------------------------------------------------------------------
IO.puts("\n[1] Backend Status")
IO.puts(String.duplicate("-", 40))

stats = Memory.stats()
IO.puts("  Backend: #{inspect(stats[:backend])}")
IO.puts("  Rust Ready: #{inspect(stats[:rust_ready])}")
IO.puts("  Qdrant Ready: #{inspect(stats[:qdrant_ready])}")

# ----------------------------------------------------------------------------
# 2. Store Episodic (Rust HNSW)
# ----------------------------------------------------------------------------
IO.puts("\n[2] Store Episodic (Rust HNSW)")
IO.puts(String.duplicate("-", 40))

{time_episodic, result_episodic} = :timer.tc(fn ->
  Memory.store("Test episodic event at #{DateTime.utc_now()}", %{
    type: :episodic,
    importance: 0.7,
    emotion: %{pleasure: 0.3, arousal: 0.2, dominance: 0.1}
  })
end)

case result_episodic do
  {:ok, id} ->
    IO.puts("  Stored: #{id}")
    IO.puts("  Time: #{time_episodic / 1000}ms")
  {:error, reason} ->
    IO.puts("  ERROR: #{inspect(reason)}")
end

# ----------------------------------------------------------------------------
# 3. Store Semantic (Qdrant)
# ----------------------------------------------------------------------------
IO.puts("\n[3] Store Semantic (Qdrant)")
IO.puts(String.duplicate("-", 40))

{time_semantic, result_semantic} = :timer.tc(fn ->
  Memory.store("General knowledge about Elixir concurrency patterns", %{
    type: :semantic,
    importance: 0.8
  })
end)

case result_semantic do
  {:ok, id} ->
    IO.puts("  Stored: #{id}")
    IO.puts("  Time: #{time_semantic / 1000}ms")
  {:error, reason} ->
    IO.puts("  ERROR: #{inspect(reason)}")
end

# ----------------------------------------------------------------------------
# 4. Hybrid Search
# ----------------------------------------------------------------------------
IO.puts("\n[4] Hybrid Search")
IO.puts(String.duplicate("-", 40))

{time_search, results} = :timer.tc(fn ->
  Memory.search("episodic event", limit: 5, types: [:episodic, :semantic])
end)

IO.puts("  Found: #{length(results)} results")
IO.puts("  Time: #{time_search / 1000}ms")

Enum.each(results, fn r ->
  IO.puts("    - [#{r[:type]}] #{String.slice(r[:content] || "", 0, 40)}... (#{Float.round(r[:similarity] || 0.0, 3)})")
end)

# ----------------------------------------------------------------------------
# 5. Performance Comparison
# ----------------------------------------------------------------------------
IO.puts("\n[5] Performance Comparison")
IO.puts(String.duplicate("-", 40))

# Episodic-only search
{time_epi_only, _} = :timer.tc(fn ->
  Memory.search("test", limit: 10, types: [:episodic])
end)
IO.puts("  Episodic-only: #{time_epi_only / 1000}ms")

# Semantic-only search
{time_sem_only, _} = :timer.tc(fn ->
  Memory.search("test", limit: 10, types: [:semantic])
end)
IO.puts("  Semantic-only: #{time_sem_only / 1000}ms")

IO.puts("\n" <> String.duplicate("=", 60))
IO.puts("VERIFICATION COMPLETE")
IO.puts(String.duplicate("=", 60) <> "\n")
