# VIVA GPU Benchmark
# Run with: XLA_TARGET=cuda12 mix run scripts/bench_gpu.exs

IO.puts("""

╔════════════════════════════════════════════════════════════╗
║           VIVA GPU BENCHMARK                               ║
║           RTX 4090 CUDA Performance                        ║
╚════════════════════════════════════════════════════════════╝
""")

# GPU Info
info = VivaGpu.gpu_info()
{:GpuInfo, true, name, memory, compute} = info
IO.puts("GPU: #{name}")
IO.puts("VRAM: #{memory} MB")
IO.puts("Compute: #{compute}")
IO.puts("")

# ============================================================================
# Benchmark: Batch PAD Operations
# ============================================================================

IO.puts("─── BATCH PAD OPERATIONS (100 souls) ───")

# Create batch of 100 PADs
pads = for i <- 1..100 do
  f = i / 100.0
  {:Pad, f, -f, f * 0.5}
end

batch_data = Enum.flat_map(pads, fn {:Pad, p, a, d} -> [p, a, d] end)
batch = {:PadBatch, batch_data, 100}
delta = {:Pad, 0.1, 0.1, 0.1}

# Warmup
_ = VivaGpu.batch_apply_delta(batch, delta)

# Benchmark
iterations = 10_000

{time_us, _} = :timer.tc(fn ->
  for _ <- 1..iterations do
    VivaGpu.batch_apply_delta(batch, delta)
  end
end)

ops_per_sec = iterations / (time_us / 1_000_000)
IO.puts("batch_apply_delta: #{Float.round(ops_per_sec, 0)} ops/sec")
IO.puts("  (#{iterations} iterations in #{Float.round(time_us/1000, 2)} ms)")

# ============================================================================
# Benchmark: Batch Resonance (O(n²))
# ============================================================================

IO.puts("\n─── BATCH RESONANCE (n² pairwise) ───")

for n <- [10, 50, 100] do
  pads = for i <- 1..n do
    f = i / n
    {:Pad, f - 0.5, :rand.uniform() - 0.5, :rand.uniform() - 0.5}
  end

  # Warmup
  _ = VivaGpu.batch_resonance(pads)

  {time_us, _} = :timer.tc(fn ->
    for _ <- 1..100 do
      VivaGpu.batch_resonance(pads)
    end
  end)

  ops_per_sec = 100 / (time_us / 1_000_000)
  pairs = n * n
  pairs_per_sec = ops_per_sec * pairs
  IO.puts("n=#{n}: #{Float.round(ops_per_sec, 0)} ops/sec (#{Float.round(pairs_per_sec/1000, 0)}K pairs/sec)")
end

# ============================================================================
# Benchmark: Matmul Comparison
# ============================================================================

IO.puts("\n─── MATMUL (GPU vs CPU) ───")

for size <- [100, 500, 1000] do
  a = Nx.iota({size, size}, type: :f32)
  b = Nx.iota({size, size}, type: :f32)

  # Warmup GPU
  _ = Nx.dot(a, b)

  # GPU
  {gpu_time, _} = :timer.tc(fn ->
    for _ <- 1..10 do
      Nx.dot(a, b)
    end
  end)
  gpu_ms = gpu_time / 10 / 1000

  # CPU (BinaryBackend)
  a_cpu = Nx.backend_transfer(a, Nx.BinaryBackend)
  b_cpu = Nx.backend_transfer(b, Nx.BinaryBackend)
  {cpu_time, _} = :timer.tc(fn -> Nx.dot(a_cpu, b_cpu) end)
  cpu_ms = cpu_time / 1000

  speedup = Float.round(cpu_ms / gpu_ms, 1)
  IO.puts("#{size}x#{size}: GPU #{Float.round(gpu_ms, 2)}ms | CPU #{Float.round(cpu_ms, 2)}ms | #{speedup}x ⚡")
end

IO.puts("""

════════════════════════════════════════════════════════════
GPU BENCHMARK COMPLETE
════════════════════════════════════════════════════════════
""")
