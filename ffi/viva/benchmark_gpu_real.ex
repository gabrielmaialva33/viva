defmodule Viva.BenchmarkGpuReal do
  @moduledoc """
  Benchmark que mostra onde GPU realmente brilha:
  - Tensores GRANDES (100K+ elementos)
  - Operações batched mantidas na GPU
  - Pipeline compilado com Nx.Defn

  Uso: mix run -e "Viva.BenchmarkGpuReal.run()"
  """

  import Nx.Defn

  def run do
    IO.puts("")
    IO.puts("╔══════════════════════════════════════════════════════════════╗")
    IO.puts("║       GPU BENCHMARK - Where RTX 4090 REALLY Shines          ║")
    IO.puts("╚══════════════════════════════════════════════════════════════╝")
    IO.puts("")

    IO.puts("Testing CPU vs CUDA on LARGE tensors...")
    IO.puts("")

    bench_large_matmul()
    bench_large_batch()
    bench_defn_pipeline()

    IO.puts("")
    IO.puts("════════════════════════════════════════════════════════════")
    IO.puts("BENCHMARK COMPLETE")
    IO.puts("════════════════════════════════════════════════════════════")
  end

  # ============================================================================
  # LARGE MATMUL - Where GPU dominates
  # ============================================================================

  defp bench_large_matmul do
    IO.puts("═══════════════════════════════════════════════════════════")
    IO.puts("[1/3] LARGE MATMUL (GPU sweet spot)")
    IO.puts("───────────────────────────────────────────────────────────")

    sizes = [2048, 4096, 8192]

    Enum.each(sizes, fn size ->
      IO.puts("")
      IO.puts("  Matrix: #{size}x#{size}")

      # CPU benchmark
      Nx.default_backend({EXLA.Backend, client: :host})
      a_cpu = Nx.Random.uniform(Nx.Random.key(42), shape: {size, size}) |> elem(0)
      b_cpu = Nx.Random.uniform(Nx.Random.key(43), shape: {size, size}) |> elem(0)

      # Warmup
      _ = Nx.dot(a_cpu, b_cpu)

      {cpu_time, _} = :timer.tc(fn ->
        Nx.dot(a_cpu, b_cpu)
      end)

      cpu_gflops = (2 * size * size * size) / cpu_time  # FLOPS / microseconds = MFLOPS

      # GPU benchmark
      try do
        Nx.default_backend({EXLA.Backend, client: :cuda})
        a_gpu = Nx.Random.uniform(Nx.Random.key(42), shape: {size, size}) |> elem(0)
        b_gpu = Nx.Random.uniform(Nx.Random.key(43), shape: {size, size}) |> elem(0)

        # Warmup (JIT compile)
        _ = Nx.dot(a_gpu, b_gpu)

        {gpu_time, _} = :timer.tc(fn ->
          result = Nx.dot(a_gpu, b_gpu)
          # Force sync
          Nx.to_number(Nx.sum(result))
        end)

        gpu_gflops = (2 * size * size * size) / gpu_time

        speedup = gpu_gflops / cpu_gflops

        IO.puts("    CPU:     #{fmt_time(cpu_time)} (#{Float.round(cpu_gflops, 1)} GFLOPS)")
        IO.puts("    GPU:     #{fmt_time(gpu_time)} (#{Float.round(gpu_gflops, 1)} GFLOPS)")
        IO.puts("    Speedup: #{Float.round(speedup, 1)}x #{if speedup > 1, do: "⚡", else: ""}")
      rescue
        e ->
          IO.puts("    CPU:     #{fmt_time(cpu_time)} (#{Float.round(cpu_gflops, 1)} GFLOPS)")
          IO.puts("    GPU:     Error: #{inspect(e)}")
      end
    end)
  end

  # ============================================================================
  # LARGE BATCH - Neural network inference at scale
  # ============================================================================

  defp bench_large_batch do
    IO.puts("")
    IO.puts("═══════════════════════════════════════════════════════════")
    IO.puts("[2/3] LARGE BATCH INFERENCE (Real-world scenario)")
    IO.puts("───────────────────────────────────────────────────────────")

    # Network: 1024 -> 2048 -> 1024
    IO.puts("  Network: 1024 -> 2048 -> 1024")

    batch_sizes = [1024, 4096, 16384]

    Enum.each(batch_sizes, fn batch_size ->
      IO.puts("")
      IO.puts("  Batch: #{batch_size} samples")

      # CPU
      Nx.default_backend({EXLA.Backend, client: :host})
      w1_cpu = Nx.Random.uniform(Nx.Random.key(1), shape: {2048, 1024}) |> elem(0)
      w2_cpu = Nx.Random.uniform(Nx.Random.key(2), shape: {1024, 2048}) |> elem(0)
      inputs_cpu = Nx.Random.uniform(Nx.Random.key(3), shape: {batch_size, 1024}) |> elem(0)

      # Warmup
      _ = forward_pass(inputs_cpu, w1_cpu, w2_cpu)

      {cpu_time, _} = :timer.tc(fn ->
        forward_pass(inputs_cpu, w1_cpu, w2_cpu)
      end)

      cpu_throughput = batch_size / (cpu_time / 1_000_000)

      # GPU
      try do
        Nx.default_backend({EXLA.Backend, client: :cuda})
        w1_gpu = Nx.Random.uniform(Nx.Random.key(1), shape: {2048, 1024}) |> elem(0)
        w2_gpu = Nx.Random.uniform(Nx.Random.key(2), shape: {1024, 2048}) |> elem(0)
        inputs_gpu = Nx.Random.uniform(Nx.Random.key(3), shape: {batch_size, 1024}) |> elem(0)

        # Warmup (JIT)
        _ = forward_pass(inputs_gpu, w1_gpu, w2_gpu)

        {gpu_time, _} = :timer.tc(fn ->
          result = forward_pass(inputs_gpu, w1_gpu, w2_gpu)
          # Force sync
          Nx.to_number(Nx.sum(result))
        end)

        gpu_throughput = batch_size / (gpu_time / 1_000_000)
        speedup = gpu_throughput / cpu_throughput

        IO.puts("    CPU:     #{fmt_time(cpu_time)} (#{fmt_throughput(cpu_throughput)})")
        IO.puts("    GPU:     #{fmt_time(gpu_time)} (#{fmt_throughput(gpu_throughput)})")
        IO.puts("    Speedup: #{Float.round(speedup, 1)}x #{if speedup > 1, do: "⚡", else: ""}")
      rescue
        e ->
          IO.puts("    CPU:     #{fmt_time(cpu_time)} (#{fmt_throughput(cpu_throughput)})")
          IO.puts("    GPU:     Error: #{inspect(e)}")
      end
    end)
  end

  defp forward_pass(inputs, w1, w2) do
    inputs
    |> Nx.dot(Nx.transpose(w1))
    |> Nx.max(0)  # ReLU
    |> Nx.dot(Nx.transpose(w2))
  end

  # ============================================================================
  # DEFN PIPELINE - Compiled GPU kernels
  # ============================================================================

  defp bench_defn_pipeline do
    IO.puts("")
    IO.puts("═══════════════════════════════════════════════════════════")
    IO.puts("[3/3] COMPILED PIPELINE (Nx.Defn - no transfer overhead)")
    IO.puts("───────────────────────────────────────────────────────────")

    size = 4096
    iterations = 10

    IO.puts("  Operation: 10x chained matmul + activation on #{size}x#{size}")
    IO.puts("")

    # CPU
    Nx.default_backend({EXLA.Backend, client: :host})
    a_cpu = Nx.Random.uniform(Nx.Random.key(42), shape: {size, size}) |> elem(0)

    # Warmup
    _ = chained_ops(a_cpu, iterations)

    {cpu_time, _} = :timer.tc(fn ->
      chained_ops(a_cpu, iterations)
    end)

    # GPU
    try do
      Nx.default_backend({EXLA.Backend, client: :cuda})
      a_gpu = Nx.Random.uniform(Nx.Random.key(42), shape: {size, size}) |> elem(0)

      # Warmup (JIT)
      _ = chained_ops(a_gpu, iterations)

      {gpu_time, _} = :timer.tc(fn ->
        result = chained_ops(a_gpu, iterations)
        Nx.to_number(Nx.sum(result))
      end)

      speedup = cpu_time / gpu_time

      IO.puts("    CPU:     #{fmt_time(cpu_time)}")
      IO.puts("    GPU:     #{fmt_time(gpu_time)}")
      IO.puts("    Speedup: #{Float.round(speedup, 1)}x #{if speedup > 1, do: "⚡⚡⚡", else: ""}")
    rescue
      e ->
        IO.puts("    CPU:     #{fmt_time(cpu_time)}")
        IO.puts("    GPU:     Error: #{inspect(e)}")
    end
  end

  defp chained_ops(tensor, 0), do: tensor
  defp chained_ops(tensor, n) do
    tensor
    |> Nx.dot(Nx.transpose(tensor))
    |> Nx.divide(Nx.reduce_max(tensor))  # Normalize
    |> Nx.tanh()
    |> chained_ops(n - 1)
  end

  # ============================================================================
  # FORMATTERS
  # ============================================================================

  defp fmt_time(us) when us > 1_000_000, do: "#{Float.round(us / 1_000_000, 2)}s"
  defp fmt_time(us) when us > 1_000, do: "#{Float.round(us / 1_000, 2)}ms"
  defp fmt_time(us), do: "#{us}μs"

  defp fmt_throughput(t) when t > 1_000_000, do: "#{Float.round(t / 1_000_000, 2)}M samples/s"
  defp fmt_throughput(t) when t > 1_000, do: "#{Float.round(t / 1_000, 2)}K samples/s"
  defp fmt_throughput(t), do: "#{Float.round(t, 2)} samples/s"
end
