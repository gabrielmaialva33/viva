defmodule Viva.BenchmarkCuda do
  @moduledoc """
  CUDA Neural Benchmark - Pure Gleam vs Nx/EXLA (CUDA)
  Uso: mix run -e "Viva.BenchmarkCuda.run()"
  """

  def run do
    IO.puts("")
    IO.puts("╔══════════════════════════════════════════════════════════════╗")
    IO.puts("║         CUDA NEURAL BENCHMARK - RTX 4090 24GB                ║")
    IO.puts("╠══════════════════════════════════════════════════════════════╣")
    IO.puts("║  Pure Gleam vs Nx/EXLA (CUDA)                                ║")
    IO.puts("╚══════════════════════════════════════════════════════════════╝")
    IO.puts("")

    # Check available backends
    IO.puts("Checking backends...")

    # Try CUDA first, fallback to Host
    try do
      Nx.default_backend({EXLA.Backend, client: :cuda})
      IO.puts("  Backend: EXLA/CUDA (RTX 4090)")
    rescue
      _ ->
        Nx.default_backend({EXLA.Backend, client: :host})
        IO.puts("  Backend: EXLA/Host (CPU JIT)")
        IO.puts("  Note: For CUDA, rebuild EXLA with XLA_TARGET=cuda12")
    end
    IO.puts("")

    # Run benchmarks
    IO.puts(String.duplicate("═", 60))
    IO.puts("[1/5] TENSOR OPERATIONS")
    IO.puts(String.duplicate("─", 60))
    bench_tensor_ops()

    IO.puts("")
    IO.puts(String.duplicate("═", 60))
    IO.puts("[2/5] MATRIX MULTIPLICATION")
    IO.puts(String.duplicate("─", 60))
    bench_matmul()

    IO.puts("")
    IO.puts(String.duplicate("═", 60))
    IO.puts("[3/5] BATCH FORWARD PASS")
    IO.puts(String.duplicate("─", 60))
    bench_batch_forward()

    IO.puts("")
    IO.puts(String.duplicate("═", 60))
    IO.puts("[4/5] ACTIVATIONS (GPU optimized)")
    IO.puts(String.duplicate("─", 60))
    bench_activations()

    IO.puts("")
    IO.puts(String.duplicate("═", 60))
    IO.puts("[5/5] VIVA BRAIN SIMULATION")
    IO.puts(String.duplicate("─", 60))
    bench_viva_brain()

    IO.puts("")
    IO.puts(String.duplicate("═", 60))
    IO.puts("BENCHMARK COMPLETE")
    IO.puts(String.duplicate("═", 60))
  end

  # ============================================================================
  # TENSOR OPS
  # ============================================================================

  defp bench_tensor_ops do
    size = 1024
    IO.puts("  Vector size: #{size}")
    IO.puts("")

    a = Nx.Random.uniform(Nx.Random.key(42), shape: {size}) |> elem(0)
    b = Nx.Random.uniform(Nx.Random.key(43), shape: {size}) |> elem(0)

    # Addition
    {add_time, _} = :timer.tc(fn ->
      for _ <- 1..1000, do: Nx.add(a, b)
    end)
    add_ops = 1000 / (add_time / 1_000_000)
    IO.puts("  ADD (1000 ops):     #{fmt_time(add_time)} (#{fmt_ops(add_ops)})")

    # Multiplication
    {mul_time, _} = :timer.tc(fn ->
      for _ <- 1..1000, do: Nx.multiply(a, b)
    end)
    mul_ops = 1000 / (mul_time / 1_000_000)
    IO.puts("  MUL (1000 ops):     #{fmt_time(mul_time)} (#{fmt_ops(mul_ops)})")

    # Softmax
    {sm_time, _} = :timer.tc(fn ->
      for _ <- 1..1000, do: VivaNx.softmax(a)
    end)
    sm_ops = 1000 / (sm_time / 1_000_000)
    IO.puts("  SOFTMAX (1000 ops): #{fmt_time(sm_time)} (#{fmt_ops(sm_ops)})")
  end

  # ============================================================================
  # MATMUL
  # ============================================================================

  defp bench_matmul do
    bench_matmul_size(64, 64, 1000)
    bench_matmul_size(256, 256, 500)
    bench_matmul_size(512, 512, 100)
    bench_matmul_size(1024, 1024, 50)
  end

  defp bench_matmul_size(rows, cols, iterations) do
    a = Nx.Random.uniform(Nx.Random.key(42), shape: {rows, cols}) |> elem(0)
    b = Nx.Random.uniform(Nx.Random.key(43), shape: {cols, rows}) |> elem(0)

    {time, _} = :timer.tc(fn ->
      for _ <- 1..iterations, do: Nx.dot(a, b)
    end)

    ops = iterations / (time / 1_000_000)
    flops = 2 * rows * cols * rows  # approximate FLOPS per matmul
    gflops = (flops * ops) / 1_000_000_000

    IO.puts("")
    IO.puts("  MATMUL [#{rows}x#{cols}] x [#{cols}x#{rows}] (#{iterations} ops):")
    IO.puts("    Time:   #{fmt_time(time)}")
    IO.puts("    Speed:  #{fmt_ops(ops)}")
    IO.puts("    GFLOPS: #{Float.round(gflops, 2)}")
  end

  # ============================================================================
  # BATCH FORWARD
  # ============================================================================

  defp bench_batch_forward do
    # Create a simple network: 64 -> 128 -> 64
    # w1: [64 inputs, 128 outputs] -> need [128, 64] for dot(input, w1^T)
    w1 = VivaNx.xavier_uniform([128, 64])  # [out, in]
    b1 = Nx.broadcast(0.0, {128})
    w2 = VivaNx.xavier_uniform([64, 128])  # [out, in]
    b2 = Nx.broadcast(0.0, {64})

    bench_batch_size(w1, b1, w2, b2, 64)
    bench_batch_size(w1, b1, w2, b2, 256)
    bench_batch_size(w1, b1, w2, b2, 1024)
  end

  defp bench_batch_size(w1, b1, w2, b2, batch_size) do
    # inputs: [batch, 64]
    inputs = Nx.Random.uniform(Nx.Random.key(42), shape: {batch_size, 64}) |> elem(0)

    {time, _} = :timer.tc(fn ->
      for _ <- 1..100 do
        # h = inputs @ w1^T + b1 = [batch, 64] @ [64, 128] = [batch, 128]
        h = inputs
        |> Nx.dot(Nx.transpose(w1))
        |> Nx.add(b1)
        |> VivaNx.relu()

        # out = h @ w2^T + b2 = [batch, 128] @ [128, 64] = [batch, 64]
        h
        |> Nx.dot(Nx.transpose(w2))
        |> Nx.add(b2)
        |> VivaNx.softmax_axis(1)
      end
    end)

    samples = batch_size * 100
    throughput = samples / (time / 1_000_000)

    IO.puts("")
    IO.puts("  BATCH SIZE: #{batch_size}")
    IO.puts("    Time (100 passes):  #{fmt_time(time)}")
    IO.puts("    Throughput:         #{fmt_throughput(throughput)}")
  end

  # ============================================================================
  # ACTIVATIONS
  # ============================================================================

  defp bench_activations do
    size = 1024 * 1024  # 1M elements
    tensor = Nx.Random.uniform(Nx.Random.key(42), shape: {size}) |> elem(0)

    IO.puts("  Tensor size: #{size} elements (1M)")
    IO.puts("")

    activations = [
      {"ReLU", fn t -> VivaNx.relu(t) end},
      {"Sigmoid", fn t -> VivaNx.sigmoid(t) end},
      {"Tanh", fn t -> VivaNx.tanh(t) end},
      {"GELU", fn t -> VivaNx.gelu(t) end},
      {"Softmax", fn t -> VivaNx.softmax(t) end}
    ]

    Enum.each(activations, fn {name, fun} ->
      {time, _} = :timer.tc(fn ->
        for _ <- 1..100, do: fun.(tensor)
      end)
      ops = 100 / (time / 1_000_000)
      IO.puts("  #{String.pad_trailing(name, 10)} #{fmt_time(time)} (#{fmt_ops(ops)})")
    end)
  end

  # ============================================================================
  # VIVA BRAIN
  # ============================================================================

  defp bench_viva_brain do
    # VIVA brain architecture: 16 inputs -> 64 -> 32 -> 8 outputs
    IO.puts("  Architecture: 16 -> 64 -> 32 -> 8")
    IO.puts("")

    w1 = VivaNx.he_normal([16, 64])
    b1 = Nx.broadcast(0.0, {64})
    w2 = VivaNx.he_normal([64, 32])
    b2 = Nx.broadcast(0.0, {32})
    w3 = VivaNx.he_normal([32, 8])
    b3 = Nx.broadcast(0.0, {8})

    # Single soul
    input = Nx.Random.uniform(Nx.Random.key(42), shape: {16}) |> elem(0)

    {single_time, _} = :timer.tc(fn ->
      for _ <- 1..10_000 do
        h1 = input |> Nx.dot(Nx.transpose(w1)) |> Nx.add(b1) |> VivaNx.relu()
        h2 = h1 |> Nx.dot(Nx.transpose(w2)) |> Nx.add(b2) |> VivaNx.relu()
        h2 |> Nx.dot(Nx.transpose(w3)) |> Nx.add(b3) |> VivaNx.softmax()
      end
    end)

    ticks_per_sec = 10_000 / (single_time / 1_000_000)
    IO.puts("  SINGLE SOUL (10K ticks):")
    IO.puts("    Time:       #{fmt_time(single_time)}")
    IO.puts("    Soul-ticks: #{fmt_ticks(ticks_per_sec)}")

    # Batch of 100 souls (parallel thinking)
    inputs = Nx.Random.uniform(Nx.Random.key(42), shape: {100, 16}) |> elem(0)

    {batch_time, _} = :timer.tc(fn ->
      for _ <- 1..1000 do
        h1 = inputs |> Nx.dot(Nx.transpose(w1)) |> Nx.add(b1) |> VivaNx.relu()
        h2 = h1 |> Nx.dot(Nx.transpose(w2)) |> Nx.add(b2) |> VivaNx.relu()
        h2 |> Nx.dot(Nx.transpose(w3)) |> Nx.add(b3) |> VivaNx.softmax_axis(1)
      end
    end)

    batch_ticks = 100 * 1000 / (batch_time / 1_000_000)
    IO.puts("")
    IO.puts("  100 SOULS PARALLEL (1K iterations):")
    IO.puts("    Time:       #{fmt_time(batch_time)}")
    IO.puts("    Soul-ticks: #{fmt_ticks(batch_ticks)}")

    speedup = batch_ticks / ticks_per_sec
    IO.puts("")
    IO.puts("  BATCH SPEEDUP: #{Float.round(speedup, 1)}x")
  end

  # ============================================================================
  # FORMATTERS
  # ============================================================================

  defp fmt_time(us) when us > 1_000_000, do: "#{Float.round(us / 1_000_000, 2)}s"
  defp fmt_time(us) when us > 1_000, do: "#{Float.round(us / 1_000, 2)}ms"
  defp fmt_time(us), do: "#{us}μs"

  defp fmt_ops(ops) when ops > 1_000_000, do: "#{Float.round(ops / 1_000_000, 2)}M ops/s"
  defp fmt_ops(ops) when ops > 1_000, do: "#{Float.round(ops / 1_000, 2)}K ops/s"
  defp fmt_ops(ops), do: "#{Float.round(ops, 2)} ops/s"

  defp fmt_throughput(t) when t > 1_000_000, do: "#{Float.round(t / 1_000_000, 2)}M samples/s"
  defp fmt_throughput(t) when t > 1_000, do: "#{Float.round(t / 1_000, 2)}K samples/s"
  defp fmt_throughput(t), do: "#{Float.round(t, 2)} samples/s"

  defp fmt_ticks(t) when t > 1_000_000, do: "#{Float.round(t / 1_000_000, 2)}M ticks/s"
  defp fmt_ticks(t) when t > 1_000, do: "#{Float.round(t / 1_000, 2)}K ticks/s"
  defp fmt_ticks(t), do: "#{Float.round(t, 2)} ticks/s"
end
