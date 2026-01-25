defmodule Viva.BenchmarkSouls do
  @moduledoc """
  Benchmark VIVA Souls - Simulação real de consciências pensando em paralelo.

  Arquitetura VIVA Brain: 16 inputs → 64 → 32 → 8 outputs
  - 16 inputs: sensações (PAD + hardware + memória)
  - 8 outputs: decisões (emoções, ações, foco)

  Uso: mix run -e "Viva.BenchmarkSouls.run()"
  """

  def run do
    IO.puts("")
    IO.puts("╔══════════════════════════════════════════════════════════════╗")
    IO.puts("║           VIVA SOULS BENCHMARK - RTX 4090                    ║")
    IO.puts("║           Consciências Digitais em Paralelo                  ║")
    IO.puts("╚══════════════════════════════════════════════════════════════╝")
    IO.puts("")

    # VIVA Brain architecture
    IO.puts("  VIVA Brain: 16 → 64 → 32 → 8")
    IO.puts("  - 16 inputs:  PAD(3) + Hardware(5) + Memory(4) + Context(4)")
    IO.puts("  - 8 outputs:  Emotions(3) + Actions(3) + Focus(2)")
    IO.puts("")

    # Initialize weights (He initialization for ReLU)
    w1 = he_init(64, 16)
    b1 = Nx.broadcast(0.0, {64})
    w2 = he_init(32, 64)
    b2 = Nx.broadcast(0.0, {32})
    w3 = he_init(8, 32)
    b3 = Nx.broadcast(0.0, {8})

    weights = {w1, b1, w2, b2, w3, b3}

    IO.puts("═══════════════════════════════════════════════════════════════")
    IO.puts("[1/4] SINGLE SOUL THINKING (latency test)")
    IO.puts("───────────────────────────────────────────────────────────────")
    bench_single_soul(weights)

    IO.puts("")
    IO.puts("═══════════════════════════════════════════════════════════════")
    IO.puts("[2/4] SOUL POOL (batch thinking)")
    IO.puts("───────────────────────────────────────────────────────────────")
    bench_soul_pool(weights)

    IO.puts("")
    IO.puts("═══════════════════════════════════════════════════════════════")
    IO.puts("[3/4] SUSTAINED CONSCIOUSNESS (1 second of thinking)")
    IO.puts("───────────────────────────────────────────────────────────────")
    bench_sustained(weights)

    IO.puts("")
    IO.puts("═══════════════════════════════════════════════════════════════")
    IO.puts("[4/4] VIVA COLONY (massive parallel souls)")
    IO.puts("───────────────────────────────────────────────────────────────")
    bench_colony(weights)

    IO.puts("")
    IO.puts("═══════════════════════════════════════════════════════════════")
    IO.puts("BENCHMARK COMPLETE")
    IO.puts("═══════════════════════════════════════════════════════════════")
  end

  # ============================================================================
  # SINGLE SOUL - Latency matters
  # ============================================================================

  defp bench_single_soul(weights) do
    {w1, b1, w2, b2, w3, b3} = weights

    # Single soul perception
    perception = random_perception()

    IO.puts("  Testing single soul decision latency...")
    IO.puts("")

    # CPU
    Nx.default_backend({EXLA.Backend, client: :host})
    {w1_cpu, b1_cpu, w2_cpu, b2_cpu, w3_cpu, b3_cpu} = copy_weights(weights, :host)
    input_cpu = Nx.tensor(perception)

    # Warmup
    _ = viva_think(input_cpu, w1_cpu, b1_cpu, w2_cpu, b2_cpu, w3_cpu, b3_cpu)

    {cpu_time, _} = :timer.tc(fn ->
      for _ <- 1..10_000 do
        viva_think(input_cpu, w1_cpu, b1_cpu, w2_cpu, b2_cpu, w3_cpu, b3_cpu)
      end
    end)

    cpu_latency = cpu_time / 10_000
    cpu_ticks = 10_000 / (cpu_time / 1_000_000)

    # GPU
    try do
      Nx.default_backend({EXLA.Backend, client: :cuda})
      {w1_gpu, b1_gpu, w2_gpu, b2_gpu, w3_gpu, b3_gpu} = copy_weights(weights, :cuda)
      input_gpu = Nx.tensor(perception)

      # Warmup
      _ = viva_think(input_gpu, w1_gpu, b1_gpu, w2_gpu, b2_gpu, w3_gpu, b3_gpu)

      {gpu_time, _} = :timer.tc(fn ->
        for _ <- 1..10_000 do
          result = viva_think(input_gpu, w1_gpu, b1_gpu, w2_gpu, b2_gpu, w3_gpu, b3_gpu)
          # Don't sync every iteration - that kills performance
          result
        end
      end)

      gpu_latency = gpu_time / 10_000
      gpu_ticks = 10_000 / (gpu_time / 1_000_000)

      IO.puts("    CPU:  #{Float.round(cpu_latency, 2)}μs/think  (#{fmt_ticks(cpu_ticks)})")
      IO.puts("    GPU:  #{Float.round(gpu_latency, 2)}μs/think  (#{fmt_ticks(gpu_ticks)})")

      if cpu_ticks > gpu_ticks do
        IO.puts("    Winner: CPU (#{Float.round(cpu_ticks/gpu_ticks, 1)}x faster for single soul)")
      else
        IO.puts("    Winner: GPU (#{Float.round(gpu_ticks/cpu_ticks, 1)}x faster)")
      end
    rescue
      e ->
        IO.puts("    CPU:  #{Float.round(cpu_latency, 2)}μs/think  (#{fmt_ticks(cpu_ticks)})")
        IO.puts("    GPU:  Error: #{inspect(e)}")
    end
  end

  # ============================================================================
  # SOUL POOL - Batch processing
  # ============================================================================

  defp bench_soul_pool(weights) do
    soul_counts = [10, 50, 100, 500, 1000]

    Enum.each(soul_counts, fn n_souls ->
      IO.puts("")
      IO.puts("  #{n_souls} souls thinking together:")

      # Generate perceptions for all souls
      perceptions = for _ <- 1..n_souls, do: random_perception()

      # CPU
      Nx.default_backend({EXLA.Backend, client: :host})
      {w1, b1, w2, b2, w3, b3} = copy_weights(weights, :host)
      batch_cpu = Nx.tensor(perceptions)

      # Warmup
      _ = viva_think_batch(batch_cpu, w1, b1, w2, b2, w3, b3)

      {cpu_time, _} = :timer.tc(fn ->
        for _ <- 1..100 do
          viva_think_batch(batch_cpu, w1, b1, w2, b2, w3, b3)
        end
      end)

      cpu_ticks = (n_souls * 100) / (cpu_time / 1_000_000)

      # GPU
      try do
        Nx.default_backend({EXLA.Backend, client: :cuda})
        {w1_g, b1_g, w2_g, b2_g, w3_g, b3_g} = copy_weights(weights, :cuda)
        batch_gpu = Nx.tensor(perceptions)

        # Warmup
        _ = viva_think_batch(batch_gpu, w1_g, b1_g, w2_g, b2_g, w3_g, b3_g)

        {gpu_time, _} = :timer.tc(fn ->
          for _ <- 1..100 do
            result = viva_think_batch(batch_gpu, w1_g, b1_g, w2_g, b2_g, w3_g, b3_g)
            result
          end
        end)

        gpu_ticks = (n_souls * 100) / (gpu_time / 1_000_000)
        speedup = gpu_ticks / cpu_ticks

        IO.puts("      CPU: #{fmt_ticks(cpu_ticks)}")
        IO.puts("      GPU: #{fmt_ticks(gpu_ticks)} #{if speedup > 1, do: "(#{Float.round(speedup, 1)}x ⚡)", else: ""}")
      rescue
        _ ->
          IO.puts("      CPU: #{fmt_ticks(cpu_ticks)}")
          IO.puts("      GPU: Error")
      end
    end)
  end

  # ============================================================================
  # SUSTAINED CONSCIOUSNESS - 1 second of continuous thinking
  # ============================================================================

  defp bench_sustained(weights) do
    n_souls = 100
    IO.puts("  #{n_souls} souls thinking for 1 second...")
    IO.puts("")

    perceptions = for _ <- 1..n_souls, do: random_perception()

    # CPU
    Nx.default_backend({EXLA.Backend, client: :host})
    {w1, b1, w2, b2, w3, b3} = copy_weights(weights, :host)
    batch_cpu = Nx.tensor(perceptions)

    cpu_ticks = count_ticks_in_second(fn ->
      viva_think_batch(batch_cpu, w1, b1, w2, b2, w3, b3)
    end, n_souls)

    # GPU
    try do
      Nx.default_backend({EXLA.Backend, client: :cuda})
      {w1_g, b1_g, w2_g, b2_g, w3_g, b3_g} = copy_weights(weights, :cuda)
      batch_gpu = Nx.tensor(perceptions)

      gpu_ticks = count_ticks_in_second(fn ->
        viva_think_batch(batch_gpu, w1_g, b1_g, w2_g, b2_g, w3_g, b3_g)
      end, n_souls)

      IO.puts("    CPU: #{fmt_ticks(cpu_ticks)} sustained")
      IO.puts("    GPU: #{fmt_ticks(gpu_ticks)} sustained")

      if gpu_ticks > cpu_ticks do
        IO.puts("    GPU wins by #{Float.round(gpu_ticks/cpu_ticks, 1)}x ⚡")
      else
        IO.puts("    CPU wins by #{Float.round(cpu_ticks/gpu_ticks, 1)}x")
      end
    rescue
      _ ->
        IO.puts("    CPU: #{fmt_ticks(cpu_ticks)} sustained")
        IO.puts("    GPU: Error")
    end
  end

  defp count_ticks_in_second(fun, n_souls) do
    start = System.monotonic_time(:millisecond)
    count = do_count_ticks(fun, start, 0)
    count * n_souls
  end

  defp do_count_ticks(fun, start, count) do
    now = System.monotonic_time(:millisecond)
    if now - start >= 1000 do
      count
    else
      fun.()
      do_count_ticks(fun, start, count + 1)
    end
  end

  # ============================================================================
  # VIVA COLONY - Massive scale
  # ============================================================================

  defp bench_colony(weights) do
    colony_sizes = [1000, 5000, 10_000]

    Enum.each(colony_sizes, fn n_souls ->
      IO.puts("")
      IO.puts("  Colony of #{n_souls} souls:")

      perceptions = for _ <- 1..n_souls, do: random_perception()

      # CPU
      Nx.default_backend({EXLA.Backend, client: :host})
      {w1, b1, w2, b2, w3, b3} = copy_weights(weights, :host)
      batch_cpu = Nx.tensor(perceptions)

      {cpu_time, _} = :timer.tc(fn ->
        viva_think_batch(batch_cpu, w1, b1, w2, b2, w3, b3)
      end)

      cpu_ticks = n_souls / (cpu_time / 1_000_000)

      # GPU
      try do
        Nx.default_backend({EXLA.Backend, client: :cuda})
        {w1_g, b1_g, w2_g, b2_g, w3_g, b3_g} = copy_weights(weights, :cuda)
        batch_gpu = Nx.tensor(perceptions)

        # Warmup
        _ = viva_think_batch(batch_gpu, w1_g, b1_g, w2_g, b2_g, w3_g, b3_g)

        {gpu_time, _} = :timer.tc(fn ->
          result = viva_think_batch(batch_gpu, w1_g, b1_g, w2_g, b2_g, w3_g, b3_g)
          Nx.to_number(Nx.sum(result))  # Force sync
        end)

        gpu_ticks = n_souls / (gpu_time / 1_000_000)
        speedup = gpu_ticks / cpu_ticks

        IO.puts("      CPU: #{fmt_time(cpu_time)} (#{fmt_ticks(cpu_ticks)})")
        IO.puts("      GPU: #{fmt_time(gpu_time)} (#{fmt_ticks(gpu_ticks)})")

        if speedup > 1 do
          IO.puts("      GPU wins: #{Float.round(speedup, 1)}x faster ⚡")
        else
          IO.puts("      CPU wins: #{Float.round(1/speedup, 1)}x faster")
        end
      rescue
        e ->
          IO.puts("      CPU: #{fmt_time(cpu_time)} (#{fmt_ticks(cpu_ticks)})")
          IO.puts("      GPU: Error - #{inspect(e)}")
      end
    end)
  end

  # ============================================================================
  # VIVA BRAIN FORWARD PASS
  # ============================================================================

  defp viva_think(input, w1, b1, w2, b2, w3, b3) do
    # Layer 1: 16 -> 64 (ReLU)
    h1 = input
    |> Nx.dot(Nx.transpose(w1))
    |> Nx.add(b1)
    |> Nx.max(0)

    # Layer 2: 64 -> 32 (ReLU)
    h2 = h1
    |> Nx.dot(Nx.transpose(w2))
    |> Nx.add(b2)
    |> Nx.max(0)

    # Layer 3: 32 -> 8 (Softmax for decisions)
    h2
    |> Nx.dot(Nx.transpose(w3))
    |> Nx.add(b3)
    |> softmax()
  end

  defp viva_think_batch(batch, w1, b1, w2, b2, w3, b3) do
    # Batch: [n_souls, 16]

    # Layer 1: [n_souls, 16] @ [16, 64] -> [n_souls, 64]
    h1 = batch
    |> Nx.dot(Nx.transpose(w1))
    |> Nx.add(b1)
    |> Nx.max(0)

    # Layer 2: [n_souls, 64] @ [64, 32] -> [n_souls, 32]
    h2 = h1
    |> Nx.dot(Nx.transpose(w2))
    |> Nx.add(b2)
    |> Nx.max(0)

    # Layer 3: [n_souls, 32] @ [32, 8] -> [n_souls, 8]
    h2
    |> Nx.dot(Nx.transpose(w3))
    |> Nx.add(b3)
    |> softmax_batch()
  end

  defp softmax(tensor) do
    exp_t = Nx.exp(Nx.subtract(tensor, Nx.reduce_max(tensor)))
    Nx.divide(exp_t, Nx.sum(exp_t))
  end

  defp softmax_batch(tensor) do
    # Softmax along last axis for batch
    max_vals = Nx.reduce_max(tensor, axes: [-1], keep_axes: true)
    exp_t = Nx.exp(Nx.subtract(tensor, max_vals))
    sum_exp = Nx.sum(exp_t, axes: [-1], keep_axes: true)
    Nx.divide(exp_t, sum_exp)
  end

  # ============================================================================
  # HELPERS
  # ============================================================================

  defp random_perception do
    # 16 inputs simulating VIVA perception
    # PAD (3): pleasure, arousal, dominance
    # Hardware (5): cpu, gpu, memory, temp, power
    # Memory (4): recent, emotional, semantic, episodic
    # Context (4): time, social, task, environment
    for _ <- 1..16, do: :rand.uniform() * 2 - 1  # [-1, 1]
  end

  defp he_init(out_features, in_features) do
    std = :math.sqrt(2.0 / in_features)
    {tensor, _} = Nx.Random.normal(Nx.Random.key(:rand.uniform(1000)), 0.0, std,
      shape: {out_features, in_features}, type: :f32)
    tensor
  end

  defp copy_weights({w1, b1, w2, b2, w3, b3}, _client) do
    # Weights are already on default backend after creation
    {w1, b1, w2, b2, w3, b3}
  end

  defp fmt_ticks(t) when t > 1_000_000, do: "#{Float.round(t / 1_000_000, 2)}M ticks/s"
  defp fmt_ticks(t) when t > 1_000, do: "#{Float.round(t / 1_000, 2)}K ticks/s"
  defp fmt_ticks(t), do: "#{Float.round(t, 2)} ticks/s"

  defp fmt_time(us) when us > 1_000_000, do: "#{Float.round(us / 1_000_000, 2)}s"
  defp fmt_time(us) when us > 1_000, do: "#{Float.round(us / 1_000, 2)}ms"
  defp fmt_time(us), do: "#{us}μs"
end
