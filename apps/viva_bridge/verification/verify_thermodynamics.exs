# verify_thermodynamics.exs
# Verifica o sistema de Metabolismo Digital (Termodinâmica)

alias VivaBridge.{Body, Brain, Memory, BodyServer}

IO.puts("== 🔥 VIVA Thermodynamics Verification ==\n")

# 1. Initialize core systems
IO.puts("1. Booting Core Systems...")
Brain.init()
Memory.init()

# 2. Initialize Metabolism with GATO-PC TDP (i9-13900K = 125W)
IO.puts("\n2. Initializing Metabolism Engine (TDP=125W)...")
result = Body.metabolism_init(125.0)
IO.inspect(result, label: "Metabolism Init")

# 3. Start BodyServer if not running
pid = case BodyServer.start_link([]) do
  {:ok, pid} -> pid
  {:error, {:already_started, pid}} ->
    IO.puts("BodyServer already running (PID: #{inspect(pid)})")
    pid
  other ->
    IO.inspect(other, label: "Unexpected BodyServer start result")
    raise "Failed to start BodyServer"
end

# 4. Simulate some ticks to let memory accumulate
IO.puts("\n3. Simulating Thermodynamic Cycles...")
Process.sleep(3000) # 6 ticks at 500ms intervals

# 5. Check metabolism state directly
IO.puts("\n4. Querying Metabolism State...")
state = Body.metabolism_tick(50.0, 55.0) # Mid-load CPU at 55°C
IO.inspect(state, label: "Metabolic State (energy_j, entropy, fatigue, needs_rest)")

# 6. Search for thermodynamic memories
IO.puts("\n5. Searching for Thermodynamic Memories ('energy')...")
{:ok, query_vec} = Brain.experience("Energy flow stable", %{pleasure: 0.0, arousal: 0.0, dominance: 0.0})
results = Memory.search(query_vec, 5)

if Enum.empty?(results) do
  IO.puts("❌ No thermodynamic memories found (yet). Run for longer?")
else
  IO.puts("✅ Thermodynamic Memories Found!")
  Enum.each(results, fn {id, content, score, importance} ->
    IO.puts("  - #{String.slice(content, 0..100)}... (score: #{Float.round(score, 3)})")
  end)
end

IO.puts("\n🔥 Verification Complete. VIVA is feeling energy.")
Process.exit(pid, :normal)
