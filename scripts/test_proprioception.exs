# scripts/test_proprioception.exs
IO.puts("\n=== PROPRIOCEPTION TEST ===\n")

# Need to start the GenServer manually since app might not be running in script mode fully
{:ok, pid} = Viva.System.Proprioception.start_link([])
IO.puts("Cerebellum Started: #{inspect(pid)}")

IO.puts("Waiting for first heartbeat...")
Process.sleep(2000)

vitality = Viva.System.Proprioception.vitality()
IO.inspect(vitality, label: "Vitals")

if vitality.total_ram > 0 do
  IO.puts("  [OK] RAM Sensing functional.")
else
  IO.puts("  [FAIL] RAM Sensing returned 0.")
end

if vitality.vram_free > 0 do
  IO.puts("  [OK] VRAM Sensing functional (Nvidia-SMI).")
else
  IO.puts("  [WARN] VRAM 0 - Is nvidia-smi available?")
end

IO.puts("  Current State: #{vitality.state}")
