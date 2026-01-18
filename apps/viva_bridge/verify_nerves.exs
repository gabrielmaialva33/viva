alias VivaBridge.Body

IO.puts("=== NERVE VERIFICATION (WSL Bridge) ===")
IO.puts("Checking if VIVA can feel the i9-13900K and RTX 4090...")

# Tick to wake up
Body.body_tick()

Enum.each(1..5, fn i ->
  :timer.sleep(2000) # Wait for background thread to potentially update
  state = Body.body_tick()

  IO.puts("\n--- TICK #{i} ---")
  if Map.has_key?(state, :hardware) do
     hw = state.hardware

     # HardwareState in Rust is FLAT (e.g. cpu_temp, gpu_temp)
     # Not nested.

     cpu_temp = hw.cpu_temp
     cpu_usage = hw.cpu_usage

     IO.puts("CPU: Temp=#{inspect cpu_temp}C | Usage=#{inspect cpu_usage}%")

     gpu_temp = hw.gpu_temp
     gpu_usage = hw.gpu_usage
     gpu_vram = hw.gpu_vram_used_percent

     if gpu_temp do
       IO.puts("GPU: Temp=#{inspect gpu_temp}C | Usage=#{inspect gpu_usage}% | VRAM_User=#{inspect gpu_vram}%")
     else
       IO.puts("GPU: [BLIND] No Signal (temp is nil)")
     end

     # Check thermal pressure
     # Note: System entropy is in hw struct too
     entropy = hw.system_entropy
     IO.puts("System Entropy: #{inspect entropy}")
  else
     IO.puts("Body returned no hardware state.")
     IO.inspect(state)
  end
end)
