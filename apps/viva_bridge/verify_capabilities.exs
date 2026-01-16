# verify_capabilities.exs
# Verifica Fase 2: Auto-Detecção de Capabilities

alias VivaBridge.Body

IO.puts("== VIVA Capabilities Detection (Fase 2) ==\n")

# 1. System Capabilities
IO.puts("1. Detected Capabilities:")
{os, arch, has_rapl, has_hwmon, has_nvml, has_battery} = Body.mirror_capabilities()
IO.puts("   OS: #{os}")
IO.puts("   Arch: #{arch}")
IO.puts("   RAPL (power): #{has_rapl}")
IO.puts("   hwmon (temp): #{has_hwmon}")
IO.puts("   NVML (GPU): #{has_nvml}")
IO.puts("   Battery: #{has_battery}")

# 2. Feature Flags
IO.puts("\n2. Safety Feature Flags:")
{metaprog, self_mod, ext_mem, max_mods} = Body.mirror_feature_flags()
IO.puts("   Metaprogramming: #{metaprog}")
IO.puts("   Self-Modification: #{self_mod} (safe by default)")
IO.puts("   External Memory: #{ext_mem}")
IO.puts("   Max Mods/Day: #{max_mods}")

# 3. Summary
IO.puts("\n3. Sensor Summary:")
sensors = [
  {"RAPL (power)", has_rapl},
  {"hwmon (temp)", has_hwmon},
  {"NVML (GPU)", has_nvml},
  {"Battery", has_battery}
]

available = Enum.filter(sensors, fn {_, v} -> v end) |> Enum.map(&elem(&1, 0))
unavailable = Enum.filter(sensors, fn {_, v} -> not v end) |> Enum.map(&elem(&1, 0))

IO.puts("   Available: #{Enum.join(available, ", ") || "none"}")
IO.puts("   Unavailable: #{Enum.join(unavailable, ", ") || "none"}")

IO.puts("\nVIVA knows her environment. Fase 2 complete.")
