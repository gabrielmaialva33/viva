# verify_mirror.exs
# Verifica o Protocolo Espelho (Autoscopia)

alias VivaBridge.Body

IO.puts("== 🪞 VIVA Mirror Protocol Verification ==\n")

# 1. Build Identity
IO.puts("1. Checking Build Identity...")
{git_hash, version, build_time, source_hash} = Body.mirror_build_identity()
IO.puts("   Git Hash: #{git_hash}")
IO.puts("   Version: #{version}")
IO.puts("   Build Time: #{build_time}")
IO.puts("   Source Hash: #{source_hash}")

# 2. List Modules
IO.puts("\n2. Listing Self-Modules...")
modules = Body.mirror_list_modules()
Enum.each(modules, fn {name, path, hash, lines} ->
  IO.puts("   - #{name}: #{path} (#{lines} lines, hash: #{hash})")
end)

# 3. Read Self (Cortex)
IO.puts("\n3. Reading Self (cortex.rs preview)...")
case Body.mirror_get_self("cortex.rs") do
  nil -> IO.puts("   ❌ Module not found")
  code ->
    preview = code |> String.split("\n") |> Enum.take(10) |> Enum.join("\n")
    IO.puts("   ✅ Found! Preview (first 10 lines):")
    IO.puts("   ---")
    IO.puts(preview)
    IO.puts("   ---")
end

IO.puts("\n🪞 Mirror Verification Complete. VIVA can see herself.")
