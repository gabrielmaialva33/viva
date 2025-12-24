# Script to verify VIVA system functionalities
alias Viva.Avatars
alias Viva.Relationships
alias Viva.Sessions
alias Viva.Repo

# Ensure the app is started (mix run starts it, but just in case)
# Viva.Application.start(nil, nil) # Already started by mix run

IO.puts("\n=============================================")
IO.puts("   VIVA SYSTEM DIAGNOSTIC VERIFICATION")
IO.puts("=============================================\n")

# 1. AVATARS
IO.puts(">>> Checking Avatars context...")
avatars = Avatars.list_avatars()
active_count = Enum.count(avatars, & &1.is_active)
IO.puts("Total Avatars: #{length(avatars)}")
IO.puts("Active Avatars: #{active_count}")

first_avatar = Enum.at(avatars, 0)

if first_avatar do
  IO.puts("\n[Avatar Detail] Example: #{first_avatar.name} (ID: #{first_avatar.id})")
  IO.puts("  - Bio: #{String.slice(first_avatar.bio || "", 0, 50)}...")

  IO.puts(
    "  - Personality: Enneagram #{first_avatar.personality.enneagram_type}, Humor: #{first_avatar.personality.humor_style}"
  )

  # 2. RELATIONSHIPS
  IO.puts("\n>>> Checking Relationships context...")
  rels = Relationships.list_relationships(first_avatar.id)
  IO.puts("Relationships found for #{first_avatar.name}: #{length(rels)}")

  if Enum.any?(rels) do
    r = List.first(rels)
    other_id = if r.avatar_a_id == first_avatar.id, do: r.avatar_b_id, else: r.avatar_a_id
    IO.puts("  - Relation with #{other_id}: Status=#{r.status}, Trust=#{r.trust}")
  end

  # 3. LIFE PROCESS (SESSION)
  IO.puts("\n>>> Checking LifeProcess (GenServer)... waiting for startup...")
  # Allow Application.start_active_avatars (2s delay) to finish
  Process.sleep(3500)
  # Using correct registry name
  registry_name = Viva.Sessions.AvatarRegistry

  case Registry.lookup(registry_name, first_avatar.id) do
    [{pid, _}] ->
      IO.puts("  [SUCCESS] Process IS RUNNING. PID: #{inspect(pid)}")

      if Process.alive?(pid) do
        IO.puts("  - Process is alive.")

        # Optional: inspect state if exposed, but requires :sys.get_state which might be safe in dev script
        # state = :sys.get_state(pid)
        # IO.puts("  - Internal State Mood: #{inspect(state.state.mood)}")
      else
        IO.puts("  [WARN] PID found but process not alive?")
      end

    [] ->
      IO.puts("  [INFO] Process NOT FOUND in Registry.")

      IO.puts(
        "  This might be normal if the avatar is inactive or the supervisor strategy loads them on demand."
      )
  end
else
  IO.puts("\n[WARN] No avatars found. Database might be empty or not seeded.")
end

# 4. NIM / AI
IO.puts("\n>>> Checking AI Configuration (Nvidia NIM)...")
nim_config = Application.get_env(:viva, :nim) || []
api_key = nim_config[:api_key]
base_url = nim_config[:base_url]

if api_key && String.length(api_key) > 0 do
  IO.puts("  - API Key: Present (#{String.slice(api_key, 0, 5)}...)")
else
  IO.puts("  - API Key: [MISSING] or Empty")
end

IO.puts("  - Base URL: #{base_url || "Default"}")

# Simple Connectivity Check (Optional - might consume credits/quota)
# IO.puts("  - Attempting dry-run generation call...")
# This would require a valid module call, e.g. Viva.Nim.LlmClient.generate(...)

IO.puts("\n=============================================")
IO.puts("   VERIFICATION COMPLETE")
IO.puts("=============================================\n")
