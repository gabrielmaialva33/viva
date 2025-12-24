# VIVA - Comprehensive Feature Test Script
# Run with: mix run test_all_features.exs
# Or in IEx: Code.eval_file("test_all_features.exs")

defmodule FeatureTest do
  @moduledoc "Tests all major VIVA features"

  def run do
    IO.puts("\n" <> String.duplicate("=", 60))
    IO.puts("   VIVA COMPREHENSIVE FEATURE TEST")
    IO.puts(String.duplicate("=", 60) <> "\n")

    results = [
      test_database(),
      test_avatars(),
      test_world_clock(),
      test_life_process(),
      test_matchmaker(),
      test_relationships(),
      test_memories(),
      test_nim_config(),
      test_nim_llm(),
      test_conversations()
    ]

    print_summary(results)
  end

  # ============================================================================
  # 1. DATABASE
  # ============================================================================
  defp test_database do
    IO.puts(">>> [1/10] Testing Database Connection...")

    try do
      result = Viva.Repo.query!("SELECT 1 as ok")
      [[1]] = result.rows
      IO.puts("    [OK] Database connection successful")
      {:database, :ok}
    rescue
      e ->
        IO.puts("    [FAIL] #{inspect(e)}")
        {:database, :error}
    end
  end

  # ============================================================================
  # 2. AVATARS
  # ============================================================================
  defp test_avatars do
    IO.puts("\n>>> [2/10] Testing Avatars Context...")

    try do
      avatars = Viva.Avatars.list_avatars()
      count = length(avatars)
      IO.puts("    Found #{count} avatars")

      if count > 0 do
        avatar = List.first(avatars)
        IO.puts("    Sample: #{avatar.name}")
        IO.puts("      - Big Five Openness: #{avatar.personality.openness}")
        IO.puts("      - Conscientiousness: #{avatar.personality.conscientiousness}")
        IO.puts("      - Extraversion: #{avatar.personality.extraversion}")
        IO.puts("      - Agreeableness: #{avatar.personality.agreeableness}")
        IO.puts("      - Neuroticism: #{avatar.personality.neuroticism}")
        IO.puts("      - Enneagram Type: #{avatar.personality.enneagram_type}")
        IO.puts("      - Humor Style: #{avatar.personality.humor_style}")
        IO.puts("      - Is Active: #{avatar.is_active}")
        {:avatars, :ok, avatar}
      else
        IO.puts("    [WARN] No avatars found - run mix ecto.reset to seed")
        {:avatars, :empty}
      end
    rescue
      e ->
        IO.puts("    [FAIL] #{Exception.message(e)}")
        {:avatars, :error}
    end
  end

  # ============================================================================
  # 3. WORLD CLOCK
  # ============================================================================
  defp test_world_clock do
    IO.puts("\n>>> [3/10] Testing World.Clock GenServer...")

    try do
      case GenServer.whereis(Viva.World.Clock) do
        nil ->
          IO.puts("    [WARN] Clock not running, starting...")
          {:ok, _pid} = Viva.World.Clock.start_link([])
          :timer.sleep(100)
          test_clock_features()

        pid when is_pid(pid) ->
          IO.puts("    Clock running at PID: #{inspect(pid)}")
          test_clock_features()
      end
    rescue
      e ->
        IO.puts("    [FAIL] #{Exception.message(e)}")
        {:world_clock, :error}
    end
  end

  defp test_clock_features do
    world_time = Viva.World.Clock.now()
    time_scale = Viva.World.Clock.time_scale()

    # Get internal state using :sys.get_state
    state = :sys.get_state(Viva.World.Clock)
    is_running = state.is_running

    IO.puts("    World Time: #{DateTime.to_string(world_time)}")
    IO.puts("    Time Scale: #{time_scale}x (1 real min = #{time_scale} sim min)")
    IO.puts("    Is Running: #{is_running}")

    {:world_clock, :ok}
  end

  # ============================================================================
  # 4. LIFE PROCESS
  # ============================================================================
  defp test_life_process do
    IO.puts("\n>>> [4/10] Testing LifeProcess (Avatar Simulation)...")

    try do
      # Wait for avatars to be started
      :timer.sleep(2000)

      running = Viva.Sessions.Supervisor.list_running_avatars()
      IO.puts("    Running avatar processes: #{length(running)}")

      if length(running) > 0 do
        avatar_id = List.first(running)
        IO.puts("    Testing avatar: #{avatar_id}")

        # Get state - returns struct directly, not {:ok, state}
        state = Viva.Sessions.LifeProcess.get_state(avatar_id)

        IO.puts("    [OK] Got state successfully")
        IO.puts("      - Avatar: #{state.avatar.name}")
        IO.puts("      - Current Activity: #{state.state.current_activity}")
        IO.puts("      - Current Desire: #{state.state.current_desire || "none"}")
        IO.puts("      - Bio Dopamine: #{Float.round(state.state.bio.dopamine, 2)}")
        IO.puts("      - Bio Oxytocin: #{Float.round(state.state.bio.oxytocin, 2)}")
        IO.puts("      - Bio Cortisol: #{Float.round(state.state.bio.cortisol, 2)}")
        IO.puts("      - Bio Adenosine: #{Float.round(state.state.bio.adenosine, 2)}")
        IO.puts("      - Emotional Pleasure: #{Float.round(state.state.emotional.pleasure, 2)}")
        IO.puts("      - Emotional Arousal: #{Float.round(state.state.emotional.arousal, 2)}")
        IO.puts("      - Mood Label: #{state.state.emotional.mood_label}")
        {:life_process, :ok}
      else
        IO.puts("    [WARN] No running processes. Starting one...")
        avatars = Viva.Avatars.list_avatars() |> Enum.filter(& &1.is_active)

        if length(avatars) > 0 do
          avatar = List.first(avatars)
          Viva.Sessions.Supervisor.start_avatar(avatar.id)
          :timer.sleep(500)
          IO.puts("    Started #{avatar.name}")
          {:life_process, :started}
        else
          {:life_process, :no_avatars}
        end
      end
    rescue
      e ->
        IO.puts("    [FAIL] #{Exception.message(e)}")
        {:life_process, :error}
    end
  end

  # ============================================================================
  # 5. MATCHMAKER
  # ============================================================================
  defp test_matchmaker do
    IO.puts("\n>>> [5/10] Testing Matchmaker.Engine...")

    try do
      case GenServer.whereis(Viva.Matching.Engine) do
        nil ->
          IO.puts("    [WARN] Matchmaker not running")
          {:matchmaker, :not_running}

        pid ->
          IO.puts("    Matchmaker running at: #{inspect(pid)}")

          # Try finding matches
          avatars = Viva.Avatars.list_avatars()

          if length(avatars) >= 2 do
            avatar = List.first(avatars)
            IO.puts("    Finding matches for: #{avatar.name}...")

            case Viva.Matching.Engine.find_matches(avatar.id, limit: 3) do
              {:ok, matches} ->
                IO.puts("    [OK] Found #{length(matches)} matches")

                Enum.each(matches, fn m ->
                  score_pct = Float.round(m.score.total * 100, 1)
                  IO.puts("      - #{m.avatar.name}: #{score_pct}% compatible")
                end)

                {:matchmaker, :ok}

              {:error, reason} ->
                IO.puts("    [FAIL] #{inspect(reason)}")
                {:matchmaker, :error}
            end
          else
            IO.puts("    [WARN] Need at least 2 avatars to test matching")
            {:matchmaker, :insufficient_data}
          end
      end
    rescue
      e ->
        IO.puts("    [FAIL] #{Exception.message(e)}")
        {:matchmaker, :error}
    end
  end

  # ============================================================================
  # 6. RELATIONSHIPS
  # ============================================================================
  defp test_relationships do
    IO.puts("\n>>> [6/10] Testing Relationships Context...")

    try do
      avatars = Viva.Avatars.list_avatars()

      if length(avatars) >= 2 do
        avatar = List.first(avatars)
        rels = Viva.Relationships.list_relationships(avatar.id)
        IO.puts("    #{avatar.name} has #{length(rels)} relationships")

        if length(rels) > 0 do
          rel = List.first(rels)
          other_id = if rel.avatar_a_id == avatar.id, do: rel.avatar_b_id, else: rel.avatar_a_id
          other = Viva.Avatars.get_avatar!(other_id)

          IO.puts("    Sample relationship with #{other.name}:")
          IO.puts("      - Status: #{rel.status}")
          IO.puts("      - Trust: #{Float.round(rel.trust, 2)}")
          IO.puts("      - Affection: #{Float.round(rel.affection, 2)}")
          IO.puts("      - Familiarity: #{Float.round(rel.familiarity, 2)}")
          IO.puts("      - Attraction: #{Float.round(rel.attraction, 2)}")
          {:relationships, :ok}
        else
          IO.puts("    [INFO] No relationships yet - avatars need to interact")
          {:relationships, :empty}
        end
      else
        {:relationships, :insufficient_data}
      end
    rescue
      e ->
        IO.puts("    [FAIL] #{Exception.message(e)}")
        {:relationships, :error}
    end
  end

  # ============================================================================
  # 7. MEMORIES
  # ============================================================================
  defp test_memories do
    IO.puts("\n>>> [7/10] Testing Memory System...")

    try do
      avatars = Viva.Avatars.list_avatars()

      if length(avatars) > 0 do
        avatar = List.first(avatars)
        memories = Viva.Avatars.list_memories(avatar.id, limit: 5)
        IO.puts("    #{avatar.name} has memories: #{length(memories)} (showing up to 5)")

        if length(memories) > 0 do
          mem = List.first(memories)
          IO.puts("    Sample memory:")
          IO.puts("      - Type: #{mem.type}")
          IO.puts("      - Content: #{String.slice(mem.content, 0, 50)}...")
          IO.puts("      - Importance: #{Float.round(mem.importance, 2)}")
          IO.puts("      - Strength: #{Float.round(mem.strength, 2)}")
          IO.puts("      - Has Embedding: #{mem.embedding != nil}")
          {:memories, :ok}
        else
          IO.puts("    [INFO] No memories yet - avatar needs experiences")
          {:memories, :empty}
        end
      else
        {:memories, :no_avatars}
      end
    rescue
      e ->
        IO.puts("    [FAIL] #{Exception.message(e)}")
        {:memories, :error}
    end
  end

  # ============================================================================
  # 8. NIM CONFIG
  # ============================================================================
  defp test_nim_config do
    IO.puts("\n>>> [8/10] Testing NVIDIA NIM Configuration...")

    try do
      nim_config = Application.get_env(:viva, :nim, [])

      api_key = nim_config[:api_key]
      base_url = nim_config[:base_url]
      llm_model = nim_config[:llm_model]

      has_key = api_key && String.length(api_key) > 10

      if has_key do
        IO.puts("    [OK] API Key: #{String.slice(api_key, 0, 10)}...")
      else
        IO.puts("    [WARN] API Key missing or too short")
      end

      IO.puts("    Base URL: #{base_url || "not set"}")
      IO.puts("    LLM Model: #{llm_model || "default"}")

      if has_key, do: {:nim_config, :ok}, else: {:nim_config, :missing_key}
    rescue
      e ->
        IO.puts("    [FAIL] #{inspect(e)}")
        {:nim_config, :error}
    end
  end

  # ============================================================================
  # 9. NIM LLM CLIENT
  # ============================================================================
  defp test_nim_llm do
    IO.puts("\n>>> [9/10] Testing NIM LLM Client (live call)...")

    try do
      nim_config = Application.get_env(:viva, :nim, [])
      has_key = nim_config[:api_key] && String.length(nim_config[:api_key]) > 10

      if has_key do
        IO.puts("    Sending test prompt to LLM...")

        messages = [
          %{role: "system", content: "You are a helpful assistant. Respond in one short sentence."},
          %{role: "user", content: "Say hello in Portuguese."}
        ]

        # Note: LlmClient not LLMClient
        case Viva.Nim.LlmClient.chat(messages, max_tokens: 50) do
          {:ok, response} ->
            IO.puts("    [OK] LLM Response: #{response}")
            {:nim_llm, :ok}

          {:error, reason} ->
            IO.puts("    [FAIL] LLM Error: #{inspect(reason)}")
            {:nim_llm, :error}
        end
      else
        IO.puts("    [SKIP] No API key configured")
        {:nim_llm, :skipped}
      end
    rescue
      e ->
        IO.puts("    [FAIL] #{Exception.message(e)}")
        {:nim_llm, :error}
    end
  end

  # ============================================================================
  # 10. CONVERSATIONS
  # ============================================================================
  defp test_conversations do
    IO.puts("\n>>> [10/10] Testing Conversations Context...")

    try do
      avatars = Viva.Avatars.list_avatars()

      if length(avatars) > 0 do
        avatar = List.first(avatars)
        # list_conversations requires avatar_id as first arg
        convs = Viva.Conversations.list_conversations(avatar.id, limit: 5)
        IO.puts("    Found #{length(convs)} conversations for #{avatar.name}")

        if length(convs) > 0 do
          conv = List.first(convs)
          IO.puts("    Sample conversation:")
          IO.puts("      - Type: #{conv.type}")
          IO.puts("      - Status: #{conv.status}")
          IO.puts("      - Topic: #{conv.topic || "none"}")

          messages = Viva.Conversations.list_messages(conv.id, limit: 3)
          IO.puts("      - Messages: #{length(messages)}")
          {:conversations, :ok}
        else
          IO.puts("    [INFO] No conversations yet")
          {:conversations, :empty}
        end
      else
        {:conversations, :no_avatars}
      end
    rescue
      e ->
        IO.puts("    [FAIL] #{Exception.message(e)}")
        {:conversations, :error}
    end
  end

  # ============================================================================
  # SUMMARY
  # ============================================================================
  defp print_summary(results) do
    IO.puts("\n" <> String.duplicate("=", 60))
    IO.puts("   TEST SUMMARY")
    IO.puts(String.duplicate("=", 60))

    Enum.each(results, fn
      {name, :ok} -> IO.puts("    [PASS] #{name}")
      {name, :ok, _} -> IO.puts("    [PASS] #{name}")
      {name, :empty} -> IO.puts("    [WARN] #{name} - no data")
      {name, :skipped} -> IO.puts("    [SKIP] #{name}")
      {name, :started} -> IO.puts("    [INFO] #{name} - started")
      {name, status} -> IO.puts("    [FAIL] #{name} - #{status}")
    end)

    ok_count =
      Enum.count(results, fn
        {_, :ok} -> true
        {_, :ok, _} -> true
        _ -> false
      end)

    IO.puts("\n    #{ok_count}/#{length(results)} tests passed")
    IO.puts(String.duplicate("=", 60) <> "\n")
  end
end

# Run the tests
FeatureTest.run()
