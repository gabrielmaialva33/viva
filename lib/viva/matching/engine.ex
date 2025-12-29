defmodule Viva.Matching.Engine do
  @moduledoc """
  Matchmaker engine that finds compatible avatars.
  Uses personality compatibility, shared interests, and values alignment.

  ## Cache Strategy

  Uses Cachex with automatic TTL expiration to prevent memory leaks:
  - Match results cached for 24 hours
  - Automatic eviction of expired entries
  - Periodic cleanup of inactive avatars
  - Stats available via `stats/0`

  ## Configuration

      config :viva, Viva.Matching.Engine,
        cache_ttl_hours: 24,
        refresh_interval_hours: 1,
        max_cache_size: 10_000
  """
  use GenServer

  import Ecto.Query

  require Cachex.Spec
  require Logger

  alias Viva.Avatars.Avatar
  alias Viva.Avatars.Enneagram
  alias Viva.Avatars.Personality
  alias Viva.Repo

  # === Types ===

  @type avatar_id :: Ecto.UUID.t()
  @type score :: %{
          total: float(),
          personality: float(),
          enneagram: float(),
          temperament: float(),
          interests: float(),
          values: float(),
          communication: float()
        }
  @type match_result :: %{avatar: Avatar.t(), score: score(), explanation: String.t()}
  @type stats :: %{
          hits: non_neg_integer(),
          misses: non_neg_integer(),
          evictions: non_neg_integer(),
          expirations: non_neg_integer(),
          size: non_neg_integer(),
          hit_rate: float()
        }

  @cache_name :matchmaker_cache
  @default_cache_ttl_hours 24
  @default_refresh_interval_hours 1
  @default_max_cache_size 10_000
  @cleanup_interval :timer.hours(6)

  # === Client API ===

  @spec start_link(keyword()) :: GenServer.on_start()
  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @doc "Find top matches for an avatar"
  @spec find_matches(avatar_id(), keyword()) :: {:ok, [match_result()]}
  def find_matches(avatar_id, opts \\ []) do
    GenServer.call(__MODULE__, {:find_matches, avatar_id, opts}, 30_000)
  end

  @doc "Calculate compatibility between two avatars"
  @spec calculate_compatibility(avatar_id(), avatar_id()) :: {:ok, score()} | {:error, term()}
  def calculate_compatibility(avatar_a_id, avatar_b_id) do
    GenServer.call(__MODULE__, {:calculate_compatibility, avatar_a_id, avatar_b_id})
  end

  @doc "Force refresh matches for an avatar"
  @spec refresh_matches(avatar_id()) :: :ok
  def refresh_matches(avatar_id) do
    GenServer.cast(__MODULE__, {:refresh_matches, avatar_id})
  end

  @doc "Invalidate cache for an avatar (e.g., when personality changes)"
  @spec invalidate(avatar_id()) :: {:ok, boolean()}
  def invalidate(avatar_id) do
    Cachex.del(@cache_name, avatar_id)
  end

  @doc "Clear all cached matches"
  @spec clear_cache() :: {:ok, non_neg_integer()}
  def clear_cache do
    Cachex.clear(@cache_name)
  end

  @doc "Get cache statistics"
  @spec stats() :: stats() | %{error: :stats_unavailable}
  def stats do
    case Cachex.stats(@cache_name) do
      {:ok, stats} ->
        {:ok, size} = Cachex.size(@cache_name)

        %{
          hits: Map.get(stats, :hits, 0),
          misses: Map.get(stats, :misses, 0),
          evictions: Map.get(stats, :evictions, 0),
          expirations: Map.get(stats, :expirations, 0),
          size: size,
          hit_rate: calculate_hit_rate(stats)
        }

      {:error, _} ->
        %{error: :stats_unavailable}
    end
  end

  # === Server Callbacks ===

  @impl GenServer
  def init(opts) do
    # Start Cachex for match caching
    cache_opts = [
      stats: true,
      expiration:
        Cachex.Spec.expiration(
          default: :timer.hours(cache_ttl_hours()),
          interval: :timer.minutes(5),
          lazy: true
        ),
      limit:
        Cachex.Spec.limit(
          size: max_cache_size(),
          policy: Cachex.Policy.LRW,
          reclaim: 0.1
        )
    ]

    case Cachex.start_link(@cache_name, cache_opts) do
      {:ok, _} ->
        Logger.info("Matchmaker cache started")

      {:error, {:already_started, _}} ->
        Logger.debug("Matchmaker cache already running")
    end

    state = %{
      last_refresh_at: nil,
      last_cleanup_at: nil,
      opts: opts
    }

    schedule_refresh()
    schedule_cleanup()

    {:ok, state}
  end

  @impl GenServer
  def handle_call({:find_matches, avatar_id, opts}, _, state) do
    limit = Keyword.get(opts, :limit, 10)
    exclude_ids = Keyword.get(opts, :exclude, [])

    result =
      case Cachex.get(@cache_name, avatar_id) do
        {:ok, nil} ->
          # Cache miss - calculate and store
          matches = calculate_matches(avatar_id, 20, [])
          Cachex.put(@cache_name, avatar_id, matches)
          filter_matches(matches, exclude_ids, limit)

        {:ok, cached_matches} ->
          # Cache hit
          filter_matches(cached_matches, exclude_ids, limit)

        {:error, reason} ->
          Logger.error("Matchmaker cache error: #{inspect(reason)}")
          # Fallback to direct calculation
          calculate_matches(avatar_id, limit, exclude_ids)
      end

    {:reply, {:ok, result}, state}
  end

  @impl GenServer
  def handle_call({:calculate_compatibility, avatar_a_id, avatar_b_id}, _, state) do
    result = do_calculate_compatibility(avatar_a_id, avatar_b_id)
    {:reply, result, state}
  end

  @impl GenServer
  def handle_cast({:refresh_matches, avatar_id}, state) do
    # Run calculation in supervised task to avoid blocking
    Task.Supervisor.start_child(Viva.Sessions.TaskSupervisor, fn ->
      matches = calculate_matches(avatar_id, 20, [])
      Cachex.put(@cache_name, avatar_id, matches)
      Logger.debug("Refreshed matches for avatar #{avatar_id}")
    end)

    {:noreply, state}
  end

  @impl GenServer
  def handle_info(:refresh_all, state) do
    Logger.debug("Starting periodic match refresh for active avatars")

    # Only refresh for currently active avatars
    active_avatars = Viva.Sessions.Supervisor.list_running_avatars()

    Enum.each(active_avatars, fn avatar_id ->
      # Stagger refreshes to avoid thundering herd
      delay = :rand.uniform(5_000)

      Task.Supervisor.start_child(Viva.Sessions.TaskSupervisor, fn ->
        Process.sleep(delay)
        matches = calculate_matches(avatar_id, 20, [])
        Cachex.put(@cache_name, avatar_id, matches)
      end)
    end)

    schedule_refresh()
    {:noreply, %{state | last_refresh_at: DateTime.utc_now()}}
  end

  @impl GenServer
  def handle_info(:cleanup_inactive, state) do
    Logger.debug("Starting matchmaker cache cleanup")

    # Get all cached avatar IDs and remove those that are no longer active
    active_avatars = MapSet.new(Viva.Sessions.Supervisor.list_running_avatars())

    {:ok, keys} = Cachex.keys(@cache_name)
    inactive_keys = Enum.reject(keys, &MapSet.member?(active_avatars, &1))

    removed_count =
      Enum.reduce(inactive_keys, 0, fn key, count ->
        case Cachex.del(@cache_name, key) do
          {:ok, true} -> count + 1
          _ -> count
        end
      end)

    if removed_count > 0 do
      Logger.info("Matchmaker cache cleanup: removed #{removed_count} inactive avatar entries")
    end

    schedule_cleanup()
    {:noreply, %{state | last_cleanup_at: DateTime.utc_now()}}
  end

  # === Private Functions ===

  defp schedule_refresh do
    interval = :timer.hours(refresh_interval_hours())
    Process.send_after(self(), :refresh_all, interval)
  end

  defp schedule_cleanup do
    Process.send_after(self(), :cleanup_inactive, @cleanup_interval)
  end

  defp filter_matches(matches, exclude_ids, limit) do
    matches
    |> Enum.reject(fn m -> m.avatar.id in exclude_ids end)
    |> Enum.take(limit)
  end

  defp calculate_hit_rate(%{hits: hits, misses: misses}) when hits + misses > 0 do
    Float.round(hits / (hits + misses) * 100, 1)
  end

  defp calculate_hit_rate(_), do: 0.0

  # Configuration helpers
  defp cache_ttl_hours do
    config = Application.get_env(:viva, __MODULE__, [])
    Keyword.get(config, :cache_ttl_hours, @default_cache_ttl_hours)
  end

  defp refresh_interval_hours do
    config = Application.get_env(:viva, __MODULE__, [])
    Keyword.get(config, :refresh_interval_hours, @default_refresh_interval_hours)
  end

  defp max_cache_size do
    config = Application.get_env(:viva, __MODULE__, [])
    Keyword.get(config, :max_cache_size, @default_max_cache_size)
  end

  # === Match Calculation ===

  defp calculate_matches(avatar_id, limit, exclude_ids) do
    case Repo.get(Avatar, avatar_id) do
      nil ->
        Logger.warning("Avatar #{avatar_id} not found for match calculation")
        []

      avatar ->
        candidates =
          Avatar.active()
          |> exclude_self(avatar_id)
          |> exclude_ids(exclude_ids)
          |> Repo.all()

        candidates
        |> Task.async_stream(
          fn candidate ->
            score = calculate_score(avatar, candidate)
            %{avatar: candidate, score: score}
          end,
          max_concurrency: 10,
          timeout: 10_000,
          on_timeout: :kill_task
        )
        |> Enum.flat_map(fn
          {:ok, result} -> [result]
          {:exit, _} -> []
        end)
        |> Enum.sort_by(& &1.score.total, :desc)
        |> Enum.take(limit)
        |> Enum.map(&add_explanation/1)
    end
  end

  defp exclude_self(query, avatar_id) do
    from(a in query, where: a.id != ^avatar_id)
  end

  defp exclude_ids(query, []), do: query

  defp exclude_ids(query, ids) do
    from(a in query, where: a.id not in ^ids)
  end

  defp calculate_score(avatar_a, avatar_b) do
    personality = personality_compatibility(avatar_a.personality, avatar_b.personality)
    interests = interest_overlap(avatar_a.personality.interests, avatar_b.personality.interests)
    values = value_alignment(avatar_a.personality.values, avatar_b.personality.values)
    style = communication_style_match(avatar_a.personality, avatar_b.personality)
    enneagram = enneagram_compatibility(avatar_a.personality, avatar_b.personality)
    temperament = temperament_compatibility(avatar_a.personality, avatar_b.personality)

    total =
      personality * 0.25 +
        enneagram * 0.20 +
        interests * 0.15 +
        values * 0.15 +
        style * 0.10 +
        temperament * 0.15

    %{
      total: total,
      personality: personality,
      enneagram: enneagram,
      temperament: temperament,
      interests: interests,
      values: values,
      communication: style
    }
  end

  defp personality_compatibility(pa, pb) do
    openness_sim = 1.0 - abs(pa.openness - pb.openness)
    consc_sim = 1.0 - abs(pa.conscientiousness - pb.conscientiousness)
    extra_compat = extraversion_compatibility(pa.extraversion, pb.extraversion)
    agree_score = (pa.agreeableness + pb.agreeableness) / 2
    stability = 1.0 - (pa.neuroticism + pb.neuroticism) / 2
    attach_score = attachment_compatibility(pa.attachment_style, pb.attachment_style)

    openness_sim * 0.15 +
      consc_sim * 0.15 +
      extra_compat * 0.15 +
      agree_score * 0.2 +
      stability * 0.15 +
      attach_score * 0.2
  end

  defp extraversion_compatibility(ea, eb) do
    similarity = 1.0 - abs(ea - eb)
    complementary = if (ea < 0.4 and eb > 0.6) or (eb < 0.4 and ea > 0.6), do: 0.7, else: 0.0
    max(similarity, complementary)
  end

  defp attachment_compatibility(style_a, style_b) do
    case {style_a, style_b} do
      {:secure, :secure} -> 1.0
      {:secure, _} -> 0.75
      {_, :secure} -> 0.75
      {:anxious, :anxious} -> 0.5
      {:avoidant, :avoidant} -> 0.4
      {:anxious, :avoidant} -> 0.25
      {:avoidant, :anxious} -> 0.25
      {:fearful, _} -> 0.3
      {_, :fearful} -> 0.3
      _ -> 0.5
    end
  end

  defp interest_overlap(interests_a, interests_b) do
    set_a = MapSet.new(interests_a || [])
    set_b = MapSet.new(interests_b || [])

    intersection =
      set_a
      |> MapSet.intersection(set_b)
      |> MapSet.size()

    union =
      set_a
      |> MapSet.union(set_b)
      |> MapSet.size()

    if union == 0, do: 0.5, else: intersection / union
  end

  defp value_alignment(values_a, values_b) do
    set_a = MapSet.new(values_a || [])
    set_b = MapSet.new(values_b || [])

    intersection =
      set_a
      |> MapSet.intersection(set_b)
      |> MapSet.size()

    union =
      set_a
      |> MapSet.union(set_b)
      |> MapSet.size()

    if union == 0, do: 0.5, else: intersection / union
  end

  defp communication_style_match(pa, pb) do
    humor_match =
      if pa.humor_style == pb.humor_style,
        do: 1.0,
        else: humor_compatibility(pa.humor_style, pb.humor_style)

    love_lang_match = if pa.love_language == pb.love_language, do: 1.0, else: 0.5
    humor_match * 0.6 + love_lang_match * 0.4
  end

  defp humor_compatibility(style_a, style_b) do
    compatible_pairs = [
      {:witty, :sarcastic},
      {:wholesome, :witty},
      {:absurd, :witty},
      {:dark, :sarcastic}
    ]

    pair = {style_a, style_b}
    reverse = {style_b, style_a}

    if pair in compatible_pairs or reverse in compatible_pairs, do: 0.7, else: 0.3
  end

  defp enneagram_compatibility(pa, pb) do
    Enneagram.compatibility(pa.enneagram_type, pb.enneagram_type)
  end

  # Temperament compatibility lookup (symmetric pairs)
  @temperament_scores %{
    {:sanguine, :melancholic} => 0.8,
    {:choleric, :phlegmatic} => 0.85,
    {:sanguine, :choleric} => 0.7,
    {:phlegmatic, :melancholic} => 0.75,
    {:sanguine, :phlegmatic} => 0.6,
    {:choleric, :melancholic} => 0.5
  }

  defp temperament_compatibility(pa, pb) do
    temp_a = Personality.temperament(pa)
    temp_b = Personality.temperament(pb)

    if temp_a == temp_b do
      0.7
    else
      lookup_temperament_score(temp_a, temp_b)
    end
  end

  defp lookup_temperament_score(temp_a, temp_b) do
    Map.get(@temperament_scores, {temp_a, temp_b}) ||
      Map.get(@temperament_scores, {temp_b, temp_a}) ||
      0.6
  end

  defp add_explanation(match) do
    explanation = generate_explanation(match)
    Map.put(match, :explanation, explanation)
  end

  defp generate_explanation(match) do
    score = match.score

    cond do
      score.total > 0.8 ->
        "Exceptional match! Strong personality alignment and shared interests."

      score.total > 0.7 ->
        "Great potential! #{highlight_strength(score)}"

      score.total > 0.6 ->
        "Good compatibility. #{highlight_strength(score)}"

      score.total > 0.5 ->
        "Decent match with some differences that could be interesting."

      true ->
        "Different personalities, but opposites can attract!"
    end
  end

  defp highlight_strength(score) do
    cond do
      score.enneagram > 0.8 -> "Deep psychological compatibility."
      score.temperament > 0.8 -> "Complementary temperaments that balance each other."
      score.personality > 0.7 -> "Very compatible personalities."
      score.interests > 0.6 -> "Lots of shared interests to explore."
      score.values > 0.7 -> "Strong alignment on what matters most."
      score.communication > 0.7 -> "Great communication chemistry."
      true -> "A balanced match across all areas."
    end
  end

  defp do_calculate_compatibility(avatar_a_id, avatar_b_id) do
    with {:ok, avatar_a} <- fetch_avatar(avatar_a_id),
         {:ok, avatar_b} <- fetch_avatar(avatar_b_id) do
      score = calculate_score(avatar_a, avatar_b)
      {:ok, score}
    end
  end

  defp fetch_avatar(id) do
    case Repo.get(Avatar, id) do
      nil -> {:error, :not_found}
      avatar -> {:ok, avatar}
    end
  end
end
