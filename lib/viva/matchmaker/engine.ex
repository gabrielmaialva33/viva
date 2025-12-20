defmodule Viva.Matchmaker.Engine do
  @moduledoc """
  Matchmaker engine that finds compatible avatars.
  Uses personality compatibility, shared interests, and values alignment.
  Runs as a GenServer to cache and update match scores.
  """
  use GenServer
  require Logger

  alias Viva.Avatars.Avatar
  alias Viva.Repo

  import Ecto.Query

  @refresh_interval :timer.hours(1)
  @match_cache_ttl :timer.hours(24)

  defstruct [
    :match_cache,
    :last_refresh_at
  ]

  # === Client API ===

  def start_link(_opts) do
    GenServer.start_link(__MODULE__, %{}, name: __MODULE__)
  end

  @doc "Find top matches for an avatar"
  def find_matches(avatar_id, opts \\ []) do
    GenServer.call(__MODULE__, {:find_matches, avatar_id, opts}, 30_000)
  end

  @doc "Calculate compatibility between two avatars"
  def calculate_compatibility(avatar_a_id, avatar_b_id) do
    GenServer.call(__MODULE__, {:calculate_compatibility, avatar_a_id, avatar_b_id})
  end

  @doc "Force refresh matches for an avatar"
  def refresh_matches(avatar_id) do
    GenServer.cast(__MODULE__, {:refresh_matches, avatar_id})
  end

  # === Server Callbacks ===

  @impl true
  def init(_opts) do
    state = %__MODULE__{
      match_cache: %{},
      last_refresh_at: nil
    }

    schedule_refresh()
    {:ok, state}
  end

  @impl true
  def handle_call({:find_matches, avatar_id, opts}, _from, state) do
    limit = Keyword.get(opts, :limit, 10)
    exclude_ids = Keyword.get(opts, :exclude, [])

    case Map.get(state.match_cache, avatar_id) do
      nil ->
        # Calculate fresh matches
        matches = calculate_matches(avatar_id, limit, exclude_ids)
        new_cache = Map.put(state.match_cache, avatar_id, {matches, DateTime.utc_now()})
        {:reply, {:ok, matches}, %{state | match_cache: new_cache}}

      {matches, cached_at} ->
        if cache_expired?(cached_at) do
          # Refresh in background, return cached
          GenServer.cast(self(), {:refresh_matches, avatar_id})
        end

        filtered = Enum.reject(matches, fn m -> m.avatar.id in exclude_ids end)
        {:reply, {:ok, Enum.take(filtered, limit)}, state}
    end
  end

  @impl true
  def handle_call({:calculate_compatibility, avatar_a_id, avatar_b_id}, _from, state) do
    result = do_calculate_compatibility(avatar_a_id, avatar_b_id)
    {:reply, result, state}
  end

  @impl true
  def handle_cast({:refresh_matches, avatar_id}, state) do
    matches = calculate_matches(avatar_id, 20, [])
    new_cache = Map.put(state.match_cache, avatar_id, {matches, DateTime.utc_now()})
    {:noreply, %{state | match_cache: new_cache}}
  end

  @impl true
  def handle_info(:refresh_all, state) do
    # Refresh matches for active avatars
    active_avatars = Viva.Sessions.Supervisor.list_running_avatars()

    Enum.each(active_avatars, fn avatar_id ->
      Task.start(fn ->
        GenServer.cast(__MODULE__, {:refresh_matches, avatar_id})
      end)
    end)

    schedule_refresh()
    {:noreply, %{state | last_refresh_at: DateTime.utc_now()}}
  end

  # === Private Functions ===

  defp schedule_refresh do
    Process.send_after(self(), :refresh_all, @refresh_interval)
  end

  defp cache_expired?(cached_at) do
    DateTime.diff(DateTime.utc_now(), cached_at, :millisecond) > @match_cache_ttl
  end

  defp calculate_matches(avatar_id, limit, exclude_ids) do
    avatar = Repo.get!(Avatar, avatar_id)

    # Get candidate avatars
    candidates =
      Avatar.active()
      |> exclude_self(avatar_id)
      |> exclude_ids(exclude_ids)
      |> Repo.all()

    # Calculate compatibility for each
    candidates
    |> Task.async_stream(
      fn candidate ->
        score = calculate_score(avatar, candidate)
        %{avatar: candidate, score: score}
      end,
      max_concurrency: 10,
      timeout: 10_000
    )
    |> Enum.map(fn {:ok, result} -> result end)
    |> Enum.sort_by(& &1.score.total, :desc)
    |> Enum.take(limit)
    |> Enum.map(&add_explanation/1)
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

    total =
      personality * 0.35 +
        interests * 0.25 +
        values * 0.25 +
        style * 0.15

    %{
      total: total,
      personality: personality,
      interests: interests,
      values: values,
      communication: style
    }
  end

  defp personality_compatibility(pa, pb) do
    # Similar traits
    openness_sim = 1.0 - abs(pa.openness - pb.openness)
    consc_sim = 1.0 - abs(pa.conscientiousness - pb.conscientiousness)

    # Complementary extraversion can work
    extra_compat = extraversion_compatibility(pa.extraversion, pb.extraversion)

    # Both agreeable is good
    agree_score = (pa.agreeableness + pb.agreeableness) / 2

    # Lower neuroticism is generally better
    stability = 1.0 - (pa.neuroticism + pb.neuroticism) / 2

    # Attachment compatibility
    attach_score = attachment_compatibility(pa.attachment_style, pb.attachment_style)

    openness_sim * 0.15 +
      consc_sim * 0.15 +
      extra_compat * 0.15 +
      agree_score * 0.2 +
      stability * 0.15 +
      attach_score * 0.2
  end

  defp extraversion_compatibility(ea, eb) do
    # Similar levels work well
    similarity = 1.0 - abs(ea - eb)

    # But introvert + extrovert can also work (complementary)
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

    intersection = MapSet.intersection(set_a, set_b) |> MapSet.size()
    union = MapSet.union(set_a, set_b) |> MapSet.size()

    if union == 0, do: 0.5, else: intersection / union
  end

  defp value_alignment(values_a, values_b) do
    set_a = MapSet.new(values_a || [])
    set_b = MapSet.new(values_b || [])

    intersection = MapSet.intersection(set_a, set_b) |> MapSet.size()
    union = MapSet.union(set_a, set_b) |> MapSet.size()

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

    if pair in compatible_pairs or reverse in compatible_pairs do
      0.7
    else
      0.3
    end
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
