defmodule Viva.Avatars do
  @moduledoc """
  The Avatars context.
  Manages avatar lifecycle, memories, and state.
  """

  import Ecto.Query

  alias Viva.Avatars.Avatar
  alias Viva.Avatars.InternalState
  alias Viva.Avatars.Memory
  alias Viva.Avatars.Personality
  alias Viva.Avatars.Visuals
  alias Viva.Repo

  # === Types ===

  @type avatar_id :: Ecto.UUID.t()
  @type user_id :: Ecto.UUID.t()

  # === Avatar CRUD ===

  @spec list_avatars(keyword()) :: [Avatar.t()]
  def list_avatars(opts \\ []) do
    Avatar
    |> maybe_filter_active(opts[:active])
    |> maybe_filter_user(opts[:user_id])
    |> maybe_preload(opts[:preload])
    |> Repo.all()
  end

  @spec list_active_avatar_ids() :: [avatar_id()]
  def list_active_avatar_ids do
    Avatar
    |> where([a], a.is_active == true)
    |> select([a], a.id)
    |> Repo.all()
  end

  @spec get_avatar(avatar_id()) :: Avatar.t() | nil
  def get_avatar(id), do: Repo.get(Avatar, id)

  @spec get_avatar!(avatar_id()) :: Avatar.t()
  def get_avatar!(id), do: Repo.get!(Avatar, id)

  @spec get_avatar_by_user(user_id(), avatar_id()) :: Avatar.t() | nil
  def get_avatar_by_user(user_id, avatar_id) do
    Avatar
    |> where([a], a.id == ^avatar_id and a.user_id == ^user_id)
    |> Repo.one()
  end

  @spec create_avatar(user_id(), map()) :: {:ok, Avatar.t()} | {:error, Ecto.Changeset.t()}
  def create_avatar(user_id, attrs) do
    %Avatar{}
    |> Avatar.create_changeset(Map.put(attrs, :user_id, user_id))
    |> Repo.insert()
  end

  @spec update_avatar(Avatar.t(), map()) :: {:ok, Avatar.t()} | {:error, Ecto.Changeset.t()}
  def update_avatar(%Avatar{} = avatar, attrs) do
    avatar
    |> Avatar.changeset(attrs)
    |> Repo.update()
  end

  @spec delete_avatar(Avatar.t()) :: {:ok, Avatar.t()} | {:error, Ecto.Changeset.t()}
  def delete_avatar(%Avatar{} = avatar) do
    Repo.delete(avatar)
  end

  @spec update_internal_state(avatar_id(), InternalState.t()) ::
          {:ok, Avatar.t()} | {:error, Ecto.Changeset.t()}
  def update_internal_state(avatar_id, %InternalState{} = state) do
    avatar = get_avatar!(avatar_id)
    # Convert everything to plain maps/strings for the changeset cast
    params = %{"internal_state" => sanitize_for_db(state)}

    avatar
    |> Avatar.changeset(params)
    |> Repo.update()
  end

  @spec mark_active(avatar_id()) :: {non_neg_integer(), nil}
  def mark_active(avatar_id) do
    now = DateTime.utc_now(:second)

    Avatar
    |> where([a], a.id == ^avatar_id)
    |> Repo.update_all(
      set: [
        last_active_at: now
      ]
    )
  end

  # === Memories ===

  @spec list_memories(avatar_id(), keyword()) :: [Memory.t()]
  def list_memories(avatar_id, opts \\ []) do
    limit = Keyword.get(opts, :limit, 20)
    type = Keyword.get(opts, :type)

    avatar_id
    |> Memory.for_avatar()
    |> maybe_filter_memory_type(type)
    |> Memory.by_importance()
    |> limit(^limit)
    |> Repo.all()
  end

  @spec create_memory(avatar_id(), map()) :: {:ok, Memory.t()} | {:error, Ecto.Changeset.t()}
  def create_memory(avatar_id, attrs) do
    %Memory{}
    |> Memory.changeset(Map.put(attrs, :avatar_id, avatar_id))
    |> Repo.insert()
  end

  @spec search_memories(avatar_id(), [float()], pos_integer()) :: [Memory.t()]
  def search_memories(avatar_id, query_embedding, limit \\ 10) do
    avatar_id
    |> Memory.for_avatar()
    |> Memory.similar_to(query_embedding, limit)
    |> Repo.all()
  end

  @spec decay_old_memories(avatar_id()) :: :ok
  def decay_old_memories(avatar_id) do
    now = DateTime.utc_now()
    # Only process memories older than 24h
    cutoff = DateTime.add(now, -24, :hour)

    query =
      from m in Memory,
        where: m.avatar_id == ^avatar_id and m.inserted_at < ^cutoff,
        # Select all fields required for potential re-insertion (upsert)
        # or just enough for calculation if we used individual updates.
        # For bulk upsert, we need all non-nullable fields: avatar_id, content.
        select: %{
          id: m.id,
          avatar_id: m.avatar_id,
          content: m.content,
          inserted_at: m.inserted_at,
          strength: m.strength,
          importance: m.importance,
          times_recalled: m.times_recalled
        }

    Repo.transaction(fn ->
      query
      |> Repo.stream()
      |> Stream.chunk_every(500)
      |> Enum.each(fn batch ->
        {to_delete_ids, to_update} =
          Enum.reduce(batch, {[], []}, &classify_memory(&1, &2, now))

        unless to_delete_ids == [] do
          Repo.delete_all(from m in Memory, where: m.id in ^to_delete_ids)
        end

        unless to_update == [] do
          Repo.insert_all(Memory, to_update,
            on_conflict: {:replace, [:strength, :updated_at]},
            conflict_target: [:id]
          )
        end
      end)
    end)

    :ok
  end

  # === Personality Generation ===

  @spec generate_random_personality() :: Personality.t()
  def generate_random_personality do
    Personality.random()
  end

  # === Visual Generation (NIM) ===

  @doc """
  Generate complete visual package for avatar.
  Includes profile image and expression variations.

  ## Options

  - `:include_3d` - Also generate 3D model (expensive)
  - `:style` - Art style: "realistic", "anime", "illustration"
  """
  @spec generate_visuals(Avatar.t(), keyword()) :: {:ok, Avatar.t()} | {:error, term()}
  def generate_visuals(avatar, opts \\ []) do
    Visuals.generate_complete(avatar, opts)
  end

  @doc """
  Generate profile image for avatar.
  """
  @spec generate_profile_image(Avatar.t(), keyword()) :: {:ok, Avatar.t()} | {:error, term()}
  def generate_profile_image(avatar, opts \\ []) do
    Visuals.generate_profile(avatar, opts)
  end

  @doc """
  Generate 3D model for avatar.
  """
  @spec generate_3d_avatar(Avatar.t(), keyword()) :: {:ok, Avatar.t()} | {:error, term()}
  def generate_3d_avatar(avatar, opts \\ []) do
    Visuals.generate_3d_model(avatar, opts)
  end

  @doc """
  Update avatar's expression based on current emotion.
  Returns the avatar and appropriate image URL.
  """
  @spec update_expression(Avatar.t(), atom()) :: {:ok, Avatar.t(), String.t()} | {:error, term()}
  def update_expression(avatar, emotion) do
    Visuals.update_expression(avatar, emotion)
  end

  @doc """
  Get current expression image URL for avatar.
  """
  @spec get_expression_image(Avatar.t()) :: String.t() | nil
  def get_expression_image(avatar) do
    Visuals.get_expression_image(avatar, avatar.current_expression)
  end

  @doc """
  Generate lipsync animation from audio.
  """
  @spec generate_lipsync(Avatar.t(), binary(), keyword()) :: {:ok, map()} | {:error, term()}
  def generate_lipsync(avatar, audio_data, opts \\ []) do
    Visuals.generate_lipsync(avatar, audio_data, opts)
  end

  # === Private Helpers ===

  defp maybe_filter_active(query, nil), do: query
  defp maybe_filter_active(query, true), do: where(query, [a], a.is_active == true)
  defp maybe_filter_active(query, false), do: where(query, [a], a.is_active == false)

  defp maybe_filter_user(query, nil), do: query
  defp maybe_filter_user(query, user_id), do: where(query, [a], a.user_id == ^user_id)

  defp maybe_preload(query, nil), do: query
  defp maybe_preload(query, preloads), do: preload(query, ^preloads)

  defp maybe_filter_memory_type(query, nil), do: query
  defp maybe_filter_memory_type(query, type), do: Memory.of_type(query, type)

  defp classify_memory(data, {del_acc, upd_acc}, now) do
    memory = struct(Memory, data)
    hours_passed = DateTime.diff(now, memory.inserted_at, :hour)
    decayed = Memory.decay_strength(memory, hours_passed)

    if decayed.strength < 0.1 do
      {[memory.id | del_acc], upd_acc}
    else
      update_map = %{
        id: memory.id,
        avatar_id: memory.avatar_id,
        content: memory.content,
        strength: decayed.strength,
        updated_at: DateTime.truncate(now, :second)
      }

      {del_acc, [update_map | upd_acc]}
    end
  end

  # Recursively sanitize nested structs/maps for DB storage
  defp sanitize_for_db(%DateTime{} = dt), do: DateTime.to_iso8601(dt)
  defp sanitize_for_db(%Date{} = d), do: Date.to_iso8601(d)
  defp sanitize_for_db(%Time{} = t), do: Time.to_iso8601(t)

  defp sanitize_for_db(%_{} = struct) do
    struct
    |> Map.from_struct()
    |> sanitize_for_db()
  end

  defp sanitize_for_db(%{} = map) do
    Map.new(map, fn {k, v} -> {k, sanitize_for_db(v)} end)
  end

  defp sanitize_for_db(list) when is_list(list) do
    Enum.map(list, &sanitize_for_db/1)
  end

  defp sanitize_for_db(value), do: value
end
