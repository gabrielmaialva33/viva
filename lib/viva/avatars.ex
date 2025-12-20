defmodule Viva.Avatars do
  @moduledoc """
  The Avatars context.
  Manages avatar lifecycle, memories, and state.
  """

  import Ecto.Query
  alias Viva.Repo
  alias Viva.Avatars.{Avatar, Memory, Personality, InternalState, Visuals}

  # === Avatar CRUD ===

  def list_avatars(opts \\ []) do
    Avatar
    |> maybe_filter_active(opts[:active])
    |> maybe_filter_user(opts[:user_id])
    |> maybe_preload(opts[:preload])
    |> Repo.all()
  end

  def list_active_avatar_ids do
    Avatar
    |> where([a], a.is_active == true)
    |> select([a], a.id)
    |> Repo.all()
  end

  def get_avatar(id), do: Repo.get(Avatar, id)

  def get_avatar!(id), do: Repo.get!(Avatar, id)

  def get_avatar_by_user(user_id, avatar_id) do
    Avatar
    |> where([a], a.id == ^avatar_id and a.user_id == ^user_id)
    |> Repo.one()
  end

  def create_avatar(user_id, attrs) do
    %Avatar{}
    |> Avatar.create_changeset(Map.put(attrs, :user_id, user_id))
    |> Repo.insert()
  end

  def update_avatar(%Avatar{} = avatar, attrs) do
    avatar
    |> Avatar.changeset(attrs)
    |> Repo.update()
  end

  def delete_avatar(%Avatar{} = avatar) do
    Repo.delete(avatar)
  end

  def update_internal_state(avatar_id, %InternalState{} = state) do
    Avatar
    |> where([a], a.id == ^avatar_id)
    |> Repo.update_all(
      set: [
        internal_state: Map.from_struct(state),
        updated_at: DateTime.utc_now()
      ]
    )
  end

  def mark_active(avatar_id) do
    Avatar
    |> where([a], a.id == ^avatar_id)
    |> Repo.update_all(
      set: [
        last_active_at: DateTime.utc_now()
      ]
    )
  end

  # === Memories ===

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

  def create_memory(avatar_id, attrs) do
    %Memory{}
    |> Memory.changeset(Map.put(attrs, :avatar_id, avatar_id))
    |> Repo.insert()
  end

  def search_memories(avatar_id, query_embedding, limit \\ 10) do
    avatar_id
    |> Memory.for_avatar()
    |> Memory.similar_to(query_embedding, limit)
    |> Repo.all()
  end

  def decay_old_memories(avatar_id) do
    now = DateTime.utc_now()

    avatar_id
    |> Memory.for_avatar()
    |> Repo.all()
    |> Enum.each(fn memory ->
      hours_passed = DateTime.diff(now, memory.inserted_at, :hour)

      if hours_passed > 24 do
        decayed = Memory.decay_strength(memory, hours_passed)

        if decayed.strength < 0.1 do
          Repo.delete(memory)
        else
          memory
          |> Ecto.Changeset.change(strength: decayed.strength)
          |> Repo.update()
        end
      end
    end)
  end

  # === Personality Generation ===

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
  def generate_visuals(avatar, opts \\ []) do
    Visuals.generate_complete(avatar, opts)
  end

  @doc """
  Generate profile image for avatar.
  """
  def generate_profile_image(avatar, opts \\ []) do
    Visuals.generate_profile(avatar, opts)
  end

  @doc """
  Generate 3D model for avatar.
  """
  def generate_3d_avatar(avatar, opts \\ []) do
    Visuals.generate_3d_model(avatar, opts)
  end

  @doc """
  Update avatar's expression based on current emotion.
  Returns the avatar and appropriate image URL.
  """
  def update_expression(avatar, emotion) do
    Visuals.update_expression(avatar, emotion)
  end

  @doc """
  Get current expression image URL for avatar.
  """
  def get_expression_image(avatar) do
    Visuals.get_expression_image(avatar, avatar.current_expression)
  end

  @doc """
  Generate lipsync animation from audio.
  """
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
end
