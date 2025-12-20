defmodule Viva.Accounts do
  @moduledoc """
  The Accounts context.
  Manages user accounts and authentication.
  """

  import Ecto.Query
  alias Viva.Repo
  alias Viva.Accounts.User

  # === User CRUD ===

  def list_users(opts \\ []) do
    active_only = Keyword.get(opts, :active, false)
    limit = Keyword.get(opts, :limit, 50)

    User
    |> maybe_filter_active_users(active_only)
    |> limit(^limit)
    |> Repo.all()
  end

  def get_user(id), do: Repo.get(User, id)

  def get_user!(id), do: Repo.get!(User, id)

  def get_user_by_email(email) do
    User.by_email(email)
    |> Repo.one()
  end

  def get_user_by_username(username) do
    User.by_username(username)
    |> Repo.one()
  end

  def create_user(attrs) do
    %User{}
    |> User.registration_changeset(attrs)
    |> Repo.insert()
  end

  def update_user(%User{} = user, attrs) do
    user
    |> User.changeset(attrs)
    |> Repo.update()
  end

  def delete_user(%User{} = user) do
    Repo.delete(user)
  end

  # === Authentication ===

  def authenticate_by_email(email, password) do
    user = get_user_by_email(email)

    if User.verify_password(user, password) do
      {:ok, user}
    else
      {:error, :invalid_credentials}
    end
  end

  def authenticate_by_username(username, password) do
    user = get_user_by_username(username)

    if User.verify_password(user, password) do
      {:ok, user}
    else
      {:error, :invalid_credentials}
    end
  end

  # === User Status ===

  def mark_user_seen(%User{} = user) do
    user
    |> Ecto.Changeset.change(last_seen_at: DateTime.utc_now())
    |> Repo.update()
  end

  def deactivate_user(%User{} = user) do
    user
    |> Ecto.Changeset.change(is_active: false)
    |> Repo.update()
  end

  def verify_user(%User{} = user) do
    user
    |> Ecto.Changeset.change(is_verified: true)
    |> Repo.update()
  end

  # === User Preferences ===

  def update_preferences(%User{} = user, preferences) do
    merged = Map.merge(user.preferences || %{}, preferences)

    user
    |> Ecto.Changeset.change(preferences: merged)
    |> Repo.update()
  end

  def get_preference(%User{} = user, key, default \\ nil) do
    get_in(user.preferences || %{}, [key]) || default
  end

  # === User Avatars ===

  def list_user_avatars(user_id, opts \\ []) do
    Viva.Avatars.list_avatars(Keyword.put(opts, :user_id, user_id))
  end

  def get_user_avatar(user_id, avatar_id) do
    Viva.Avatars.get_avatar_by_user(user_id, avatar_id)
  end

  # === Private Helpers ===

  defp maybe_filter_active_users(query, false), do: query
  defp maybe_filter_active_users(query, true), do: where(query, [u], u.is_active == true)
end
