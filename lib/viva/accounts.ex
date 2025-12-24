defmodule Viva.Accounts do
  @moduledoc """
  The Accounts context.
  Manages user accounts and authentication.
  """

  import Ecto.Query

  alias Viva.Accounts.User
  alias Viva.Repo

  # === Types ===

  @type user_id :: Ecto.UUID.t()

  # === User CRUD ===

  @spec list_users(keyword()) :: [User.t()]
  def list_users(opts \\ []) do
    active_only = Keyword.get(opts, :active, false)
    limit = Keyword.get(opts, :limit, 50)

    User
    |> maybe_filter_active_users(active_only)
    |> limit(^limit)
    |> Repo.all()
  end

  @spec get_user(user_id()) :: User.t() | nil
  def get_user(id), do: Repo.get(User, id)

  @spec get_user!(user_id()) :: User.t()
  def get_user!(id), do: Repo.get!(User, id)

  @spec get_user_by_email(String.t()) :: User.t() | nil
  def get_user_by_email(email) do
    email
    |> User.by_email()
    |> Repo.one()
  end

  @spec get_user_by_username(String.t()) :: User.t() | nil
  def get_user_by_username(username) do
    username
    |> User.by_username()
    |> Repo.one()
  end

  @spec create_user(map()) :: {:ok, User.t()} | {:error, Ecto.Changeset.t()}
  def create_user(attrs) do
    %User{}
    |> User.registration_changeset(attrs)
    |> Repo.insert()
  end

  @spec update_user(User.t(), map()) :: {:ok, User.t()} | {:error, Ecto.Changeset.t()}
  def update_user(%User{} = user, attrs) do
    user
    |> User.changeset(attrs)
    |> Repo.update()
  end

  @spec delete_user(User.t()) :: {:ok, User.t()} | {:error, Ecto.Changeset.t()}
  def delete_user(%User{} = user) do
    Repo.delete(user)
  end

  # === Authentication ===

  @spec authenticate_by_email(String.t(), String.t()) ::
          {:ok, User.t()} | {:error, :invalid_credentials}
  def authenticate_by_email(email, password) do
    user = get_user_by_email(email)

    if User.verify_password(user, password) do
      {:ok, user}
    else
      {:error, :invalid_credentials}
    end
  end

  @spec authenticate_by_username(String.t(), String.t()) ::
          {:ok, User.t()} | {:error, :invalid_credentials}
  def authenticate_by_username(username, password) do
    user = get_user_by_username(username)

    if User.verify_password(user, password) do
      {:ok, user}
    else
      {:error, :invalid_credentials}
    end
  end

  # === User Status ===

  @spec mark_user_seen(User.t()) :: {:ok, User.t()} | {:error, Ecto.Changeset.t()}
  def mark_user_seen(%User{} = user) do
    user
    |> Ecto.Changeset.change(last_seen_at: DateTime.utc_now(:second))
    |> Repo.update()
  end

  @spec deactivate_user(User.t()) :: {:ok, User.t()} | {:error, Ecto.Changeset.t()}
  def deactivate_user(%User{} = user) do
    user
    |> Ecto.Changeset.change(is_active: false)
    |> Repo.update()
  end

  @spec verify_user(User.t()) :: {:ok, User.t()} | {:error, Ecto.Changeset.t()}
  def verify_user(%User{} = user) do
    user
    |> Ecto.Changeset.change(is_verified: true)
    |> Repo.update()
  end

  # === User Preferences ===

  @spec update_preferences(User.t(), map()) :: {:ok, User.t()} | {:error, Ecto.Changeset.t()}
  def update_preferences(%User{} = user, preferences) do
    merged = Map.merge(user.preferences || %{}, preferences)

    user
    |> Ecto.Changeset.change(preferences: merged)
    |> Repo.update()
  end

  @spec get_preference(User.t(), atom() | String.t(), term()) :: term()
  def get_preference(%User{} = user, key, default \\ nil) do
    get_in(user.preferences || %{}, [key]) || default
  end

  # === User Avatars ===

  @spec list_user_avatars(user_id(), keyword()) :: [Viva.Avatars.Avatar.t()]
  def list_user_avatars(user_id, opts \\ []) do
    opts_with_user = Keyword.put(opts, :user_id, user_id)
    Viva.Avatars.list_avatars(opts_with_user)
  end

  @spec get_user_avatar(user_id(), Ecto.UUID.t()) :: Viva.Avatars.Avatar.t() | nil
  def get_user_avatar(user_id, avatar_id) do
    Viva.Avatars.get_avatar_by_user(user_id, avatar_id)
  end

  # === Private Helpers ===

  defp maybe_filter_active_users(query, false), do: query
  defp maybe_filter_active_users(query, true), do: where(query, [u], u.is_active == true)
end
