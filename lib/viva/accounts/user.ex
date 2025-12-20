defmodule Viva.Accounts.User do
  @moduledoc """
  User schema for avatar owners.
  """
  use Ecto.Schema
  import Ecto.Changeset
  import Ecto.Query

  alias Viva.Avatars.Avatar

  @primary_key {:id, :binary_id, autogenerate: true}
  @foreign_key_type :binary_id
  schema "users" do
    field :email, :string
    field :username, :string
    field :display_name, :string
    field :avatar_url, :string
    field :hashed_password, :string
    field :password, :string, virtual: true, redact: true

    # Profile
    field :bio, :string
    field :date_of_birth, :date
    field :timezone, :string, default: "UTC"
    field :locale, :string, default: "en"

    # Status
    field :is_active, :boolean, default: true
    field :is_verified, :boolean, default: false
    field :last_seen_at, :utc_datetime

    # Preferences
    field :preferences, :map, default: %{}

    # Avatars owned by this user
    has_many :avatars, Avatar

    timestamps(type: :utc_datetime)
  end

  def changeset(user, attrs) do
    user
    |> cast(attrs, [
      :email,
      :username,
      :display_name,
      :avatar_url,
      :bio,
      :date_of_birth,
      :timezone,
      :locale,
      :is_active,
      :last_seen_at,
      :preferences
    ])
    |> validate_required([:email, :username])
    |> validate_email()
    |> validate_username()
  end

  def registration_changeset(user, attrs) do
    user
    |> changeset(attrs)
    |> cast(attrs, [:password])
    |> validate_required([:password])
    |> validate_password()
    |> hash_password()
  end

  defp validate_email(changeset) do
    changeset
    |> validate_required([:email])
    |> validate_format(:email, ~r/^[^\s]+@[^\s]+$/, message: "must be a valid email")
    |> validate_length(:email, max: 160)
    |> unsafe_validate_unique(:email, Viva.Repo)
    |> unique_constraint(:email)
    |> update_change(:email, &String.downcase/1)
  end

  defp validate_username(changeset) do
    changeset
    |> validate_required([:username])
    |> validate_format(:username, ~r/^[a-zA-Z0-9_]+$/,
      message: "only letters, numbers, and underscores"
    )
    |> validate_length(:username, min: 3, max: 30)
    |> unsafe_validate_unique(:username, Viva.Repo)
    |> unique_constraint(:username)
    |> update_change(:username, &String.downcase/1)
  end

  defp validate_password(changeset) do
    changeset
    |> validate_required([:password])
    |> validate_length(:password, min: 8, max: 72)
    |> validate_format(:password, ~r/[a-z]/, message: "must have at least one lowercase letter")
    |> validate_format(:password, ~r/[A-Z]/, message: "must have at least one uppercase letter")
    |> validate_format(:password, ~r/[0-9]/, message: "must have at least one digit")
  end

  defp hash_password(changeset) do
    case get_change(changeset, :password) do
      nil ->
        changeset

      password ->
        hashed = Bcrypt.hash_pwd_salt(password)
        put_change(changeset, :hashed_password, hashed)
    end
  end

  def verify_password(%__MODULE__{hashed_password: hashed}, password) do
    Bcrypt.verify_pass(password, hashed)
  end

  def verify_password(nil, _password) do
    Bcrypt.no_user_verify()
    false
  end

  # Queries
  def active, do: from(u in __MODULE__, where: u.is_active == true)

  def by_email(email) do
    from(u in __MODULE__, where: u.email == ^String.downcase(email))
  end

  def by_username(username) do
    from(u in __MODULE__, where: u.username == ^String.downcase(username))
  end
end
