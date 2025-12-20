defmodule Viva.Repo.Migrations.CreateUsers do
  use Ecto.Migration

  def change do
    create table(:users, primary_key: false) do
      add :id, :binary_id, primary_key: true
      add :email, :string, null: false
      add :username, :string, null: false
      add :display_name, :string
      add :avatar_url, :string
      add :hashed_password, :string

      # Profile
      add :bio, :text
      add :date_of_birth, :date
      add :timezone, :string, default: "UTC"
      add :locale, :string, default: "en"

      # Status
      add :is_active, :boolean, default: true
      add :is_verified, :boolean, default: false
      add :last_seen_at, :utc_datetime

      # Preferences
      add :preferences, :map, default: %{}

      timestamps(type: :utc_datetime)
    end

    create unique_index(:users, [:email])
    create unique_index(:users, [:username])
    create index(:users, [:is_active])
  end
end
