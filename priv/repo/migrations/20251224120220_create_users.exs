defmodule Viva.Repo.Migrations.CreateUsers do
  use Ecto.Migration

  def change do
    create table(:users, primary_key: false) do
      add :id, :binary_id, primary_key: true
      add :username, :string, null: false
      add :locale, :string, default: "pt-BR"
      add :email, :string, null: false
      add :display_name, :string
      add :hashed_password, :string, null: false
      add :bio, :text
      add :timezone, :string
      add :is_active, :boolean, default: true
      add :is_verified, :boolean, default: false
      add :preferences, :map, default: %{}

      timestamps(type: :utc_datetime)
    end

    create unique_index(:users, [:email])
    create unique_index(:users, [:username])
    create index(:users, [:is_active])
  end
end
