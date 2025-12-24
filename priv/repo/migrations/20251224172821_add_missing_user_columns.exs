defmodule Viva.Repo.Migrations.AddMissingUserColumns do
  use Ecto.Migration

  def change do
    alter table(:users) do
      add :avatar_url, :string
      add :date_of_birth, :date
      add :last_seen_at, :utc_datetime
    end
  end
end
