defmodule Viva.Repo.Migrations.CreateConversations do
  use Ecto.Migration

  def change do
    create table(:conversations, primary_key: false) do
      add :id, :binary_id, primary_key: true

      add :avatar_a_id, references(:avatars, type: :binary_id, on_delete: :delete_all),
        null: false

      add :avatar_b_id, references(:avatars, type: :binary_id, on_delete: :delete_all),
        null: false

      add :type, :string, default: "interactive"
      add :status, :string, default: "active"
      add :context, :map, default: %{}
      add :topic, :string
      add :started_at, :utc_datetime
      add :ended_at, :utc_datetime
      add :duration_minutes, :integer
      add :message_count, :integer, default: 0
      add :analysis, :map

      timestamps(type: :utc_datetime)
    end

    create index(:conversations, [:avatar_a_id, :status])
    create index(:conversations, [:avatar_b_id, :status])
    create index(:conversations, [:status, :type])
    create index(:conversations, [:started_at])
  end
end
