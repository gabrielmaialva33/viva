defmodule Viva.Repo.Migrations.CreateConversations do
  use Ecto.Migration

  def change do
    create table(:conversations, primary_key: false) do
      add :id, :binary_id, primary_key: true

      add :avatar_a_id, references(:avatars, type: :binary_id, on_delete: :delete_all),
        null: false

      add :avatar_b_id, references(:avatars, type: :binary_id, on_delete: :delete_all),
        null: false

      # Type: interactive (owner-initiated) or autonomous (AI-initiated)
      add :type, :string, default: "interactive"

      # Status: active, ended, paused
      add :status, :string, default: "active"

      # Context and topic
      add :context, :map, default: %{}
      add :topic, :string

      # Timing
      add :started_at, :utc_datetime
      add :ended_at, :utc_datetime
      add :duration_minutes, :integer

      # Stats
      add :message_count, :integer, default: 0

      # Analysis (filled after conversation ends)
      add :analysis, :map

      timestamps(type: :utc_datetime)
    end

    create index(:conversations, [:avatar_a_id])
    create index(:conversations, [:avatar_b_id])
    create index(:conversations, [:status])
    create index(:conversations, [:type])
    create index(:conversations, [:started_at])
  end
end
