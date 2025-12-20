defmodule Viva.Repo.Migrations.CreateAvatars do
  use Ecto.Migration

  def change do
    create table(:avatars, primary_key: false) do
      add :id, :binary_id, primary_key: true
      add :user_id, references(:users, type: :binary_id, on_delete: :delete_all), null: false

      add :name, :string, null: false
      add :bio, :text
      add :gender, :string
      add :age, :integer, default: 25

      # Visual appearance
      add :avatar_url, :string
      add :avatar_model_id, :string

      # Personality (embedded as JSONB)
      add :personality, :map, null: false

      # Current internal state (embedded as JSONB)
      add :internal_state, :map

      # System prompt for LLM
      add :system_prompt, :text

      # Voice settings
      add :voice_id, :string
      add :voice_sample_url, :string

      # Stats
      add :total_conversations, :integer, default: 0
      add :total_matches, :integer, default: 0

      # Status
      add :is_active, :boolean, default: true
      add :last_active_at, :utc_datetime
      add :created_at, :utc_datetime

      timestamps(type: :utc_datetime)
    end

    create index(:avatars, [:user_id])
    create index(:avatars, [:is_active])
    create index(:avatars, [:last_active_at])
  end
end
