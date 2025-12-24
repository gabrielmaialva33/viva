defmodule Viva.Repo.Migrations.CreateAvatars do
  use Ecto.Migration

  def change do
    create table(:avatars, primary_key: false) do
      add :id, :binary_id, primary_key: true
      add :name, :string, null: false
      add :bio, :text
      add :gender, :string
      add :age, :integer, default: 25

      # Visuals
      add :avatar_url, :string
      add :avatar_model_id, :string
      add :profile_image_url, :string
      add :avatar_3d_model_url, :string
      add :current_expression, :string, default: "neutral"
      add :expression_images, :map, default: %{}

      # Core
      add :user_id, references(:users, type: :binary_id, on_delete: :delete_all), null: false
      # Big Five + Enneagram
      add :personality, :map, null: false
      # Bio + Emotional + Cognitive
      add :internal_state, :map, null: false
      add :system_prompt, :text

      # Audio
      add :voice_id, :string
      add :voice_sample_url, :string

      # Stats
      add :total_conversations, :integer, default: 0
      add :total_matches, :integer, default: 0

      # Meta
      add :is_active, :boolean, default: true
      add :last_active_at, :utc_datetime
      add :created_at, :utc_datetime

      timestamps(type: :utc_datetime)
    end

    create index(:avatars, [:user_id, :is_active])
    create index(:avatars, [:name])
    create index(:avatars, [:is_active])
  end
end
