defmodule Viva.Repo.Migrations.CreateMessages do
  use Ecto.Migration

  def change do
    create table(:messages, primary_key: false) do
      add :id, :binary_id, primary_key: true

      add :conversation_id, references(:conversations, type: :binary_id, on_delete: :delete_all),
        null: false

      add :speaker_id, references(:avatars, type: :binary_id, on_delete: :nilify_all)

      # Content
      add :content, :text, null: false
      add :content_type, :string, default: "text"

      # Emotional context
      add :emotional_tone, :string
      add :emotions, :map, default: %{}

      # Audio (if voice message)
      add :audio_url, :string
      add :duration_seconds, :float

      # Timing
      add :timestamp, :utc_datetime, null: false

      # Generation metadata
      add :generated_at, :bigint

      timestamps(type: :utc_datetime)
    end

    create index(:messages, [:conversation_id])
    create index(:messages, [:speaker_id])
    create index(:messages, [:timestamp])
  end
end
