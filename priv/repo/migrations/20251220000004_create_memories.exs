defmodule Viva.Repo.Migrations.CreateMemories do
  use Ecto.Migration

  def change do
    create table(:memories, primary_key: false) do
      add :id, :binary_id, primary_key: true
      add :avatar_id, references(:avatars, type: :binary_id, on_delete: :delete_all), null: false

      # Memory content
      add :content, :text, null: false
      add :summary, :string

      # Type of memory
      add :type, :string, default: "interaction"

      # Participants
      add :participant_ids, {:array, :binary_id}, default: []

      # Emotional context
      add :emotions_felt, :map, default: %{}

      # Importance and strength
      add :importance, :float, default: 0.5
      add :strength, :float, default: 1.0

      # Vector embedding for semantic search (1024 dimensions for nv-embedqa-e5-v5)
      add :embedding, :vector, size: 1024

      # Recall tracking
      add :times_recalled, :integer, default: 0
      add :last_recalled_at, :utc_datetime

      # Metadata
      add :context, :map, default: %{}

      timestamps(type: :utc_datetime)
    end

    create index(:memories, [:avatar_id])
    create index(:memories, [:type])
    create index(:memories, [:importance])
    create index(:memories, [:inserted_at])

    # Vector similarity search index (HNSW for fast approximate search)
    execute """
            CREATE INDEX memories_embedding_idx ON memories
            USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64)
            """,
            "DROP INDEX IF EXISTS memories_embedding_idx"
  end
end
