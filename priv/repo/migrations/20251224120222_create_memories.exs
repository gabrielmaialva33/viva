defmodule Viva.Repo.Migrations.CreateMemories do
  use Ecto.Migration

  def change do
    create table(:memories, primary_key: false) do
      add :id, :binary_id, primary_key: true
      add :avatar_id, references(:avatars, type: :binary_id, on_delete: :delete_all), null: false
      add :content, :text, null: false
      add :summary, :string
      add :type, :string, default: "interaction"
      add :participant_ids, {:array, :binary_id}, default: []
      add :emotions_felt, :map, default: %{}
      add :importance, :float, default: 0.5
      add :strength, :float, default: 1.0
      # nv-embedqa-e5-v5
      add :embedding, :vector, size: 1024
      add :times_recalled, :integer, default: 0
      add :last_recalled_at, :utc_datetime
      add :context, :map, default: %{}

      timestamps(type: :utc_datetime)
    end

    create index(:memories, [:avatar_id, :type])

    create index(:memories, [:avatar_id, :importance, :strength],
             name: :memories_avatar_priority_idx
           )

    create index(:memories, [:strength])

    execute "CREATE INDEX memories_participant_ids_idx ON memories USING GIN (participant_ids)"

    execute "CREATE INDEX memories_embedding_idx ON memories USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)"
  end
end
