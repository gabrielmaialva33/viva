defmodule Viva.Repo.Migrations.CreateRelationships do
  use Ecto.Migration

  def change do
    create table(:relationships, primary_key: false) do
      add :id, :binary_id, primary_key: true

      add :avatar_a_id, references(:avatars, type: :binary_id, on_delete: :delete_all),
        null: false

      add :avatar_b_id, references(:avatars, type: :binary_id, on_delete: :delete_all),
        null: false

      # Core metrics (0.0 to 1.0)
      add :familiarity, :float, default: 0.0
      add :trust, :float, default: 0.5
      add :affection, :float, default: 0.0
      add :attraction, :float, default: 0.0
      add :compatibility_score, :float

      # Relationship status
      add :status, :string, default: "strangers"

      # Individual feelings (embedded as JSONB)
      add :a_feelings, :map
      add :b_feelings, :map

      # Interaction history
      add :first_interaction_at, :utc_datetime
      add :last_interaction_at, :utc_datetime
      add :interaction_count, :integer, default: 0
      add :shared_memories_count, :integer, default: 0
      add :total_conversation_minutes, :integer, default: 0

      # Milestones
      add :milestones, {:array, :map}, default: []

      # Conflict tracking
      add :unresolved_conflicts, :integer, default: 0
      add :last_conflict_at, :utc_datetime

      timestamps(type: :utc_datetime)
    end

    create index(:relationships, [:avatar_a_id])
    create index(:relationships, [:avatar_b_id])
    create index(:relationships, [:status])
    create index(:relationships, [:last_interaction_at])

    # Ensure unique pairs (order independent)
    create unique_index(:relationships, [:avatar_a_id, :avatar_b_id],
             name: :relationships_unique_pair
           )
  end
end
