defmodule Viva.Repo.Migrations.CreateRelationships do
  use Ecto.Migration

  def change do
    create table(:relationships, primary_key: false) do
      add :id, :binary_id, primary_key: true

      add :avatar_a_id, references(:avatars, type: :binary_id, on_delete: :delete_all),
        null: false

      add :avatar_b_id, references(:avatars, type: :binary_id, on_delete: :delete_all),
        null: false

      add :familiarity, :float, default: 0.0
      add :trust, :float, default: 0.5
      add :affection, :float, default: 0.0
      add :attraction, :float, default: 0.0
      add :compatibility_score, :float
      add :status, :string, default: "strangers"
      add :a_feelings, :map
      add :b_feelings, :map
      add :first_interaction_at, :utc_datetime
      add :last_interaction_at, :utc_datetime
      add :interaction_count, :integer, default: 0
      add :shared_memories_count, :integer, default: 0
      add :total_conversation_minutes, :integer, default: 0
      add :milestones, {:array, :map}, default: []
      add :unresolved_conflicts, :integer, default: 0
      add :last_conflict_at, :utc_datetime
      add :matched_at, :utc_datetime

      timestamps(type: :utc_datetime)
    end

    create unique_index(:relationships, [:avatar_a_id, :avatar_b_id],
             name: :relationships_unique_pair
           )

    create index(:relationships, [:avatar_a_id, :status])
    create index(:relationships, [:avatar_b_id, :status])
    create index(:relationships, [:compatibility_score])
    create index(:relationships, [:affection])
    create index(:relationships, [:attraction])

    create index(:relationships, [:status, :familiarity],
             name: :relationships_status_familiarity_idx
           )
  end
end
