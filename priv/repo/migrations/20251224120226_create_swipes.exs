defmodule Viva.Repo.Migrations.CreateSwipes do
  use Ecto.Migration

  def change do
    create table(:swipes, primary_key: false) do
      add :id, :binary_id, primary_key: true

      add :actor_avatar_id, references(:avatars, type: :binary_id, on_delete: :delete_all),
        null: false

      add :target_avatar_id, references(:avatars, type: :binary_id, on_delete: :delete_all),
        null: false

      add :action, :string, null: false
      add :metadata, :map, default: %{}

      timestamps(type: :utc_datetime)
    end

    create unique_index(:swipes, [:actor_avatar_id, :target_avatar_id],
             name: :swipes_actor_target_unique
           )

    create index(:swipes, [:actor_avatar_id])

    # Partial index for fast reciprocal like checking
    create index(:swipes, [:target_avatar_id, :actor_avatar_id],
             where: "action IN ('like', 'superlike')",
             name: :swipes_reciprocal_likes_index
           )
  end
end
