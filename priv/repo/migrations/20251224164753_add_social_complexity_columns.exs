defmodule Viva.Repo.Migrations.AddSocialComplexityColumns do
  use Ecto.Migration

  def change do
    alter table(:avatars) do
      # Uses :map which maps to JSONB in Postgres
      add :social_persona, :map, default: "{}"
      add :moral_flexibility, :float, default: 0.3
    end

    alter table(:relationships) do
      add :social_leverage, :float, default: 0.0
    end
  end
end
