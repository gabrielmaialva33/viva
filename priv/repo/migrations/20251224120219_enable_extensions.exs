defmodule Viva.Repo.Migrations.EnableExtensions do
  use Ecto.Migration

  def change do
    execute "CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\""
    execute "CREATE EXTENSION IF NOT EXISTS vector"
  end
end
