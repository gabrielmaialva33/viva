defmodule Viva.Repo do
  @moduledoc """
  Repository for database interactions.
  """
  use Ecto.Repo,
    otp_app: :viva,
    adapter: Ecto.Adapters.Postgres
end
