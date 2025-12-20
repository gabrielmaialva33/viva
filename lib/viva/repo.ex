defmodule Viva.Repo do
  use Ecto.Repo,
    otp_app: :viva,
    adapter: Ecto.Adapters.Postgres
end
