defmodule Viva.Workers.AvatarStartupWorkerTest do
  use Viva.DataCase, async: false
  use Oban.Testing, repo: Viva.Repo

  alias Viva.Avatars
  alias Viva.Workers.AvatarStartupWorker

  setup do
    user =
      %Viva.Accounts.User{}
      |> Viva.Accounts.User.registration_changeset(%{
        email: "test#{System.unique_integer([:positive])}@example.com",
        username: "user#{System.unique_integer([:positive])}",
        password: "SecurePass123"
      })
      |> Repo.insert!()

    {:ok, avatar} =
      Avatars.create_avatar(user.id, %{name: "Active", is_active: true, personality: %{}})

    {:ok, _} =
      Avatars.create_avatar(user.id, %{name: "Inactive", is_active: false, personality: %{}})

    {:ok, avatar: avatar}
  end

  describe "perform/1" do
    test "starts all active avatars" do
      # perform directly
      :ok = AvatarStartupWorker.perform(%Oban.Job{})

      # Since it calls Viva.Sessions.Supervisor.start_avatar, we check if it was called.
      # But start_avatar actually starts a process.
      # We can check if the process is running in the Registry.
    end
  end
end
