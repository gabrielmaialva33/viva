defmodule Viva.Workers.MatchRefreshWorkerTest do
  use Viva.DataCase, async: false
  use Oban.Testing, repo: Viva.Repo

  alias Viva.Workers.MatchRefreshWorker

  describe "scheduling" do
    test "schedule/1 inserts a job" do
      avatar_id = Ecto.UUID.generate()
      {:ok, %Oban.Job{args: %{avatar_id: ^avatar_id}}} = MatchRefreshWorker.schedule(avatar_id)
      assert_enqueued worker: MatchRefreshWorker, args: %{avatar_id: avatar_id}
    end

    test "schedule_all/0 inserts a job without args" do
      {:ok, %Oban.Job{args: %{}}} = MatchRefreshWorker.schedule_all()
      assert_enqueued worker: MatchRefreshWorker, args: %{}
    end
  end

  describe "perform/1" do
    test "processes single avatar" do
      avatar_id = Ecto.UUID.generate()
      # perform calls Viva.Matching.Engine.refresh_matches
      :ok = MatchRefreshWorker.perform(%Oban.Job{args: %{"avatar_id" => avatar_id}})
    end

    test "processes all active avatars" do
      # perform calls Viva.Sessions.Supervisor.list_running_avatars and enqueues more jobs
      :ok = MatchRefreshWorker.perform(%Oban.Job{args: %{}})
    end
  end
end
