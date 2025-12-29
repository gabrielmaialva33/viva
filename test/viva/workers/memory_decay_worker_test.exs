defmodule Viva.Workers.MemoryDecayWorkerTest do
  use Viva.DataCase, async: false
  use Oban.Testing, repo: Viva.Repo

  alias Viva.Workers.MemoryDecayWorker

  describe "scheduling" do
    test "schedule/1" do
      avatar_id = Ecto.UUID.generate()
      {:ok, _} = MemoryDecayWorker.schedule(avatar_id)
      assert_enqueued worker: MemoryDecayWorker, args: %{avatar_id: avatar_id}
    end
  end

  describe "perform/1" do
    test "processes single avatar" do
      avatar_id = Ecto.UUID.generate()
      # perform calls Viva.Avatars.decay_old_memories
      :ok = MemoryDecayWorker.perform(%Oban.Job{args: %{"avatar_id" => avatar_id}})
    end
  end
end
