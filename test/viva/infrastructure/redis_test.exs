defmodule Viva.Infrastructure.RedisTest do
  use ExUnit.Case, async: true

  alias Viva.Infrastructure.Redis

  describe "module structure" do
    test "exports expected functions" do
      functions = Redis.__info__(:functions)

      assert {:child_spec, 1} in functions
      assert {:start_link, 0} in functions
      assert {:set_avatar_view_state, 2} in functions
      assert {:get_avatar_view_state, 1} in functions
    end
  end

  describe "child_spec/1" do
    test "returns valid child spec" do
      spec = Redis.child_spec([])

      assert spec.id == Redis
      assert spec.start == {Redis, :start_link, []}
      assert spec.type == :worker
      assert spec.restart == :permanent
      assert spec.shutdown == 500
    end
  end
end
