defmodule VivaCore.InnerMonologueTest do
  use ExUnit.Case

  alias VivaCore.InnerMonologue

  setup do
    # Start monologue for testing (if not in supervision tree)
    # Since we added it to supervision tree, we might need to stop it first or just use the name
    # But usually unit tests run in :test env where app might be started or not.
    # We'll try to start it with a unique name for isolation if possible, or just use the globally registered one.

    # In integration tests we might rely on the global one.
    # For unit tests, let's just assert on the module functions assuming it's running or start a temporary one.

    # For safety in concurrent tests, we'll try to start a specific linked instance
    # But the module uses a fixed name in start_link.
    # So we'll just check if it's alive or start it.

    pid = Process.whereis(VivaCore.InnerMonologue)

    if pid && Process.alive?(pid) do
      %{pid: pid}
    else
      {:ok, pid} = InnerMonologue.start_link([])
      %{pid: pid}
    end
  end

  @tag :external
  test "generates narrative" do
    narrative = InnerMonologue.generate()
    assert is_binary(narrative)
    assert String.length(narrative) > 5
  end

  @tag :external
  test "history returns recent entries" do
    # Force some generations
    InnerMonologue.generate()
    InnerMonologue.generate()

    history = InnerMonologue.history(2)
    assert length(history) == 2
    assert Enum.all?(history, &Map.has_key?(&1, :narrative))
  end

  test "reflect generates reflection narrative" do
    narrative = InnerMonologue.reflect("teste")
    assert is_binary(narrative)
    assert String.contains?(narrative, "teste")
  end

  test "sets mode" do
    assert InnerMonologue.set_mode(:llm) == :ok
    # Wait a bit for cast
    :timer.sleep(10)
    # Restore
    InnerMonologue.set_mode(:template)
  end
end
