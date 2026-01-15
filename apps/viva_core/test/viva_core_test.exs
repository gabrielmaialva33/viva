defmodule VivaCoreTest do
  use ExUnit.Case
  doctest VivaCore

  test "greets the world" do
    assert VivaCore.hello() == :world
  end
end
