defmodule VivaWeb.UserSocketTest do
  use ExUnit.Case, async: true

  alias VivaWeb.UserSocket

  describe "module structure" do
    test "exports expected functions" do
      functions = UserSocket.__info__(:functions)

      # Phoenix.Socket callbacks
      assert {:connect, 3} in functions
      assert {:id, 1} in functions
    end
  end

  describe "connect/3" do
    test "returns error without token" do
      socket = %Phoenix.Socket{}
      result = UserSocket.connect(%{}, socket, %{})

      assert result == :error
    end

    test "returns error with invalid token" do
      socket = %Phoenix.Socket{}
      result = UserSocket.connect(%{"token" => "invalid"}, socket, %{})

      assert result == :error
    end
  end
end
