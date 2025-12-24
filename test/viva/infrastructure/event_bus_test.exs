defmodule Viva.Infrastructure.EventBusTest do
  use ExUnit.Case, async: true

  alias Viva.Infrastructure.EventBus

  describe "module structure" do
    test "exports expected functions" do
      functions = EventBus.__info__(:functions)

      assert {:publish_thought, 1} in functions
    end
  end
end
