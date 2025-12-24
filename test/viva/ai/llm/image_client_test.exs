defmodule Viva.AI.LLM.ImageClientTest do
  use ExUnit.Case, async: true

  alias Viva.AI.LLM.ImageClient

  describe "module structure" do
    test "exports expected functions" do
      functions = ImageClient.__info__(:functions)

      assert {:generate_profile, 1} in functions or {:generate_profile, 2} in functions
      assert {:generate_expression, 2} in functions or {:generate_expression, 3} in functions
      assert {:edit_image, 2} in functions or {:edit_image, 3} in functions
      assert {:change_expression, 2} in functions or {:change_expression, 3} in functions
      assert {:stylize, 2} in functions or {:stylize, 3} in functions

      assert {:generate_expression_pack, 1} in functions or
               {:generate_expression_pack, 2} in functions or
               {:generate_expression_pack, 3} in functions
    end

    test "module is loaded correctly" do
      assert {:module, ImageClient} = Code.ensure_loaded(ImageClient)
    end
  end
end
