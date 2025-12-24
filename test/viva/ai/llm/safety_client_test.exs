defmodule Viva.AI.LLM.SafetyClientTest do
  use ExUnit.Case, async: true

  alias Viva.AI.LLM.SafetyClient

  describe "categories/0" do
    test "returns all safety categories" do
      categories = SafetyClient.categories()

      assert is_list(categories)
      assert :violence in categories
      assert :hate_speech in categories
      assert :sexual_content in categories
      assert :harassment in categories
      assert :self_harm in categories
      assert :dangerous_activities in categories
      assert :misinformation in categories
      assert :personal_info in categories
    end

    test "returns 8 categories" do
      categories = SafetyClient.categories()
      assert length(categories) == 8
    end

    test "all categories are atoms" do
      categories = SafetyClient.categories()

      Enum.each(categories, fn cat ->
        assert is_atom(cat)
      end)
    end
  end

  describe "module structure" do
    test "exports expected functions" do
      functions = SafetyClient.__info__(:functions)

      assert {:check_content, 1} in functions or {:check_content, 2} in functions
      assert {:detect_jailbreak, 1} in functions or {:detect_jailbreak, 2} in functions
      assert {:check_topic, 2} in functions or {:check_topic, 3} in functions
      assert {:check_multimodal, 2} in functions or {:check_multimodal, 3} in functions
      assert {:sanitize_input, 1} in functions or {:sanitize_input, 2} in functions
      assert {:check_avatar_message, 2} in functions or {:check_avatar_message, 3} in functions
      assert {:categories, 0} in functions
    end

    test "implements Pipeline.Stage behaviour" do
      behaviours = SafetyClient.__info__(:attributes)[:behaviour] || []
      assert Viva.AI.Pipeline.Stage in behaviours
    end
  end
end
