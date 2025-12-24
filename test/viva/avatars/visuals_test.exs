defmodule Viva.Avatars.VisualsTest do
  use ExUnit.Case, async: true

  alias Viva.Avatars.Visuals

  describe "module structure" do
    test "exports expected functions" do
      functions = Visuals.__info__(:functions)

      assert {:generate_complete, 2} in functions
      assert {:generate_profile, 2} in functions
      assert {:generate_expressions, 3} in functions
      assert {:generate_3d_model, 2} in functions
      assert {:update_expression, 2} in functions
      assert {:get_expression_image, 2} in functions
      assert {:generate_lipsync, 3} in functions
      assert {:stream_lipsync, 3} in functions
      assert {:needs_generation?, 1} in functions
      assert {:has_3d_model?, 1} in functions
    end
  end

  describe "needs_generation?/1" do
    test "returns true when profile_image_url is nil" do
      avatar = %{profile_image_url: nil}
      assert Visuals.needs_generation?(avatar) == true
    end

    test "returns false when profile_image_url exists" do
      avatar = %{profile_image_url: "/uploads/avatars/test/profile.png"}
      assert Visuals.needs_generation?(avatar) == false
    end
  end

  describe "has_3d_model?/1" do
    test "returns true when avatar_3d_model_url exists" do
      avatar = %{avatar_3d_model_url: "/uploads/avatars/test/model.glb"}
      assert Visuals.has_3d_model?(avatar) == true
    end

    test "returns false when avatar_3d_model_url is nil" do
      avatar = %{avatar_3d_model_url: nil}
      assert Visuals.has_3d_model?(avatar) == false
    end
  end

  describe "get_expression_image/2" do
    test "returns expression image when available" do
      avatar = %{
        expression_images: %{"happy" => "/uploads/happy.png"},
        profile_image_url: "/uploads/profile.png",
        avatar_url: "/uploads/avatar.png"
      }

      assert Visuals.get_expression_image(avatar, :happy) == "/uploads/happy.png"
    end

    test "falls back to profile_image_url when expression not available" do
      avatar = %{
        expression_images: %{},
        profile_image_url: "/uploads/profile.png",
        avatar_url: "/uploads/avatar.png"
      }

      assert Visuals.get_expression_image(avatar, :sad) == "/uploads/profile.png"
    end

    test "falls back to avatar_url when profile_image_url is nil" do
      avatar = %{
        expression_images: nil,
        profile_image_url: nil,
        avatar_url: "/uploads/avatar.png"
      }

      assert Visuals.get_expression_image(avatar, :neutral) == "/uploads/avatar.png"
    end
  end
end
