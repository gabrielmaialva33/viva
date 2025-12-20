defmodule Viva.Repo.Migrations.AddAvatarVisualFields do
  use Ecto.Migration

  def change do
    alter table(:avatars) do
      # AI-generated profile image URL (from Stable Diffusion)
      add :profile_image_url, :string

      # 3D model URL (from TRELLIS)
      add :avatar_3d_model_url, :string

      # Current facial expression for display
      add :current_expression, :string, default: "neutral"

      # Cached expression images map (expression => url)
      add :expression_images, :map, default: %{}
    end
  end
end
