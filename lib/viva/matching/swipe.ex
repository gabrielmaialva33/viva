defmodule Viva.Matching.Swipe do
  @moduledoc """
  Represents a swipe action between two avatars.
  """
  use Ecto.Schema
  import Ecto.Changeset

  @primary_key {:id, :binary_id, autogenerate: true}
  @foreign_key_type :binary_id
  schema "swipes" do
    belongs_to :actor_avatar, Viva.Avatars.Avatar
    belongs_to :target_avatar, Viva.Avatars.Avatar

    field :action, Ecto.Enum, values: [:like, :pass, :superlike]
    field :metadata, :map, default: %{}

    timestamps(type: :utc_datetime)
  end

  def changeset(swipe, attrs) do
    swipe
    |> cast(attrs, [:actor_avatar_id, :target_avatar_id, :action, :metadata])
    |> validate_required([:actor_avatar_id, :target_avatar_id, :action])
    |> validate_different_avatars()
    |> unique_constraint([:actor_avatar_id, :target_avatar_id], name: :swipes_actor_target_index)
  end

  defp validate_different_avatars(changeset) do
    actor_id = get_field(changeset, :actor_avatar_id)
    target_id = get_field(changeset, :target_avatar_id)

    if actor_id && target_id && actor_id == target_id do
      add_error(changeset, :target_avatar_id, "cannot swipe on yourself")
    else
      changeset
    end
  end
end
