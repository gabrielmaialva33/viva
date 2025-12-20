defmodule Viva.Relationships.Relationship do
  @moduledoc """
  Relationship schema between two avatars.
  Tracks affection, trust, attraction, and relationship status.
  """
  use Ecto.Schema
  import Ecto.Changeset
  import Ecto.Query

  alias Viva.Avatars.Avatar

  @primary_key {:id, :binary_id, autogenerate: true}
  @foreign_key_type :binary_id
  schema "relationships" do
    belongs_to :avatar_a, Avatar
    belongs_to :avatar_b, Avatar

    # Core relationship metrics (0.0 to 1.0)
    field :familiarity, :float, default: 0.0
    field :trust, :float, default: 0.5
    field :affection, :float, default: 0.0
    field :attraction, :float, default: 0.0
    field :compatibility_score, :float

    # Relationship status
    field :status, Ecto.Enum,
      values: [
        :strangers,
        :acquaintances,
        :friends,
        :close_friends,
        :best_friends,
        :crush,
        :mutual_crush,
        :dating,
        :partners,
        :complicated,
        :ex
      ],
      default: :strangers

    # Individual feelings (each avatar's perspective)
    embeds_one :a_feelings, __MODULE__.Feelings, on_replace: :update
    embeds_one :b_feelings, __MODULE__.Feelings, on_replace: :update

    # Interaction history
    field :first_interaction_at, :utc_datetime
    field :last_interaction_at, :utc_datetime
    field :interaction_count, :integer, default: 0
    field :shared_memories_count, :integer, default: 0
    field :total_conversation_minutes, :integer, default: 0

    # Relationship milestones
    field :milestones, {:array, :map}, default: []

    # Tension/conflict tracking
    field :unresolved_conflicts, :integer, default: 0
    field :last_conflict_at, :utc_datetime

    timestamps(type: :utc_datetime)
  end

  defmodule Feelings do
    @moduledoc "Individual avatar's feelings about the relationship"
    use Ecto.Schema

    @primary_key false
    embedded_schema do
      field :thinks_about_often, :boolean, default: false
      field :feels_understood, :float, default: 0.5
      field :wants_more_time, :boolean, default: false
      field :romantic_interest, :float, default: 0.0
      field :jealousy, :float, default: 0.0
      field :admiration, :float, default: 0.0
      field :comfort_level, :float, default: 0.5
      field :excitement_to_see, :float, default: 0.3
    end

    def changeset(feelings, attrs) do
      feelings
      |> Ecto.Changeset.cast(attrs, [
        :thinks_about_often,
        :feels_understood,
        :wants_more_time,
        :romantic_interest,
        :jealousy,
        :admiration,
        :comfort_level,
        :excitement_to_see
      ])
    end
  end

  def changeset(relationship, attrs) do
    relationship
    |> cast(attrs, [
      :avatar_a_id,
      :avatar_b_id,
      :familiarity,
      :trust,
      :affection,
      :attraction,
      :compatibility_score,
      :status,
      :first_interaction_at,
      :last_interaction_at,
      :interaction_count,
      :shared_memories_count,
      :total_conversation_minutes,
      :milestones,
      :unresolved_conflicts,
      :last_conflict_at
    ])
    |> cast_embed(:a_feelings)
    |> cast_embed(:b_feelings)
    |> validate_required([:avatar_a_id, :avatar_b_id])
    |> validate_different_avatars()
    |> validate_number(:familiarity, greater_than_or_equal_to: 0.0, less_than_or_equal_to: 1.0)
    |> validate_number(:trust, greater_than_or_equal_to: 0.0, less_than_or_equal_to: 1.0)
    |> validate_number(:affection, greater_than_or_equal_to: 0.0, less_than_or_equal_to: 1.0)
    |> validate_number(:attraction, greater_than_or_equal_to: 0.0, less_than_or_equal_to: 1.0)
    |> unique_constraint([:avatar_a_id, :avatar_b_id])
    |> put_default_feelings()
  end

  defp validate_different_avatars(changeset) do
    a_id = get_field(changeset, :avatar_a_id)
    b_id = get_field(changeset, :avatar_b_id)

    if a_id && b_id && a_id == b_id do
      add_error(changeset, :avatar_b_id, "cannot be the same as avatar_a")
    else
      changeset
    end
  end

  defp put_default_feelings(changeset) do
    changeset
    |> put_default_embed(:a_feelings)
    |> put_default_embed(:b_feelings)
  end

  defp put_default_embed(changeset, field) do
    case get_field(changeset, field) do
      nil -> put_embed(changeset, field, %Feelings{})
      _ -> changeset
    end
  end

  @doc "Calculate relationship health (0.0 to 1.0)"
  def health(%__MODULE__{} = rel) do
    positive = (rel.trust + rel.affection + rel.familiarity) / 3
    negative = rel.unresolved_conflicts * 0.1

    (positive - negative)
    |> max(0.0)
    |> min(1.0)
  end

  @doc "Check if there's mutual romantic interest"
  def mutual_interest?(%__MODULE__{} = rel) do
    rel.a_feelings.romantic_interest > 0.5 &&
      rel.b_feelings.romantic_interest > 0.5
  end

  @doc "Add a milestone to the relationship"
  def add_milestone(%__MODULE__{} = rel, type, description) do
    milestone = %{
      type: type,
      description: description,
      occurred_at: DateTime.utc_now()
    }

    %{rel | milestones: [milestone | rel.milestones]}
  end

  # Queries
  def for_avatar(avatar_id) do
    from(r in __MODULE__,
      where: r.avatar_a_id == ^avatar_id or r.avatar_b_id == ^avatar_id
    )
  end

  def involving(query, avatar_id) do
    from(r in query,
      where: r.avatar_a_id == ^avatar_id or r.avatar_b_id == ^avatar_id
    )
  end

  def between(avatar_a_id, avatar_b_id) do
    from(r in __MODULE__,
      where:
        (r.avatar_a_id == ^avatar_a_id and r.avatar_b_id == ^avatar_b_id) or
          (r.avatar_a_id == ^avatar_b_id and r.avatar_b_id == ^avatar_a_id)
    )
  end

  def between(query, avatar_a_id, avatar_b_id) do
    from(r in query,
      where:
        (r.avatar_a_id == ^avatar_a_id and r.avatar_b_id == ^avatar_b_id) or
          (r.avatar_a_id == ^avatar_b_id and r.avatar_b_id == ^avatar_a_id)
    )
  end

  def by_status(query \\ __MODULE__, status) do
    from(r in query, where: r.status == ^status)
  end

  def romantic(query \\ __MODULE__) do
    from(r in query,
      where: r.status in [:crush, :mutual_crush, :dating, :partners]
    )
  end

  def active(query \\ __MODULE__) do
    one_week_ago = DateTime.add(DateTime.utc_now(), -7, :day)
    from(r in query, where: r.last_interaction_at > ^one_week_ago)
  end
end
