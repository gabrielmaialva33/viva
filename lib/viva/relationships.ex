defmodule Viva.Relationships do
  @moduledoc """
  The Relationships context.
  Manages relationships between avatars.
  """

  import Ecto.Query

  alias Viva.Relationships.Relationship
  alias Viva.Repo

  # === Types ===

  @type avatar_id :: Ecto.UUID.t()
  @type interaction_type :: atom()
  @type deltas :: %{optional(atom()) => number()}

  # === Relationship CRUD ===

  @spec list_relationships(avatar_id(), keyword()) :: [Relationship.t()]
  def list_relationships(avatar_id, opts \\ []) do
    status = Keyword.get(opts, :status)
    limit = Keyword.get(opts, :limit, 50)

    Relationship
    |> Relationship.involving(avatar_id)
    |> maybe_filter_status(status)
    |> order_by([r], desc: r.last_interaction_at)
    |> limit(^limit)
    |> Repo.all()
  end

  @spec get_relationship(Ecto.UUID.t()) :: Relationship.t() | nil
  def get_relationship(id), do: Repo.get(Relationship, id)

  @spec get_relationship!(Ecto.UUID.t()) :: Relationship.t()
  def get_relationship!(id), do: Repo.get!(Relationship, id)

  @spec get_relationship_between(avatar_id(), avatar_id()) :: Relationship.t() | nil
  def get_relationship_between(avatar_a_id, avatar_b_id) do
    Relationship
    |> Relationship.between(avatar_a_id, avatar_b_id)
    |> Repo.one()
  end

  @spec get_or_create_relationship(avatar_id(), avatar_id()) ::
          {:ok, Relationship.t()} | {:error, Ecto.Changeset.t()}
  def get_or_create_relationship(avatar_a_id, avatar_b_id) do
    # Ensure consistent ordering (lower ID is always avatar_a)
    {a_id, b_id} = order_avatar_ids(avatar_a_id, avatar_b_id)

    insert_result =
      %Relationship{}
      |> Relationship.changeset(%{
        avatar_a_id: a_id,
        avatar_b_id: b_id,
        status: :strangers,
        first_interaction_at: DateTime.utc_now()
      })
      |> Repo.insert(
        on_conflict: :nothing,
        conflict_target: [:avatar_a_id, :avatar_b_id],
        returning: true
      )

    case insert_result do
      {:ok, %{id: nil}} ->
        # Conflict happened, fetch the existing one
        {:ok, get_relationship_between(a_id, b_id)}

      result ->
        result
    end
  end

  @spec create_relationship(avatar_id(), avatar_id()) ::
          {:ok, Relationship.t()} | {:error, Ecto.Changeset.t()}
  def create_relationship(avatar_a_id, avatar_b_id) do
    get_or_create_relationship(avatar_a_id, avatar_b_id)
  end

  @spec update_relationship(Relationship.t(), map()) ::
          {:ok, Relationship.t()} | {:error, Ecto.Changeset.t()}
  def update_relationship(%Relationship{} = rel, attrs) do
    rel
    |> Relationship.changeset(attrs)
    |> Repo.update()
  end

  @spec delete_relationship(Relationship.t()) ::
          {:ok, Relationship.t()} | {:error, Ecto.Changeset.t()}
  def delete_relationship(%Relationship{} = rel) do
    Repo.delete(rel)
  end

  # === Relationship Updates ===

  @spec record_interaction(avatar_id(), avatar_id(), interaction_type()) ::
          {:ok, Relationship.t()} | {:error, term()}
  def record_interaction(avatar_a_id, avatar_b_id, interaction_type) do
    with {:ok, rel} <- get_or_create_relationship(avatar_a_id, avatar_b_id) do
      update_relationship(rel, %{
        interaction_count: rel.interaction_count + 1,
        last_interaction_at: DateTime.utc_now(),
        last_interaction_type: interaction_type
      })
    end
  end

  @spec update_feelings(Relationship.t(), avatar_id(), map()) ::
          {:ok, Relationship.t()} | {:error, Ecto.Changeset.t()}
  def update_feelings(rel, avatar_id, feelings) do
    {a_id, _} = order_avatar_ids(rel.avatar_a_id, rel.avatar_b_id)

    field = if avatar_id == a_id, do: :feelings_a_to_b, else: :feelings_b_to_a

    update_relationship(rel, %{field => feelings})
  end

  @spec evolve_relationship(Relationship.t(), deltas()) ::
          {:ok, Relationship.t()} | {:error, Ecto.Changeset.t()}
  def evolve_relationship(%Relationship{} = rel, deltas) do
    new_familiarity = clamp(rel.familiarity + Map.get(deltas, :familiarity, 0), 0.0, 1.0)
    new_trust = clamp(rel.trust + Map.get(deltas, :trust, 0), 0.0, 1.0)
    new_affection = clamp(rel.affection + Map.get(deltas, :affection, 0), 0.0, 1.0)
    new_attraction = clamp(rel.attraction + Map.get(deltas, :attraction, 0), 0.0, 1.0)
    conflict_delta = Map.get(deltas, :conflict, 0)
    new_conflicts = max(0, rel.unresolved_conflicts + conflict_delta)

    new_status =
      determine_status(new_familiarity, new_trust, new_affection, new_attraction, new_conflicts)

    update_relationship(rel, %{
      familiarity: new_familiarity,
      trust: new_trust,
      affection: new_affection,
      attraction: new_attraction,
      unresolved_conflicts: new_conflicts,
      status: new_status
    })
  end

  # === Matching ===

  @spec list_potential_matches(avatar_id(), keyword()) :: [Relationship.t()]
  def list_potential_matches(avatar_id, opts \\ []) do
    limit = Keyword.get(opts, :limit, 20)
    min_attraction = Keyword.get(opts, :min_attraction, 0.6)

    Relationship
    |> Relationship.involving(avatar_id)
    |> where([r], r.attraction >= ^min_attraction)
    |> where([r], r.status in [:acquaintances, :friends, :close_friends])
    |> order_by([r], desc: r.attraction, desc: r.affection)
    |> limit(^limit)
    |> Repo.all()
  end

  @spec list_mutual_matches(avatar_id()) :: [Relationship.t()]
  def list_mutual_matches(avatar_id) do
    Relationship
    |> Relationship.involving(avatar_id)
    |> where([r], r.status == :matched)
    |> Repo.all()
  end

  @spec attempt_match(avatar_id(), avatar_id()) ::
          {:ok, Relationship.t()} | {:error, term()}
  def attempt_match(avatar_a_id, avatar_b_id) do
    with {:ok, rel} <- get_or_create_relationship(avatar_a_id, avatar_b_id) do
      {a_id, _} = order_avatar_ids(rel.avatar_a_id, rel.avatar_b_id)

      # Check if both avatars are interested
      a_interested = get_in(rel.feelings_a_to_b, [:romantic_interest]) || false
      b_interested = get_in(rel.feelings_b_to_a, [:romantic_interest]) || false

      if a_interested && b_interested do
        update_relationship(rel, %{
          status: :matched,
          matched_at: DateTime.utc_now()
        })
      else
        # Update interest for the requesting avatar
        field = if avatar_a_id == a_id, do: :feelings_a_to_b, else: :feelings_b_to_a

        current_feelings =
          if avatar_a_id == a_id, do: rel.feelings_a_to_b, else: rel.feelings_b_to_a

        update_relationship(rel, %{
          field => Map.put(current_feelings, :romantic_interest, true)
        })
      end
    end
  end

  # === Statistics ===

  @spec relationship_stats(avatar_id()) :: map()
  def relationship_stats(avatar_id) do
    # Get general counts and averages in one query
    stats_query =
      from(r in Relationship,
        where: r.avatar_a_id == ^avatar_id or r.avatar_b_id == ^avatar_id,
        select: %{
          total: count(r.id),
          avg_familiarity: avg(r.familiarity),
          avg_trust: avg(r.trust),
          avg_affection: avg(r.affection),
          matches: filter(count(r.id), r.status == :matched)
        }
      )

    stats = Repo.one(stats_query)

    # Get frequencies by status
    frequencies_query =
      from(r in Relationship,
        where: r.avatar_a_id == ^avatar_id or r.avatar_b_id == ^avatar_id,
        group_by: r.status,
        select: {r.status, count(r.id)}
      )

    by_status =
      frequencies_query
      |> Repo.all()
      |> Map.new()

    %{
      total: stats.total || 0,
      by_status: by_status,
      avg_familiarity: stats.avg_familiarity || 0.0,
      avg_trust: stats.avg_trust || 0.0,
      avg_affection: stats.avg_affection || 0.0,
      matches: stats.matches || 0
    }
  end

  # === Additional Query Helpers ===

  @spec count_close_relationships(avatar_id()) :: non_neg_integer()
  def count_close_relationships(avatar_id) do
    Relationship
    |> Relationship.involving(avatar_id)
    |> where([r], r.status in [:close_friends, :best_friends, :dating, :partners])
    |> Repo.aggregate(:count, :id)
  end

  @spec list_for_avatar(avatar_id()) :: [Relationship.t()]
  def list_for_avatar(avatar_id) do
    Relationship
    |> Relationship.involving(avatar_id)
    |> Repo.all()
  end

  @doc """
  Lists all relationships in the system.
  Used for the relationship graph visualization.
  """
  @spec list_all(keyword()) :: [Relationship.t()]
  def list_all(opts \\ []) do
    limit = Keyword.get(opts, :limit, 100)
    preload = Keyword.get(opts, :preload, [:avatar_a, :avatar_b])

    Relationship
    |> order_by([r], desc: r.last_interaction_at)
    |> limit(^limit)
    |> Repo.all()
    |> Repo.preload(preload)
  end

  @spec find_available_friend(avatar_id()) :: avatar_id() | nil
  def find_available_friend(avatar_id) do
    relationship =
      Relationship
      |> Relationship.involving(avatar_id)
      |> where([r], r.status in [:friends, :close_friends, :best_friends])
      |> order_by([r], desc: r.last_interaction_at)
      |> limit(1)
      |> Repo.one()

    case relationship do
      nil -> nil
      rel -> if rel.avatar_a_id == avatar_id, do: rel.avatar_b_id, else: rel.avatar_a_id
    end
  end

  @spec get_crush(avatar_id()) :: avatar_id() | nil
  def get_crush(avatar_id) do
    relationship =
      Relationship
      |> Relationship.involving(avatar_id)
      |> where([r], r.status in [:crush, :mutual_crush])
      |> limit(1)
      |> Repo.one()

    case relationship do
      nil -> nil
      rel -> if rel.avatar_a_id == avatar_id, do: rel.avatar_b_id, else: rel.avatar_a_id
    end
  end

  # === Private Helpers ===

  defp order_avatar_ids(id_a, id_b) do
    if id_a < id_b, do: {id_a, id_b}, else: {id_b, id_a}
  end

  defp maybe_filter_status(query, nil), do: query
  defp maybe_filter_status(query, status), do: where(query, [r], r.status == ^status)

  defp clamp(value, min_val, max_val) do
    value
    |> max(min_val)
    |> min(max_val)
  end

  defp determine_status(familiarity, trust, affection, attraction, conflicts) do
    cond do
      conflicts > 5 -> :complicated
      familiarity < 0.1 -> :strangers
      familiarity < 0.3 -> :acquaintances
      trust > 0.7 && affection > 0.7 && attraction > 0.7 -> :dating
      trust > 0.5 && affection > 0.6 -> :close_friends
      familiarity > 0.4 -> :friends
      true -> :acquaintances
    end
  end
end
