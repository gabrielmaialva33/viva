defmodule Viva.Matching do
  @moduledoc """
  The Matching context.
  Handles swipes, match detection, and discovery feed.
  """
  import Ecto.Query
  alias Ecto.Multi
  alias Viva.Matching.Swipe
  alias Viva.Relationships
  alias Viva.Repo

  @doc """
  Registers a swipe action and checks for a match.
  Returns {:ok, %{swipe: swipe, match: relationship | nil}}
  """
  def swipe(actor_id, target_id, action) do
    Multi.new()
    |> Multi.insert(
      :swipe,
      Swipe.changeset(%Swipe{}, %{
        actor_avatar_id: actor_id,
        target_avatar_id: target_id,
        action: action
      })
    )
    |> Multi.run(:check_match, fn repo, %{swipe: swipe} ->
      if action in [:like, :superlike] do
        {:ok, check_reciprocal_like(repo, swipe)}
      else
        {:ok, nil}
      end
    end)
    |> Multi.run(:match, fn repo, %{check_match: is_match} ->
      if is_match do
        # Evolution: Relationships are now born from mutual likes
        create_match_relationship(repo, actor_id, target_id)
      else
        {:ok, nil}
      end
    end)
    |> Repo.transaction()
    |> handle_transaction_result()
  end

  @doc """
  Gets candidate avatars for the discovery feed.
  Prioritizes avatars the user hasn't swiped on yet.
  """
  def get_candidates(avatar_id, limit \\ 10) do
    # Subquery to get IDs of avatars already swiped on
    swiped_ids =
      from(s in Swipe,
        where: s.actor_avatar_id == ^avatar_id,
        select: s.target_avatar_id
      )

    # Fetch active avatars not including self and already swiped
    Viva.Avatars.Avatar
    |> where([a], a.id != ^avatar_id)
    |> where([a], a.id not in subquery(swiped_ids))
    |> limit(^limit)
    |> Repo.all()

    # In a real scenario, we'd pipe this to Matchmaker.Engine for scoring
  end

  # Private Helpers

  defp check_reciprocal_like(repo, swipe) do
    query =
      from(s in Swipe,
        where:
          s.actor_avatar_id == ^swipe.target_avatar_id and
            s.target_avatar_id == ^swipe.actor_avatar_id and
            s.action in [:like, :superlike]
      )

    repo.exists?(query)
  end

  defp create_match_relationship(_, actor_id, target_id) do
    # We use the existing Relationships context to ensure consistency

    case Relationships.get_or_create_relationship(actor_id, target_id) do
      {:ok, rel} ->
        Relationships.update_relationship(rel, %{
          status: :matched,
          matched_at: DateTime.utc_now()
        })

      error ->
        error
    end
  end

  defp handle_transaction_result({:ok, %{swipe: _swipe, match: match}}) do
    if match do
      notify_match(match)
    end

    {:ok, %{swipe: _swipe, match: match}}
  end

  defp handle_transaction_result({:error, _, changeset, _}) do
    {:error, changeset}
  end

  defp notify_match(relationship) do
    # Broadcast via PubSub for real-time UI updates
    Phoenix.PubSub.broadcast(
      Viva.PubSub,
      "avatar:#{relationship.avatar_a_id}",
      {:match, relationship}
    )

    Phoenix.PubSub.broadcast(
      Viva.PubSub,
      "avatar:#{relationship.avatar_b_id}",
      {:match, relationship}
    )

    # Notify LifeProcesses to update emotions (Dopamine hit!)
    Viva.Sessions.LifeProcess.trigger_thought(relationship.avatar_a_id)
    Viva.Sessions.LifeProcess.trigger_thought(relationship.avatar_b_id)
  end
end
