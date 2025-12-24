defmodule Viva.MatchingTest do
  use Viva.DataCase, async: true

  alias Viva.Matching
  alias Viva.Relationships
  alias Viva.Avatars.Avatar
  alias Viva.Avatars.Personality
  alias Viva.Accounts.User

  setup do
    user =
      %User{
        email: "user#{System.unique_integer()}@example.com",
        username: "user#{System.unique_integer()}",
        hashed_password: "password"
      }
      |> Repo.insert!()

    avatar_a = insert_avatar(user, "Avatar A")
    avatar_b = insert_avatar(user, "Avatar B")

    {:ok, avatar_a: avatar_a, avatar_b: avatar_b, user: user}
  end

  defp insert_avatar(user, name) do
    %Avatar{}
    |> Avatar.changeset(%{
      name: name,
      user_id: user.id,
      personality: %{
        openness: 0.5,
        conscientiousness: 0.5,
        extraversion: 0.5,
        agreeableness: 0.5,
        neuroticism: 0.5,
        enneagram_type: "type_1",
        humor_style: "sarcastic",
        native_language: "pt-BR"
      }
    })
    |> Repo.insert!()
  end

  describe "swipes" do
    test "single like does not create a match", %{avatar_a: a, avatar_b: b} do
      {:ok, %{swipe: swipe, match: match}} = Matching.swipe(a.id, b.id, :like)

      assert swipe.action == :like
      assert match == nil

      rel = Relationships.get_relationship_between(a.id, b.id)
      assert rel == nil
    end

    test "mutual like creates a match relationship", %{avatar_a: a, avatar_b: b} do
      # B likes A first
      Matching.swipe(b.id, a.id, :like)

      # A likes B (Mutual!)
      {:ok, %{swipe: _, match: match}} = Matching.swipe(a.id, b.id, :like)

      assert match != nil
      assert match.status == :matched

      # Verify relationship is persistent and retrievable
      rel = Relationships.get_relationship_between(a.id, b.id)
      assert rel.status == :matched
    end

    test "pass action never creates a match", %{avatar_a: a, avatar_b: b} do
      Matching.swipe(b.id, a.id, :like)
      {:ok, %{swipe: _, match: match}} = Matching.swipe(a.id, b.id, :pass)

      assert match == nil
    end
  end

  describe "discovery" do
    test "get_candidates excludes self and already swiped avatars", %{
      avatar_a: a,
      avatar_b: b,
      user: user
    } do
      avatar_c = insert_avatar(user, "Avatar C")

      # Swipe on B
      Matching.swipe(a.id, b.id, :pass)

      candidates = Matching.get_candidates(a.id)
      candidate_ids = Enum.map(candidates, & &1.id)

      assert avatar_c.id in candidate_ids
      refute a.id in candidate_ids
      refute b.id in candidate_ids
    end
  end
end
