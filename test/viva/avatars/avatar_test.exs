defmodule Viva.Avatars.AvatarTest do
  use Viva.DataCase, async: true
  alias Viva.Avatars.Avatar
  alias Viva.Avatars.Personality

  describe "changeset/2" do
    @valid_personality %{
      openness: 0.5,
      conscientiousness: 0.5,
      extraversion: 0.5,
      agreeableness: 0.5,
      neuroticism: 0.5,
      enneagram_type: :type_5
    }

    @valid_attrs %{
      name: "Neo",
      user_id: Ecto.UUID.generate(),
      personality: @valid_personality
    }

    test "validates required fields" do
      changeset = Avatar.changeset(%Avatar{}, %{})
      refute changeset.valid?

      assert %{
               name: ["can't be blank"],
               user_id: ["can't be blank"],
               personality: ["can't be blank"]
             } = errors_on(changeset)
    end

    test "validates name length" do
      changeset = Avatar.changeset(%Avatar{}, %{@valid_attrs | name: "A"})
      refute changeset.valid?
      assert %{name: ["should be at least 2 character(s)"]} = errors_on(changeset)
    end

    test "generates system prompt automatically" do
      changeset = Avatar.changeset(%Avatar{}, @valid_attrs)
      assert changeset.valid?
      prompt = get_change(changeset, :system_prompt)
      assert prompt =~ "You are Neo"
      # Type 5
      assert prompt =~ "Investigator"
    end

    test "puts default internal state and social persona" do
      changeset = Avatar.changeset(%Avatar{}, @valid_attrs)
      assert get_change(changeset, :internal_state) != nil
      assert get_change(changeset, :social_persona) != nil
    end
  end

  describe "create_changeset/2" do
    test "sets timestamps" do
      attrs = %{
        name: "Trinity",
        user_id: Ecto.UUID.generate(),
        personality: %{
          openness: 0.5,
          conscientiousness: 0.5,
          extraversion: 0.5,
          agreeableness: 0.5,
          neuroticism: 0.5,
          enneagram_type: :type_1
        }
      }

      changeset = Avatar.create_changeset(%Avatar{}, attrs)
      assert get_change(changeset, :created_at) != nil
      assert get_change(changeset, :last_active_at) != nil
    end
  end

  describe "queries" do
    test "active/0 filters by is_active" do
      query = Avatar.active()
      assert inspect(query) =~ "is_active == true"
    end

    test "by_user/1 filters by user_id" do
      user_id = Ecto.UUID.generate()
      query = Avatar.by_user(user_id)
      assert inspect(query) =~ "user_id == ^\"#{user_id}\""
    end
  end
end
