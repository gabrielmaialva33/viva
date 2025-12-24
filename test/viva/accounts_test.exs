defmodule Viva.AccountsTest do
  use Viva.DataCase, async: true

  alias Viva.Accounts
  alias Viva.Accounts.User

  describe "create_user/1" do
    test "creates user with valid data" do
      attrs = %{
        email: unique_email(),
        username: unique_username(),
        password: "SecurePass123"
      }

      assert {:ok, %User{} = user} = Accounts.create_user(attrs)
      assert user.email == attrs.email
      assert user.username == attrs.username
      assert user.hashed_password != nil
    end

    test "returns error with invalid email" do
      attrs = %{
        email: "invalid-email",
        username: unique_username(),
        password: "SecurePass123"
      }

      assert {:error, changeset} = Accounts.create_user(attrs)
      assert "must be a valid email" in errors_on(changeset).email
    end

    test "returns error with short password" do
      attrs = %{
        email: unique_email(),
        username: unique_username(),
        password: "Short1"
      }

      assert {:error, changeset} = Accounts.create_user(attrs)
      assert "should be at least 8 character(s)" in errors_on(changeset).password
    end

    test "returns error with password missing uppercase" do
      attrs = %{
        email: unique_email(),
        username: unique_username(),
        password: "lowercase123"
      }

      assert {:error, changeset} = Accounts.create_user(attrs)
      assert "must have at least one uppercase letter" in errors_on(changeset).password
    end

    test "returns error with password missing digit" do
      attrs = %{
        email: unique_email(),
        username: unique_username(),
        password: "NoDigitsHere"
      }

      assert {:error, changeset} = Accounts.create_user(attrs)
      assert "must have at least one digit" in errors_on(changeset).password
    end

    test "returns error with invalid username format" do
      attrs = %{
        email: unique_email(),
        username: "user name with spaces",
        password: "SecurePass123"
      }

      assert {:error, changeset} = Accounts.create_user(attrs)
      assert "only letters, numbers, and underscores" in errors_on(changeset).username
    end

    test "returns error with username too short" do
      attrs = %{
        email: unique_email(),
        username: "ab",
        password: "SecurePass123"
      }

      assert {:error, changeset} = Accounts.create_user(attrs)
      assert "should be at least 3 character(s)" in errors_on(changeset).username
    end

    test "normalizes email to lowercase" do
      email = String.upcase(unique_email())

      attrs = %{
        email: email,
        username: unique_username(),
        password: "SecurePass123"
      }

      assert {:ok, user} = Accounts.create_user(attrs)
      assert user.email == String.downcase(email)
    end

    test "normalizes username to lowercase" do
      username = "TestUser#{System.unique_integer([:positive])}"

      attrs = %{
        email: unique_email(),
        username: username,
        password: "SecurePass123"
      }

      assert {:ok, user} = Accounts.create_user(attrs)
      assert user.username == String.downcase(username)
    end
  end

  describe "get_user/1" do
    test "returns user by id" do
      {:ok, user} = create_user()
      found = Accounts.get_user(user.id)
      assert found.id == user.id
    end

    test "returns nil for non-existent id" do
      assert Accounts.get_user(Ecto.UUID.generate()) == nil
    end
  end

  describe "get_user!/1" do
    test "returns user by id" do
      {:ok, user} = create_user()
      found = Accounts.get_user!(user.id)
      assert found.id == user.id
    end

    test "raises for non-existent id" do
      assert_raise Ecto.NoResultsError, fn ->
        Accounts.get_user!(Ecto.UUID.generate())
      end
    end
  end

  describe "get_user_by_email/1" do
    test "returns user by email" do
      {:ok, user} = create_user()
      found = Accounts.get_user_by_email(user.email)
      assert found.id == user.id
    end

    test "is case insensitive" do
      {:ok, user} = create_user()
      upcase_email = String.upcase(user.email)
      found = Accounts.get_user_by_email(upcase_email)
      assert found.id == user.id
    end

    test "returns nil for non-existent email" do
      assert Accounts.get_user_by_email("nonexistent@example.com") == nil
    end
  end

  describe "get_user_by_username/1" do
    test "returns user by username" do
      {:ok, user} = create_user()
      found = Accounts.get_user_by_username(user.username)
      assert found.id == user.id
    end

    test "is case insensitive" do
      {:ok, user} = create_user()
      upcase_username = String.upcase(user.username)
      found = Accounts.get_user_by_username(upcase_username)
      assert found.id == user.id
    end

    test "returns nil for non-existent username" do
      assert Accounts.get_user_by_username("nonexistent") == nil
    end
  end

  describe "update_user/2" do
    test "updates user with valid data" do
      {:ok, user} = create_user()

      assert {:ok, updated} = Accounts.update_user(user, %{display_name: "New Name"})
      assert updated.display_name == "New Name"
    end

    test "returns error with invalid data" do
      {:ok, user} = create_user()

      assert {:error, changeset} = Accounts.update_user(user, %{email: "invalid"})
      assert "must be a valid email" in errors_on(changeset).email
    end
  end

  describe "delete_user/1" do
    test "deletes the user" do
      {:ok, user} = create_user()

      assert {:ok, %User{}} = Accounts.delete_user(user)
      assert Accounts.get_user(user.id) == nil
    end
  end

  describe "list_users/1" do
    test "returns users including created ones" do
      {:ok, user1} = create_user()
      {:ok, user2} = create_user()

      users = Accounts.list_users()
      user_ids = Enum.map(users, & &1.id)
      assert user1.id in user_ids
      assert user2.id in user_ids
    end

    test "respects limit option" do
      for _ <- 1..5 do
        create_user()
      end

      users = Accounts.list_users(limit: 3)
      assert length(users) == 3
    end

    test "filters active users when active: true" do
      {:ok, active} = create_user()
      {:ok, inactive} = create_user()
      Accounts.deactivate_user(inactive)

      users = Accounts.list_users(active: true)
      user_ids = Enum.map(users, & &1.id)
      assert active.id in user_ids
      refute inactive.id in user_ids
    end
  end

  describe "authenticate_by_email/2" do
    test "returns user with valid credentials" do
      {:ok, user} = create_user()

      assert {:ok, authenticated} = Accounts.authenticate_by_email(user.email, "SecurePass123")
      assert authenticated.id == user.id
    end

    test "returns error with invalid password" do
      {:ok, user} = create_user()

      assert {:error, :invalid_credentials} =
               Accounts.authenticate_by_email(user.email, "wrong")
    end

    test "returns error with non-existent email" do
      assert {:error, :invalid_credentials} =
               Accounts.authenticate_by_email(
                 "nonexistent#{System.unique_integer()}@example.com",
                 "password"
               )
    end
  end

  describe "authenticate_by_username/2" do
    test "returns user with valid credentials" do
      {:ok, user} = create_user()

      assert {:ok, authenticated} =
               Accounts.authenticate_by_username(user.username, "SecurePass123")

      assert authenticated.id == user.id
    end

    test "returns error with invalid password" do
      {:ok, user} = create_user()

      assert {:error, :invalid_credentials} =
               Accounts.authenticate_by_username(user.username, "wrong")
    end
  end

  describe "mark_user_seen/1" do
    test "updates last_seen_at" do
      {:ok, user} = create_user()
      assert user.last_seen_at == nil

      assert {:ok, updated} = Accounts.mark_user_seen(user)
      assert updated.last_seen_at != nil
    end
  end

  describe "deactivate_user/1" do
    test "sets is_active to false" do
      {:ok, user} = create_user()
      assert user.is_active == true

      assert {:ok, updated} = Accounts.deactivate_user(user)
      assert updated.is_active == false
    end
  end

  describe "verify_user/1" do
    test "sets is_verified to true" do
      {:ok, user} = create_user()
      assert user.is_verified == false

      assert {:ok, updated} = Accounts.verify_user(user)
      assert updated.is_verified == true
    end
  end

  describe "update_preferences/2" do
    test "merges preferences" do
      {:ok, user} = create_user()

      assert {:ok, updated} = Accounts.update_preferences(user, %{"theme" => "dark"})
      assert updated.preferences["theme"] == "dark"

      assert {:ok, updated2} = Accounts.update_preferences(updated, %{"language" => "pt"})
      assert updated2.preferences["theme"] == "dark"
      assert updated2.preferences["language"] == "pt"
    end
  end

  describe "get_preference/3" do
    test "returns preference value" do
      {:ok, user} = create_user()
      {:ok, updated_user} = Accounts.update_preferences(user, %{"theme" => "dark"})

      assert Accounts.get_preference(updated_user, "theme") == "dark"
    end

    test "returns default for missing preference" do
      {:ok, user} = create_user()

      assert Accounts.get_preference(user, "missing", "default") == "default"
    end
  end

  # Helper functions

  defp unique_email do
    "test#{System.unique_integer([:positive])}@example.com"
  end

  defp unique_username do
    "user#{System.unique_integer([:positive])}"
  end

  defp create_user(attrs \\ []) do
    default_attrs = %{
      email: Keyword.get(attrs, :email, unique_email()),
      username: Keyword.get(attrs, :username, unique_username()),
      password: "SecurePass123"
    }

    attrs
    |> Map.new()
    |> then(&Map.merge(default_attrs, &1))
    |> Accounts.create_user()
  end
end
