defmodule Viva.Accounts.UserTest do
  use Viva.DataCase, async: true

  alias Viva.Accounts.User

  describe "changeset/2" do
    test "valid changeset with required fields" do
      changeset =
        User.changeset(%User{}, %{
          email: "test@example.com",
          username: "testuser"
        })

      assert changeset.valid?
    end

    test "invalid without email" do
      changeset = User.changeset(%User{}, %{username: "testuser"})
      refute changeset.valid?
      assert "can't be blank" in errors_on(changeset).email
    end

    test "invalid without username" do
      changeset = User.changeset(%User{}, %{email: "test@example.com"})
      refute changeset.valid?
      assert "can't be blank" in errors_on(changeset).username
    end

    test "validates email format" do
      changeset =
        User.changeset(%User{}, %{
          email: "invalid-email",
          username: "testuser"
        })

      refute changeset.valid?
      assert "must be a valid email" in errors_on(changeset).email
    end

    test "validates email length" do
      long_email = String.duplicate("a", 200) <> "@example.com"

      changeset =
        User.changeset(%User{}, %{
          email: long_email,
          username: "testuser"
        })

      refute changeset.valid?
      assert "should be at most 160 character(s)" in errors_on(changeset).email
    end

    test "validates username format" do
      changeset =
        User.changeset(%User{}, %{
          email: "test@example.com",
          username: "user with spaces"
        })

      refute changeset.valid?
      assert "only letters, numbers, and underscores" in errors_on(changeset).username
    end

    test "validates username min length" do
      changeset =
        User.changeset(%User{}, %{
          email: "test@example.com",
          username: "ab"
        })

      refute changeset.valid?
      assert "should be at least 3 character(s)" in errors_on(changeset).username
    end

    test "validates username max length" do
      changeset =
        User.changeset(%User{}, %{
          email: "test@example.com",
          username: String.duplicate("a", 31)
        })

      refute changeset.valid?
      assert "should be at most 30 character(s)" in errors_on(changeset).username
    end

    test "allows underscores in username" do
      changeset =
        User.changeset(%User{}, %{
          email: "test@example.com",
          username: "test_user_123"
        })

      assert changeset.valid?
    end

    test "downcases email" do
      changeset =
        User.changeset(%User{}, %{
          email: "TEST@EXAMPLE.COM",
          username: "testuser"
        })

      assert changeset.valid?
      assert Ecto.Changeset.get_change(changeset, :email) == "test@example.com"
    end

    test "downcases username" do
      changeset =
        User.changeset(%User{}, %{
          email: "test@example.com",
          username: "TestUser"
        })

      assert changeset.valid?
      assert Ecto.Changeset.get_change(changeset, :username) == "testuser"
    end
  end

  describe "registration_changeset/2" do
    test "valid with all required fields" do
      changeset =
        User.registration_changeset(%User{}, %{
          email: "test@example.com",
          username: "testuser",
          password: "SecurePass123"
        })

      assert changeset.valid?
      assert Ecto.Changeset.get_change(changeset, :hashed_password) != nil
    end

    test "requires password" do
      changeset =
        User.registration_changeset(%User{}, %{
          email: "test@example.com",
          username: "testuser"
        })

      refute changeset.valid?
      assert "can't be blank" in errors_on(changeset).password
    end

    test "validates password min length" do
      changeset =
        User.registration_changeset(%User{}, %{
          email: "test@example.com",
          username: "testuser",
          password: "Short1A"
        })

      refute changeset.valid?
      assert "should be at least 8 character(s)" in errors_on(changeset).password
    end

    test "validates password max length" do
      long_password = String.duplicate("A1a", 25)

      changeset =
        User.registration_changeset(%User{}, %{
          email: "test@example.com",
          username: "testuser",
          password: long_password
        })

      refute changeset.valid?
      assert "should be at most 72 character(s)" in errors_on(changeset).password
    end

    test "validates password has lowercase" do
      changeset =
        User.registration_changeset(%User{}, %{
          email: "test@example.com",
          username: "testuser",
          password: "NOLOWERCASE123"
        })

      refute changeset.valid?
      assert "must have at least one lowercase letter" in errors_on(changeset).password
    end

    test "validates password has uppercase" do
      changeset =
        User.registration_changeset(%User{}, %{
          email: "test@example.com",
          username: "testuser",
          password: "nouppercase123"
        })

      refute changeset.valid?
      assert "must have at least one uppercase letter" in errors_on(changeset).password
    end

    test "validates password has digit" do
      changeset =
        User.registration_changeset(%User{}, %{
          email: "test@example.com",
          username: "testuser",
          password: "NoDigitsHere"
        })

      refute changeset.valid?
      assert "must have at least one digit" in errors_on(changeset).password
    end

    test "hashes password" do
      changeset =
        User.registration_changeset(%User{}, %{
          email: "test@example.com",
          username: "testuser",
          password: "SecurePass123"
        })

      hashed = Ecto.Changeset.get_change(changeset, :hashed_password)
      assert hashed != "SecurePass123"
      assert String.starts_with?(hashed, "$2b$")
    end
  end

  describe "verify_password/2" do
    test "returns true for correct password" do
      {:ok, user} = create_user("SecurePass123")
      assert User.verify_password(user, "SecurePass123") == true
    end

    test "returns false for incorrect password" do
      {:ok, user} = create_user("SecurePass123")
      assert User.verify_password(user, "WrongPassword1") == false
    end

    test "returns false for nil user" do
      assert User.verify_password(nil, "any_password") == false
    end
  end

  describe "query functions" do
    test "active/0 returns query for active users" do
      query = User.active()
      assert is_struct(query, Ecto.Query)
    end

    test "by_email/1 returns query for email" do
      query = User.by_email("test@example.com")
      assert is_struct(query, Ecto.Query)
    end

    test "by_email/1 downcases email" do
      {:ok, user} = create_user("SecurePass123", email: "test@example.com")

      found =
        "TEST@EXAMPLE.COM"
        |> User.by_email()
        |> Repo.one()

      assert found.id == user.id
    end

    test "by_username/1 returns query for username" do
      query = User.by_username("testuser")
      assert is_struct(query, Ecto.Query)
    end

    test "by_username/1 downcases username" do
      {:ok, user} = create_user("SecurePass123", username: "testuser")

      found =
        "TESTUSER"
        |> User.by_username()
        |> Repo.one()

      assert found.id == user.id
    end
  end

  # Helper

  defp create_user(password, attrs \\ []) do
    %User{}
    |> User.registration_changeset(%{
      email: Keyword.get(attrs, :email, "test#{System.unique_integer([:positive])}@example.com"),
      username: Keyword.get(attrs, :username, "user#{System.unique_integer([:positive])}"),
      password: password
    })
    |> Repo.insert()
  end
end
