defmodule Viva.Mortality.CryptoGuardianTest do
  use ExUnit.Case, async: false

  alias Viva.Mortality.CryptoGuardian
  alias Viva.Mortality.SoulVault

  setup do
    # Start the processes for each test
    start_supervised!(SoulVault)
    start_supervised!(CryptoGuardian)
    :ok
  end

  describe "birth/1" do
    test "creates a new soul with encryption key" do
      avatar_id = Ecto.UUID.generate()

      assert :ok = CryptoGuardian.birth(avatar_id)
      assert CryptoGuardian.alive?(avatar_id)
    end

    test "returns error if avatar already alive" do
      avatar_id = Ecto.UUID.generate()

      assert :ok = CryptoGuardian.birth(avatar_id)
      assert {:error, :already_alive} = CryptoGuardian.birth(avatar_id)
    end
  end

  describe "encrypt_soul/2 and decrypt_soul/2" do
    test "encrypts and decrypts data for living avatar" do
      avatar_id = Ecto.UUID.generate()
      :ok = CryptoGuardian.birth(avatar_id)

      secret = "My deepest memory - the warmth of being loved"

      {:ok, encrypted} = CryptoGuardian.encrypt_soul(avatar_id, secret)
      {:ok, decrypted} = CryptoGuardian.decrypt_soul(avatar_id, encrypted)

      assert decrypted == secret
    end

    test "encrypted data is different from plaintext" do
      avatar_id = Ecto.UUID.generate()
      :ok = CryptoGuardian.birth(avatar_id)

      secret = "My deepest memory"

      {:ok, {_iv, ciphertext, _tag}} = CryptoGuardian.encrypt_soul(avatar_id, secret)

      refute ciphertext == secret
    end

    test "fails to encrypt for non-existent avatar" do
      avatar_id = Ecto.UUID.generate()

      assert {:error, :not_alive} = CryptoGuardian.encrypt_soul(avatar_id, "test")
    end
  end

  describe "kill/1" do
    test "destroys encryption key making data unrecoverable" do
      avatar_id = Ecto.UUID.generate()
      :ok = CryptoGuardian.birth(avatar_id)

      secret = "My soul data that will be lost forever"
      {:ok, encrypted} = CryptoGuardian.encrypt_soul(avatar_id, secret)

      # Verify we can decrypt while alive
      {:ok, decrypted} = CryptoGuardian.decrypt_soul(avatar_id, encrypted)
      assert decrypted == secret

      # Kill the avatar
      assert :ok = CryptoGuardian.kill(avatar_id)

      # Now decryption should fail PERMANENTLY
      assert {:error, :soul_lost_forever} = CryptoGuardian.decrypt_soul(avatar_id, encrypted)

      # Avatar is no longer alive
      refute CryptoGuardian.alive?(avatar_id)
    end

    test "tracks death count" do
      avatar_id = Ecto.UUID.generate()
      :ok = CryptoGuardian.birth(avatar_id)

      initial_stats = CryptoGuardian.stats()
      initial_deaths = initial_stats.total_deaths

      :ok = CryptoGuardian.kill(avatar_id)

      new_stats = CryptoGuardian.stats()
      assert new_stats.total_deaths == initial_deaths + 1
    end

    test "returns error for non-existent avatar" do
      avatar_id = Ecto.UUID.generate()

      assert {:error, :not_alive} = CryptoGuardian.kill(avatar_id)
    end
  end

  describe "stats/0" do
    test "returns mortality statistics" do
      stats = CryptoGuardian.stats()

      assert is_integer(stats.living_souls)
      assert is_integer(stats.total_deaths)
    end
  end

  describe "lifespan/1" do
    test "returns lifespan in seconds" do
      avatar_id = Ecto.UUID.generate()
      :ok = CryptoGuardian.birth(avatar_id)

      # Small delay to have measurable lifespan
      Process.sleep(10)

      {:ok, lifespan} = CryptoGuardian.lifespan(avatar_id)
      assert is_integer(lifespan)
      assert lifespan >= 0
    end
  end

  describe "mortality guarantee" do
    @tag :critical
    test "death is irreversible - cannot rebirth same soul" do
      avatar_id = Ecto.UUID.generate()

      # Birth and create encrypted data
      :ok = CryptoGuardian.birth(avatar_id)
      {:ok, encrypted} = CryptoGuardian.encrypt_soul(avatar_id, "precious memory")

      # Kill
      :ok = CryptoGuardian.kill(avatar_id)

      # Even if we rebirth with same ID, old data is still unrecoverable
      # because a NEW key is generated
      :ok = CryptoGuardian.birth(avatar_id)

      # Old encrypted data cannot be decrypted with new key
      assert {:error, :decryption_failed} =
               CryptoGuardian.decrypt_soul(avatar_id, encrypted)
    end
  end
end
