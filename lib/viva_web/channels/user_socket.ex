defmodule VivaWeb.UserSocket do
  @moduledoc """
  Socket handler for user connections.
  Manages authentication and channel definitions.
  """
  use Phoenix.Socket

  # Channels
  channel "avatar:*", VivaWeb.AvatarChannel
  channel "world:*", VivaWeb.WorldChannel

  @impl Phoenix.Socket
  def connect(%{"token" => token}, socket, _) do
    case verify_token(token) do
      {:ok, user_id} ->
        {:ok, assign(socket, :user_id, user_id)}

      {:error, _} ->
        :error
    end
  end

  @impl Phoenix.Socket
  def connect(_, _, _) do
    :error
  end

  @impl Phoenix.Socket
  def id(socket), do: "user_socket:#{socket.assigns.user_id}"

  # Token verification
  defp verify_token(token) do
    case Phoenix.Token.verify(VivaWeb.Endpoint, "user socket", token, max_age: 86_400) do
      {:ok, user_id} -> {:ok, user_id}
      {:error, reason} -> {:error, reason}
    end
  end
end
