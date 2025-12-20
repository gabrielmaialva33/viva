defmodule VivaWeb.UserSocket do
  use Phoenix.Socket

  # Channels
  channel "avatar:*", VivaWeb.AvatarChannel
  channel "world:*", VivaWeb.WorldChannel

  @impl true
  def connect(%{"token" => token}, socket, _connect_info) do
    case verify_token(token) do
      {:ok, user_id} ->
        {:ok, assign(socket, :user_id, user_id)}

      {:error, _reason} ->
        :error
    end
  end

  def connect(_params, _socket, _connect_info) do
    :error
  end

  @impl true
  def id(socket), do: "user_socket:#{socket.assigns.user_id}"

  # Token verification
  defp verify_token(token) do
    case Phoenix.Token.verify(VivaWeb.Endpoint, "user socket", token, max_age: 86400) do
      {:ok, user_id} -> {:ok, user_id}
      {:error, reason} -> {:error, reason}
    end
  end
end
