defmodule VivaWeb.PageController do
  @moduledoc """
  Controller for handling static pages such as the home page.
  """
  use VivaWeb, :controller

  @spec home(Plug.Conn.t(), map()) :: Plug.Conn.t()
  def home(conn, _) do
    render(conn, :home)
  end
end
