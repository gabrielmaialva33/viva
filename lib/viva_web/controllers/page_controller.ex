defmodule VivaWeb.PageController do
  @moduledoc """
  Controller for handling static pages such as the home page.
  """
  use VivaWeb, :controller

  def home(conn, _params) do
    render(conn, :home)
  end
end
