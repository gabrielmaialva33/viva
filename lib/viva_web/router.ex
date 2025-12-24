defmodule VivaWeb.Router do
  @moduledoc """
  Defines the application's router and pipelines.
  """
  use VivaWeb, :router

  pipeline :browser do
    plug :accepts, ["html"]
    plug :fetch_session
    plug :fetch_live_flash
    plug :put_root_layout, html: {VivaWeb.Layouts, :root}
    plug :protect_from_forgery
    plug :put_secure_browser_headers
  end

  pipeline :api do
    plug :accepts, ["json"]
  end

  scope "/", VivaWeb do
    pipe_through :browser

    live "/", LandingLive
  end

  # Other scopes may use custom stacks.
  # scope "/api", VivaWeb do
  #   pipe_through :api
  # end
end
