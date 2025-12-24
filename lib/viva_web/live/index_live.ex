defmodule VivaWeb.IndexLive do
  @moduledoc """
  Main index page.
  """
  use VivaWeb, :live_view

  @impl Phoenix.LiveView
  def mount(_, _, socket) do
    {:ok, socket}
  end

  @impl Phoenix.LiveView
  def render(assigns) do
    ~H"""
    <Layouts.app flash={@flash}>
      <div class="flex flex-col items-center justify-center min-h-[60vh]">
        <h1 class="text-4xl font-bold text-base-content">VIVA</h1>
        <p class="mt-4 text-base-content/70">AI Avatar Platform</p>
      </div>
    </Layouts.app>
    """
  end
end
