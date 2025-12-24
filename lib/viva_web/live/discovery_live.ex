defmodule VivaWeb.DiscoveryLive do
  use VivaWeb, :live_view

  @impl true
  def mount(_params, _session, socket) do
    {:ok, assign(socket, page_title: "VIVA - Discover")}
  end

  @impl true
  def render(assigns) do
    ~H"""
    <div class="h-screen flex flex-col items-center justify-center bg-gray-900 text-white">
      <div class="max-w-md w-full p-6 text-center">
        <h1 class="text-3xl font-bold mb-4">VIVA</h1>
        <p class="text-gray-400">Loading potential matches...</p>
        
        <!-- Placeholder for Swipe Card -->
        <div class="mt-8 h-96 bg-gray-800 rounded-2xl flex items-center justify-center border border-gray-700">
          <span class="text-gray-500">Card Area</span>
        </div>
      </div>
    </div>
    """
  end
end
