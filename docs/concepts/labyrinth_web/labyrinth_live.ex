defmodule VivaWeb.LabyrinthLive do
  use VivaWeb, :live_view
  # Unused alias removed

  @topic "world:updates"
  # SNES Resolution scale (8x8 tiles)
  @scale 8

  @impl true
  def mount(_params, _session, socket) do
    if connected?(socket) do
      Phoenix.PubSub.subscribe(VivaWeb.PubSub, @topic)
    end

    # Fetch the Shared Reality
    world_state = VivaCore.World.Observer.get_state()

    # world_state has keys: :grid, :width, :height, :pos, :seed
    # We wrap it in a struct-like map for the template to consume easily if needed,
    # or just pass the map. The template expects @labyrinth.grid, .width etc.
    # The Observer state map has the same keys as the Generator struct + :pos.

    labyrinth = %{
      grid: world_state.grid,
      width: world_state.width,
      height: world_state.height,
      seed: world_state.seed
    }

    {:ok,
     assign(socket,
       labyrinth: labyrinth,
       agent_pos: world_state.pos,
       page_title: "VIVA :: Visual Cortex",
       scale: @scale
     )}
  end

  @impl true
  def render(assigns) do
    ~H"""
    <div class="w-full h-screen bg-black flex items-center justify-center overflow-hidden">
      <!--
        The Canvas Hook.
        "phx-hook" connects this div to assets/js/hooks/labyrinth_renderer.js
        snes-canvas class handles the specific aspect ratio and scanlines via CSS.
      -->
      <div
        id="labyrinth-viewport"
        phx-hook="LabyrinthRenderer"
        data-grid={encode_grid(@labyrinth.grid, @labyrinth.width, @labyrinth.height)}
        data-width={@labyrinth.width}
        data-height={@labyrinth.height}
        data-agent-x={elem(@agent_pos, 0)}
        data-agent-y={elem(@agent_pos, 1)}
        class="relative border-2 border-green-900 shadow-[0_0_20px_rgba(0,255,0,0.2)]"
        style={"width: #{@labyrinth.width * @scale}px; height: #{@labyrinth.height * @scale}px;"}
      >
        <canvas
          id="snes-canvas"
          width={@labyrinth.width * @scale}
          height={@labyrinth.height * @scale}
          class="block"
        >
        </canvas>
        
    <!-- CRT Overlay/Scanlines -->
        <div class="pointer-events-none absolute inset-0 bg-[url('/images/scanlines.png')] opacity-30">
        </div>
        <div class="pointer-events-none absolute inset-0 bg-green-500 Mix-blend-overlay opacity-5 animate-pulse">
        </div>
      </div>
      
    <!-- Overlay UI (Mycelial Data) -->
      <div class="absolute bottom-4 left-4 text-xs font-mono text-green-700">
        <p>SEED: {@labyrinth.seed}</p>
        <p>ENTROPY: NOMINAL</p>
        <p>SPIN_NET: {map_size(@labyrinth.grid)} NODES</p>
      </div>
    </div>
    """
  end

  @impl true
  def handle_event("keydown", %{"key" => key}, socket) do
    direction =
      case key do
        "ArrowUp" -> :up
        "w" -> :up
        "ArrowDown" -> :down
        "s" -> :down
        "ArrowLeft" -> :left
        "a" -> :left
        "ArrowRight" -> :right
        "d" -> :right
        _ -> nil
      end

    if direction do
      VivaCore.World.Observer.move(direction)
    end

    {:noreply, socket}
  end

  # Helper to compress the grid for the frontend (Map -> JSON List/Binary)
  @impl true
  def handle_info({:universe_reset, new_state}, socket) do
    # The Universe has reset. We must re-render the entire reality.
    socket =
      socket
      |> assign(:labyrinth, %{
        grid: new_state.grid,
        width: new_state.width,
        height: new_state.height,
        seed: new_state.seed
      })
      |> assign(:agent_pos, new_state.pos)

    # Push event to JS to redraw the whole canvas
    {:noreply,
     push_event(socket, "reset_labyrinth", %{
       grid: encode_grid(new_state.grid, new_state.width, new_state.height),
       width: new_state.width,
       height: new_state.height,
       agent_x: elem(new_state.pos, 0),
       agent_y: elem(new_state.pos, 1)
     })}
  end

  @impl true
  def handle_info({:observer_moved, new_pos}, socket) do
    # Push event to JS Hook to update the canvas agent drawing without full re-render
    # Or just re-assign. Re-assign is cheaper here since it's just one coordinate.
    # But wait, my template doesn't use @agent_pos... I need to update the template too.
    # For now, let's just push the event to the hook.

    {:noreply,
     socket
     |> assign(agent_pos: new_pos)
     |> push_event("update_agent", %{x: elem(new_pos, 0), y: elem(new_pos, 1)})}
  end

  defp encode_grid(grid, width, height) do
    # Simple list of lists or flat list for now.
    # Format: [tile_type, tile_type, ...] row by row
    for y <- 0..(height - 1) do
      for x <- 0..(width - 1) do
        Map.get(grid, {x, y}, 0)
      end
    end
    |> Jason.encode!()
  end
end
