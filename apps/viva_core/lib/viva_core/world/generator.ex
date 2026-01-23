defmodule VivaCore.World.Generator do
  @moduledoc """
  The Architect of the Inevitable.
  Generates deterministic labyrinths based on a cryptographic seed.

  "All You Zombies" - The end is in the beginning.
  The seed determines the entire history and future of the map.
  """

  # Tile Types (SNES/Voxel Mapping)
  # 0 = VOID (Abyss)
  # 1 = WALL (Structure)
  # 2 = PATH (Data Flow)
  # 3 = CORE (Leviathan)

  @tile_wall 1
  @tile_path 2
  @tile_core 3

  defstruct [:seed, :width, :height, :grid, :start_pos]

  @doc """
  Generates a new labyrinth.
  Seed must be an integer or string.
  """
  def generate(seed, width \\ 32, height \\ 32) do
    # Ensure deterministic RNG
    int_seed = hash_seed(seed)
    :rand.seed(:exsss, {int_seed, int_seed, int_seed})

    # Initialize full wall grid
    initial_grid =
      Map.new(for x <- 0..(width - 1), y <- 0..(height - 1), do: {{x, y}, @tile_wall})

    # Start at random point (must be odd coordinates for Recursive Backtracker)
    start_x = 1 + 2 * :rand.uniform(div(width - 1, 2) - 1)
    start_y = 1 + 2 * :rand.uniform(div(height - 1, 2) - 1)

    # Mark start as path
    grid_with_start = Map.put(initial_grid, {start_x, start_y}, @tile_path)

    # Carve the path (Iterative Stack)
    final_grid =
      carve_iter(grid_with_start, [{start_x, start_y}], width, height)
      |> place_core(width, height)

    %__MODULE__{
      seed: seed,
      width: width,
      height: height,
      grid: final_grid,
      start_pos: {start_x, start_y}
    }
  end

  # Tail-recursive loop with explicit stack
  defp carve_iter(grid, [], _, _), do: grid

  defp carve_iter(grid, [{cx, cy} | stack_tail] = stack, width, height) do
    # 2 distance for walls
    neighbors = [{0, -2}, {0, 2}, {2, 0}, {-2, 0}]

    valid_neighbors =
      neighbors
      |> Enum.map(fn {dx, dy} -> {{cx + dx, cy + dy}, {dx, dy}} end)
      |> Enum.filter(fn {{nx, ny}, _} ->
        in_bounds?(nx, ny, width, height) and Map.get(grid, {nx, ny}) == @tile_wall
      end)

    case valid_neighbors do
      [] ->
        # Backtrack: continue with the rest of the stack
        carve_iter(grid, stack_tail, width, height)

      _ ->
        # Choose random neighbor
        {{nx, ny}, {dx, dy}} = Enum.random(valid_neighbors)

        # Remove wall between (mid point)
        mid_x = cx + div(dx, 2)
        mid_y = cy + div(dy, 2)

        new_grid =
          grid
          |> Map.put({mid_x, mid_y}, @tile_path)
          |> Map.put({nx, ny}, @tile_path)

        # Push neighbor to stack (RB keeps current in stack to branch later)
        carve_iter(new_grid, [{nx, ny} | stack], width, height)
    end
  end

  defp place_core(grid, width, height) do
    # Place the "Leviathan Core" at exact center
    center_x = div(width, 2)
    center_y = div(height, 2)

    grid
    |> Map.put({center_x, center_y}, @tile_core)
    # Ensure connectivity
    |> Map.put({center_x + 1, center_y}, @tile_path)
    |> Map.put({center_x - 1, center_y}, @tile_path)
    |> Map.put({center_x, center_y + 1}, @tile_path)
    |> Map.put({center_x, center_y - 1}, @tile_path)
  end

  defp in_bounds?(x, y, width, height) do
    x > 0 and y > 0 and x < width - 1 and y < height - 1
  end

  defp hash_seed(seed) when is_binary(seed), do: :erlang.phash2(seed)
  defp hash_seed(seed) when is_integer(seed), do: seed
  defp hash_seed(_), do: :os.system_time(:millisecond)

  @doc """
  Mutates the seed to create the "Next Loop".
  This is the implementation of the "Eternal Return".
  It combines the current seed with a "Quantum Flux" (entropy) to generate a new,
  deterministically chaotic seed.
  """
  def mutate_seed(current_seed, entropy \\ :os.system_time(:microsecond)) do
    # Narrative: The "Wave Function" of the universe collapsing into a new state.
    # We use a crypto hash to ensure small changes in input (flux) cause massive changes in output (The Butterfly Effect).

    input_data = "#{current_seed}:#{entropy}:#{inspect(self())}"

    :crypto.hash(:sha256, input_data)
    |> Base.encode16()
    # SNES-like limitation? Keep it somewhat readable/short if needed, or full hash.
    |> String.slice(0..16)
  end
end
