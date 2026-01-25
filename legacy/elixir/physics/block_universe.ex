defmodule VivaCore.Physics.BlockUniverse do
  @moduledoc """
  Wheeler-DeWitt / Block Universe Implementation.

  Premise: The universe exists all at once. Time is an illusion of the observer.
  Mechanic:
  - Calculates "Light Cones" (future possibilities)
  - Implements Retrocausal Pull (future states minimizing free energy pull the present)
  """
  require VivaLog

  @doc """
  Expands the Light Cone to see possible futures within radius_ticks.
  Returns a list of possible future states (positions).
  """
  def expand_light_cone(current_pos, _grid_state, radius_ticks) do
    # Simple BFS to find reachable tiles within radius
    traverse_futures(current_pos, radius_ticks, MapSet.new())
  end

  @doc """
  Calculates the Retrocausal Pull.
  Determines which immediate move minimizes Free Energy relative to a target future goal.

  The "Goal" acts as a magnet. The decision is made by the Future pulling the Present.
  """
  def retrocausal_pull(current_pos, possible_moves, goal_pos, current_energy) do
    # Which move gets us closer to goal with minimal energy expenditure?
    possible_moves
    |> Enum.map(fn move_dir ->
      next_pos = calculate_pos(current_pos, move_dir)
      distance = distance(next_pos, goal_pos)
      cost = estimated_cost(distance, current_energy)
      {move_dir, cost}
    end)
    |> Enum.min_by(fn {_dir, cost} -> cost end, fn -> {:stay, 999.9} end)
  end

  defp traverse_futures(pos, radius, visited) do
    # Start recursion
    initial_visited = MapSet.put(visited, pos)

    traverse_bfs_recursive(pos, radius, initial_visited)
    |> MapSet.to_list()
  end

  defp traverse_bfs_recursive(_pos, 0, visited), do: visited

  defp traverse_bfs_recursive(pos, radius, visited) do
    neighbors = get_neighbors(pos)

    Enum.reduce(neighbors, visited, fn neighbor, acc ->
      if MapSet.member?(acc, neighbor) do
        acc
      else
        new_acc = MapSet.put(acc, neighbor)
        traverse_bfs_recursive(neighbor, radius - 1, new_acc)
      end
    end)
  end

  defp get_neighbors({x, y}) do
    [{x, y - 1}, {x, y + 1}, {x - 1, y}, {x + 1, y}]
  end

  defp calculate_pos({x, y}, :up), do: {x, y - 1}
  defp calculate_pos({x, y}, :down), do: {x, y + 1}
  defp calculate_pos({x, y}, :left), do: {x - 1, y}
  defp calculate_pos({x, y}, :right), do: {x + 1, y}
  defp calculate_pos(pos, _), do: pos

  defp distance({x1, y1}, {x2, y2}) do
    :math.sqrt(:math.pow(x2 - x1, 2) + :math.pow(y2 - y1, 2))
  end

  defp estimated_cost(dist, _energy), do: dist
end
