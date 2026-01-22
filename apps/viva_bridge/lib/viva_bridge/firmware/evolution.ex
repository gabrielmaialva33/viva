defmodule VivaBridge.Firmware.Evolution do
  @moduledoc """
  MAP-Elites: Quality-Diversity Algorithm for VIVA firmware evolution.

  Instead of finding ONE optimal firmware, MAP-Elites discovers a DIVERSE
  collection of high-performing firmwares across different behavioral niches.

  ## Behavioral Descriptors (2D Grid)

  1. **Energy Level** (0-1): Average normalized PWM across emotions
     - Low energy: calm, quiet, introspective firmwares
     - High energy: active, loud, expressive firmwares

  2. **Melody Complexity** (0-1): Musical richness of emotional expressions
     - Simple: few notes, narrow frequency range
     - Complex: many notes, wide frequency variations

  ## Archive Structure

      Energy →
      ┌─────┬─────┬─────┬─────┬─────┐
    C │     │     │     │     │     │ 1.0
    o ├─────┼─────┼─────┼─────┼─────┤
    m │     │ 🧘  │     │     │ 🎉  │
    p ├─────┼─────┼─────┼─────┼─────┤
    l │     │     │ 🤖  │     │     │
    e ├─────┼─────┼─────┼─────┼─────┤
    x │ 😴  │     │     │     │ 🏃  │
      └─────┴─────┴─────┴─────┴─────┘ 0.0
       0.0                      1.0

  Each cell contains the BEST firmware for that behavioral niche.

  ## Algorithm

      1. Initialize: Random population → evaluate → add to archive
      2. Loop:
         a. Select random parent from archive
         b. Mutate to create offspring
         c. Evaluate fitness + compute behavioral descriptors
         d. If cell empty OR offspring.fitness > cell.fitness:
              archive[cell] = offspring
      3. Result: Archive full of diverse, high-quality firmwares

  ## References

  Mouret & Clune (2015) "Illuminating search spaces by mapping elites"
  https://arxiv.org/abs/1504.04909
  """

  require Logger

  alias VivaBridge.Firmware.{Codegen, Uploader}

  # Grid dimensions
  # 5x5 = 25 behavioral niches
  @grid_size 5
  # energy_level, melody_complexity
  @dimensions 2

  # Evolution parameters
  @initial_population 20
  @iterations_per_run 50
  # Parameters
  # (Attributes removed as they were unused)

  # ============================================================================
  # Public API
  # ============================================================================

  @doc """
  Initialize a new MAP-Elites archive with random genotypes.
  """
  def initialize do
    Logger.info("[MAP-Elites] Initializing archive (#{@grid_size}x#{@grid_size} grid)")

    archive = create_empty_archive()

    # Generate initial random population
    genotypes = for _ <- 1..@initial_population, do: random_genotype()

    # Add each to archive
    archive =
      Enum.reduce(genotypes, archive, fn genotype, acc ->
        add_to_archive(acc, genotype)
      end)

    filled = count_filled_cells(archive)
    Logger.info("[MAP-Elites] Initial archive: #{filled}/#{@grid_size * @grid_size} cells filled")

    archive
  end

  @doc """
  Run MAP-Elites for N iterations.

  Options:
    - `:iterations` - Number of iterations (default: #{@iterations_per_run})
    - `:evaluate_live` - If true, uploads firmware and measures real Free Energy
    - `:archive` - Existing archive to continue from
  """
  def run(opts \\ []) do
    iterations = Keyword.get(opts, :iterations, @iterations_per_run)
    evaluate_live = Keyword.get(opts, :evaluate_live, false)
    archive = Keyword.get(opts, :archive) || initialize()

    Logger.info(
      "[MAP-Elites] Starting evolution: #{iterations} iterations, live=#{evaluate_live}"
    )

    final_archive =
      Enum.reduce(1..iterations, archive, fn i, acc ->
        # Select random parent from archive
        case select_parent(acc) do
          nil ->
            Logger.warning("[MAP-Elites] Archive empty, reinitializing")
            initialize()

          parent ->
            # Create offspring via mutation
            offspring = mutate(parent)

            # Evaluate fitness
            offspring =
              if evaluate_live do
                evaluate_live(offspring)
              else
                evaluate_simulated(offspring)
              end

            # Add to archive (replaces if better)
            new_archive = add_to_archive(acc, offspring)

            # Progress log every 10 iterations
            if rem(i, 10) == 0 do
              filled = count_filled_cells(new_archive)
              best = best_fitness(new_archive)

              Logger.info(
                "[MAP-Elites] Iteration #{i}: #{filled} cells, best fitness: #{Float.round(best, 4)}"
              )
            end

            new_archive
        end
      end)

    # Summary
    summary = summarize_archive(final_archive)

    Logger.info(
      "[MAP-Elites] Evolution complete: #{summary.filled_cells} cells, coverage: #{summary.coverage}%"
    )

    {:ok, final_archive, summary}
  end

  @doc """
  Get the best genotype for a specific behavioral niche.
  """
  def get_niche(archive, energy_level, complexity) when energy_level >= 0 and energy_level <= 1 do
    {row, col} = descriptor_to_cell(energy_level, complexity)
    Map.get(archive, {row, col})
  end

  @doc """
  Get all filled cells in the archive.
  """
  def get_elites(archive) do
    archive
    |> Enum.filter(fn {_cell, genotype} -> genotype != nil end)
    |> Enum.map(fn {cell, genotype} -> {cell, genotype} end)
    |> Enum.into(%{})
  end

  @doc """
  Export archive to storable format (for Qdrant/persistence).
  """
  def export(archive) do
    elites = get_elites(archive)

    %{
      grid_size: @grid_size,
      dimensions: @dimensions,
      elites:
        Enum.map(elites, fn {{row, col}, genotype} ->
          %{
            cell: {row, col},
            genotype: genotype,
            descriptors: compute_descriptors(genotype),
            fitness: genotype.fitness
          }
        end),
      exported_at: DateTime.utc_now()
    }
  end

  @doc """
  Import archive from stored format.
  """
  def import(data) do
    archive = create_empty_archive()

    Enum.reduce(data.elites, archive, fn elite, acc ->
      Map.put(acc, elite.cell, elite.genotype)
    end)
  end

  # ============================================================================
  # Archive Operations
  # ============================================================================

  defp create_empty_archive do
    for row <- 0..(@grid_size - 1),
        col <- 0..(@grid_size - 1),
        into: %{} do
      {{row, col}, nil}
    end
  end

  defp add_to_archive(archive, genotype) do
    {energy, complexity} = compute_descriptors(genotype)
    {row, col} = descriptor_to_cell(energy, complexity)

    current = Map.get(archive, {row, col})

    if current == nil or genotype.fitness > current.fitness do
      Logger.debug(
        "[MAP-Elites] New elite at (#{row},#{col}): fitness=#{Float.round(genotype.fitness, 4)}"
      )

      Map.put(archive, {row, col}, genotype)
    else
      archive
    end
  end

  defp descriptor_to_cell(energy, complexity) do
    # Map [0,1] to [0, grid_size-1]
    row = min(floor(energy * @grid_size), @grid_size - 1)
    col = min(floor(complexity * @grid_size), @grid_size - 1)
    {row, col}
  end

  defp select_parent(archive) do
    filled =
      archive
      |> Enum.filter(fn {_cell, g} -> g != nil end)
      |> Enum.map(fn {_cell, g} -> g end)

    if Enum.empty?(filled), do: nil, else: Enum.random(filled)
  end

  defp count_filled_cells(archive) do
    Enum.count(archive, fn {_cell, g} -> g != nil end)
  end

  defp best_fitness(archive) do
    archive
    |> Enum.filter(fn {_cell, g} -> g != nil end)
    |> Enum.map(fn {_cell, g} -> g.fitness end)
    |> Enum.max(fn -> 0.0 end)
  end

  # ============================================================================
  # Behavioral Descriptors
  # ============================================================================

  @doc """
  Compute behavioral descriptors for a genotype.
  Returns {energy_level, melody_complexity} both in [0, 1].
  """
  def compute_descriptors(genotype) do
    energy = compute_energy_level(genotype)
    complexity = compute_melody_complexity(genotype)
    {energy, complexity}
  end

  # Energy level: average PWM normalized to [0, 1]
  defp compute_energy_level(genotype) do
    pwms = genotype.emotions |> Enum.map(fn {_name, data} -> data.pwm end)
    avg_pwm = Enum.sum(pwms) / max(length(pwms), 1)
    avg_pwm / 255.0
  end

  # Melody complexity: based on note count and frequency variance
  defp compute_melody_complexity(genotype) do
    melodies = genotype.emotions |> Enum.map(fn {_name, data} -> data.melody end)

    # Count total notes
    total_notes = melodies |> Enum.map(&length/1) |> Enum.sum()

    # Compute frequency variance
    all_freqs =
      melodies
      |> Enum.flat_map(fn melody -> Enum.map(melody, fn {freq, _dur} -> freq end) end)

    variance =
      if length(all_freqs) > 1 do
        mean = Enum.sum(all_freqs) / length(all_freqs)
        sq_diffs = Enum.map(all_freqs, fn f -> (f - mean) * (f - mean) end)
        Enum.sum(sq_diffs) / length(sq_diffs)
      else
        0
      end

    # Normalize: notes (0-50) and variance (0-50000)
    note_score = min(total_notes / 50.0, 1.0)
    variance_score = min(variance / 50000.0, 1.0)

    # Combined complexity
    (note_score + variance_score) / 2.0
  end

  # ============================================================================
  # Genotype Operations
  # ============================================================================

  defp random_genotype do
    base = Codegen.default_genotype()

    %{
      base
      | generation: 0,
        fitness: 0.0,
        emotions: random_emotions(),
        # 1.5 to 3.0
        harmony_ratio: 1.5 + :rand.uniform() * 1.5,
        timer: %{prescaler: 1, top: random_timer_top()}
    }
  end

  defp random_emotions do
    emotion_names = [:joy, :sad, :fear, :calm, :curious, :love]

    for name <- emotion_names, into: %{} do
      {name,
       %{
         # 0-255
         pwm: :rand.uniform(256) - 1,
         # 3-7 notes
         melody: random_melody(3 + :rand.uniform(5)),
         repeat: if(name == :fear, do: 3 + :rand.uniform(5), else: nil)
       }}
    end
  end

  defp random_melody(num_notes) do
    # Musical frequencies (C4 to C6)
    base_freqs = [262, 294, 330, 349, 392, 440, 494, 523, 587, 659, 698, 784, 880, 988, 1047]

    for _ <- 1..num_notes do
      freq = Enum.random(base_freqs)
      dur = Enum.random([50, 100, 150, 200, 250, 300, 400, 500])
      {freq, dur}
    end
  end

  defp random_timer_top do
    # 20kHz to 30kHz range: TOP = 16MHz / (1 * freq) - 1
    # freq 25kHz → TOP 639
    # freq 20kHz → TOP 799
    # freq 30kHz → TOP 532
    # 532 to 800
    532 + :rand.uniform(268)
  end

  # ============================================================================
  # Mutation Operators
  # ============================================================================

  defp mutate(parent) do
    # Apply multiple mutation types
    offspring =
      parent
      |> mutate_emotions()
      |> mutate_harmony()
      |> mutate_timer()
      |> Map.put(:generation, parent.generation + 1)
      # Reset fitness for re-evaluation
      |> Map.put(:fitness, 0.0)

    offspring
  end

  defp mutate_emotions(genotype) do
    emotions =
      genotype.emotions
      |> Enum.map(fn {name, data} ->
        new_data =
          if :rand.uniform() < 0.3 do
            # Mutate PWM with Gaussian noise
            new_pwm = data.pwm + round(:rand.normal() * 30)
            new_pwm = max(0, min(255, new_pwm))

            # Possibly mutate melody
            new_melody =
              if :rand.uniform() < 0.2 do
                mutate_melody(data.melody)
              else
                data.melody
              end

            %{data | pwm: new_pwm, melody: new_melody}
          else
            data
          end

        {name, new_data}
      end)
      |> Enum.into(%{})

    %{genotype | emotions: emotions}
  end

  defp mutate_melody(melody) do
    case :rand.uniform(4) do
      1 ->
        # Add a note
        freq = Enum.random([262, 294, 330, 392, 440, 523, 587, 659])
        dur = Enum.random([100, 150, 200, 300])
        melody ++ [{freq, dur}]

      2 ->
        # Remove a note (keep at least 2)
        if length(melody) > 2 do
          idx = :rand.uniform(length(melody)) - 1
          List.delete_at(melody, idx)
        else
          melody
        end

      3 ->
        # Modify a note's frequency
        if length(melody) > 0 do
          idx = :rand.uniform(length(melody)) - 1
          {_old_freq, dur} = Enum.at(melody, idx)
          new_freq = Enum.random([262, 294, 330, 392, 440, 523, 587, 659])
          List.replace_at(melody, idx, {new_freq, dur})
        else
          melody
        end

      4 ->
        # Modify a note's duration
        if length(melody) > 0 do
          idx = :rand.uniform(length(melody)) - 1
          {freq, _old_dur} = Enum.at(melody, idx)
          new_dur = Enum.random([50, 100, 150, 200, 300, 400])
          List.replace_at(melody, idx, {freq, new_dur})
        else
          melody
        end
    end
  end

  defp mutate_harmony(genotype) do
    if :rand.uniform() < 0.1 do
      new_ratio = genotype.harmony_ratio + :rand.normal() * 0.2
      new_ratio = max(1.5, min(3.0, new_ratio))
      %{genotype | harmony_ratio: new_ratio}
    else
      genotype
    end
  end

  defp mutate_timer(genotype) do
    if :rand.uniform() < 0.05 do
      new_top = genotype.timer.top + round(:rand.normal() * 20)
      new_top = max(532, min(800, new_top))
      %{genotype | timer: %{genotype.timer | top: new_top}}
    else
      genotype
    end
  end

  # ============================================================================
  # Fitness Evaluation
  # ============================================================================

  # Simulated fitness (no hardware) - based on genotype properties
  defp evaluate_simulated(genotype) do
    # Fitness components:
    # 1. PWM diversity (different emotions = different behaviors)
    # 2. Melody coherence (notes should be musically sensible)
    # 3. Timer validity (25kHz is optimal for Intel fans)

    pwm_diversity = compute_pwm_diversity(genotype)
    melody_coherence = compute_melody_coherence(genotype)
    timer_score = compute_timer_score(genotype)

    # Combined fitness (0 to 1)
    fitness = pwm_diversity * 0.4 + melody_coherence * 0.4 + timer_score * 0.2

    %{genotype | fitness: fitness}
  end

  defp compute_pwm_diversity(genotype) do
    pwms = genotype.emotions |> Enum.map(fn {_name, data} -> data.pwm end)

    if length(pwms) > 1 do
      mean = Enum.sum(pwms) / length(pwms)
      variance = Enum.sum(Enum.map(pwms, fn p -> (p - mean) * (p - mean) end)) / length(pwms)
      # Normalize: high variance (0-5000) is good
      min(variance / 5000.0, 1.0)
    else
      0.0
    end
  end

  defp compute_melody_coherence(genotype) do
    # Check if melodies use consonant intervals
    melodies = genotype.emotions |> Enum.map(fn {_name, data} -> data.melody end)

    scores =
      Enum.map(melodies, fn melody ->
        if length(melody) > 1 do
          freqs = Enum.map(melody, fn {f, _d} -> f end)
          # Check consecutive frequency ratios
          pairs = Enum.zip(freqs, tl(freqs))
          ratios = Enum.map(pairs, fn {a, b} -> max(a, b) / min(a, b) end)

          # Consonant ratios: 1.0, 1.25, 1.33, 1.5, 2.0
          consonant_count =
            Enum.count(ratios, fn r ->
              Enum.any?([1.0, 1.25, 1.33, 1.5, 2.0], fn c -> abs(r - c) < 0.1 end)
            end)

          consonant_count / max(length(ratios), 1)
        else
          # Single note is neutral
          0.5
        end
      end)

    Enum.sum(scores) / max(length(scores), 1)
  end

  defp compute_timer_score(genotype) do
    # Optimal TOP for 25kHz = 639
    optimal = 639
    diff = abs(genotype.timer.top - optimal)
    # Score decreases as we deviate from optimal
    max(0, 1.0 - diff / 200.0)
  end

  # Live fitness evaluation - uploads firmware and measures real Free Energy
  defp evaluate_live(genotype) do
    Logger.info("[MAP-Elites] Live evaluation starting...")

    case Codegen.generate(genotype) do
      {:ok, ino_code} ->
        case Uploader.deploy(ino_code, generation: genotype.generation) do
          {:ok, _} ->
            # Wait for system to stabilize
            Process.sleep(5_000)

            # Measure Free Energy from interoception
            fitness = measure_free_energy()
            Logger.info("[MAP-Elites] Live fitness: #{Float.round(fitness, 4)}")

            %{genotype | fitness: fitness}

          {:error, reason} ->
            Logger.warning("[MAP-Elites] Deploy failed: #{inspect(reason)}, using simulated")
            evaluate_simulated(genotype)
        end

      {:error, reason} ->
        Logger.warning("[MAP-Elites] Codegen failed: #{inspect(reason)}, using simulated")
        evaluate_simulated(genotype)
    end
  end

  defp measure_free_energy do
    # Get interoception state from Music module
    case VivaBridge.Music.interoception_state() do
      %{average_prediction_error: error} when is_number(error) ->
        # Convert prediction error to fitness (lower error = higher fitness)
        # Fitness in [0, 1]
        max(0, 1.0 - error / 1000.0)

      _ ->
        # Neutral if no data
        0.5
    end
  end

  # ============================================================================
  # Summary & Visualization
  # ============================================================================

  defp summarize_archive(archive) do
    elites = get_elites(archive)
    filled = map_size(elites)
    total = @grid_size * @grid_size

    fitnesses = elites |> Enum.map(fn {_cell, g} -> g.fitness end)

    %{
      grid_size: @grid_size,
      total_cells: total,
      filled_cells: filled,
      coverage: Float.round(filled / total * 100, 1),
      best_fitness: if(Enum.empty?(fitnesses), do: 0.0, else: Enum.max(fitnesses)),
      avg_fitness:
        if(Enum.empty?(fitnesses), do: 0.0, else: Enum.sum(fitnesses) / length(fitnesses)),
      min_fitness: if(Enum.empty?(fitnesses), do: 0.0, else: Enum.min(fitnesses))
    }
  end

  @doc """
  Print a visual representation of the archive.
  """
  def visualize(archive) do
    IO.puts("\n╔═══════════════════════════════════════╗")
    IO.puts("║     MAP-Elites Archive (#{@grid_size}x#{@grid_size})          ║")
    IO.puts("║  Energy →                             ║")
    IO.puts("╠═══════════════════════════════════════╣")

    for col <- (@grid_size - 1)..0//-1 do
      row_str =
        for row <- 0..(@grid_size - 1) do
          case Map.get(archive, {row, col}) do
            nil -> "  ·  "
            g -> " #{format_fitness(g.fitness)} "
          end
        end
        |> Enum.join("│")

      complexity_label = if col == @grid_size - 1, do: "C", else: if(col == 0, do: " ", else: "│")
      IO.puts("║ #{complexity_label} │#{row_str}│")
    end

    IO.puts("╚═══════════════════════════════════════╝")
    IO.puts("  Complexity ↑  (· = empty, number = fitness)")
  end

  defp format_fitness(f) do
    f |> Float.round(2) |> Float.to_string() |> String.pad_leading(4, " ")
  end
end
