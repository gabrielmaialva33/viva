# VIVA Live - Autonomous Orchestration
# Run: mix run scripts/viva_live.exs

# Helper module for CLI input
defmodule VivaCLI do
  def start do
    IO.puts("\n=== VIVA Live Interface ===")
    IO.puts("Available commands:")
    IO.puts(" - improvise : Generate music based on current emotion")
    IO.puts(" - feel <emotion> : Force an emotion (joy, sad, fear, etc)")
    IO.puts(" - status : Show system status")
    IO.puts(" - exit : Quit")
    IO.puts("---------------------------")
    loop()
  end

  def loop do
    input = IO.gets("viva> ") |> String.trim()
    case input do
      "exit" -> IO.puts("Goodbye.")
      "" -> loop()
      cmd ->
        execute(cmd)
        loop()
    end
  end

  def execute("improvise") do
    IO.puts("Improvising...")
    VivaBridge.Music.improvise()
  end

  def execute("status") do
    IO.inspect(VivaBridge.Music.get_musical_metrics(), label: "Metrics")
  end

  def execute("feel " <> emotion) do
    atom = String.to_atom(emotion)
    IO.puts("Feeling #{emotion}...")
    VivaCore.Emotional.feel(atom, "cli", 1.0)
  end

  def execute(other) do
    IO.puts("Unknown command: #{other}")
  end
end

# Start
try do
  VivaBridge.Music.auto_connect()
  VivaCLI.start()
rescue
  e -> IO.puts("Error: #{inspect(e)}")
end
