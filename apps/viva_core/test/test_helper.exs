# Start Phoenix.PubSub for tests that need it (if not already running)
case Process.whereis(Viva.PubSub) do
  nil -> Phoenix.PubSub.Supervisor.start_link(name: Viva.PubSub)
  _pid -> :already_started
end

# Exclude external tests by default (LLM, Python services)
# Run with: mix test --include external
ExUnit.start(exclude: [:external])
