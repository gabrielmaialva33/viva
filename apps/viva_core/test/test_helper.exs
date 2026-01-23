# Start Phoenix.PubSub for tests that need it
# Use start_link with children spec for reliability
unless Process.whereis(Viva.PubSub) do
  {:ok, _} = Phoenix.PubSub.Supervisor.start_link(name: Viva.PubSub)
end

# Give PubSub time to fully initialize
Process.sleep(50)

# Exclude external tests by default (LLM, Python services)
# Run with: mix test --include external
ExUnit.start(exclude: [:external])
