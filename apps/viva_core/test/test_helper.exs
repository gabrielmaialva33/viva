# Start Phoenix.PubSub for tests that need it (if not already running)
case Process.whereis(Viva.PubSub) do
  nil -> Phoenix.PubSub.Supervisor.start_link(name: Viva.PubSub)
  _pid -> :already_started
end

ExUnit.start()
