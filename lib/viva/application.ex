defmodule Viva.Application do
  @moduledoc """
  VIVA - Virtual Intelligent Vida Autônoma
  Main application supervisor.
  """

  use Application

  @impl Application
  def start(_, _) do
    children = [
      # Telemetry
      VivaWeb.Telemetry,

      # Database
      Viva.Repo,

      # Oban for background jobs
      {Oban, Application.fetch_env!(:viva, Oban)},

      # Cache
      {Cachex, name: :viva_cache},

      # DNS Cluster
      {DNSCluster, query: Application.get_env(:viva, :dns_cluster_query) || :ignore},

      # PubSub for real-time events
      {Phoenix.PubSub, name: Viva.PubSub},

      # NIM API resilience (circuit breaker + rate limiter)
      Viva.Nim.CircuitBreaker,
      Viva.Nim.RateLimiter,

      # Avatar Sessions Supervisor (includes Registry, DynamicSupervisor, World Clock)
      Viva.Sessions.Supervisor,

      # Phoenix Endpoint (must be last)
      VivaWeb.Endpoint
    ]

    # See https://hexdocs.pm/elixir/Supervisor.html
    # for other strategies and supported options
    opts = [strategy: :one_for_one, name: Viva.Supervisor]

    result = Supervisor.start_link(children, opts)

    # Start all active avatars after supervision tree is up
    start_active_avatars()

    result
  end

  # Tell Phoenix to update the endpoint configuration
  # whenever the application is updated.
  @impl Application
  def config_change(changed, _, removed) do
    VivaWeb.Endpoint.config_change(changed, removed)
    :ok
  end

  defp start_active_avatars do
    # Delay startup to ensure everything is ready
    Task.start(fn ->
      Process.sleep(2000)
      Viva.Sessions.Supervisor.start_all_active_avatars()
    end)
  end
end
