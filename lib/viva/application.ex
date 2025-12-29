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
      Viva.AI.LLM.CircuitBreaker,
      Viva.AI.LLM.RateLimiter,

      # Infrastructure (Redis & RabbitMQ Pipeline)
      Viva.Infrastructure.Redis,
      Viva.AI.Pipeline,
      Viva.Social.NetworkAnalyst,

      # Avatar Sessions Supervisor (includes Registry, DynamicSupervisor, World Clock)
      Viva.Sessions.Supervisor,

      # Phoenix Endpoint (must be last)
      VivaWeb.Endpoint
    ]

    # See https://hexdocs.pm/elixir/Supervisor.html
    # for other strategies and supported options
    opts = [strategy: :one_for_one, name: Viva.Supervisor]

    result = Supervisor.start_link(children, opts)

    # Schedule avatar startup via Oban (replaces Process.sleep anti-pattern)
    if Application.get_env(:viva, :start_active_avatars, true) do
      schedule_avatar_startup()
    end

    result
  end

  # Tell Phoenix to update the endpoint configuration
  # whenever the application is updated.
  @impl Application
  def config_change(changed, _, removed) do
    VivaWeb.Endpoint.config_change(changed, removed)
    :ok
  end

  defp schedule_avatar_startup do
    # Schedule job with 2 second delay to ensure all processes are ready
    %{}
    |> Viva.Workers.AvatarStartupWorker.new(
      scheduled_at: DateTime.add(DateTime.utc_now(), 2, :second)
    )
    |> Oban.insert()
  end
end
