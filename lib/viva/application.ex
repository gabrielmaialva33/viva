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

      # Local LLM Queue (priority-based for avatar thoughts)
      Viva.AI.LLM.Backends.Queue,

      # Infrastructure (Redis & RabbitMQ Pipeline)
      Viva.Infrastructure.Redis,
      Viva.AI.Pipeline,
      Viva.Social.NetworkAnalyst,

      # Avatar Sessions Supervisor (includes Registry, DynamicSupervisor, World Clock)
      Viva.Sessions.Supervisor,

      # Metrics Collector for real-time simulation monitoring
      Viva.Metrics.Collector,

      # Phoenix Endpoint (must be last)
      VivaWeb.Endpoint
    ]

    # Add Mortality system if enabled (disabled in test env for controlled testing)
    children =
      if Application.get_env(:viva, :start_mortality_system, true) do
        children ++ [Viva.Mortality.Supervisor]
      else
        children
      end

    # Add GPU-dependent services only if enabled (disabled in test env)
    children =
      if Application.get_env(:viva, :start_gpu_services, true) do
        children ++
          [
            # Quantum World Engine (GPU-accelerated RL simulation)
            {Viva.Quantum.WorldEngine, training_enabled: true, auto_tick: true},
            # HiveMind - Collective Consciousness
            Viva.Collective.HiveMind
          ]
      else
        children
      end

    # See https://hexdocs.pm/elixir/Supervisor.html
    opts = [strategy: :one_for_one, name: Viva.Supervisor]

    case Supervisor.start_link(children, opts) do
      {:ok, pid} ->
        # Schedule avatar startup via Oban (only if supervisor started successfully)
        if Application.get_env(:viva, :start_active_avatars, true) do
          schedule_avatar_startup()
        end

        {:ok, pid}

      {:error, _reason} = error ->
        error
    end
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
