defmodule VivaWeb.SimulationController do
  @moduledoc """
  Controller para gerenciar a simulação de avatares.
  """
  use VivaWeb, :controller

  alias Viva.Sessions.Supervisor, as: SessionsSupervisor

  @doc """
  Inicia todos os avatares ativos.
  POST /api/simulation/start
  """
  def start(conn, _params) do
    count_before = SessionsSupervisor.count_running_avatars()
    SessionsSupervisor.start_all_active_avatars()
    count_after = SessionsSupervisor.count_running_avatars()

    json(conn, %{
      status: "ok",
      message: "Simulação iniciada",
      avatars_running: count_after,
      avatars_started: count_after - count_before
    })
  end

  @doc """
  Para todos os avatares.
  POST /api/simulation/stop
  """
  def stop(conn, _params) do
    running = SessionsSupervisor.list_running_avatars()

    Enum.each(running, fn avatar_id ->
      SessionsSupervisor.stop_avatar(avatar_id)
    end)

    json(conn, %{
      status: "ok",
      message: "Simulação parada",
      avatars_stopped: length(running)
    })
  end

  @doc """
  Status da simulação.
  GET /api/simulation/status
  """
  def status(conn, _params) do
    running_ids = SessionsSupervisor.list_running_avatars()
    count = length(running_ids)

    json(conn, %{
      status: "ok",
      avatars_running: count,
      avatar_ids: running_ids
    })
  end
end
