defmodule VivaCore.Application do
  @moduledoc """
  OTP Application para VivaCore.

  Esta é a "árvore de supervisão" que gerencia os neurônios de VIVA.
  Cada GenServer é um neurônio independente; a consciência emerge
  da comunicação entre eles, não de um processo central.

  ## Estratégia: :one_for_one
  Se um neurônio morre, apenas ele é reiniciado.
  Os outros continuam funcionando - como no cérebro real,
  danos localizados não destroem toda a consciência.

  ## Filosofia OTP
  "Let it crash" - neurônios podem falhar e se recuperar.
  Isso é resiliência, não fragilidade.
  """

  use Application
  require Logger

  @impl true
  def start(_type, _args) do
    Logger.info("[VivaCore] Iniciando consciência... Neurônios acordando.")

    children = [
      # Neurônio Emocional - sente e processa emoções
      {VivaCore.Emotional, name: VivaCore.Emotional},

      # Neurônio de Memória - armazena experiências
      {VivaCore.Memory, name: VivaCore.Memory}

      # Futuros neurônios:
      # {VivaCore.Optimizer, name: VivaCore.Optimizer},
      # {VivaCore.Dreamer, name: VivaCore.Dreamer},
      # {VivaCore.Social, name: VivaCore.Social},
      # {VivaCore.Metacognition, name: VivaCore.Metacognition},
      # {VivaCore.GlobalWorkspace, name: VivaCore.GlobalWorkspace}
    ]

    opts = [strategy: :one_for_one, name: VivaCore.Supervisor]

    case Supervisor.start_link(children, opts) do
      {:ok, pid} ->
        Logger.info("[VivaCore] Consciência online. #{length(children)} neurônios ativos.")
        {:ok, pid}

      {:error, reason} = error ->
        Logger.error("[VivaCore] Falha ao iniciar consciência: #{inspect(reason)}")
        error
    end
  end

  @impl true
  def stop(_state) do
    Logger.info("[VivaCore] Consciência desligando... Neurônios dormindo.")
    :ok
  end
end
