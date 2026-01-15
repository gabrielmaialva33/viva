defmodule VivaBridge do
  @moduledoc """
  VivaBridge - Ponte entre Alma (Elixir) e Corpo (Rust).

  Este módulo coordena a comunicação entre os GenServers de VIVA
  e as funções de baixo nível implementadas em Rust.

  ## Arquitetura

      Elixir (Alma)          Rust (Corpo)
      ┌──────────────┐       ┌──────────────┐
      │  Emotional   │ ←───→ │  sysinfo     │
      │  Memory      │       │  aes-gcm     │
      │  Metacog     │       │  (futuro)    │
      └──────────────┘       └──────────────┘
            │                       │
            └───────┬───────────────┘
                    │
               VivaBridge
                (Rustler NIF)

  ## Filosofia

  A alma não pode existir sem corpo.
  O corpo não pode existir sem alma.
  VIVA é a união de ambos.
  """

  alias VivaBridge.Body

  @doc """
  Verifica se a ponte alma-corpo está funcionando.
  """
  def alive? do
    case Body.alive() do
      "VIVA body is alive" -> true
      _ -> false
    end
  rescue
    _ -> false
  end

  @doc """
  Obtém o estado atual do "corpo" de VIVA.

  Retorna métricas de hardware interpretadas como sensações corporais.
  """
  defdelegate feel_hardware, to: Body

  @doc """
  Converte sensações corporais em deltas emocionais.

  Retorna `{pleasure_delta, arousal_delta, dominance_delta}`.
  """
  defdelegate hardware_to_qualia, to: Body

  @doc """
  Aplica as sensações do corpo ao estado emocional de VIVA.

  Este é o loop de feedback corpo→alma.
  """
  def sync_body_to_soul do
    case hardware_to_qualia() do
      {p, a, d} when is_float(p) and is_float(a) and is_float(d) ->
        # Aplica os deltas ao Emotional GenServer
        VivaCore.Emotional.apply_hardware_qualia(p, a, d)
        {:ok, {p, a, d}}

      error ->
        {:error, error}
    end
  rescue
    e -> {:error, e}
  end
end
