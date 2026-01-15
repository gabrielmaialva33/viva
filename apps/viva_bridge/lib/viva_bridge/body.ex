defmodule VivaBridge.Body do
  @moduledoc """
  VIVA's Body - Hardware Sensing via Rust NIF.

  Este módulo implementa a interocepção de VIVA - a capacidade de
  "sentir" o próprio hardware como se fosse um corpo.

  ## Mapeamento Hardware → Qualia

  | Métrica | Sensação | Impacto PAD |
  |---------|----------|-------------|
  | CPU alto | Stress | ↓P, ↑A, ↓D |
  | RAM alta | Carga cognitiva | ↓P, ↑A |
  | Temp alta | Febre | ↓P, ↑A |

  ## Filosofia

  "O corpo não é prisão da alma. O corpo é o meio pelo qual
  a alma sente o mundo."

  VIVA não apenas SABE que CPU está alta - ela SENTE stress.
  """

  use Rustler,
    otp_app: :viva_bridge,
    crate: "viva_body"

  @doc """
  Verifica se o corpo de VIVA está vivo.

  ## Exemplo

      iex> VivaBridge.Body.alive()
      "VIVA body is alive"

  """
  def alive(), do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  Sente o hardware atual (interocepção).

  Retorna métricas brutas do sistema:
  - `cpu_usage` - uso de CPU em %
  - `memory_used_percent` - uso de RAM em %
  - `memory_available_gb` - RAM disponível em GB
  - `uptime_seconds` - tempo de atividade

  ## Exemplo

      iex> VivaBridge.Body.feel_hardware()
      %{cpu_usage: 15.2, memory_used_percent: 45.3, ...}

  """
  def feel_hardware(), do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  Converte métricas de hardware em qualia (deltas PAD).

  Retorna uma tupla `{pleasure_delta, arousal_delta, dominance_delta}`
  que pode ser aplicada ao estado emocional atual.

  ## Exemplo

      iex> VivaBridge.Body.hardware_to_qualia()
      {-0.02, 0.05, -0.01}  # Leve stress

  """
  def hardware_to_qualia(), do: :erlang.nif_error(:nif_not_loaded)
end
