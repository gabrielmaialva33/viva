defmodule VivaBridge.Body do
  @moduledoc """
  VIVA's Body - Hardware Sensing via Rust NIF (Interocepção Multiplataforma).

  Este módulo implementa a interocepção de VIVA - a capacidade de
  "sentir" o próprio hardware como se fosse um corpo.

  ## Base Teórica
  - Interocepção (Craig, 2002) - percepção do estado interno
  - Embodied Cognition (Varela et al., 1991) - mente emerge do corpo
  - PAD Model (Mehrabian, 1996) - espaço emocional 3D

  ## Mapeamento Hardware → Qualia

  | Métrica | Sensação | Impacto PAD |
  |---------|----------|-------------|
  | CPU alto | Stress cardíaco | ↓P, ↑A, ↓D |
  | CPU temp alta | Febre | ↓P, ↑A |
  | RAM alta | Carga cognitiva | ↓P, ↑A |
  | GPU VRAM alta | Imaginação limitada | ↓P, ↓D |
  | Disk cheio | Digestão lenta | ↓A |
  | Load alto | Overwhelm | ↓P, ↓D |

  ## Métricas Disponíveis

  - **CPU**: usage, temp (opcional), count
  - **Memory**: used%, available_gb, total_gb, swap%
  - **GPU**: usage, vram%, temp, name (NVIDIA via NVML, opcional)
  - **Disk**: usage%, read/write bytes
  - **Network**: rx/tx bytes
  - **System**: uptime, process_count, load_avg (1m/5m/15m)

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

      VivaBridge.Body.alive()
      # => "VIVA body is alive"

  """
  def alive(), do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  Sente o hardware atual (interocepção completa).

  Retorna mapa com métricas do sistema:

  ## CPU
  - `cpu_usage` - uso de CPU em %
  - `cpu_temp` - temperatura em °C (nil se indisponível)
  - `cpu_count` - número de cores

  ## Memory
  - `memory_used_percent` - uso de RAM em %
  - `memory_available_gb` - RAM disponível em GB
  - `memory_total_gb` - RAM total em GB
  - `swap_used_percent` - uso de swap em %

  ## GPU (opcional - nil se não disponível)
  - `gpu_usage` - uso de GPU em %
  - `gpu_vram_used_percent` - uso de VRAM em %
  - `gpu_temp` - temperatura em °C
  - `gpu_name` - nome da GPU

  ## Disk
  - `disk_usage_percent` - uso de disco em %
  - `disk_read_bytes` - bytes lidos
  - `disk_write_bytes` - bytes escritos

  ## Network
  - `net_rx_bytes` - bytes recebidos
  - `net_tx_bytes` - bytes transmitidos

  ## System
  - `uptime_seconds` - tempo de atividade
  - `process_count` - número de processos
  - `load_avg_1m` - load average 1 minuto
  - `load_avg_5m` - load average 5 minutos
  - `load_avg_15m` - load average 15 minutos

  ## Exemplo

      hw = VivaBridge.Body.feel_hardware()
      # => %{cpu_usage: 15.2, cpu_temp: 45.0, memory_used_percent: 25.3, ...}

  """
  def feel_hardware(), do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  Converte métricas de hardware em qualia (deltas PAD).

  Retorna uma tupla `{pleasure_delta, arousal_delta, dominance_delta}`
  que pode ser aplicada ao estado emocional atual.

  ## Fórmula de Stress Composto

      σ = 0.15×cpu + 0.15×load + 0.20×mem + 0.05×swap + 0.20×temp + 0.15×gpu + 0.10×disk

  ## Mapeamento PAD

  - **Pleasure**: `δP = -0.08 × σ` (stress → desconforto)
  - **Arousal**: `δA = 0.12 × σ × (1 - σ/2)` (stress → ativação, com saturação)
  - **Dominance**: `δD = -0.06 × (0.4×load + 0.3×gpu + 0.3×mem)` (overwhelm → perda de controle)

  ## Exemplo

      {p, a, d} = VivaBridge.Body.hardware_to_qualia()
      # => {-0.008, 0.012, -0.005}  # Leve stress do sistema

  """
  def hardware_to_qualia(), do: :erlang.nif_error(:nif_not_loaded)
end
