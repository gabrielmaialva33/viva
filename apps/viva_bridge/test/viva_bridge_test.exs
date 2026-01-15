defmodule VivaBridgeTest do
  use ExUnit.Case

  @moduletag :bridge

  describe "Body NIF - Básico" do
    test "alive/0 retorna confirmação" do
      assert VivaBridge.Body.alive() == "VIVA body is alive"
    end
  end

  describe "Body NIF - Hardware Sensing (Interocepção)" do
    test "feel_hardware/0 retorna métricas de CPU" do
      result = VivaBridge.Body.feel_hardware()

      assert is_map(result)
      assert Map.has_key?(result, :cpu_usage)
      assert Map.has_key?(result, :cpu_count)

      # CPU usage deve estar entre 0-100
      assert result.cpu_usage >= 0.0
      assert result.cpu_usage <= 100.0

      # Deve ter pelo menos 1 CPU
      assert result.cpu_count >= 1
    end

    test "feel_hardware/0 retorna métricas de memória" do
      result = VivaBridge.Body.feel_hardware()

      assert Map.has_key?(result, :memory_used_percent)
      assert Map.has_key?(result, :memory_available_gb)
      assert Map.has_key?(result, :memory_total_gb)
      assert Map.has_key?(result, :swap_used_percent)

      # Memory percent deve estar entre 0-100
      assert result.memory_used_percent >= 0.0
      assert result.memory_used_percent <= 100.0

      # Deve ter memória disponível
      assert result.memory_total_gb > 0.0
    end

    test "feel_hardware/0 retorna temperatura (opcional)" do
      result = VivaBridge.Body.feel_hardware()

      # cpu_temp pode ser nil se não disponível
      assert Map.has_key?(result, :cpu_temp)

      case result.cpu_temp do
        nil -> :ok  # Temperatura não disponível - OK
        temp ->
          # Se disponível, deve estar em range razoável (0-150°C)
          assert is_float(temp)
          assert temp >= 0.0
          assert temp < 150.0
      end
    end

    test "feel_hardware/0 retorna métricas de GPU (opcional)" do
      result = VivaBridge.Body.feel_hardware()

      # Todas métricas de GPU podem ser nil
      assert Map.has_key?(result, :gpu_usage)
      assert Map.has_key?(result, :gpu_vram_used_percent)
      assert Map.has_key?(result, :gpu_temp)
      assert Map.has_key?(result, :gpu_name)

      # Se GPU disponível, valores devem ser válidos
      if result.gpu_usage != nil do
        assert result.gpu_usage >= 0.0
        assert result.gpu_usage <= 100.0
      end

      if result.gpu_vram_used_percent != nil do
        assert result.gpu_vram_used_percent >= 0.0
        assert result.gpu_vram_used_percent <= 100.0
      end
    end

    test "feel_hardware/0 retorna métricas de disco" do
      result = VivaBridge.Body.feel_hardware()

      assert Map.has_key?(result, :disk_usage_percent)
      assert Map.has_key?(result, :disk_read_bytes)
      assert Map.has_key?(result, :disk_write_bytes)

      # Disk usage deve estar entre 0-100
      assert result.disk_usage_percent >= 0.0
      assert result.disk_usage_percent <= 100.0
    end

    test "feel_hardware/0 retorna métricas de rede" do
      result = VivaBridge.Body.feel_hardware()

      assert Map.has_key?(result, :net_rx_bytes)
      assert Map.has_key?(result, :net_tx_bytes)

      # Bytes devem ser não-negativos
      assert result.net_rx_bytes >= 0
      assert result.net_tx_bytes >= 0
    end

    test "feel_hardware/0 retorna métricas de sistema" do
      result = VivaBridge.Body.feel_hardware()

      assert Map.has_key?(result, :uptime_seconds)
      assert Map.has_key?(result, :process_count)
      assert Map.has_key?(result, :load_avg_1m)
      assert Map.has_key?(result, :load_avg_5m)
      assert Map.has_key?(result, :load_avg_15m)

      # Uptime deve ser positivo
      assert result.uptime_seconds > 0

      # Deve ter pelo menos alguns processos
      assert result.process_count > 0

      # Load average não-negativo
      assert result.load_avg_1m >= 0.0
    end
  end

  describe "Body NIF - Qualia (Hardware → PAD)" do
    test "hardware_to_qualia/0 retorna tupla de deltas PAD" do
      {p, a, d} = VivaBridge.Body.hardware_to_qualia()

      # Todos devem ser floats
      assert is_float(p)
      assert is_float(a)
      assert is_float(d)

      # Pleasure delta: negativo ou zero (stress nunca aumenta pleasure)
      assert p <= 0.0
      assert p >= -0.1  # Max stress não passa de -0.08

      # Arousal delta: tipicamente positivo
      assert a >= 0.0
      assert a <= 0.15  # Max ~0.12

      # Dominance delta: negativo ou zero
      assert d <= 0.0
      assert d >= -0.1
    end

    test "hardware_to_qualia/0 é determinístico em curto prazo" do
      # Duas chamadas próximas devem dar resultados similares
      {p1, a1, d1} = VivaBridge.Body.hardware_to_qualia()
      Process.sleep(10)
      {p2, a2, d2} = VivaBridge.Body.hardware_to_qualia()

      # Diferença deve ser pequena (< 0.05)
      assert abs(p1 - p2) < 0.05
      assert abs(a1 - a2) < 0.05
      assert abs(d1 - d2) < 0.05
    end
  end

  describe "VivaBridge integration" do
    test "alive?/0 retorna true quando NIF carregado" do
      assert VivaBridge.alive?() == true
    end

    test "feel_hardware/0 delega para Body" do
      result = VivaBridge.feel_hardware()
      assert is_map(result)
      assert Map.has_key?(result, :cpu_usage)
      assert Map.has_key?(result, :cpu_temp)
      assert Map.has_key?(result, :gpu_usage)
    end

    test "sync_body_to_soul/0 aplica qualia ao Emotional" do
      # Inicia um Emotional isolado para teste
      {:ok, _pid} = VivaCore.Emotional.start_link(name: :test_emotional_sync)

      # Estado inicial é neutro
      initial = VivaCore.Emotional.get_state(:test_emotional_sync)
      assert initial.pleasure == 0.0
      assert initial.arousal == 0.0
      assert initial.dominance == 0.0

      # Aplica qualia diretamente (não via sync_body_to_soul que usa o global)
      {p, a, d} = VivaBridge.hardware_to_qualia()
      VivaCore.Emotional.apply_hardware_qualia(p, a, d, :test_emotional_sync)

      # Pequeno delay para o cast processar
      Process.sleep(10)

      # Estado deve ter mudado
      new_state = VivaCore.Emotional.get_state(:test_emotional_sync)

      # Se houve algum stress, pleasure deve ter diminuído
      if p < 0, do: assert(new_state.pleasure < 0)

      # Arousal deve ter aumentado (stress aumenta arousal)
      if a > 0, do: assert(new_state.arousal > 0)

      GenServer.stop(:test_emotional_sync)
    end
  end
end
