defmodule VivaCore.SensesTest do
  use ExUnit.Case

  @moduletag :senses

  describe "Senses - Inicialização" do
    test "start_link/1 inicia com estado padrão" do
      {:ok, pid} = VivaCore.Senses.start_link(name: :test_senses_init, enabled: false)

      state = VivaCore.Senses.get_state(:test_senses_init)

      assert state.interval_ms == 1000
      assert state.enabled == false
      assert state.heartbeat_count == 0
      assert state.last_reading == nil
      assert state.last_qualia == nil

      GenServer.stop(pid)
    end

    test "start_link/1 aceita opções customizadas" do
      {:ok, pid} = VivaCore.Senses.start_link(
        name: :test_senses_custom,
        interval_ms: 500,
        enabled: false
      )

      state = VivaCore.Senses.get_state(:test_senses_custom)
      assert state.interval_ms == 500

      GenServer.stop(pid)
    end
  end

  describe "Senses - Heartbeat" do
    test "pulse/1 força leitura imediata" do
      # Inicia Emotional isolado para o teste
      {:ok, emotional_pid} = VivaCore.Emotional.start_link(name: :test_emotional_pulse)

      {:ok, senses_pid} = VivaCore.Senses.start_link(
        name: :test_senses_pulse,
        emotional_server: :test_emotional_pulse,
        enabled: false
      )

      # Estado inicial do Emotional é neutro
      initial = VivaCore.Emotional.get_state(:test_emotional_pulse)
      assert initial.pleasure == 0.0
      assert initial.arousal == 0.0
      assert initial.dominance == 0.0

      # Força um pulse
      {:ok, {p, a, d}} = VivaCore.Senses.pulse(:test_senses_pulse)

      # Qualia deve ter sido calculada
      assert is_float(p)
      assert is_float(a)
      assert is_float(d)

      # Estado do Senses deve ter sido atualizado
      state = VivaCore.Senses.get_state(:test_senses_pulse)
      assert state.heartbeat_count == 1
      assert state.last_qualia == {p, a, d}
      assert state.last_reading != nil

      # Pequeno delay para o cast processar
      Process.sleep(10)

      # Emotional deve ter recebido a qualia
      new_state = VivaCore.Emotional.get_state(:test_emotional_pulse)

      # Se houve stress do hardware, pleasure deve ter mudado
      if p != 0.0, do: assert(new_state.pleasure != 0.0)

      GenServer.stop(senses_pid)
      GenServer.stop(emotional_pid)
    end

    test "heartbeat automático incrementa contador" do
      {:ok, emotional_pid} = VivaCore.Emotional.start_link(name: :test_emotional_auto)

      {:ok, senses_pid} = VivaCore.Senses.start_link(
        name: :test_senses_auto,
        emotional_server: :test_emotional_auto,
        interval_ms: 100,  # 100ms para teste rápido
        enabled: true
      )

      # Espera 3 heartbeats (~300ms + margem)
      Process.sleep(350)

      state = VivaCore.Senses.get_state(:test_senses_auto)
      assert state.heartbeat_count >= 3

      GenServer.stop(senses_pid)
      GenServer.stop(emotional_pid)
    end
  end

  describe "Senses - Controle" do
    test "pause/1 para o sensing automático" do
      {:ok, emotional_pid} = VivaCore.Emotional.start_link(name: :test_emotional_pause)

      {:ok, senses_pid} = VivaCore.Senses.start_link(
        name: :test_senses_pause,
        emotional_server: :test_emotional_pause,
        interval_ms: 100,
        enabled: true
      )

      # Espera alguns heartbeats
      Process.sleep(250)

      state1 = VivaCore.Senses.get_state(:test_senses_pause)
      count1 = state1.heartbeat_count

      # Pausa
      VivaCore.Senses.pause(:test_senses_pause)

      # Espera mais tempo
      Process.sleep(250)

      state2 = VivaCore.Senses.get_state(:test_senses_pause)
      count2 = state2.heartbeat_count

      # Contador não deve ter aumentado (ou no máximo +1 se estava no meio)
      assert count2 <= count1 + 1
      assert state2.enabled == false

      GenServer.stop(senses_pid)
      GenServer.stop(emotional_pid)
    end

    test "resume/1 retoma o sensing" do
      {:ok, emotional_pid} = VivaCore.Emotional.start_link(name: :test_emotional_resume)

      {:ok, senses_pid} = VivaCore.Senses.start_link(
        name: :test_senses_resume,
        emotional_server: :test_emotional_resume,
        interval_ms: 100,
        enabled: false  # Inicia pausado
      )

      state1 = VivaCore.Senses.get_state(:test_senses_resume)
      assert state1.enabled == false

      # Resume
      VivaCore.Senses.resume(:test_senses_resume)
      Process.sleep(250)

      state2 = VivaCore.Senses.get_state(:test_senses_resume)
      assert state2.enabled == true
      assert state2.heartbeat_count >= 1

      GenServer.stop(senses_pid)
      GenServer.stop(emotional_pid)
    end

    test "set_interval/2 altera frequência em runtime" do
      {:ok, senses_pid} = VivaCore.Senses.start_link(
        name: :test_senses_interval,
        enabled: false
      )

      state1 = VivaCore.Senses.get_state(:test_senses_interval)
      assert state1.interval_ms == 1000

      VivaCore.Senses.set_interval(500, :test_senses_interval)

      state2 = VivaCore.Senses.get_state(:test_senses_interval)
      assert state2.interval_ms == 500

      GenServer.stop(senses_pid)
    end
  end

  describe "Senses - Integração com Hardware" do
    test "last_reading contém métricas de hardware" do
      {:ok, emotional_pid} = VivaCore.Emotional.start_link(name: :test_emotional_hw)

      {:ok, senses_pid} = VivaCore.Senses.start_link(
        name: :test_senses_hw,
        emotional_server: :test_emotional_hw,
        enabled: false
      )

      # Força um pulse para ter leitura
      VivaCore.Senses.pulse(:test_senses_hw)

      state = VivaCore.Senses.get_state(:test_senses_hw)

      # last_reading deve conter métricas de hardware
      assert is_map(state.last_reading)
      assert Map.has_key?(state.last_reading, :cpu_usage)
      assert Map.has_key?(state.last_reading, :memory_used_percent)
      assert Map.has_key?(state.last_reading, :gpu_usage)

      GenServer.stop(senses_pid)
      GenServer.stop(emotional_pid)
    end
  end
end
