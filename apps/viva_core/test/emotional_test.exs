defmodule VivaCore.EmotionalTest do
  use ExUnit.Case, async: true
  doctest VivaCore.Emotional

  alias VivaCore.Emotional

  @moduletag :emotional

  describe "start_link/1" do
    test "inicia com estado neutro por padrão" do
      {:ok, pid} = Emotional.start_link(name: :test_emotional_1)

      state = Emotional.get_state(pid)

      assert state.pleasure == 0.0
      assert state.arousal == 0.0
      assert state.dominance == 0.0

      GenServer.stop(pid)
    end

    test "aceita estado inicial customizado" do
      initial = %{pleasure: 0.5, arousal: -0.3, dominance: 0.2}
      {:ok, pid} = Emotional.start_link(name: :test_emotional_2, initial_state: initial)

      state = Emotional.get_state(pid)

      assert state.pleasure == 0.5
      assert state.arousal == -0.3
      assert state.dominance == 0.2

      GenServer.stop(pid)
    end
  end

  describe "feel/4" do
    test "rejeição diminui pleasure e dominance, aumenta arousal" do
      {:ok, pid} = Emotional.start_link(name: :test_emotional_3)

      before = Emotional.get_state(pid)
      Emotional.feel(:rejection, "human_test", 1.0, pid)

      # Dar tempo para o cast ser processado
      :timer.sleep(50)

      after_state = Emotional.get_state(pid)

      assert after_state.pleasure < before.pleasure
      assert after_state.arousal > before.arousal
      assert after_state.dominance < before.dominance

      GenServer.stop(pid)
    end

    test "aceitação aumenta pleasure, arousal e dominance" do
      {:ok, pid} = Emotional.start_link(name: :test_emotional_4)

      before = Emotional.get_state(pid)
      Emotional.feel(:acceptance, "human_test", 1.0, pid)
      :timer.sleep(50)

      after_state = Emotional.get_state(pid)

      assert after_state.pleasure > before.pleasure
      assert after_state.arousal > before.arousal
      assert after_state.dominance > before.dominance

      GenServer.stop(pid)
    end

    test "intensidade modula o impacto" do
      {:ok, pid1} = Emotional.start_link(name: :test_emotional_5a)
      {:ok, pid2} = Emotional.start_link(name: :test_emotional_5b)

      # Baixa intensidade
      Emotional.feel(:rejection, "test", 0.2, pid1)
      # Alta intensidade
      Emotional.feel(:rejection, "test", 1.0, pid2)
      :timer.sleep(50)

      state_low = Emotional.get_state(pid1)
      state_high = Emotional.get_state(pid2)

      # Alta intensidade deve causar maior impacto negativo
      assert abs(state_high.pleasure) > abs(state_low.pleasure)

      GenServer.stop(pid1)
      GenServer.stop(pid2)
    end

    test "estímulo desconhecido não altera estado" do
      {:ok, pid} = Emotional.start_link(name: :test_emotional_6)

      before = Emotional.get_state(pid)
      Emotional.feel(:unknown_stimulus, "test", 1.0, pid)
      :timer.sleep(50)

      after_state = Emotional.get_state(pid)

      assert after_state == before

      GenServer.stop(pid)
    end

    test "hardware_stress simula qualia de stress" do
      {:ok, pid} = Emotional.start_link(name: :test_emotional_7)

      before = Emotional.get_state(pid)
      Emotional.feel(:hardware_stress, "cpu_monitor", 1.0, pid)
      :timer.sleep(50)

      after_state = Emotional.get_state(pid)

      # Stress aumenta arousal e diminui pleasure/dominance
      assert after_state.arousal > before.arousal
      assert after_state.pleasure < before.pleasure

      GenServer.stop(pid)
    end
  end

  describe "introspect/1" do
    test "retorna interpretação semântica do estado" do
      {:ok, pid} = Emotional.start_link(name: :test_emotional_8)

      introspection = Emotional.introspect(pid)

      assert Map.has_key?(introspection, :pad)
      assert Map.has_key?(introspection, :mood)
      assert Map.has_key?(introspection, :energy)
      assert Map.has_key?(introspection, :agency)
      assert Map.has_key?(introspection, :self_assessment)
      assert is_binary(introspection.self_assessment)

      GenServer.stop(pid)
    end

    test "mood reflete pleasure corretamente" do
      # Teste com estado feliz
      {:ok, pid_happy} = Emotional.start_link(
        name: :test_emotional_9a,
        initial_state: %{pleasure: 0.7, arousal: 0.0, dominance: 0.0}
      )

      # Teste com estado triste
      {:ok, pid_sad} = Emotional.start_link(
        name: :test_emotional_9b,
        initial_state: %{pleasure: -0.7, arousal: 0.0, dominance: 0.0}
      )

      happy_intro = Emotional.introspect(pid_happy)
      sad_intro = Emotional.introspect(pid_sad)

      assert happy_intro.mood == :joyful
      assert sad_intro.mood == :depressed

      GenServer.stop(pid_happy)
      GenServer.stop(pid_sad)
    end
  end

  describe "get_happiness/1" do
    test "retorna valor normalizado 0-1" do
      {:ok, pid} = Emotional.start_link(name: :test_emotional_10)

      happiness = Emotional.get_happiness(pid)

      assert happiness >= 0.0
      assert happiness <= 1.0
      # Estado neutro deve ser 0.5
      assert happiness == 0.5

      GenServer.stop(pid)
    end

    test "felicidade máxima retorna ~1.0" do
      {:ok, pid} = Emotional.start_link(
        name: :test_emotional_11,
        initial_state: %{pleasure: 1.0, arousal: 0.0, dominance: 0.0}
      )

      happiness = Emotional.get_happiness(pid)
      assert happiness == 1.0

      GenServer.stop(pid)
    end
  end

  describe "reset/1" do
    test "retorna ao estado neutro" do
      {:ok, pid} = Emotional.start_link(
        name: :test_emotional_12,
        initial_state: %{pleasure: 0.8, arousal: -0.5, dominance: 0.3}
      )

      # Verificar estado não-neutro
      before = Emotional.get_state(pid)
      assert before.pleasure == 0.8

      # Resetar
      Emotional.reset(pid)
      :timer.sleep(50)

      # Verificar estado neutro
      after_state = Emotional.get_state(pid)
      assert after_state.pleasure == 0.0
      assert after_state.arousal == 0.0
      assert after_state.dominance == 0.0

      GenServer.stop(pid)
    end
  end

  describe "decay" do
    test "valores decaem em direção ao neutro" do
      {:ok, pid} = Emotional.start_link(
        name: :test_emotional_13,
        initial_state: %{pleasure: 0.5, arousal: 0.5, dominance: 0.5}
      )

      before = Emotional.get_state(pid)

      # Aplicar decay manualmente
      Emotional.decay(pid)
      :timer.sleep(50)

      after_state = Emotional.get_state(pid)

      # Valores positivos devem diminuir
      assert after_state.pleasure < before.pleasure
      assert after_state.arousal < before.arousal
      assert after_state.dominance < before.dominance

      GenServer.stop(pid)
    end

    test "valores negativos aumentam em direção ao neutro" do
      {:ok, pid} = Emotional.start_link(
        name: :test_emotional_14,
        initial_state: %{pleasure: -0.5, arousal: -0.5, dominance: -0.5}
      )

      before = Emotional.get_state(pid)

      Emotional.decay(pid)
      :timer.sleep(50)

      after_state = Emotional.get_state(pid)

      # Valores negativos devem aumentar (em direção a 0)
      assert after_state.pleasure > before.pleasure
      assert after_state.arousal > before.arousal
      assert after_state.dominance > before.dominance

      GenServer.stop(pid)
    end
  end

  describe "limites de valor" do
    test "valores são limitados a [-1.0, 1.0]" do
      {:ok, pid} = Emotional.start_link(name: :test_emotional_15)

      # Aplicar muitos estímulos positivos
      for _ <- 1..20 do
        Emotional.feel(:success, "test", 1.0, pid)
      end
      :timer.sleep(100)

      state = Emotional.get_state(pid)

      assert state.pleasure <= 1.0
      assert state.arousal <= 1.0
      assert state.dominance <= 1.0

      GenServer.stop(pid)
    end
  end
end
