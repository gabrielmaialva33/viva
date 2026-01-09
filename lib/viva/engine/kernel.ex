defmodule Viva.Engine.Kernel do
  @moduledoc """
  Consciousness Kernel - GPU-Accelerated Avatar State Processing.

  Processa a fisiologia e emoção de N avatares em paralelo usando
  Nx tensors compilados para CUDA via EXLA.

  Arquitetura Soul-Body Split:
  - Este módulo é o "sistema nervoso" computacional
  - Processa milhares de avatares em microssegundos
  - Não faz rendering visual, apenas computação de estados

  Tensors:
  - bio: [batch_size, 5] - dopamine, cortisol, oxytocin, adenosine, libido
  - emotion: [batch_size, 3] - pleasure, arousal, dominance (PAD)
  - traits: [batch_size, 5] - openness, conscientiousness, extraversion, agreeableness, neuroticism
  """

  import Nx.Defn

  # Índices do tensor bio
  @dopamine_idx 0
  @cortisol_idx 1
  @oxytocin_idx 2
  @adenosine_idx 3
  @libido_idx 4

  # Índices do tensor emotion (PAD)
  @pleasure_idx 0
  @arousal_idx 1
  @dominance_idx 2

  # Índices do tensor traits (Big Five)
  @openness_idx 0
  @conscientiousness_idx 1
  @extraversion_idx 2
  @agreeableness_idx 3
  @neuroticism_idx 4

  # ============================================================================
  # Main Tick Function - Processa todos avatares de uma vez
  # ============================================================================

  @doc """
  Tick de Simulação Massivo - GPU Accelerated.

  Processa o estado de N avatares em uma única chamada.
  Retorna os novos estados biológicos e emocionais.

  ## Parameters
    - bio: Tensor [batch_size, 5] - estados biológicos
    - emotion: Tensor [batch_size, 3] - estados emocionais (PAD)
    - traits: Tensor [batch_size, 5] - traços de personalidade (Big Five)
    - dt: float - delta tempo (fração de hora simulada)

  ## Returns
    - {new_bio, new_emotion} - estados atualizados
  """
  defn tick(bio, emotion, traits, dt) do
    # 1. Atualiza biologia (decay, homeostase)
    new_bio = update_biology(bio, traits, dt)

    # 2. Atualiza emoções baseado na nova biologia
    new_emotion = update_emotions(emotion, new_bio, traits, dt)

    {new_bio, new_emotion}
  end

  # ============================================================================
  # Biology Update - Decaimento e Homeostase
  # ============================================================================

  defnp update_biology(bio, traits, dt) do
    # Extrai valores atuais
    dopamine = bio[[.., @dopamine_idx]]
    cortisol = bio[[.., @cortisol_idx]]
    oxytocin = bio[[.., @oxytocin_idx]]
    adenosine = bio[[.., @adenosine_idx]]
    libido = bio[[.., @libido_idx]]

    # Extrai traços relevantes
    neuroticism = traits[[.., @neuroticism_idx]]
    extraversion = traits[[.., @extraversion_idx]]

    # === Dopamina ===
    # Hedonic treadmill: decai naturalmente
    # Extravertidos mantêm dopamina por mais tempo
    dopamine_decay = 0.03 * (1.0 - extraversion * 0.3)
    dopamine_floor = 0.15  # Nunca vai a zero
    new_dopamine = Nx.max(dopamine * (1.0 - dopamine_decay * dt), dopamine_floor)

    # === Cortisol ===
    # Decai naturalmente, mas neuróticos decaem mais devagar
    cortisol_decay = 0.04 * (1.0 - neuroticism * 0.4)
    cortisol_baseline = 0.15 + neuroticism * 0.1  # Neuróticos têm baseline maior
    new_cortisol = cortisol + (cortisol_baseline - cortisol) * cortisol_decay * dt

    # === Oxitocina ===
    # Decai sem interação social
    oxytocin_decay = 0.05
    oxytocin_floor = 0.1
    new_oxytocin = Nx.max(oxytocin * (1.0 - oxytocin_decay * dt), oxytocin_floor)

    # === Adenosina ===
    # Acumula com tempo acordado (fadiga)
    # Conscientiousness reduz acúmulo (mais disciplina = menos fadiga)
    conscientiousness = traits[[.., @conscientiousness_idx]]
    adenosine_buildup = 0.02 * (1.0 - conscientiousness * 0.2)
    adenosine_cap = 1.0
    new_adenosine = Nx.min(adenosine + adenosine_buildup * dt, adenosine_cap)

    # === Libido ===
    # Flutua baseado em dopamina e stress
    libido_target = 0.4 + new_dopamine * 0.3 - new_cortisol * 0.2
    libido_rate = 0.1
    new_libido = libido + (libido_target - libido) * libido_rate * dt
    new_libido = Nx.clip(new_libido, 0.0, 1.0)

    # Reconstrói tensor bio
    Nx.stack(
      [new_dopamine, new_cortisol, new_oxytocin, new_adenosine, new_libido],
      axis: 1
    )
  end

  # ============================================================================
  # Emotion Update - Integração Bio -> PAD
  # ============================================================================

  defnp update_emotions(emotion, bio, traits, dt) do
    # Extrai PAD atual
    pleasure = emotion[[.., @pleasure_idx]]
    arousal = emotion[[.., @arousal_idx]]
    dominance = emotion[[.., @dominance_idx]]

    # Extrai biologia
    dopamine = bio[[.., @dopamine_idx]]
    cortisol = bio[[.., @cortisol_idx]]
    oxytocin = bio[[.., @oxytocin_idx]]
    adenosine = bio[[.., @adenosine_idx]]

    # Extrai traços
    neuroticism = traits[[.., @neuroticism_idx]]
    extraversion = traits[[.., @extraversion_idx]]
    agreeableness = traits[[.., @agreeableness_idx]]

    # === Pleasure (Valence) ===
    # Influenciado por: +dopamina, +oxitocina, -cortisol, -adenosina
    pleasure_target =
      dopamine * 0.4 +
      oxytocin * 0.3 -
      cortisol * 0.4 -
      adenosine * 0.2

    # Neuróticos têm pleasure mais volátil e tendência negativa
    pleasure_bias = -neuroticism * 0.15
    pleasure_target = pleasure_target + pleasure_bias

    # Elasticidade: quão rápido o humor muda
    pleasure_elasticity = 0.15 + neuroticism * 0.1
    new_pleasure = pleasure + (pleasure_target - pleasure) * pleasure_elasticity * dt
    new_pleasure = Nx.clip(new_pleasure, -1.0, 1.0)

    # === Arousal (Ativação) ===
    # Influenciado por: +cortisol (stress ativa), -adenosina (fadiga desativa)
    # Extravertidos têm arousal baseline maior
    arousal_target =
      cortisol * 0.5 -
      adenosine * 0.4 +
      extraversion * 0.2

    arousal_elasticity = 0.2
    new_arousal = arousal + (arousal_target - arousal) * arousal_elasticity * dt
    new_arousal = Nx.clip(new_arousal, -1.0, 1.0)

    # === Dominance (Poder) ===
    # Influenciado por: +dopamina (confiança), -cortisol (medo reduz)
    # Agreeableness reduz dominance (mais submisso)
    dominance_target =
      dopamine * 0.3 -
      cortisol * 0.3 -
      agreeableness * 0.15

    dominance_elasticity = 0.1
    new_dominance = dominance + (dominance_target - dominance) * dominance_elasticity * dt
    new_dominance = Nx.clip(new_dominance, -1.0, 1.0)

    # Reconstrói tensor emotion
    Nx.stack([new_pleasure, new_arousal, new_dominance], axis: 1)
  end

  # ============================================================================
  # Reward Signal - Para Reinforcement Learning
  # ============================================================================

  @doc """
  Calcula sinal de recompensa para RL training.

  Reward alto = avatar está "bem":
  - Pleasure alto
  - Dopamina adequada
  - Cortisol baixo
  - Não exausto
  """
  defn compute_reward(bio, emotion) do
    pleasure = emotion[[.., @pleasure_idx]]
    dopamine = bio[[.., @dopamine_idx]]
    cortisol = bio[[.., @cortisol_idx]]
    adenosine = bio[[.., @adenosine_idx]]

    # Normaliza pleasure de [-1,1] para [0,1]
    pleasure_norm = (pleasure + 1.0) / 2.0

    # Reward composto
    reward =
      pleasure_norm * 0.4 +
      dopamine * 0.2 +
      (1.0 - cortisol) * 0.2 +
      (1.0 - adenosine) * 0.2

    Nx.clip(reward, 0.0, 1.0)
  end

  # ============================================================================
  # State Vectorization - Converte structs para tensores
  # ============================================================================

  @doc """
  Converte uma lista de InternalStates para tensores GPU.

  Útil para inicialização do Orchestrator.
  """
  def states_to_tensors(internal_states) do
    bio_data =
      Enum.map(internal_states, fn state ->
        bio = state.bio
        [bio.dopamine, bio.cortisol, bio.oxytocin, bio.adenosine, bio.libido]
      end)

    emotion_data =
      Enum.map(internal_states, fn state ->
        emo = state.emotional
        [emo.pleasure, emo.arousal, emo.dominance]
      end)

    bio_tensor = Nx.tensor(bio_data, type: :f32)
    emotion_tensor = Nx.tensor(emotion_data, type: :f32)

    {bio_tensor, emotion_tensor}
  end

  @doc """
  Converte uma lista de Personalities para tensor de traits.
  """
  def personalities_to_tensor(personalities) do
    traits_data =
      Enum.map(personalities, fn p ->
        [p.openness, p.conscientiousness, p.extraversion, p.agreeableness, p.neuroticism]
      end)

    Nx.tensor(traits_data, type: :f32)
  end

  @doc """
  Extrai valores de tensores de volta para structs (para sync com GenServers).

  Retorna lista de maps com os valores atualizados.
  """
  def tensors_to_updates(bio_tensor, emotion_tensor) do
    bio_list = Nx.to_list(bio_tensor)
    emotion_list = Nx.to_list(emotion_tensor)

    Enum.zip(bio_list, emotion_list)
    |> Enum.map(fn {bio_row, emotion_row} ->
      [dopamine, cortisol, oxytocin, adenosine, libido] = bio_row
      [pleasure, arousal, dominance] = emotion_row

      %{
        bio: %{
          dopamine: dopamine,
          cortisol: cortisol,
          oxytocin: oxytocin,
          adenosine: adenosine,
          libido: libido
        },
        emotional: %{
          pleasure: pleasure,
          arousal: arousal,
          dominance: dominance
        }
      }
    end)
  end
end
