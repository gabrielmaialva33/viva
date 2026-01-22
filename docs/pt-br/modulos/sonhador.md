# Sonhador - Consolidação de Memória

> *"Sonhos não são ruído - são a alma se reorganizando."*

## Teoria

Implementa o mecanismo de reflexão de **Park et al. (2023) "Generative Agents"** adaptado para a arquitetura emocional de VIVA.

Na reflexão, experiências dispersas se tornam significado coerente.

---

## Fundamentos Matemáticos

### Pontuação de Recuperação (DRE - Dream Retrieval Engine)

```
S(m, q) = w_r · D(m) + w_s · Sim(m, q) + w_i · I(m) + w_e · E(m)
```

| Componente | Peso | Descrição |
|------------|------|-----------|
| **D(m)** | 0.2 | Recência com repetição espaçada |
| **Sim(m, q)** | 0.4 | Similaridade semântica |
| **I(m)** | 0.2 | Importância da memória |
| **E(m)** | 0.2 | Ressonância emocional |

### Função de Decaimento (com Repetição Espaçada)

```
D(m) = e^(-idade/τ) × (1 + min(0.5, log(1 + access_count) / κ))
```

Onde:
- τ = 604.800 segundos (1 semana)
- κ = 10.0 (divisor de boost de repetição)
- Boost máximo de acesso limitado a 50%

### Ressonância Emocional

```
E(m) = max(0, 1 - ||PAD_m - PAD_atual|| / √12)
```

Distância no espaço PAD normalizada para [0, 1].

---

## Referência da API

### `VivaCore.Dreamer.status/0`
Obtém estatísticas atuais do Dreamer.

```elixir
VivaCore.Dreamer.status()
# => %{
#      importance_accumulator: 8.5,
#      threshold: 15.0,
#      progress_percent: 56.7,
#      last_reflection: ~U[2024-01-15 14:00:00Z],
#      reflection_count: 42,
#      thoughts_count: 156,
#      ...
#    }
```

### `VivaCore.Dreamer.reflect_now/0`
Força ciclo de reflexão imediato.

```elixir
VivaCore.Dreamer.reflect_now()
# => %{
#      focal_points: [%{question: "O que aprendi sobre...", ...}],
#      insights: [%{insight: "Refletindo sobre...", depth: 1, ...}],
#      trigger: :manual
#    }
```

### `VivaCore.Dreamer.sleep_cycle/0`
Inicia reflexão profunda (múltiplas iterações + meta-reflexão).

```elixir
{:ok, ref} = VivaCore.Dreamer.sleep_cycle()
# Roda assincronamente, atualiza estado quando completo
```

### `VivaCore.Dreamer.recent_thoughts/1`
Obtém reflexões recentes.

```elixir
VivaCore.Dreamer.recent_thoughts(5)
# => [
#      %{insight: "...", depth: 1, importance: 0.7, ...},
#      ...
#    ]
```

### `VivaCore.Dreamer.retrieve_with_scoring/2`
Recupera memórias com pontuação composta completa.

```elixir
VivaCore.Dreamer.retrieve_with_scoring("ações bem sucedidas", limit: 10)
# => [%{content: "...", composite_score: 0.85, ...}, ...]
```

### `VivaCore.Dreamer.hallucinate_goal/1`
Inferência Ativa: Gera estado PAD alvo (onde VIVA *quer* estar).

```elixir
context = %{pleasure: -0.1, arousal: 0.2, dominance: 0.0}
VivaCore.Dreamer.hallucinate_goal(context)
# => %{pleasure: 0.2, arousal: 0.15, dominance: 0.1}
```

### `VivaCore.Dreamer.on_memory_stored/2`
Notifica Dreamer de nova memória (chamado pelo módulo Memory).

```elixir
VivaCore.Dreamer.on_memory_stored("mem_12345", 0.8)
# => :ok (acumula importância, pode disparar reflexão)
```

---

## Gatilhos de Reflexão

Reflexão é disparada quando QUALQUER condição é atendida:

| Gatilho | Threshold | Descrição |
|---------|-----------|-----------|
| **Importância** | Σ importance ≥ 15.0 | Importância acumulada de novas memórias |
| **Tempo** | > 1 hora desde última | Limite de tempo de atividade |
| **Sono** | Manual/Circadiano | Ciclo de reflexão profunda |

---

## Profundidade de Reflexão

| Profundidade | Tipo | Descrição |
|--------------|------|-----------|
| 0 | Evento | Memória direta (experiência bruta) |
| 1 | Insight | Reflexão de primeira ordem (reconhecimento de padrões) |
| 2 | Meta-cognição | Segunda ordem (reflexão sobre reflexões) |

---

## O Processo de Reflexão

```
1. GERAR PONTOS FOCAIS
   └── Extrair tópicos de memórias recentes
   └── "O que aprendi sobre {tópico}?"

2. RECUPERAR MEMÓRIAS RELEVANTES
   └── Usar pontuação composta (DRE)
   └── Ranquear por recência + similaridade + importância + emoção

3. SINTETIZAR INSIGHTS
   └── Gerar observações de memórias recuperadas
   └── Armazenar como pensamentos depth=1

4. (APENAS CICLO DE SONO) META-REFLEXÃO
   └── Refletir sobre pensamentos recentes
   └── Gerar insights depth=2

5. CONSOLIDAÇÃO DE MEMÓRIA
   └── Promoção Episódico → Semântico
   └── Memórias importantes se tornam conhecimento de longo prazo
```

---

## Geração de Meta Homeostática

Em vez de seleção aleatória de metas, Dreamer usa memória para encontrar o que funcionou:

```elixir
def calculate_personal_baseline(state) do
  # Busca memórias com resultados emocionais positivos
  {:ok, memories} = Memory.search("estados positivos felicidade alívio sucesso", limit: 10)

  # Calcula média dos estados PAD bem-sucedidos
  pads = Enum.map(memories, & &1.emotion)
  %{
    pleasure: mean(Enum.map(pads, & &1.pleasure)),
    arousal: mean(Enum.map(pads, & &1.arousal)),
    dominance: mean(Enum.map(pads, & &1.dominance))
  }
end
```

### Arousal Ótimo Yerkes-Dodson

```elixir
def calculate_optimal_arousal(current_pad) do
  cond do
    # Alta dominância + positivo → pode estar excitada
    dominance > 0.3 and pleasure > 0 -> 0.4
    # Alta dominância + negativo → precisa ativação para corrigir
    dominance > 0.3 and pleasure < 0 -> 0.3
    # Baixa dominância → precisa calma para recuperar
    dominance < -0.3 -> 0.0
    # Padrão: leve engajamento
    true -> 0.15
  end
end
```

---

## Consolidação de Memória (DRE)

Durante o ciclo de sono, memórias episódicas são promovidas para semânticas:

### Score de Consolidação

```elixir
score = Mathematics.consolidation_score(
  memory_pad,      # Estado emocional da memória
  baseline_pad,    # Baseline pessoal
  importance,      # 0.0 - 1.0
  age_seconds,     # Tempo desde criação
  access_count     # Quantas vezes acessada
)
```

### Threshold de Consolidação

Memórias com score ≥ **0.7** são promovidas:

```elixir
Memory.store(content, %{
  type: :semantic,        # Armazenamento de longo prazo
  importance: importance * 0.9,
  consolidated_from: original_id,
  consolidated_at: DateTime.utc_now()
})
```

---

## Loop de Feedback Emocional

Dreamer afeta estado Emocional baseado na valência da memória:

```elixir
# Calcula pleasure médio das memórias recuperadas
avg_pleasure = memories |> Enum.map(& &1.emotion.pleasure) |> mean()

feedback = cond do
  avg_pleasure > 0.1 -> :lucid_insight     # Reflexão positiva
  avg_pleasure < -0.1 -> :grim_realization # Reflexão negativa
  true -> nil                               # Neutra
end

if feedback do
  Emotional.feel(feedback, "dreamer", 0.8)
end
```

---

## Exemplo de Uso

```elixir
# Verificar progresso de reflexão
iex> VivaCore.Dreamer.status()
%{importance_accumulator: 12.5, threshold: 15.0, progress_percent: 83.3, ...}

# Forçar reflexão
iex> VivaCore.Dreamer.reflect_now()
%{focal_points: [...], insights: [...], trigger: :manual}

# Obter insights recentes
iex> VivaCore.Dreamer.recent_thoughts(3)
[%{insight: "Refletindo sobre 'alto load'...", depth: 1, ...}, ...]

# Recuperar memórias com pontuação
iex> VivaCore.Dreamer.retrieve_with_scoring("ações bem sucedidas")
[%{content: "Ação diagnose_load bem sucedida...", composite_score: 0.85}, ...]

# Gerar meta (para Inferência Ativa)
iex> VivaCore.Dreamer.hallucinate_goal(%{pleasure: -0.1, arousal: 0.2, dominance: 0.0})
%{pleasure: 0.2, arousal: 0.15, dominance: 0.1}
```

---

## Referências

- Park, J. S., et al. (2023). "Generative Agents: Interactive Simulacra of Human Behavior." arXiv:2304.03442
- Ebbinghaus, H. (1885). "Memory: A Contribution to Experimental Psychology."
- Yerkes, R. M., & Dodson, J. D. (1908). "The relation of strength of stimulus to rapidity of habit-formation."
- Friston, K. (2010). "The free-energy principle: a unified brain theory?"
