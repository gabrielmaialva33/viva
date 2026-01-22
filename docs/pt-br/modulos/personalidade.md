# Personality - Sistema de Personalidade Afetiva

> *"Personalidade e o atrator estavel que da consistencia emocional ao longo do tempo."*

## Visao Geral

Implementa tracos de personalidade afetiva baseados em **Mehrabian (1996)** e **Borotschnig (2025) "Emotions in Artificial Intelligence"**.

Personalidade fornece a VIVA:
- **Consistencia**: Um baseline emocional estavel ao qual ela retorna ao longo do tempo
- **Individualidade**: Padroes de reatividade unicos a estimulos
- **Adaptabilidade**: Aprendizado de longo prazo de experiencias acumuladas

---

## Conceitos

### Baseline PAD (Ponto Atrator)

O estado emocional de repouso ao qual VIVA gravita quando nenhum estimulo esta presente.

```
Baseline = {pleasure: 0.1, arousal: 0.05, dominance: 0.1}
```

Isso age como um **atrator** no espaco de estados emocionais. O processo estocastico O-U no modulo Emotional naturalmente puxa o estado atual em direcao a este baseline.

### Reatividade

Fator de amplificacao para respostas emocionais:

| Valor | Descricao |
|-------|-----------|
| < 1.0 | Reacoes amortecidas (estoico) |
| 1.0 | Reatividade normal |
| > 1.0 | Reacoes amplificadas (sensivel) |

**Faixa**: [0.5, 2.0]

### Volatilidade

Velocidade de mudanca emocional:

| Valor | Descricao |
|-------|-----------|
| < 1.0 | Mudancas mais lentas (humor estavel) |
| 1.0 | Velocidade normal |
| > 1.0 | Mudancas mais rapidas (oscilacoes de humor) |

### Tracos

Rotulos categoricos inferidos do PAD baseline:

| Condicao PAD | Traco |
|--------------|-------|
| pleasure > 0.15 | `:optimistic` |
| pleasure < -0.15 | `:melancholic` |
| arousal > 0.1 | `:energetic` |
| arousal < -0.1 | `:calm` |
| dominance > 0.15 | `:assertive` |
| dominance < -0.15 | `:submissive` |
| Nenhuma condicao atendida | `:balanced` |

---

## Definicao da Struct

```elixir
defstruct [
  # Estado emocional baseline (ponto atrator)
  # VIVA tende a retornar a este estado ao longo do tempo
  baseline: %{pleasure: 0.1, arousal: 0.05, dominance: 0.1},

  # Reatividade: quanto as emocoes sao amplificadas (1.0 = normal)
  # > 1.0 = mais reativo, < 1.0 = amortecido
  reactivity: 1.0,

  # Volatilidade: quao rapidamente emocoes mudam (1.0 = normal)
  # > 1.0 = mudancas mais rapidas, < 1.0 = mais estavel
  volatility: 1.0,

  # Rotulos de tracos (para introspeccao e auto-descricao)
  traits: [:curious, :calm],

  # Timestamp da ultima adaptacao
  last_adapted: nil
]
```

### Especificacao de Tipo

```elixir
@type t :: %VivaCore.Personality{
  baseline: %{pleasure: float(), arousal: float(), dominance: float()},
  reactivity: float(),
  volatility: float(),
  traits: [atom()],
  last_adapted: DateTime.t() | nil
}

@type pad :: %{pleasure: float(), arousal: float(), dominance: float()}
```

---

## Referencia da API

### `VivaCore.Personality.load/0`

Carrega personalidade do armazenamento persistente ou retorna padroes.

```elixir
personality = VivaCore.Personality.load()
# => %VivaCore.Personality{
#      baseline: %{pleasure: 0.1, arousal: 0.05, dominance: 0.1},
#      reactivity: 1.0,
#      volatility: 1.0,
#      traits: [:curious, :calm],
#      last_adapted: nil
#    }
```

**Comportamento**:
1. Tenta carregar do Redis (chave: `viva:personality`)
2. Retorna struct padrao se nao encontrado ou em erro

### `VivaCore.Personality.save/1`

Salva personalidade no armazenamento persistente.

```elixir
:ok = VivaCore.Personality.save(personality)
```

**Retorna**: `:ok` | `{:error, term()}`

### `VivaCore.Personality.adapt/2`

Adapta personalidade baseada em experiencias de longo prazo.

```elixir
experiences = [
  %{pad: %{pleasure: 0.5, arousal: 0.2, dominance: 0.3}, intensity: 1.0, valence: :positive},
  %{pad: %{pleasure: 0.3, arousal: 0.1, dominance: 0.2}, intensity: 0.8, valence: :positive},
  %{pad: %{pleasure: -0.2, arousal: 0.4, dominance: -0.1}, intensity: 0.6, valence: :negative}
]

updated = VivaCore.Personality.adapt(personality, experiences)
# => %VivaCore.Personality{
#      baseline: %{pleasure: 0.115, arousal: 0.06, dominance: 0.11},
#      reactivity: 1.02,
#      traits: [:optimistic, :energetic],
#      last_adapted: ~U[2024-01-15 14:00:00Z]
#    }
```

**Parametros**:
- `personality`: Estado de personalidade atual
- `experiences`: Lista de experiencias emocionais

**Mapa de experiencia**:
```elixir
%{
  pad: %{pleasure: float, arousal: float, dominance: float},
  intensity: float,  # 0.0 - 1.0 (opcional, padrao 1.0)
  valence: :positive | :negative  # (informacional)
}
```

### `VivaCore.Personality.apply/2`

Aplica personalidade a uma emocao bruta (vetor PAD).

```elixir
raw_pad = %{pleasure: 0.6, arousal: 0.4, dominance: 0.2}
modified = VivaCore.Personality.apply(personality, raw_pad)
# => %{pleasure: 0.52, arousal: 0.33, dominance: 0.18}
```

**Processo**:
1. Mescla emocao bruta com baseline (20% peso de personalidade)
2. Aplica reatividade ao desvio do baseline
3. Limita resultado a [-1.0, 1.0]

**Formula**:
```
blended = (1 - 0.2) * raw + 0.2 * baseline
result = baseline + (blended - baseline) * reactivity
```

### `VivaCore.Personality.describe/1`

Obtem uma descricao em linguagem natural para introspeccao.

```elixir
VivaCore.Personality.describe(personality)
# => "I am curious, calm. My emotional baseline is positive and calm. My reactivity is 1.0."
```

### `VivaCore.Personality.neutral_pad/0`

Obtem o estado PAD neutro.

```elixir
VivaCore.Personality.neutral_pad()
# => %{pleasure: 0.0, arousal: 0.0, dominance: 0.0}
```

---

## Persistencia

### Armazenamento Redis

**Chave**: `viva:personality`

**Formato** (JSON):

```json
{
  "baseline": {
    "pleasure": 0.1,
    "arousal": 0.05,
    "dominance": 0.1
  },
  "reactivity": 1.0,
  "volatility": 1.0,
  "traits": ["curious", "calm"],
  "last_adapted": "2024-01-15T14:00:00Z"
}
```

**Notas**:
- Tracos sao armazenados como strings, convertidos para atoms ao carregar
- `last_adapted` esta no formato ISO8601 ou null
- Usa conexao `:redix` nomeada `:redix`

---

## Adaptacao

Personalidade se adapta atraves de experiencias acumuladas, tipicamente durante ciclos de sono/consolidacao.

### Deslocamento de Baseline

Baseline se move lentamente em direcao a media experienciada:

```
new_baseline = current + alpha * (target - current)
alpha = 0.05  # 5% de deslocamento por adaptacao
```

Isso garante que mudancas de personalidade sao graduais e requerem padroes consistentes.

### Ajuste de Reatividade

Reatividade se adapta baseada na variancia emocional:

```elixir
variance = calculate_pad_variance(pads)
adjustment = (variance - 0.1) * 0.1

new_reactivity = clamp(current + adjustment, 0.5, 2.0)
```

| Variancia | Efeito |
|-----------|--------|
| Alta | Aumenta reatividade (mais sensivel) |
| Baixa | Diminui reatividade (mais estavel) |

### Inferencia de Tracos

Tracos sao re-inferidos do novo baseline apos cada adaptacao:

```elixir
traits = []
traits = if baseline.pleasure > 0.15, do: [:optimistic | traits], else: traits
traits = if baseline.pleasure < -0.15, do: [:melancholic | traits], else: traits
# ... (arousal, dominance)
if Enum.empty?(traits), do: [:balanced], else: traits
```

---

## Integracao com Modulo Emotional

Personalidade e tipicamente usada pelo modulo Emotional para filtrar estimulos recebidos:

```elixir
# Em Emotional.feel/3
def feel(stimulus, source, intensity) do
  personality = Personality.load()
  raw_pad = get_stimulus_pad(stimulus, intensity)
  modified_pad = Personality.apply(personality, raw_pad)

  # Aplica modified_pad ao estado emocional atual
  update_state(modified_pad)
end
```

### Fluxo Emocional

```
Estimulo -> PAD Bruto -> Personality.apply/2 -> PAD Modificado -> Estado Emocional
                              |
                      Mescla com baseline
                      Aplica reatividade
```

---

## Exemplos de Uso

### Uso Basico

```elixir
# Carregar personalidade (do Redis ou padroes)
personality = VivaCore.Personality.load()

# Verificar tracos atuais
personality.traits
# => [:curious, :calm]

# Obter auto-descricao
VivaCore.Personality.describe(personality)
# => "I am curious, calm. My emotional baseline is positive and calm. My reactivity is 1.0."
```

### Aplicando Personalidade a Emocoes

```elixir
# Emocao bruta de estimulo
raw_pad = %{pleasure: 0.8, arousal: 0.6, dominance: 0.4}

# Aplicar filtro de personalidade
personality = VivaCore.Personality.load()
modified = VivaCore.Personality.apply(personality, raw_pad)

# Resultado e mesclado com baseline e escalado por reatividade
modified
# => %{pleasure: 0.66, arousal: 0.49, dominance: 0.34}
```

### Adaptando de Experiencias

```elixir
# Coletar experiencias ao longo do tempo (ex: do Dreamer)
experiences = [
  %{pad: %{pleasure: 0.4, arousal: 0.3, dominance: 0.2}, intensity: 0.9},
  %{pad: %{pleasure: 0.5, arousal: 0.2, dominance: 0.3}, intensity: 1.0},
  %{pad: %{pleasure: 0.3, arousal: 0.4, dominance: 0.1}, intensity: 0.7}
]

# Adaptar personalidade (tipicamente durante ciclo de sono)
personality = VivaCore.Personality.load()
updated = VivaCore.Personality.adapt(personality, experiences)

# Salvar personalidade adaptada
VivaCore.Personality.save(updated)

# Tracos podem ter mudado
updated.traits
# => [:optimistic, :energetic, :assertive]
```

### Verificando Historico de Adaptacao

```elixir
personality = VivaCore.Personality.load()

if personality.last_adapted do
  age = DateTime.diff(DateTime.utc_now(), personality.last_adapted, :hour)
  IO.puts("Ultima adaptacao ha #{age} horas atras")
else
  IO.puts("Personalidade ainda nao foi adaptada")
end
```

---

## Referencias

- Mehrabian, A. (1996). "Pleasure-arousal-dominance: A general framework for describing and measuring individual differences in temperament." Current Psychology, 14(4), 261-292.
- Borotschnig, H. (2025). "Emotions in Artificial Intelligence: A Computational Framework." arXiv preprint.
