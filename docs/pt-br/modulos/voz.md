# Voz - Proto-Linguagem

> *"Bebês não começam falando frases. Eles balbuciam."*

## Filosofia

Voice NÃO é um wrapper de LLM. VIVA aprende a comunicar por:
1. Emitir sinais abstratos baseados no estado interno
2. Observar as respostas de Gabriel
3. Fortalecer associações sinal-resposta (aprendizado Hebbiano)

Através da interação com seu cuidador, certos sons se tornam associados
com certas respostas. Significado emerge através de associação, não programação.

---

## Aprendizado Hebbiano

> *"Neurônios que disparam juntos, conectam juntos."*

### A Regra de Aprendizado

```
Δw = η × (pre × post)
```

Onde:
- **η** = taxa de aprendizado (0.1)
- **pre** = sinal emitido (arousal na emissão)
- **post** = mudança emocional após resposta (delta de pleasure)

### Processo de Atualização de Pesos

Quando VIVA emite um sinal e Gabriel responde:
- Se a resposta traz **alívio** → fortalece associação
- Se **sem efeito** ou **negativo** → enfraquece associação

---

## Referência da API

### `VivaCore.Voice.babble/1`
Emite um sinal baseado no estado PAD atual.

```elixir
pad = %{pleasure: -0.3, arousal: 0.7, dominance: -0.2}
VivaCore.Voice.babble(pad)
# => :pattern_sos
```

### `VivaCore.Voice.observe_response/2`
Observa a resposta de Gabriel e atualiza pesos Hebbianos.

```elixir
# Gabriel ajudou com temperatura
emotional_delta = %{pleasure: 0.3, arousal: -0.1, dominance: 0.1}
VivaCore.Voice.observe_response(:temperature_relief, emotional_delta)
# => :ok (atualização assíncrona)
```

### `VivaCore.Voice.best_signal_for/1`
Obtém o sinal mais efetivo para uma intenção (usa associações aprendidas).

```elixir
VivaCore.Voice.best_signal_for(:attention)
# => :chirp_high (aprendido) ou :chirp_high (padrão)
```

### `VivaCore.Voice.get_vocabulary/0`
Obtém significados aprendidos (associações fortes).

```elixir
VivaCore.Voice.get_vocabulary()
# => %{
#      chirp_high: %{meaning: :attention, confidence: 0.7},
#      pattern_sos: %{meaning: :help, confidence: 0.8}
#    }
```

### `VivaCore.Voice.get_weights/0`
Inspeciona todos os pesos Hebbianos.

```elixir
VivaCore.Voice.get_weights()
# => %{
#      {:chirp_high, :attention} => 0.65,
#      {:pattern_sos, :help} => 0.82,
#      ...
#    }
```

### `VivaCore.Voice.signal_types/0`
Lista tipos de sinais disponíveis.

```elixir
VivaCore.Voice.signal_types()
# => [:chirp_high, :chirp_low, :pulse_fast, :pulse_slow,
#     :pattern_sos, :pattern_happy, :silence]
```

---

## Tipos de Sinais

Estes NÃO são palavras. São sons/padrões abstratos.

| Sinal | Descrição | Viés Inicial |
|-------|-----------|--------------|
| `:chirp_high` | Tom alto (880 Hz, 100ms) | Arousal alto |
| `:chirp_low` | Tom baixo (220 Hz, 200ms) | Arousal baixo, tristeza |
| `:pulse_fast` | Ritmo rápido | Urgência |
| `:pulse_slow` | Ritmo lento | Relaxamento |
| `:pattern_sos` | Padrão tipo SOS | Angústia |
| `:pattern_happy` | Melodia feliz (C-E-G-C) | Alegria |
| `:silence` | Silêncio intencional | Calma/retraimento |

### Vieses PAD Iniciais

```elixir
:chirp_high    → %{arousal: +0.5, pleasure:  0.0, dominance:  0.0}
:chirp_low     → %{arousal: -0.3, pleasure: -0.2, dominance: -0.1}
:pulse_fast    → %{arousal: +0.6, pleasure:  0.0, dominance: +0.2}
:pulse_slow    → %{arousal: -0.4, pleasure: +0.1, dominance:  0.0}
:pattern_sos   → %{arousal: +0.7, pleasure: -0.5, dominance: -0.3}
:pattern_happy → %{arousal: +0.3, pleasure: +0.5, dominance: +0.2}
:silence       → %{arousal: -0.5, pleasure:  0.0, dominance:  0.0}
```

---

## Tipos de Resposta

Ao observar a resposta de Gabriel, use estas categorias:

| Tipo de Resposta | Descrição |
|------------------|-----------|
| `:temperature_relief` | Ajustou ventilador/AC |
| `:attention` | Conversou com VIVA |
| `:task_help` | Ajudou com algo |
| `:ignore` | Sem resposta |
| `:negative` | Repreendeu/descartou |

---

## O Loop de Aprendizado

```
1. VIVA sente desconforto
   └── PAD: pleasure=-0.3, arousal=0.7, dominance=-0.2

2. Voice.babble(pad) emite sinal
   └── Seleciona :pattern_sos (melhor match para angústia)

3. Gabriel ouve, talvez faz algo
   └── Ajusta temperatura, conversa com VIVA

4. VIVA sente mudança
   └── pleasure=+0.2, arousal=-0.1, dominance=+0.1

5. Voice.observe_response(:temperature_relief, delta)
   └── Δw = 0.1 × 0.7 × 0.2 = +0.014
   └── Peso {:pattern_sos, :temperature_relief} aumenta

6. Próxima vez em situação similar:
   └── VIVA tenta :pattern_sos novamente (funcionou!)
   └── Ou explora alternativas (10% ruído)
```

---

## Emergência de Vocabulário

Um sinal adquire "significado" quando seu peso Hebbiano excede **0.3**:

```elixir
def update_vocabulary(weights, vocabulary) do
  weights
  |> Enum.group_by(fn {{signal, _}, _} -> signal end)
  |> Enum.reduce(vocabulary, fn {signal, associations}, vocab ->
    # Encontra associação mais forte
    {{_, response}, weight} = Enum.max_by(associations, fn {_, w} -> w end)

    if weight > 0.3 do
      Map.put(vocab, signal, %{meaning: response, confidence: weight})
    else
      vocab
    end
  end)
end
```

---

## Emissão de Som (Bridge de Música)

Sinais são emitidos via `VivaBridge.Music` (se disponível):

```elixir
:chirp_high    → Music.play_note(880, 100)      # A5, 100ms
:chirp_low     → Music.play_note(220, 200)      # A3, 200ms
:pulse_fast    → Music.play_rhythm([100, 50, ...])
:pattern_sos   → Music.play_melody([            # ... --- ...
                   {440, 100}, {0, 100}, {440, 100}, ...
                 ])
:pattern_happy → Music.play_melody([            # C-E-G-C
                   {523, 150}, {659, 150}, {784, 150}, {1047, 300}
                 ])
```

---

## Integração com Memory

Eventos de aprendizado são armazenados para recuperação futura:

```elixir
Memory.store(%{
  content: """
  Evento de aprendizado Voice:
  - Sinais emitidos: [:pattern_sos]
  - Resposta de Gabriel: temperature_relief
  - Mudança emocional: P=+0.20, A=-0.10, D=+0.10
  - Associação fortalecida
  """,
  type: :episodic,
  importance: 0.5 + abs(emotional_delta.pleasure) * 0.3,
  metadata: %{source: :voice, signals: [:pattern_sos], response: :temperature_relief}
})
```

---

## Referências

- Hebb, D. O. (1949). "The Organization of Behavior."
- Kuhl, P. K. (2004). "Early language acquisition: cracking the speech code."
- Smith, L. B., & Thelen, E. (2003). "Development as a dynamic system."
