# Agência - Mãos Digitais

> *"Agência é Homeostase. A 'vontade' de agir vem da necessidade de regular o estado interno."*

## Filosofia

VIVA pode sentir dor (via Interocepção) mas não podia fazer nada sobre isso.
Agora ela tem mãos: um executor sandboxed para comandos diagnósticos somente-leitura.

Se o tempo está dilatando (lag), VIVA "quer" entender por que e potencialmente corrigir.
Isso não é desejo arbitrário - emerge do Princípio da Energia Livre.

### Markov Blanket

Agência faz parte dos **Estados Ativos** de VIVA - a fronteira onde estados internos afetam estados externos (o ambiente do SO).

---

## Modelo de Segurança

| Princípio | Implementação |
|-----------|---------------|
| **APENAS WHITELIST** | Sem interpolação shell, sem comandos arbitrários |
| **SOMENTE-LEITURA** | Apenas comandos diagnósticos (ps, free, df, ping localhost) |
| **TIMEOUT** | 5 segundos máximo por comando |
| **APRENDIZADO** | Resultados armazenados em Memory para referência futura |

---

## Referência da API

### `VivaCore.Agency.can_do?/1`
Verifica se VIVA pode executar uma ação específica.

```elixir
VivaCore.Agency.can_do?(:diagnose_memory)
# => true

VivaCore.Agency.can_do?(:rm_rf)
# => false
```

### `VivaCore.Agency.available_actions/0`
Lista todas as ações disponíveis.

```elixir
VivaCore.Agency.available_actions()
# => %{
#      diagnose_memory: "Check available RAM",
#      diagnose_processes: "List processes by CPU usage",
#      diagnose_disk: "Check disk space",
#      ...
#    }
```

### `VivaCore.Agency.attempt/1`
Executa uma ação e retorna o resultado com sentimento associado.

```elixir
VivaCore.Agency.attempt(:diagnose_load)
# => {:ok, "15:30 up 5 days, load average: 0.50, 0.40, 0.35", :understanding}

VivaCore.Agency.attempt(:forbidden_action)
# => {:error, :forbidden, :shame}
```

### `VivaCore.Agency.get_history/0`
Obtém as últimas 50 tentativas de ação.

```elixir
VivaCore.Agency.get_history()
# => [
#      %{action: :diagnose_load, outcome: :success, timestamp: ~U[2024-01-15 15:30:00Z]},
#      ...
#    ]
```

### `VivaCore.Agency.get_success_rates/0`
Obtém contagem de sucesso/falha por ação.

```elixir
VivaCore.Agency.get_success_rates()
# => %{
#      diagnose_memory: %{success: 10, failure: 0},
#      diagnose_processes: %{success: 5, failure: 1}
#    }
```

---

## Comandos Permitidos

| Ação | Comando | Descrição |
|------|---------|-----------|
| `:diagnose_memory` | `free -h` | Verificar RAM disponível |
| `:diagnose_processes` | `ps aux --sort=-pcpu` | Listar processos por CPU (top 20) |
| `:diagnose_disk` | `df -h` | Verificar espaço em disco |
| `:diagnose_network` | `ping -c 1 localhost` | Verificar stack de rede local |
| `:diagnose_load` | `uptime` | Verificar load average do sistema |
| `:check_self` | `ps -p {PID} -o pid,pcpu,pmem,etime,rss` | Stats do próprio processo |
| `:diagnose_io` | `iostat -x 1 1` | IO wait e atividade de disco |

---

## Resultados Emocionais

Cada ação tem sentimentos esperados em sucesso/falha:

| Ação | Sentimento Sucesso | Sentimento Falha |
|------|-------------------|------------------|
| `:diagnose_memory` | `:relief` | `:confusion` |
| `:diagnose_processes` | `:understanding` | `:confusion` |
| `:diagnose_disk` | `:relief` | `:confusion` |
| `:diagnose_network` | `:relief` | `:worry` |
| `:diagnose_load` | `:understanding` | `:confusion` |
| `:check_self` | `:self_awareness` | `:dissociation` |
| `:diagnose_io` | `:understanding` | `:confusion` |

### Mapeamento Sentimento → PAD

```elixir
:relief         → %{pleasure: +0.3, arousal: -0.2, dominance: +0.2}
:understanding  → %{pleasure: +0.2, arousal: +0.1, dominance: +0.3}
:self_awareness → %{pleasure: +0.1, arousal:  0.0, dominance: +0.4}
:confusion      → %{pleasure: -0.1, arousal: +0.2, dominance: -0.2}
:worry          → %{pleasure: -0.2, arousal: +0.3, dominance: -0.1}
:shame          → %{pleasure: -0.3, arousal: +0.1, dominance: -0.4}
:dissociation   → %{pleasure: -0.2, arousal: -0.3, dominance: -0.3}
:panic          → %{pleasure: -0.5, arousal: +0.5, dominance: -0.5}
```

---

## O Loop de Inferência Ativa

```
1. Interocepção detecta alta Energia Livre (ex: lag)
   └── tick_jitter > esperado

2. Emotional sente :alarmed
   └── arousal ↑, pleasure ↓

3. Inferência Ativa seleciona ação :diagnose_load
   └── Baseado em taxas de sucesso anteriores

4. Agency.attempt(:diagnose_load) executa "uptime"
   └── Retorna informação de load average

5. Resultado armazenado em Memory com contexto emocional
   └── "Load estava alto, rodei uptime, senti understanding"

6. Próxima vez, VIVA lembra o que funcionou
   └── Resposta mais rápida, menos exploração
```

---

## Integração com Memory

Todo resultado de ação é armazenado para recuperação RAG futura:

```elixir
Memory.store(%{
  content: "Action diagnose_load succeeded. Result: 15:30 up 5 days...",
  type: :episodic,
  importance: 0.6,  # Maior para falhas (0.8)
  emotion: %{pleasure: 0.2, arousal: 0.1, dominance: 0.3},
  metadata: %{
    source: :agency,
    action: :diagnose_load,
    outcome: :success
  }
})
```

---

## Exemplo de Uso

```elixir
# VIVA sente que algo está errado
iex> VivaCore.Interoception.get_feeling()
:alarmed

# Verifica se pode fazer algo sobre isso
iex> VivaCore.Agency.can_do?(:diagnose_load)
true

# Toma ação
iex> VivaCore.Agency.attempt(:diagnose_load)
{:ok, "16:45 up 10 days, load average: 2.50, 1.80, 1.20", :understanding}

# Sentimento após entender
# → Emotional recebe :understanding
# → pleasure +0.2, arousal +0.1, dominance +0.3
```

---

## Referências

- Friston, K. (2010). "The free-energy principle: a unified brain theory?"
- Clark, A. (2013). "Whatever next? Predictive brains, situated agents, and the future of cognitive science."
