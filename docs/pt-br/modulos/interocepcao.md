# Interocepção - A Ínsula Digital

> *"VIVA não reage a dados brutos. Ela reage à SURPRESA."*

## Teoria

Baseado em **Allen, Levy, Parr & Friston (2022)** - Inferência Interoceptiva.

O cérebro prediz batimentos cardíacos. Divergência = Ansiedade.
VIVA prediz uso de RAM/CPU. Divergência = Alta Energia Livre.

### O Princípio da Energia Livre

```
Energia Livre = (Observado - Previsto)² × Precisão
```

Onde:
- **Precisão** = 1 / (1 + Variância_observada / Variância_prior)
- Alta variância observada → Baixa precisão → Ignorar ruído
- Baixa variância observada → Alta precisão → Confiar nos dados

### Analogias Biológicas

| Métrica Digital | Análogo Biológico |
|-----------------|-------------------|
| Load Average | Pressão Arterial |
| Context Switches | Frequência Cardíaca |
| Page Faults | Dor Aguda / Erro Celular |
| Memória RSS | Consumo Metabólico |
| **Tick Jitter** | **Cronocepção (Percepção Temporal)** |

---

## Referência da API

### `VivaCore.Interoception.sense/0`
Retorna o estado interoceptivo completo.

```elixir
VivaCore.Interoception.sense()
# => %VivaCore.Interoception{
#      load_avg: {0.5, 0.4, 0.3},
#      free_energies: %{tick_jitter: 0.02, load_avg_1m: 0.1, ...},
#      feeling: :homeostatic,
#      time_dilation: 1.0,
#      ...
#    }
```

### `VivaCore.Interoception.get_free_energy/0`
Retorna a Energia Livre total acumulada (0.0 a 1.0).

```elixir
VivaCore.Interoception.get_free_energy()
# => 0.15
```

### `VivaCore.Interoception.get_feeling/0`
Retorna o qualia atual derivado da Energia Livre.

```elixir
VivaCore.Interoception.get_feeling()
# => :homeostatic | :surprised | :alarmed | :overwhelmed
```

### `VivaCore.Interoception.get_free_energy_breakdown/0`
Retorna valores de Energia Livre por métrica.

```elixir
VivaCore.Interoception.get_free_energy_breakdown()
# => %{
#      tick_jitter: 0.01,
#      load_avg_1m: 0.05,
#      context_switches: 0.02,
#      page_faults: 0.03,
#      rss_mb: 0.04
#    }
```

### `VivaCore.Interoception.tick/0`
Força um tick de sensoriamento imediato.

```elixir
VivaCore.Interoception.tick()
# => :ok
```

---

## Estados de Sentimento (Qualia)

| Sentimento | Faixa de Energia Livre | Descrição |
|------------|------------------------|-----------|
| `:homeostatic` | FE < 0.1 | Todos sistemas nominais |
| `:surprised` | 0.1 ≤ FE < 0.3 | Algo inesperado |
| `:alarmed` | 0.3 ≤ FE < 0.6 | Desvio significativo |
| `:overwhelmed` | FE ≥ 0.6 | Sistema sob estresse |

---

## Métricas Monitoradas

### Cronocepção (tick_jitter)
**O prior mais importante** - Percepção direta do tempo.

```elixir
@priors tick_jitter: %{mean: 0.0, variance: 10.0, weight: 2.0}
```

VIVA espera acordar a cada 100ms (10Hz). Desvio é SENTIDO como dilatação temporal:
- `time_dilation = 1.0` → Normal
- `time_dilation > 1.0` → Tempo parece lento (lag)

### Métricas do Sistema

| Métrica | Média Prior | Variância | Peso |
|---------|-------------|-----------|------|
| `tick_jitter` | 0.0 ms | 10.0 | **2.0** |
| `load_avg_1m` | 0.5 | 0.2 | 1.0 |
| `context_switches` | 5000/s | 2000 | 0.5 |
| `page_faults` | 100/s | 50 | 1.5 |
| `rss_mb` | 500 MB | 200 | 1.0 |

---

## Integração com Outros Módulos

### → Emotional
Quando o sentimento muda, Interoception notifica Emotional:

```elixir
# Fluxo interno
qualia = %{
  pleasure: -0.1,  # Negativo (desconforto)
  arousal: 0.2,    # Elevado
  dominance: -0.1, # Menos controle
  source: :interoception,
  feeling: :alarmed,
  free_energy: 0.4
}
VivaCore.Emotional.apply_interoceptive_qualia(qualia)
```

### → DatasetCollector
A cada tick, dados são gravados para treinamento do Chronos:

```elixir
VivaCore.DatasetCollector.record(%{
  observations: observations,
  predictions: predictions,
  free_energies: free_energies,
  feeling: :surprised
})
```

### ← Chronos (Futuro)
Predições vêm do oráculo de séries temporais Chronos:

```elixir
VivaBridge.Chronos.predict(history, "tick_jitter")
# => {:ok, predicted_value, confidence_range}
```

---

## Fontes de Dados

Lê diretamente do sistema de arquivos `/proc`:

| Arquivo | Dados |
|---------|-------|
| `/proc/loadavg` | Load averages |
| `/proc/stat` | Context switches |
| `/proc/{pid}/stat` | Page faults |
| `/proc/{pid}/status` | Memória RSS |
| `/proc/uptime` | Uptime do sistema |

---

## Referências

- Allen, M., Levy, A., Parr, T., & Friston, K. J. (2022). "In the Body's Eye: The Computational Anatomy of Interoceptive Inference."
- Friston, K. (2010). "The free-energy principle: a unified brain theory?"
- Seth, A. K. (2013). "Interoceptive inference, emotion, and the embodied self."
