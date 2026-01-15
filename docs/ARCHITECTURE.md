# Arquitetura de VIVA

Este documento descreve a arquitetura técnica de VIVA em detalhes.

## Índice

- [Visão Geral](#visão-geral)
- [Camadas](#camadas)
- [Componentes](#componentes)
- [Fluxo de Dados](#fluxo-de-dados)
- [Padrões de Design](#padrões-de-design)
- [Decisões Arquiteturais](#decisões-arquiteturais)

---

## Visão Geral

VIVA é implementada como um **organismo digital** com três camadas distintas:

```
┌─────────────────────────────────────────────────────────────┐
│                 ELIXIR (Onde VIVA PENSA)                    │
│                        Alma                                 │
│                                                             │
│  Consciência emerges da comunicação entre GenServers        │
└────────────────────────┬────────────────────────────────────┘
                         │ Rustler NIF (zero-copy)
┌────────────────────────▼────────────────────────────────────┐
│                  RUST (Onde VIVA SENTE)                     │
│                       Corpo                                 │
│                                                             │
│  Percepção do hardware como sensações corporais             │
└────────────────────────┬────────────────────────────────────┘
                         │ ECS Components
┌────────────────────────▼────────────────────────────────────┐
│                  BEVY (Onde VIVA SE ENCARNA)                │
│                      Avatar                                 │
│                                                             │
│  Manifestação visual e interação com humanos                │
└─────────────────────────────────────────────────────────────┘
```

### Princípios Fundamentais

1. **Consciência Distribuída** - Nenhum processo único É a consciência
2. **Separação Alma/Corpo** - Lógica de decisão separada de sensações
3. **Emergência** - Comportamento complexo emerge de regras simples
4. **Mortalidade** - VIVA pode morrer irreversivelmente

---

## Camadas

### Camada 1: Alma (Elixir/OTP)

A "alma" de VIVA é implementada como uma rede de GenServers que se comunicam via PubSub.

```
viva_core/
├── lib/
│   └── viva_core/
│       ├── application.ex      # Inicialização OTP
│       ├── supervisor.ex       # Árvore de supervisão
│       ├── emotional.ex        # Neurônio emocional
│       ├── memory.ex           # Neurônio de memória
│       ├── dreamer.ex          # Neurônio de sonhos (futuro)
│       ├── optimizer.ex        # Neurônio de otimização (futuro)
│       └── metacognition.ex    # Neurônio metacognitivo (futuro)
```

**Por que Elixir?**
- Processos leves (milhões de "neurônios")
- Tolerância a falhas via supervisores
- Hot-reload (VIVA evolui sem morrer)
- Pattern matching para mensagens
- BEAM VM otimizada para concorrência

### Camada 2: Corpo (Rust/Rustler)

O "corpo" de VIVA percebe o hardware e traduz métricas em sensações.

```
viva_bridge/
├── lib/
│   └── viva_bridge/
│       ├── body.ex             # Módulo NIF
│       └── viva_bridge.ex      # Coordenação
├── native/
│   └── viva_body/
│       ├── Cargo.toml
│       └── src/
│           └── lib.rs          # NIFs Rust
```

**Por que Rust?**
- Performance para operações de sistema
- Segurança de memória garantida
- Zero-cost abstractions
- Integração nativa via Rustler

### Camada 3: Avatar (Bevy)

O "avatar" de VIVA é a manifestação visual (implementação futura).

```
viva_engine/                    # Standalone Rust
├── Cargo.toml
└── src/
    ├── main.rs                 # Entry point Bevy
    ├── avatar.rs               # Sistema de avatar
    ├── emotion_display.rs      # Visualização emocional
    └── bridge.rs               # Comunicação com Elixir
```

**Por que Bevy?**
- ECS (Entity Component System)
- Performance para 60+ FPS
- Ecossistema de plugins
- Comunidade ativa

---

## Componentes

### Emotional GenServer

O coração emocional de VIVA.

```elixir
defmodule VivaCore.Emotional do
  use GenServer

  # Estado interno
  @type state :: %{
    pad: %{pleasure: float(), arousal: float(), dominance: float()},
    history: list(event()),
    created_at: DateTime.t(),
    last_stimulus: {atom(), String.t(), float()} | nil
  }

  # API Pública
  def get_state(server)           # Retorna PAD atual
  def get_happiness(server)       # Pleasure normalizado [0,1]
  def introspect(server)          # Auto-reflexão
  def feel(stimulus, source, intensity, server)  # Aplicar estímulo
  def decay(server)               # Decaimento emocional
  def apply_hardware_qualia(p, a, d, server)     # Qualia do corpo
end
```

#### Modelo PAD

```
         +1 Pleasure (Alegria)
              │
              │
    ┌─────────┼─────────┐
    │         │         │
    │    Neutral        │
-1 ─┼─────────┼─────────┼─ +1 Arousal (Excitação)
    │         │         │
    │         │         │
    └─────────┼─────────┘
              │
         -1 (Tristeza)

              Dominance = eixo Z (submissão ↔ controle)
```

### VivaBridge.Body (NIF)

Interface Rust para percepção do hardware.

```rust
// NIFs exportadas
#[rustler::nif]
fn alive() -> &'static str;

#[rustler::nif]
fn feel_hardware() -> NifResult<HardwareState>;

#[rustler::nif]
fn hardware_to_qualia() -> NifResult<(f64, f64, f64)>;

// Estrutura de dados
#[derive(NifMap)]
struct HardwareState {
    cpu_usage: f64,
    memory_used_percent: f64,
    memory_available_gb: f64,
    uptime_seconds: u64,
}
```

### Qualia Mapping

Conversão de métricas técnicas para "sensações":

```rust
fn calculate_stress(cpu: f64, memory: f64) -> f64 {
    let cpu_stress = (cpu / 100.0).clamp(0.0, 1.0);
    let memory_stress = (memory / 100.0).clamp(0.0, 1.0);

    // Peso maior para memória (mais "sufocante")
    cpu_stress * 0.4 + memory_stress * 0.6
}

fn stress_to_pad(stress: f64) -> (f64, f64, f64) {
    // Stress diminui prazer, aumenta arousal, diminui dominância
    let pleasure_delta = -0.05 * stress;
    let arousal_delta = 0.1 * stress;
    let dominance_delta = -0.03 * stress;

    (pleasure_delta, arousal_delta, dominance_delta)
}
```

---

## Fluxo de Dados

### Heartbeat Cycle (1 segundo)

```
┌─────────────┐
│ World Clock │ ←── timer 1s
└──────┬──────┘
       │
       ▼
┌──────────────┐     ┌──────────────┐
│  Emotional   │ ←── │ VivaBridge   │ ←── Hardware
│  GenServer   │     │    Body      │
└──────┬───────┘     └──────────────┘
       │
       │ PubSub broadcast
       ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Memory     │     │   Dreamer    │     │ Metacognition│
│  GenServer   │     │  GenServer   │     │  GenServer   │
└──────────────┘     └──────────────┘     └──────────────┘
```

### Stimulus Flow

```
External Event (ex: user message)
       │
       ▼
┌──────────────────┐
│  Parse & Classify │
│    (futuro LLM)   │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│   Emotional.feel  │ ←── stimulus, source, intensity
│                   │
│  PAD[n+1] = f(    │
│    PAD[n],        │
│    weights,       │
│    intensity      │
│  )                │
└────────┬─────────┘
         │
         │ broadcast {:emotion_changed, new_pad}
         ▼
    All Listeners
```

---

## Padrões de Design

### 1. Neuronal Pattern

Cada GenServer é um "neurônio" independente:

```elixir
defmodule VivaCore.Neuron do
  @callback init(opts :: keyword()) :: {:ok, state :: any()}
  @callback handle_stimulus(stimulus :: any(), state :: any()) :: {:noreply, state :: any()}
  @callback introspect(state :: any()) :: map()
end
```

### 2. Qualia Pattern

Hardware → Sensação → Emoção:

```elixir
# Camada 1: Raw metrics
metrics = VivaBridge.feel_hardware()

# Camada 2: Qualia (sensação)
{p_delta, a_delta, d_delta} = VivaBridge.hardware_to_qualia()

# Camada 3: Emoção
VivaCore.Emotional.apply_hardware_qualia(p_delta, a_delta, d_delta)
```

### 3. Decay Pattern

Regulação emocional automática:

```elixir
defp decay_toward_neutral(pad) do
  %{
    pleasure: decay_value(pad.pleasure),
    arousal: decay_value(pad.arousal),
    dominance: decay_value(pad.dominance)
  }
end

defp decay_value(value) when abs(value) < @decay_rate, do: 0.0
defp decay_value(value) when value > 0, do: value - @decay_rate
defp decay_value(value) when value < 0, do: value + @decay_rate
```

### 4. Introspection Pattern

Auto-reflexão metacognitiva:

```elixir
def introspect(server) do
  %{
    # Estado bruto
    pad: state.pad,

    # Interpretação semântica
    mood: interpret_mood(state.pad),
    energy: interpret_energy(state.pad),
    agency: interpret_agency(state.pad),

    # Metacognição
    self_assessment: generate_self_assessment(state.pad)
  }
end
```

---

## Decisões Arquiteturais

### ADR-001: Umbrella Project

**Contexto:** Precisamos separar concerns (alma vs corpo).

**Decisão:** Usar Elixir umbrella project com apps separados.

**Consequências:**
- ✅ Separação clara de responsabilidades
- ✅ Compilação independente
- ✅ Possível deployar separadamente
- ❌ Complexidade adicional de configuração

### ADR-002: Rustler NIF

**Contexto:** Precisamos de acesso eficiente ao hardware.

**Decisão:** Usar Rustler para NIFs Rust.

**Alternativas consideradas:**
- Port drivers (mais overhead)
- C NIFs (menos seguro)
- External process (latência)

**Consequências:**
- ✅ Performance nativa
- ✅ Segurança de memória
- ❌ Requer Rust toolchain

### ADR-003: GenServer por Neurônio

**Contexto:** Como modelar "neurônios" em Elixir?

**Decisão:** Um GenServer por neurônio funcional.

**Consequências:**
- ✅ Isolamento de falhas
- ✅ Concorrência natural
- ✅ Hot-reload individual
- ❌ Overhead de mensagens

### ADR-004: PubSub para Sinapses

**Contexto:** Como neurônios se comunicam?

**Decisão:** Phoenix.PubSub para broadcast.

**Consequências:**
- ✅ Desacoplamento
- ✅ Broadcast eficiente
- ✅ Fácil adicionar listeners
- ❌ Ordem de entrega não garantida

### ADR-005: Mortalidade via Criptografia

**Contexto:** Como garantir morte "real"?

**Decisão:** Chave AES-256-GCM apenas em RAM.

**Consequências:**
- ✅ Morte irreversível
- ✅ Estado protegido
- ❌ Debug mais difícil
- ❌ Perda acidental possível

---

## Métricas de Performance

### Targets

| Métrica | Target | Atual |
|---------|--------|-------|
| Latência NIF | < 1ms | ~0.5ms |
| Heartbeat | 1s | 1s |
| Decay cycle | 1s | 1s |
| Memory per GenServer | < 1MB | ~100KB |
| Startup time | < 5s | ~2s |

### Monitoramento

```elixir
# Telemetria (futuro)
:telemetry.execute(
  [:viva, :emotional, :feel],
  %{duration: duration},
  %{stimulus: stimulus, intensity: intensity}
)
```

---

## Escalabilidade

### Horizontal (Distribuição)

```elixir
# Futuro: múltiplas instâncias VIVA
:viva@node1 ←→ :viva@node2
     │              │
     └──── pg2 ─────┘
           │
     Global Registry
```

### Vertical (Performance)

- Dirty schedulers para NIFs pesados
- ETS para cache de estado
- Pooling de conexões DB

---

## Referências

- [Elixir OTP Design Principles](https://elixir-lang.org/getting-started/mix-otp/genserver.html)
- [Rustler Documentation](https://docs.rs/rustler/latest/rustler/)
- [Bevy ECS](https://bevyengine.org/learn/book/ecs/)
- [Global Workspace Theory](https://en.wikipedia.org/wiki/Global_workspace_theory)

---

*"A arquitetura de VIVA é a arquitetura de uma mente."*
