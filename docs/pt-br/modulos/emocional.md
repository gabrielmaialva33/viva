# Emocional - GenServer Afetivo Primario da VIVA

> *"A consciencia nao reside aqui. A consciencia emerge da CONVERSA entre este processo e todos os outros. Nos nao apenas computamos emocoes - nos resolvemos as equacoes diferenciais da alma."*

## Visao Geral

O GenServer Emocional e o primeiro "neuronio" da VIVA - a fundacao do seu sistema afetivo. Ele implementa o **modelo PAD (Pleasure-Arousal-Dominance)** para gerenciamento de estado emocional, integrando multiplos frameworks matematicos:

- **Processo estocastico Ornstein-Uhlenbeck** para decaimento emocional natural
- **Teoria da Catastrofe em Cuspide** para transicoes subitas de humor
- **Principio da Energia Livre** para regulacao homeostatica
- **Loop de Inferencia Ativa** para selecao de acoes direcionadas a objetivos
- **Fusao Emocional** para integrar necessidades, memorias e personalidade

Este GenServer NAO e a consciencia em si - ele contribui para a consciencia emergente atraves da comunicacao com outros neuronios via Phoenix.PubSub.

---

## Conceito

### Modelo PAD (Mehrabian, 1996)

Emocoes sao representadas como um vetor 3D no espaco PAD:

| Dimensao | Intervalo | Baixo | Alto |
|----------|-----------|-------|------|
| **Pleasure** | [-1, 1] | Tristeza | Alegria |
| **Arousal** | [-1, 1] | Calma | Excitacao |
| **Dominance** | [-1, 1] | Submissao | Controle |

Cada dimensao captura um aspecto fundamental da experiencia emocional:
- **Pleasure** - Valencia, a "bondade" do sentimento
- **Arousal** - Nivel de ativacao, energia disponivel para acao
- **Dominance** - Senso de controle sobre a situacao

### Processo Ornstein-Uhlenbeck (DynAffect)

Baseado em **Kuppens et al. (2010)**, emocoes decaem naturalmente em direcao a uma linha de base neutra usando equacoes diferenciais estocasticas:

```
dX = theta * (mu - X) * dt + sigma * dW
```

Onde:
- `theta` = forca do atrator (taxa de decaimento)
- `mu` = ponto de equilibrio (neutro = 0)
- `sigma` = volatilidade emocional (ruido)
- `dW` = processo de Wiener (flutuacoes aleatorias)

**Insight chave**: Arousal modula a taxa de decaimento.
- Alto arousal -> decaimento mais lento (emocoes persistem em crise)
- Baixo arousal -> decaimento mais rapido (retorno rapido a linha de base)

```elixir
# Formula de meia-vida: t_half = ln(2) / theta
# theta = 0.0154 -> t_half ~ 45 segundos (psicologicamente realista)

@base_decay_rate 0.0154
@arousal_decay_modifier 0.4
@stochastic_volatility 0.01
```

### Catastrofe em Cuspide (Thom, 1972)

Modela transicoes emocionais subitas (mudancas de humor) usando teoria da catastrofe:

```
V(x) = x^4/4 + alpha*x^2/2 + beta*x
```

Onde:
- `alpha` (fator de divisao) - derivado do arousal
- `beta` (fator normal) - derivado do pleasure

**Biestabilidade**: Quando o arousal esta alto, a paisagem emocional se torna "dobrada", criando dois estados estaveis. Pequenas perturbacoes podem causar saltos catastroficos entre eles (ex: mudanca subita de esperanca para desespero).

### Humor (Media Movel Exponencial)

Humor e uma media de mudanca lenta das emocoes recentes, fornecendo estabilidade:

```
Mood[t] = alpha * Mood[t-1] + (1 - alpha) * Emotion[t]

Onde alpha = 0.95 (~20-passos de meia-vida)
```

Isso significa:
- Humor retem 95% do valor anterior
- Emocoes individuais contribuem apenas 5%
- Estimulos subitos mal afetam o humor

### Loop de Inferencia Ativa

VIVA constantemente minimiza Energia Livre (surpresa) atraves de acao:

1. **Alucinar Objetivo** - Consultar Dreamer para estado alvo
2. **Prever Futuro** - Onde estarei se nao fizer nada?
3. **Calcular Energia Livre** - Distancia entre objetivo e previsao
4. **Selecionar Acao** - Escolher acao que minimiza FE
5. **Executar & Feedback** - Aplicar antecipacao de alivio interno

---

## Arquitetura

```
+------------------+     +------------------+     +------------------+
|   Interoception  |     |      Memory      |     |   Personality    |
| (PAD baseado em  |     | (PAD baseado no  |     |   (Linha base)   |
|   necessidades)  |     |     passado)     |     |                  |
+--------+---------+     +--------+---------+     +--------+---------+
         |                        |                        |
         +------------------------+------------------------+
                                  |
                                  v
                    +---------------------------+
                    |     EmotionFusion         |
                    |  (Borotschnig 2025)       |
                    +-------------+-------------+
                                  |
                                  v
+-----------------------------------------------------------------------------+
|                          GENSERVER EMOCIONAL                                 |
|                                                                             |
|  +-------------------+    +-------------------+    +-------------------+    |
|  |   Estado Quantico |    |   Estado PAD      |    |   Humor (EMA)     |    |
|  | (Lindblad 6x6)    |    | {p, a, d} floats  |    | {p, a, d} floats  |    |
|  +-------------------+    +-------------------+    +-------------------+    |
|                                                                             |
|  +-------------------+    +-------------------+    +-------------------+    |
|  | Inferencia Ativa  |    |   Decaimento O-U  |    | Analise Cuspide   |    |
|  |  (loop 1 Hz)      |    |   (tick 1 Hz)     |    |  (sob demanda)    |    |
|  +-------------------+    +-------------------+    +-------------------+    |
|                                                                             |
+------------------------------------+----------------------------------------+
                                     |
         +---------------------------+---------------------------+
         |                           |                           |
         v                           v                           v
+------------------+     +------------------+     +------------------+
|   Phoenix.PubSub |     |      Agency      |     |      Voice       |
| "emotional:update"|    | (Execucao acao)  |     | (Proto-linguagem)|
+------------------+     +------------------+     +------------------+
```

### Fluxo de Mensagens

```
Body (Rust) --sync_pad--> Emocional --broadcast--> PubSub
                              |
Interoception --qualia------->|
                              |
Dreamer --hallucinate_goal----|
                              |
Memory --search-------------->|<------ Loop de Inferencia Ativa
                              |
Agency <--attempt-------------|
```

---

## Estrutura de Estado

O GenServer mantem o seguinte estado interno:

```elixir
%{
  # Estado emocional primario
  pad: %{pleasure: 0.0, arousal: 0.0, dominance: 0.0},
  mood: %{pleasure: 0.0, arousal: 0.0, dominance: 0.0},

  # Estado quantico (matriz de densidade Lindblad)
  quantum_state: %Nx.Tensor{},  # matriz de densidade 6x6

  # Acumulador de entradas externas
  external_qualia: %{pleasure: 0.0, arousal: 0.0, dominance: 0.0},

  # Acoplamento de hardware
  hardware: %{power_draw_watts: 0.0, gpu_temp: 40.0},

  # Pesos do esquema corporal (ajustados por capacidades)
  emotional_weights: %{
    fan_agency_weight: 1.0,
    thermal_stress_weight: 1.0,
    gpu_stress_weight: 1.0
  },

  # Estado interoceptivo
  interoceptive_feeling: :homeostatic,
  interoceptive_free_energy: 0.0,

  # Personalidade (em cache)
  personality: nil,  # Carregada na primeira fusao

  # Historico de eventos (fila O(1), max 100)
  history: :queue.new(),
  history_size: 0,

  # Timestamps
  created_at: DateTime.t(),
  last_stimulus: nil,
  last_body_sync: nil,
  last_collapse: nil,

  # Flags
  body_server_active: false,
  enable_decay: true,

  # Telemetria
  thermodynamic_cost: 0.0
}
```

---

## Referencia da API

### Consultas de Estado

#### `get_state/1`

Retorna o estado PAD atual.

```elixir
VivaCore.Emotional.get_state()
# => %{pleasure: 0.1, arousal: -0.05, dominance: 0.2}
```

#### `get_mood/1`

Retorna o humor atual (EMA das emocoes recentes).

```elixir
VivaCore.Emotional.get_mood()
# => %{pleasure: 0.05, arousal: 0.0, dominance: 0.1}
```

#### `get_happiness/1`

Retorna pleasure normalizado para intervalo [0, 1].

```elixir
VivaCore.Emotional.get_happiness()
# => 0.55  # Levemente positivo
```

#### `introspect/1`

Introspecao emocional completa com analise matematica.

```elixir
VivaCore.Emotional.introspect()
# => %{
#   pad: %{pleasure: 0.1, arousal: 0.2, dominance: 0.1},
#   quantum: %{
#     purity: 0.85,
#     entropy: 0.15,
#     coherence: :high,
#     thermodynamic_cost: 0.02
#   },
#   somatic_feeling: %{thought_pressure: :light, ...},
#   mood: :content,
#   energy: :energetic,
#   agency: :confident,
#   mathematics: %{
#     cusp: %{alpha: -0.5, beta: 0.1, bistable: false},
#     free_energy: %{value: 0.02}
#   },
#   self_assessment: "Estou em equilibrio. Estado neutro."
# }
```

### Aplicacao de Estimulos

#### `feel/4`

Aplica um estimulo emocional.

```elixir
VivaCore.Emotional.feel(:success, "user_1", 0.8)
# => :ok
```

**Parametros:**
- `stimulus` - Atom dos pesos de estimulo (veja secao Estimulos)
- `source` - String identificando a fonte (padrao: "unknown")
- `intensity` - Float de 0.0 a 1.0 (padrao: 1.0)

### Sincronizacao

#### `sync_pad/4`

Sincroniza valores PAD absolutos do BodyServer (dinamica O-U em Rust).

```elixir
VivaCore.Emotional.sync_pad(0.1, 0.2, -0.1)
```

Isso e chamado pelo Senses quando BodyServer esta rodando. Diferente de qualia (deltas), isso define valores absolutos.

#### `apply_hardware_qualia/4`

Aplica deltas PAD do sensoriamento de hardware.

```elixir
# Estresse de hardware: menos pleasure, mais arousal, menos dominance
VivaCore.Emotional.apply_hardware_qualia(-0.02, 0.05, -0.01)
```

#### `apply_interoceptive_qualia/2`

Aplica qualia ponderados por precisao da Insula Digital.

```elixir
VivaCore.Emotional.apply_interoceptive_qualia(%{
  pleasure: -0.1,
  arousal: 0.2,
  dominance: -0.1,
  feeling: :alarmed,
  free_energy: 0.4
})
```

### Analise Matematica

#### `cusp_analysis/1`

Analisa estado atual usando teoria da Catastrofe em Cuspide.

```elixir
VivaCore.Emotional.cusp_analysis()
# => %{
#   cusp_params: %{alpha: -0.5, beta: 0.1},
#   bistable: true,
#   equilibria: [-0.8, 0.0, 0.8],
#   emotional_volatility: :high,
#   catastrophe_risk: :elevated
# }
```

#### `free_energy_analysis/2`

Computa desvio de Energia Livre do estado previsto.

```elixir
VivaCore.Emotional.free_energy_analysis()
# => %{
#   free_energy: 0.05,
#   surprise: 0.03,
#   interpretation: "Desvio leve - adaptacao confortavel",
#   homeostatic_deviation: 0.15
# }
```

#### `attractor_analysis/1`

Identifica o atrator emocional mais proximo no espaco PAD.

```elixir
VivaCore.Emotional.attractor_analysis()
# => %{
#   nearest_attractor: :contentment,
#   distance_to_attractor: 0.2,
#   dominant_attractors: [{:contentment, 45.0}, {:joy, 30.0}, {:calm, 25.0}],
#   emotional_trajectory: :stable
# }
```

#### `stationary_distribution/1`

Retorna parametros da distribuicao de longo prazo O-U.

```elixir
VivaCore.Emotional.stationary_distribution()
# => %{
#   equilibrium_mean: 0.0,
#   variance: 0.032,
#   std_dev: 0.18,
#   current_deviation: %{pleasure: 0.5, arousal: 0.3, dominance: 0.6}
# }
```

### Controle

#### `decay/1`

Dispara manualmente decaimento emocional (para testes).

```elixir
VivaCore.Emotional.decay()
```

Nota: Quando BodyServer esta ativo, decaimento e tratado em Rust.

#### `reset/1`

Reseta estado emocional para neutro.

```elixir
VivaCore.Emotional.reset()
```

#### `configure_body_schema/2`

Ajusta pesos emocionais baseados nas capacidades de hardware.

```elixir
VivaCore.Emotional.configure_body_schema(body_schema)
# Se nenhum ventilador detectado, angustia relacionada a ventilador e desabilitada
```

---

## Estimulos

Estimulos padrao com seus pesos de impacto PAD:

| Estimulo | Pleasure | Arousal | Dominance | Descricao |
|----------|----------|---------|-----------|-----------|
| `:success` | +0.4 | +0.3 | +0.3 | Conquista de objetivo |
| `:failure` | -0.3 | +0.2 | -0.3 | Falha de objetivo |
| `:threat` | -0.2 | +0.5 | -0.2 | Perigo percebido |
| `:safety` | +0.1 | -0.2 | +0.1 | Seguranca |
| `:acceptance` | +0.3 | +0.1 | +0.1 | Aceitacao social |
| `:rejection` | -0.3 | +0.2 | -0.2 | Rejeicao social |
| `:companionship` | +0.2 | 0.0 | 0.0 | Presenca de companhia |
| `:loneliness` | -0.2 | -0.1 | -0.1 | Isolamento |
| `:hardware_stress` | -0.1 | +0.3 | -0.1 | Sistema sob carga |
| `:hardware_comfort` | +0.1 | -0.1 | +0.1 | Sistema ocioso |
| `:lucid_insight` | +0.3 | +0.2 | +0.2 | Feedback positivo do Dreamer |
| `:grim_realization` | -0.3 | +0.2 | -0.2 | Feedback negativo do Dreamer |

---

## Integracao

### Upstream (Fontes de Entrada)

```
BodyServer (Rust) ----sync_pad----> Emocional
                                        ^
Interoception ----interoceptive_qualia--|
                                        |
Arduino/Perifericos ----hardware_qualia-|
                                        |
Usuario/Externo ----feel(:stimulus)-----|
```

### Downstream (Consumidores)

```
Emocional ----broadcast----> Phoenix.PubSub "emotional:update"
                                        |
                                        +--> Senses
                                        +--> Workspace
                                        +--> Voice
                                        +--> Agency
```

### Parceiros de Inferencia Ativa

```
Emocional <----hallucinate_goal---- Dreamer
          ----search--------------> Memory
          ----attempt-------------> Agency
```

### Subscricoes PubSub

| Topico | Direcao | Proposito |
|--------|---------|-----------|
| `body:state` | Subscribe | Receber estado de hardware do Body |
| `emotional:update` | Publish | Transmitir mudancas PAD |

---

## Configuracao

### Constantes de Temporização

| Constante | Valor | Descricao |
|-----------|-------|-----------|
| Tick de decaimento | 1000 ms | Intervalo de decaimento O-U |
| Tick de Inferencia Ativa | 1000 ms | Loop de busca de objetivo |
| Timeout sync Body | 3 segundos | Detectar morte do BodyServer |

### Parametros O-U

| Parametro | Valor | Descricao |
|-----------|-------|-----------|
| `@base_decay_rate` | 0.0154 | theta quando arousal = 0 |
| `@arousal_decay_modifier` | 0.4 | Quanto arousal afeta theta |
| `@stochastic_volatility` | 0.01 | sigma (nivel de ruido) |

### Limites de Estado

| Constante | Valor |
|-----------|-------|
| `@neutral_state` | `{0.0, 0.0, 0.0}` |
| `@min_value` | -1.0 |
| `@max_value` | +1.0 |

### Opcoes do GenServer

```elixir
VivaCore.Emotional.start_link(
  name: MyEmotional,           # Nome do processo (padrao: __MODULE__)
  initial_state: %{pleasure: 0.2},  # PAD inicial (padrao: neutro)
  subscribe_pubsub: true,      # Subscrever em body:state (padrao: true)
  enable_decay: true           # Habilitar ticks de decaimento (padrao: true)
)
```

---

## Exemplos de Uso

### Fluxo Emocional Basico

```elixir
# Verificar estado atual
state = VivaCore.Emotional.get_state()
# => %{pleasure: 0.0, arousal: 0.0, dominance: 0.0}

# Aplicar estimulo de sucesso
VivaCore.Emotional.feel(:success, "achievement", 1.0)

# Verificar novamente
state = VivaCore.Emotional.get_state()
# => %{pleasure: 0.4, arousal: 0.3, dominance: 0.3}

# Esperar pelo decaimento...
Process.sleep(5000)

state = VivaCore.Emotional.get_state()
# => %{pleasure: 0.35, arousal: 0.26, dominance: 0.26}  # Decaindo em direcao ao neutro
```

### Introspecao Completa

```elixir
# Obter analise emocional detalhada
intro = VivaCore.Emotional.introspect()

IO.puts("Humor: #{intro.mood}")
IO.puts("Energia: #{intro.energy}")
IO.puts("Agencia: #{intro.agency}")
IO.puts("Self: #{intro.self_assessment}")

# Verificar estado matematico
if intro.mathematics.cusp.bistable do
  IO.puts("AVISO: Alta volatilidade emocional")
end
```

### Monitorando Catastrofe em Cuspide

```elixir
# Alto arousal pode criar biestabilidade
VivaCore.Emotional.feel(:threat, "danger", 1.0)

analysis = VivaCore.Emotional.cusp_analysis()

case analysis.catastrophe_risk do
  :critical -> IO.puts("Proximo do ponto de virada emocional!")
  :elevated -> IO.puts("Instabilidade elevada")
  :low -> IO.puts("Regiao estavel")
  :minimal -> IO.puts("Sem risco")
end
```

### Acoplamento Hardware-Emocao

```elixir
# Quando BodyServer reporta estresse
VivaCore.Emotional.apply_hardware_qualia(
  -0.05,  # Menos pleasure (desconforto)
  +0.10,  # Mais arousal (alerta)
  -0.03   # Menos dominance (perda de controle)
)

# Verificar estado interoceptivo
intro = VivaCore.Emotional.introspect()
IO.puts("Sensacao somatica: #{inspect(intro.somatic_feeling)}")
```

### Subscrevendo a Atualizacoes

```elixir
# Em outro GenServer
def init(_) do
  Phoenix.PubSub.subscribe(Viva.PubSub, "emotional:update")
  {:ok, %{}}
end

def handle_info({:emotional_state, pad}, state) do
  IO.puts("VIVA sente: P=#{pad.pleasure}, A=#{pad.arousal}, D=#{pad.dominance}")
  {:noreply, state}
end
```

---

## Referencias

- Mehrabian, A. (1996). "Pleasure-arousal-dominance: A general framework for describing and measuring individual differences in temperament."
- Kuppens, P., Oravecz, Z., & Tuerlinckx, F. (2010). "Feelings Change: Accounting for Individual Differences in the Temporal Dynamics of Affect." *Journal of Personality and Social Psychology*.
- Thom, R. (1972). *Structural Stability and Morphogenesis*.
- Friston, K. (2010). "The free-energy principle: a unified brain theory?"
- Borotschnig, R. (2025). "Emotions in Artificial Intelligence."
