# Referencia da API de Raciocinio Ultra
> *Grafo de Conhecimento & Inferencia Profunda*

**Ultra** e o "Motor de Raciocinio". Ele usa Redes Neurais de Grafos (GNN) para inferir links ausentes na memoria da VIVA e prever relacionamentos causais.

## Funcionalidades
- **Previsao de Link Zero-Shot**: Pode adivinhar `(Sujeito, Relacao, ?)` sem treinamento explicito naquele fato especifico.
- **Embedding Narrativo**: Converte texto em vetores semanticos compativeis com o Cortex Liquido.
- **CogGNN**: Rede Neural de Grafos Cognitiva para raciocinio emocional com integracao ao Global Workspace.
- **Protecao de Memoria EWC**: Elastic Weight Consolidation para prevenir esquecimento catastrofico.
- **Processamento Temporal Mamba-2**: Modelagem de sequencias em tempo linear para historico de memorias.
- **Fine-Tuning DoRA**: Adaptacao de baixo rank com decomposicao de pesos para embeddings emocionais.

---

## API Elixir (`VivaBridge.Ultra`)

### Funcoes Principais

#### `ping/0`
Verifica disponibilidade do servico.
```elixir
VivaBridge.Ultra.ping()
# Retorna: %{"status" => "pong", "loaded" => true}
```

#### `infer_relations/2`
Extrai/Infere relacionamentos de um texto.
```elixir
{:ok, relations} = VivaBridge.Ultra.infer_relations("Gabriel", "Ventilador")
# Retorna: [%{head: "Gabriel", relation: "repair", tail: "Ventilador"}]
```

#### `predict_links/3`
Preve a cauda de uma tripla.
```elixir
{:ok, predictions} = VivaBridge.Ultra.predict_links("VIVA", "sente", 10)
# Retorna: %{"triples" => [%{head: "VIVA", relation: "sente", tail: "Feliz", score: 0.95}, ...]}
```

#### `embed/1`
Obtem vetor de embedding para texto (384-dim MiniLM).
```elixir
{:ok, vector} = VivaBridge.Ultra.embed("Eu me sinto viva.")
# Retorna: {:ok, [0.123, -0.456, ...]} (384 dimensoes)
```

#### `find_path/3`
Encontra caminho de raciocinio multi-hop entre entidades.
```elixir
{:ok, path} = VivaBridge.Ultra.find_path("VIVA", "Gabriel", 3)
# Retorna: %{"path" => [%{head: "VIVA", relation: "conhece", tail: "Gabriel"}]}
```

#### `build_graph/1`
Atualiza o Grafo de Conhecimento com novas memorias.
```elixir
{:ok, stats} = VivaBridge.Ultra.build_graph(memories)
# Retorna: %{"stats" => %{nodes: 150, edges: 300}}
```

---

## CogGNN (Rede Neural de Grafos Cognitiva)

Arquitetura GNN de 3 camadas inspirada no NeuCFlow (arXiv:1905.13049) que modela consciencia como passagem de mensagens atraves de um grafo de conhecimento.

### Arquitetura
```
Camada 1 (Inconsciente): Fusao sensorial de fundo via GAT (4 cabecas)
Camada 2 (Consciente):   Raciocinio ativo com modulacao emocional (2 cabecas)
Camada 3 (Atencao):      Selecao de foco para broadcast do Global Workspace
```

A rede integra o estado emocional PAD em todas as representacoes de nos, permitindo que o contexto emocional module os padroes de atencao do grafo.

### `init_cog_gnn/2`
Inicializa a GNN Cognitiva para raciocinio de grafos emocionais.

**Parametros:**
- `in_dim` - Dimensao do embedding de entrada (padrao: 384 para MiniLM)
- `hidden_dim` - Dimensao da camada oculta (padrao: 64)

```elixir
{:ok, true} = VivaBridge.Ultra.init_cog_gnn()
# Com dimensoes customizadas:
{:ok, true} = VivaBridge.Ultra.init_cog_gnn(768, 128)
```

### `propagate/3`
Executa passagem de mensagens GNN com contexto emocional (estado PAD).

Propaga um pensamento atraves do grafo de conhecimento, usando o estado emocional PAD para modular a atencao. Retorna os nos mais atendidos representando o "foco consciente".

**Parametros:**
- `concept` - O conceito/pensamento a propagar (string)
- `pad` - Mapa do estado emocional PAD com chaves `:pleasure`, `:arousal`, `:dominance`
- `top_k` - Numero de nos mais atendidos a retornar (padrao: 5)

```elixir
{:ok, result} = VivaBridge.Ultra.propagate(
  "medo",
  %{pleasure: -0.3, arousal: 0.7, dominance: -0.2}
)
# Retorna:
# {:ok, %{
#   "attended_nodes" => ["mem_fear_1", "mem_anxiety_2"],
#   "attention_scores" => [0.85, 0.72],
#   "updated_concept" => "mem_fear_1"
# }}
```

### `propagate_query/3`
Propagacao condicionada por query atraves do grafo de conhecimento.

Combina atencao GNN com similaridade de query para recuperacao focada. Util quando buscando conceitos especificos em contexto emocional.

**Parametros:**
- `query` - String de consulta para encontrar nos relevantes
- `pad` - Mapa do estado emocional PAD
- `top_k` - Numero de resultados (padrao: 5)

```elixir
{:ok, result} = VivaBridge.Ultra.propagate_query(
  "o que me faz feliz?",
  %{pleasure: 0.2, arousal: 0.3, dominance: 0.1}
)
# Retorna:
# {:ok, %{
#   "query" => "o que me faz feliz?",
#   "results" => [
#     %{"entity" => "musica", "combined_score" => 0.89, "attention" => 0.75, "similarity" => 0.92},
#     ...
#   ]
# }}
```

### `conscious_focus/0`
Obtem o foco consciente atual da atencao GNN.

Retorna os nos com maior atencao da ultima propagacao, representando o "foco consciente" atual no Global Workspace.

```elixir
{:ok, focus} = VivaBridge.Ultra.conscious_focus()
# Retorna: {:ok, ["mem_fear_1", "mem_anxiety_2", "emotion_sad"]}
```

---

## EWC (Elastic Weight Consolidation)

Implementa protecao de memoria para prevenir esquecimento catastrofico durante aprendizado continuo (Kirkpatrick et al. 2017).

### Conceitos-Chave
- **Informacao de Fisher**: Mede a importancia de cada dimensao do embedding
- **Score de Consolidacao**: Quao importante e uma memoria (do scoring DRE do Dreamer)
- **Penalidade EWC**: `L_ewc = lambda/2 * SUM(F_i * (theta_i - theta*_i)^2)`

### Configuracao
| Parametro | Padrao | Descricao |
|-----------|--------|-----------|
| `lambda_ewc` | 0.4 | Forca da regularizacao |
| `min_consolidation_score` | 0.7 | Score DRE minimo para proteger |
| `max_protected_memories` | 1000 | Contagem maxima de memorias protegidas |
| `decay_rate` | 0.01 | Decaimento da info de Fisher por ciclo |

### `protect_memory/4`
Protege uma memoria consolidada com EWC.

Chamado pelo Dreamer apos consolidacao de memoria. Usa Informacao de Fisher para identificar dimensoes importantes do embedding e protege-las.

**Parametros:**
- `memory_id` - ID do ponto no Qdrant
- `embedding` - Embedding da memoria (lista de 384 floats)
- `related_embeddings` - Embeddings de memorias relacionadas (lista de listas)
- `consolidation_score` - Score DRE do Dreamer (0.0 - 1.0)

```elixir
{:ok, result} = VivaBridge.Ultra.protect_memory(
  "mem_abc123",
  embedding,          # [0.1, -0.2, ...] (384 dims)
  related_embeddings, # [[0.1, ...], [0.2, ...]]
  0.85                # Score de consolidacao alto
)
# Retorna se protegido:
# {:ok, %{
#   "protected" => true,
#   "qdrant_payload" => %{
#     "ewc_fisher_info" => [...],
#     "ewc_baseline_embedding" => [...],
#     "ewc_consolidation_score" => 0.85,
#     "ewc_consolidated_at" => 1705936800.0
#   }
# }}
# Retorna se nao protegido:
# {:ok, %{"protected" => false, "reason" => "score 0.50 < min 0.7"}}
```

### `ewc_penalty/2`
Calcula penalidade EWC para um embedding novo/modificado.

Usado para avaliar quanto um novo embedding afetaria memorias protegidas.

**Parametros:**
- `embedding` - O novo embedding a avaliar (lista de floats)
- `affected_memory_ids` - Memorias especificas a verificar (nil = todas)

```elixir
{:ok, result} = VivaBridge.Ultra.ewc_penalty(new_embedding)
# Retorna:
# {:ok, %{
#   "penalty" => 0.0234,
#   "details" => %{
#     "total_memories_checked" => 15,
#     "top_contributions" => [
#       %{"memory_id" => "mem_xyz", "penalty" => 0.012, "score" => 0.9},
#       ...
#     ]
#   }
# }}
```

### `ewc_stats/0`
Obtem estatisticas do gerenciador EWC.

```elixir
{:ok, stats} = VivaBridge.Ultra.ewc_stats()
# Retorna:
# {:ok, %{
#   "protected_count" => 42,
#   "avg_consolidation_score" => 0.82,
#   "max_consolidation_score" => 0.98,
#   "total_fisher_mean" => 0.45,
#   "lambda_ewc" => 0.4
# }}
```

### `ewc_decay/0`
Aplica decaimento de Fisher para permitir alguma plasticidade para memorias antigas.

Deve ser chamado periodicamente (ex: durante ciclos de sono/sonho).

```elixir
:ok = VivaBridge.Ultra.ewc_decay()
```

---

## Mamba-2 (Processamento Temporal de Memoria)

Implementa processamento de sequencias em tempo linear O(n) para historico de memoria usando Modelos de Espaco de Estados (SSM). Alternativa a atencao Transformer (O(n^2)).

### Beneficios Principais
- **Complexidade linear**: Pode processar 100+ memorias sem explosao de VRAM
- **Memoria implicita**: Estado oculto captura padroes temporais
- **Inferencia eficiente**: Passagem unica, sem cache KV

### Arquitetura
```
Embeddings de memoria [t-100:t] -> Mamba-2 -> contexto[60] -> Cortex
```

### Configuracao
| Parametro | Padrao | Descricao |
|-----------|--------|-----------|
| `d_model` | 384 | Dimensao de entrada (MiniLM) |
| `d_state` | 64 | Dimensao do estado SSM |
| `n_layers` | 2 | Numero de camadas Mamba |
| `output_dim` | 60 | Dimensao do vetor de contexto |
| `max_seq_len` | 128 | Comprimento maximo da sequencia |

### `init_mamba/3`
Inicializa o processador temporal Mamba.

**Parametros:**
- `d_model` - Dimensao do embedding de entrada (padrao: 384)
- `n_layers` - Numero de camadas Mamba (padrao: 2)
- `output_dim` - Dimensao do contexto de saida (padrao: 60)

```elixir
{:ok, %{"success" => true}} = VivaBridge.Ultra.init_mamba()
# Com config customizada:
{:ok, %{"success" => true}} = VivaBridge.Ultra.init_mamba(384, 4, 128)
```

### `process_sequence/2`
Processa uma sequencia de embeddings de memoria atraves do Mamba.

Recebe uma lista de embeddings de memoria e retorna um vetor de contexto compacto capturando padroes temporais.

**Parametros:**
- `embeddings` - Lista de vetores de embedding `[[e1], [e2], ...]`
- `timestamps` - Lista opcional de timestamps para ordenacao temporal

```elixir
embeddings = [
  [0.1, -0.2, ...],  # Memoria em t-2
  [0.3, 0.1, ...],   # Memoria em t-1
  [0.2, 0.0, ...]    # Memoria em t
]

{:ok, result} = VivaBridge.Ultra.process_sequence(embeddings)
# Retorna:
# {:ok, %{
#   "context" => [0.5, -0.1, ...],  # vetor de contexto 60-dim
#   "metadata" => %{
#     "seq_len" => 3,
#     "d_model" => 384,
#     "output_dim" => 60,
#     "has_timestamps" => false
#   }
# }}
```

### `mamba_stats/0`
Obtem estatisticas do processador Mamba.

```elixir
{:ok, stats} = VivaBridge.Ultra.mamba_stats()
# Retorna:
# {:ok, %{
#   "available" => true,
#   "d_model" => 384,
#   "n_layers" => 2,
#   "sequences_processed" => 150
# }}
```

**Nota:** Se `mamba-ssm` nao estiver instalado, um fallback usando media ponderada exponencial e usado automaticamente.

---

## DoRA (Fine-Tuning com Decomposicao de Pesos)

Implementa Adaptacao de Baixo Rank com Decomposicao de Pesos (DoRA) para fine-tuning do modelo de embedding MiniLM no espaco semantico emocional da VIVA (Liu et al., 2024).

### Conceitos-Chave
- **DoRA = LoRA + Decomposicao de Pesos**: Decompoe pesos em componentes de magnitude e direcao
- **Treinamento mais estavel** que LoRA vanilla
- **Melhor preservacao** de features pre-treinadas
- **~9% parametros treinaveis** (2M / 22M)

### Casos de Uso
- Adaptar embeddings MiniLM ao vocabulario emocional da VIVA
- Aprendizado contrastivo: emocoes similares -> embeddings similares

### Configuracao
| Parametro | Padrao | Descricao |
|-----------|--------|-----------|
| `model_name` | `all-MiniLM-L6-v2` | Modelo base |
| `r` | 8 | Rank do LoRA |
| `lora_alpha` | 16 | Fator de escala LoRA |
| `lora_dropout` | 0.1 | Dropout para camadas LoRA |
| `use_dora` | true | Habilitar decomposicao de pesos |
| `learning_rate` | 2e-4 | Taxa de aprendizado |
| `temperature` | 0.07 | Temperatura InfoNCE |

### `dora_setup/0`
Configura o fine-tuner DoRA e inicializa o modelo com adaptadores.

```elixir
{:ok, %{"success" => true}} = VivaBridge.Ultra.dora_setup()
```

### `dora_train/1`
Treina o modelo com amostras emocionais usando aprendizado contrastivo.

**Parametros:**
- `samples` - Lista de amostras de treinamento, cada uma com:
  - `text` - Texto de entrada
  - `pad` - Estado emocional PAD `[pleasure, arousal, dominance]`
  - `label` - Label categorico opcional

```elixir
samples = [
  %{text: "Estou tao feliz hoje!", pad: [0.8, 0.6, 0.4]},
  %{text: "Isso e frustrante", pad: [-0.5, 0.7, -0.3]},
  %{text: "Manha tranquila", pad: [0.4, -0.2, 0.3], label: "calmo"}
]

{:ok, result} = VivaBridge.Ultra.dora_train(samples)
# Retorna:
# {:ok, %{
#   "epochs" => 3,
#   "final_loss" => 0.234,
#   "best_loss" => 0.198,
#   "total_steps" => 150
# }}
```

### `dora_encode/1`
Codifica textos usando o modelo fine-tuned.

**Parametros:**
- `texts` - Lista de textos a codificar

```elixir
{:ok, result} = VivaBridge.Ultra.dora_encode(["Estou feliz", "Estou triste"])
# Retorna:
# {:ok, %{
#   "embeddings" => [
#     [0.12, -0.34, ...],  # 384 dims
#     [0.45, 0.23, ...]
#   ]
# }}
```

### `dora_save/1`
Salva pesos do adaptador DoRA em disco.

**Parametros:**
- `path` - Caminho do diretorio para salvar pesos

```elixir
{:ok, %{"success" => true}} = VivaBridge.Ultra.dora_save("/caminho/para/checkpoints")
```

### `dora_load/1`
Carrega pesos do adaptador DoRA do disco.

**Parametros:**
- `path` - Caminho do diretorio contendo pesos salvos

```elixir
{:ok, %{"success" => true}} = VivaBridge.Ultra.dora_load("/caminho/para/checkpoints")
```

### `dora_stats/0`
Obtem estatisticas do fine-tuning DoRA.

```elixir
{:ok, stats} = VivaBridge.Ultra.dora_stats()
# Retorna:
# {:ok, %{
#   "model" => "sentence-transformers/all-MiniLM-L6-v2",
#   "use_dora" => true,
#   "rank" => 8,
#   "alpha" => 16,
#   "training" => %{
#     "epochs_completed" => 3,
#     "total_steps" => 150,
#     "best_loss" => 0.198
#   },
#   "model_initialized" => true
# }}
```

---

## Dependencias

### Obrigatorias
- **Python 3.9+**
- **sentence-transformers** - Embeddings
- **torch** - Backend PyTorch
- **torch-geometric** - Redes neurais de grafos CogGNN
- **numpy** - Operacoes numericas

### Opcionais (Recursos Avancados)
| Pacote | Recurso | Instalacao |
|--------|---------|------------|
| `mamba-ssm>=2.0.0` | Processamento temporal Mamba-2 | `pip install mamba-ssm causal-conv1d>=1.2.0` |
| `peft>=0.10.0` | Fine-tuning DoRA | `pip install peft` |

**Nota:** Sem os pacotes opcionais, implementacoes fallback sao usadas automaticamente.

---

## Referencias

- **ULTRA**: arXiv:2310.04562 - Raciocinio em Grafos de Conhecimento
- **NeuCFlow**: arXiv:1905.13049 - Arquitetura de Circuitos Neurais
- **EWC**: Kirkpatrick et al. 2017 - Superando esquecimento catastrofico
- **Mamba-2**: Gu & Dao, 2024 - Modelagem de Sequencias em Tempo Linear
- **DoRA**: Liu et al., 2024 - Adaptacao de Baixo Rank com Decomposicao de Pesos
- **LoRA**: Hu et al., 2021 - Adaptacao de Baixo Rank para LLMs
