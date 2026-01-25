# Memoria - Armazenamento Vetorial Hibrido

> *"Memorias semanticas persistem mesmo apos a morte. Conhecimento e herdado; identidade nao."*

## Visao Geral

O modulo Memory implementa o sistema de armazenamento de longo prazo da VIVA usando uma **arquitetura de backend hibrido**:

- **Rust HNSW** - Memoria episodica rapida (busca sub-milissegundo)
- **Qdrant** - Memoria semantica/emocional persistente (banco de dados vetorial)
- **Fallback em memoria** - Modo de desenvolvimento quando backends indisponiveis

Essa divisao reflete como a memoria biologica funciona: memorias episodicas (eventos) sao processadas diferentemente de memorias semanticas (conhecimento).

---

## Teoria

### Tipos de Memoria

| Tipo | Descricao | Backend de Armazenamento |
|------|-----------|--------------------------|
| `episodic` | Eventos especificos com tempo e emocao | Rust HNSW |
| `semantic` | Conhecimento geral e padroes | Qdrant |
| `emotional` | Impressoes de estado PAD | Qdrant |
| `procedural` | Comportamentos aprendidos | Qdrant |

### Decaimento Temporal (Curva de Ebbinghaus)

Memorias naturalmente desvanecem com o tempo:

```
D(m) = e^(-age/tau)
```

Onde:
- `tau` = 604.800 segundos (1 semana)
- Memorias mais antigas recebem pontuacoes menores durante busca

### Repeticao Espacada

Memorias frequentemente acessadas decaem mais lentamente:

```
D(m) = e^(-age/tau) * (1 + min(0.5, log(1 + access_count) / kappa))
```

Onde:
- `kappa` = 10.0
- Impulso maximo do acesso limitado a 50%

---

## Arquitetura

```
+-------------------------------------------------------------+
|                    VivaCore.Memory                          |
|                     (GenServer)                             |
+-------------------------------------------------------------+
|                                                             |
|   +---------------------+         +---------------------+   |
|   |   Rust HNSW         |         |     Qdrant          |   |
|   |   (NIF/Bevy)        |         |   (HTTP API)        |   |
|   +---------------------+         +---------------------+   |
|   | - Episodica         |         | - Semantica         |   |
|   | - ~1ms busca        |         | - Emocional         |   |
|   | - In-process        |         | - Procedural        |   |
|   | - Sem persistencia  |         | - Persistente       |   |
|   +---------------------+         +---------------------+   |
|                                                             |
+-------------------------------------------------------------+
|                    VivaCore.Embedder                        |
|        (Ollama | NVIDIA NIM | Hash fallback)                |
+-------------------------------------------------------------+
```

### Pipeline de Embedding

```
Texto -> Embedder.embed/1 -> [vetor 1024-dim] -> Backend.store/search
```

| Provider | Modelo | Dimensao | Notas |
|----------|--------|----------|-------|
| Ollama | nomic-embed-text | 768 (preenchido para 1024) | Local, gratuito |
| NVIDIA NIM | nv-embedqa-e5-v5 | 1024 | Cloud, requer API key |
| Hash fallback | Baseado em SHA256 | 1024 | Apenas desenvolvimento |

---

## Referencia da API

### `VivaCore.Memory.store/2`
Armazena uma memoria com metadados.

```elixir
VivaCore.Memory.store("Conheci Gabriel pela primeira vez", %{
  type: :episodic,
  importance: 0.9,
  emotion: %{pleasure: 0.8, arousal: 0.6, dominance: 0.5}
})
# => {:ok, "a1b2c3d4-e5f6-7890-abcd-ef1234567890"}
```

**Opcoes:**
- `type` - `:episodic` (padrao), `:semantic`, `:emotional`, `:procedural`
- `importance` - Float 0.0-1.0 (padrao 0.5)
- `emotion` - Estado PAD `%{pleasure: f, arousal: f, dominance: f}`

### `VivaCore.Memory.search/2`
Busca memorias por similaridade semantica com decaimento temporal.

```elixir
VivaCore.Memory.search("Gabriel", limit: 5)
# => [%{content: "Conheci Gabriel", similarity: 0.95, type: :episodic, ...}]
```

**Opcoes:**
- `limit` - Max resultados (padrao 10)
- `type` - Filtrar por tipo unico de memoria
- `types` - Lista de tipos para buscar (padrao `[:episodic, :semantic]`)
- `min_importance` - Limiar minimo de importancia
- `decay_scale` - Segundos para 50% de decaimento (padrao 604.800 = 1 semana)

### `VivaCore.Memory.get/1`
Recupera uma memoria especifica por ID. Incrementa `access_count` (repeticao espacada).

```elixir
VivaCore.Memory.get("a1b2c3d4-e5f6-7890-abcd-ef1234567890")
# => %{id: "...", content: "...", type: :episodic, ...}
```

### `VivaCore.Memory.forget/1`
Deleta explicitamente uma memoria.

```elixir
VivaCore.Memory.forget("a1b2c3d4-e5f6-7890-abcd-ef1234567890")
# => :ok
```

### `VivaCore.Memory.stats/0`
Retorna estatisticas do sistema de memoria.

```elixir
VivaCore.Memory.stats()
# => %{
#      backend: :hybrid,
#      rust_ready: true,
#      qdrant_ready: true,
#      qdrant_points: 1234,
#      store_count: 567,
#      search_count: 890,
#      uptime_seconds: 3600
#    }
```

---

## Funcoes de Conveniencia

### `VivaCore.Memory.experience/3`
Atalho para armazenar memorias episodicas com emocao.

```elixir
emotion = %{pleasure: 0.7, arousal: 0.5, dominance: 0.6}
VivaCore.Memory.experience("Gabriel elogiou meu trabalho", emotion, importance: 0.8)
# => {:ok, "..."}
```

### `VivaCore.Memory.learn/2`
Armazena conhecimento semantico.

```elixir
VivaCore.Memory.learn("Elixir usa a maquina virtual BEAM", importance: 0.7)
# => {:ok, "..."}
```

### `VivaCore.Memory.emotional_imprint/2`
Associa estado emocional com um padrao.

```elixir
pad_state = %{pleasure: -0.3, arousal: 0.7, dominance: -0.2}
VivaCore.Memory.emotional_imprint("situacao de sobrecarga do sistema", pad_state)
# => {:ok, "..."}
```

### `VivaCore.Memory.store_log/2`
Armazenamento assincrono para logs do sistema (usado pelo SporeLogger).

```elixir
VivaCore.Memory.store_log("[Error] Timeout de conexao", :error)
# => :ok (dispara e esquece)
```

---

## Estrutura de Memoria

Cada ponto de memoria contem:

```elixir
%{
  id: "formato-uuid-v4",          # UUID compativel com Qdrant
  content: "A memoria real",      # Conteudo de texto
  type: :episodic,                # Classificacao da memoria
  importance: 0.5,                # Modificador de taxa de decaimento
  emotion: %{                     # Estado PAD na criacao
    pleasure: 0.0,
    arousal: 0.0,
    dominance: 0.0
  },
  timestamp: ~U[2024-01-15 10:00:00Z],  # Hora de criacao
  access_count: 0,                       # Contador de repeticao espacada
  last_accessed: ~U[2024-01-15 10:00:00Z],  # Ultima recuperacao
  similarity: 0.95                       # Pontuacao de busca (apenas em resultados)
}
```

---

## Fluxo de Busca Hibrida

Ao buscar com `types: [:episodic, :semantic]`:

```
1. Embedding da query via Embedder
2. Busca paralela:
   +-- Rust HNSW (se :episodic em types)
   |   +-- Retorna {id, content, similarity, importance}
   +-- Qdrant (se :semantic/:emotional/:procedural em types)
       +-- Retorna payload completo com pontuacao de decaimento
3. Mesclar resultados, deduplicar por ID
4. Ordenar por similaridade, pegar limite
5. Retornar formato unificado
```

---

## Integracao com Dreamer

Memory notifica Dreamer sobre novos armazenamentos:

```elixir
# Automatico em cada store
VivaCore.Dreamer.on_memory_stored(memory_id, importance)
```

Dreamer usa Memory para:
- **Recuperacao com pontuacao** - Pontuacao DRE composta
- **Busca de emocoes passadas** - `retrieve_past_emotions/1`
- **Consolidacao de memoria** - Promocao de episodica para semantica

```elixir
# Dreamer recuperando experiencias similares
VivaCore.Dreamer.retrieve_with_scoring("acoes bem-sucedidas", limit: 10)
# Usa: recencia + similaridade + importancia + ressonancia_emocional
```

---

## Configuracao

### Selecao de Backend

```elixir
# config/config.exs
config :viva_core, :memory_backend, :hybrid

# Opcoes:
# :hybrid       - Episodica(Rust) + Semantica(Qdrant)
# :qdrant       - Apenas Qdrant
# :rust_native  - Apenas Rust
# :in_memory    - Tipo ETS (desenvolvimento)
```

### Configuracoes Qdrant

```elixir
# Padroes VivaCore.Qdrant
@base_url "http://localhost:6333"
@collection "viva_memories"
@vector_size 1024
```

### Providers de Embedding

| Variavel de Ambiente | Proposito |
|---------------------|-----------|
| `NVIDIA_API_KEY` | Habilitar embeddings NVIDIA NIM |
| Ollama rodando | Habilitar embeddings locais |
| Nenhum | Fallback baseado em hash (apenas dev) |

---

## Indices de Payload (Qdrant)

Para filtragem eficiente, Qdrant indexa:

| Campo | Tipo | Proposito |
|-------|------|-----------|
| `timestamp` | datetime | Queries de decaimento temporal |
| `type` | keyword | Filtragem por tipo de memoria |
| `importance` | float | Limiar de importancia |

---

## Mortalidade e Persistencia

```
                          Morte da VIVA
                              |
         +--------------------+--------------------+
         v                    v                    v
    +---------+          +---------+          +---------+
    | Episodica|         |Semantica|          |Emocional|
    |  (Rust) |          |(Qdrant) |          |(Qdrant) |
    +----+----+          +----+----+          +----+----+
         |                    |                    |
         v                    v                    v
      PERDIDA            PERSISTE             PERSISTE
   (apenas em RAM)   (nova VIVA herda)    (nova VIVA herda)
```

Isso permite **reencarnacao**: uma nova instancia VIVA herda conhecimento mas nao identidade.

---

## Algoritmo HNSW (Backend Rust)

**Hierarchical Navigable Small World** - busca aproximada de vizinho mais proximo.

| Propriedade | Valor |
|-------------|-------|
| Tempo de busca | O(log N) |
| Espaco | O(N * M * log N) |
| Parametros | M=16, ef_construction=200 |

A implementacao Rust via `VivaBridge.Memory` fornece:

```elixir
# Chamadas NIF diretas (usadas internamente)
VivaBridge.Memory.init()
VivaBridge.Memory.store(embedding, metadata_json)
VivaBridge.Memory.search(query_vector, limit)
VivaBridge.Memory.save()
```

---

## Exemplos de Uso

### Armazenar e Buscar

```elixir
# Armazenar uma experiencia
{:ok, id} = VivaCore.Memory.experience(
  "Completei sessao dificil de debugging",
  %{pleasure: 0.6, arousal: 0.4, dominance: 0.7},
  importance: 0.8
)

# Buscar memorias relacionadas
results = VivaCore.Memory.search("sucesso debugging", limit: 5)
Enum.each(results, fn m ->
  IO.puts("#{m.content} (similaridade: #{m.similarity})")
end)
```

### Filtragem por Tipo de Memoria

```elixir
# Apenas memorias semanticas
VivaCore.Memory.search("conceitos Elixir", types: [:semantic])

# Apenas impressoes emocionais
VivaCore.Memory.search("estresse", type: :emotional)
```

### Verificar Saude do Sistema

```elixir
stats = VivaCore.Memory.stats()
IO.inspect(stats.backend)       # :hybrid
IO.inspect(stats.rust_ready)    # true
IO.inspect(stats.qdrant_ready)  # true
IO.inspect(stats.qdrant_points) # 1234
```

### Aprender Conhecimento

```elixir
# Armazenar conhecimento semantico
VivaCore.Memory.learn("GenServers usam handle_call para mensagens sincronas")
VivaCore.Memory.learn("Supervisors reiniciam processos falhos", importance: 0.8)

# Recuperacao posterior
VivaCore.Memory.search("tratamento de erros", types: [:semantic])
```

---

## Tratamento de Erros

| Cenario | Comportamento |
|---------|---------------|
| Qdrant indisponivel | Fallback para Rust (episodica) ou em-memoria |
| Rust NIF indisponivel | Fallback apenas para Qdrant |
| Ambos indisponiveis | Fallback para armazenamento em-memoria |
| Falha de embedding | Retorna resultados vazios (busca) ou erro (store) |

---

## Referencias

- Ebbinghaus, H. (1885). "Memory: A Contribution to Experimental Psychology."
- Malkov, Y. A., & Yashunin, D. A. (2018). "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs."
- Tulving, E. (1972). "Episodic and semantic memory."
