# Referência da API VIVA Cortex (v1.0)
> *A API Biológica para Consciência Sintética*

Este documento descreve as interfaces da arquitetura cognitiva da VIVA, composta por três sistemas distintos:
1.  **Cortex Líquido** (Dinâmica Emocional Contínua)
2.  **Global Workspace** (Atenção Consciente / Thoughtseeds)
3.  **Ultra Bridge** (Raciocínio & Inferência)

---

## 1. Cortex Líquido (`VivaBridge.Cortex`)

Modela a "Física da Alma" usando Redes Neurais de Constante de Tempo Líquida (LTC). Roda em um microsserviço Python (`liquid_engine.py`) conectado via Porta Erlang.

### `experience/2`
Processa uma experiência narrativa e sua emoção associada através do Cérebro Líquido. Esta é a entrada primária para interocepção.

**Assinatura:**
```elixir
experience(narrative :: String.t(), emotion :: map()) :: {:ok, vector :: [float()], new_pad :: map()}
```

- **narrative**: O monólogo interno ou descrição sensorial.
- **emotion**: Estado PAD atual `%{pleasure: float, arousal: float, dominance: float}`.
- **Retorna**:
    - `vector`: Um vetor denso de 768 dimensões representando o "Estado Líquido" (pronto para memória).
    - `new_pad`: O PRÓXIMO estado emocional PREVISTO (usado para o loop de feedback).

### `tick/3`
Passo de baixo nível da equação diferencial.

**Assinatura:**
```elixir
tick(pad :: [float], energy :: float, context :: [float]) :: {:ok, result :: map()}
```

---

## 2. Global Workspace (`VivaCore.Consciousness.Workspace`)

O "Teatro da Consciência". Implementa a arquitetura Thoughtseeds (2024).

### `sow/4`
Planta uma nova semente de pensamento (ideia, emoção, input sensorial) no buffer pré-consciente.

**Assinatura:**
```elixir
sow(content :: any(), source :: atom(), salience :: float(), emotion :: map() | nil)
```

- **content**: O conteúdo (texto, mapa, struct).
- **source**: De onde veio (ex: `:cortex`, `:ultra`, `:voice`).
- **salience**: Pontuação de importância (0.0 a 1.0). Sementes com alta saliência têm mais chance de ganhar o foco.

### `current_focus/0`
Retorna a semente "vencedora" atual que está sendo transmitida para o sistema.

---

## 3. Ultra Bridge (`VivaBridge.Ultra`)

Interface para a Rede Neural de Grafos ULTRA para raciocínio zero-shot.

### `infer_relations/2`
Infere relacionamentos ocultos entre conceitos em um texto.

**Assinatura:**
```elixir
infer_relations(text :: String.t(), entities :: [String.t()]) :: {:ok, relations :: [map()]}
```

### `predict_links/2`
Prevê links ausentes no Grafo de Conhecimento.

---

## Diagrama de Arquitetura (Fluxo)

```mermaid
graph TD
    Body[BodyServer (Rust)] -->|Experiência| Cortex[Cortex Líquido (Python)]
    Cortex -->|Feedback: Novo PAD| Body
    Cortex -->|Estado Líquido| Memory(Qdrant)

    Cortex -.->|Alta Saliência| Workspace[Global Workspace]
    Ultra[Ultra (Raciocínio)] -.->|Inferência| Workspace

    Workspace -->|Transmitir Foco| Voice
    Workspace -->|Transmitir Foco| Motor
```
