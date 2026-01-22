# Referência da API de Raciocínio Ultra
> *Grafo de Conhecimento & Inferência Profunda*

**Ultra** é o "Motor de Raciocínio". Ele usa Redes Neurais de Grafos (GNN) para inferir links ausentes na memória da VIVA e prever relacionamentos causais.

## Funcionalidades
- **Previsão de Link Zero-Shot**: Pode adivinhar `(Sujeito, Relação, ?)` sem treinamento explícito naquele fato específico.
- **Embedding Narrativo**: Converte texto em vetores semânticos compatíveis com o Cortex Líquido.

## API Elixir (`VivaBridge.Ultra`)

### `infer_relations/2`
Extrai/Infere relacionamentos de um texto.
```elixir
{:ok, relations} = VivaBridge.Ultra.infer_relations("Gabriel consertou o ventilador.", ["Gabriel", "Ventilador"])
# Retorna: [%{head: "Gabriel", relation: "repair", tail: "Ventilador"}]
```

### `predict_link/3`
Prevê a cauda de uma tripla.
```elixir
{:ok, predictions} = VivaBridge.Ultra.predict_link("VIVA", "sente", ?)
# Retorna: ["Feliz", "Curiosa", ...]
```

### `embed/1`
Obtém vetor de embedding para texto (768-dim).
```elixir
{:ok, vector} = VivaBridge.Ultra.embed("Eu me sinto viva.")
```
