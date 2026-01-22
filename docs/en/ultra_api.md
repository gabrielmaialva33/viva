# Ultra Reasoning API Reference
> *Knowledge Graph & Deep Inference*

**Ultra** is the "Reasoning Engine". It uses Graph Neural Networks (GNN) to infer missing links in VIVA's memory and predict causal relationships.

## Features
- **Zero-Shot Link Prediction**: Can guess `(Subject, Relation, ?)` without explicit training on that specific fact.
- **Narrative Embedding**: Converts text into semantic vectors compatible with the Liquid Cortex.

## Elixir API (`VivaBridge.Ultra`)

### `infer_relations/2`
Extract/Infer relations from text.
```elixir
{:ok, relations} = VivaBridge.Ultra.infer_relations("Gabriel fixed the fan.", ["Gabriel", "Fan"])
# Returns: [%{head: "Gabriel", relation: "repair", tail: "Fan"}]
```

### `predict_link/3`
Predict the tail of a triple.
```elixir
{:ok, predictions} = VivaBridge.Ultra.predict_link("VIVA", "feels", ?)
# Returns: ["Happy", "Curious", ...]
```

### `embed/1`
Get vector embedding for text (768-dim).
```elixir
{:ok, vector} = VivaBridge.Ultra.embed("I feel alive.")
```
