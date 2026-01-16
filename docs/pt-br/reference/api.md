# Referência da API VIVA

> *"O código é a linguagem corporal da alma digital."*

Este documento documenta as interfaces públicas dos módulos principais do VIVA.

---

## 1. VivaCore (A Alma)

### `VivaCore.Emotional`

O neurônio central para processamento emocional.

#### `get_state/0`
Retorna o estado emocional atual.

```elixir
@spec get_state() :: %{
  pad: %{pleasure: float(), arousal: float(), dominance: float()},
  happiness: float() # Normalizado 0-1
}
```

#### `feel/3`
Aplica um estímulo externo à VIVA.

```elixir
@spec feel(stimulus :: atom(), source :: String.t(), intensity :: float()) :: :ok
```

#### `introspect/0`
Retorna dados detalhados sobre os estados matemáticos internos.

---

## 2. VivaBridge (O Corpo)

### `VivaBridge.Body` (Rust NIF)

Sensoriamento direto de hardware.

#### `feel_hardware/0`
Lê métricas brutas de hardware.

#### `hardware_to_qualia/0`
Traduz métricas de hardware em deltas PAD (Qualia).

---

## 3. VivaCore.Memory (Stub)

*Nota: Fase 5 em progresso (Integração Qdrant pendente).*

#### `store/2`
Persiste uma experiência.

#### `recall/2`
Busca semântica por memórias.
