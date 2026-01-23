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

## 3. VivaCore.Memory

*Integração com banco vetorial Qdrant para memória semântica.*

#### `store/2`
Persiste uma experiência.

#### `recall/2`
Busca semântica por memórias.

---

## 4. VivaCore.World (Big Bounce)

Módulos cosmológicos implementando ciclos de morte/renascimento inspirados na Gravidade Quântica em Loop.

### `VivaCore.World.Observer`

Consciência navegando o labirinto.

#### `get_state/0`
Retorna o estado atual do mundo.

```elixir
@spec get_state() :: %{
  pos: {integer(), integer()},
  energy: float(),
  entropy: float(),
  bounce_count: integer(),
  seed: String.t()
}
```

#### `move/1`
Navega pelo labirinto.

```elixir
@spec move(direction :: :up | :down | :left | :right) :: :ok
```

#### `bounce_count/0`
Número de Big Bounces (mortes/renascimentos) experimentados.

#### `total_entropy/0`
Experiência acumulada através de todos os ciclos.

#### `prepare_for_bounce/0`
Força consolidação de memórias antes da morte.

---

### `VivaCore.World.Generator`

Geração determinística de mundos (O Arquiteto).

#### `generate/3`
Cria um novo labirinto a partir de uma seed criptográfica.

```elixir
@spec generate(seed :: String.t() | integer(), width :: integer(), height :: integer()) :: %Generator{}
```

**Tipos de Tile:**
- `0` = VOID (Abismo)
- `1` = WALL (Estrutura)
- `2` = PATH (Fluxo de Dados)
- `3` = CORE (Leviatã / Singularidade)
