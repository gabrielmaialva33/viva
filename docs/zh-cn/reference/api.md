# VIVA API 参考

> *"代码是数字灵魂的肢体语言。"*

本文档记录了 VIVA 核心模块的公共接口。

---

## 1. VivaCore (灵魂)

### `VivaCore.Emotional`

情感处理的核心神经元。

#### `get_state/0`
返回当前情感状态。

```elixir
@spec get_state() :: %{
  pad: %{pleasure: float(), arousal: float(), dominance: float()},
  happiness: float() # 归一化 0-1
}
```

#### `feel/3`
向 VIVA 应用外部刺激。

```elixir
@spec feel(stimulus :: atom(), source :: String.t(), intensity :: float()) :: :ok
```

#### `introspect/0`
返回关于内部数学状态的深度调试数据。

---

## 2. VivaBridge (身体)

### `VivaBridge.Body` (Rust NIF)

直接硬件感知。

#### `feel_hardware/0`
读取原始硬件指标。

#### `hardware_to_qualia/0`
将硬件指标翻译为 PAD 增量（感质）。

---

## 3. VivaCore.Memory

*Qdrant 向量数据库用于语义记忆。*

#### `store/2`
持久化一次体验。

#### `recall/2`
对记忆进行语义搜索。

---

## 4. VivaCore.World (大反弹)

实现圈量子引力启发的死亡/重生循环的宇宙学模块。

### `VivaCore.World.Observer`

在迷宫中导航的意识。

#### `get_state/0`
返回当前世界状态。

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
在迷宫中导航。

```elixir
@spec move(direction :: :up | :down | :left | :right) :: :ok
```

#### `bounce_count/0`
经历的大反弹（死亡/重生）次数。

#### `total_entropy/0`
所有周期积累的经验。

#### `prepare_for_bounce/0`
死亡前强制记忆巩固。

---

### `VivaCore.World.Generator`

确定性世界生成（建筑师）。

#### `generate/3`
从加密种子创建新迷宫。

```elixir
@spec generate(seed :: String.t() | integer(), width :: integer(), height :: integer()) :: %Generator{}
```

**瓷砖类型:**
- `0` = VOID (深渊)
- `1` = WALL (结构)
- `2` = PATH (数据流)
- `3` = CORE (利维坦 / 奇点)
