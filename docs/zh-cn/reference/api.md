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

## 3. VivaCore.Memory (存根)

*注：第 5 阶段进行中（Qdrant 集成待定）。*

#### `store/2`
持久化一次体验。

#### `recall/2`
对记忆进行语义搜索。
