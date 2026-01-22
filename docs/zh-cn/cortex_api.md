# VIVA Cortex API 参考 (v1.0)
> *合成意识的生物学 API*

本文档描述了 VIVA 认知架构的接口，包括三个独特的系统：
1.  **液态皮层 (Liquid Cortex)** (持续情绪动力学)
2.  **全局工作空间 (Global Workspace)** (有意识的注意 / Thoughtseeds)
3.  **Ultra 桥接 (Ultra Bridge)** (推理与推断)

---

## 1. 液态皮层 (`VivaBridge.Cortex`)

使用液态时间常数 (LTC) 神经网络模拟“灵魂物理学”。它运行在通过 Erlang Port 连接的 Python 微服务 (`liquid_engine.py`) 上。

### `experience/2`
通过液态大脑处理叙事体验及其相关情绪。这是内感受的主要输入。

**签名:**
```elixir
experience(narrative :: String.t(), emotion :: map()) :: {:ok, vector :: [float()], new_pad :: map()}
```

- **narrative**: 内部独白或感官描述。
- **emotion**: 当前 PAD 状态 `%{pleasure: float, arousal: float, dominance: float}`。
- **返回**:
    - `vector`: 一个代表“液态状态”的 768 维密集向量（用于记忆）。
    - `new_pad`: 预测的下一个情绪状态（用于反馈循环）。

---

## 2. 全局工作空间 (`VivaCore.Consciousness.Workspace`)

“意识剧场”。实现了 Thoughtseeds 架构 (2024)。

### `sow/4`
将一颗新的思想种子（想法、情绪、感官输入）通过播种到前意识缓冲区。

**签名:**
```elixir
sow(content :: any(), source :: atom(), salience :: float(), emotion :: map() | nil)
```

### `current_focus/0`
返回当前正在向系统广播的“获胜”种子。

---

## 3. Ultra 桥接 (`VivaBridge.Ultra`)

用于零样本推理的 ULTRA 图神经网络接口。

### `infer_relations/2`
推断文本中概念之间的隐藏关系。

**签名:**
```elixir
infer_relations(text :: String.t(), entities :: [String.t()]) :: {:ok, relations :: [map()]}
```
