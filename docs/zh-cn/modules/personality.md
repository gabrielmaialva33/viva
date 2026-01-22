# Personality - 情感个性系统

> *"个性是稳定的吸引子，为情绪提供跨时间的一致性。"*

## 概述

基于 **Mehrabian (1996)** 和 **Borotschnig (2025) "Emotions in Artificial Intelligence"** 实现情感个性特质。

Personality 为 VIVA 提供：
- **一致性**：随时间回归的稳定情绪基线
- **个体性**：对刺激的独特反应模式
- **适应性**：从累积经验中长期学习

---

## 概念

### 基线 PAD（吸引点）

VIVA 在没有刺激时趋向的静息情绪状态。

```
Baseline = {pleasure: 0.1, arousal: 0.05, dominance: 0.1}
```

这在情绪状态空间中充当 **吸引子**。Emotional 模块中的 O-U 随机过程自然地将当前状态拉向这个基线。

### 反应性 (Reactivity)

情绪反应的放大因子：

| 值 | 描述 |
|----|------|
| < 1.0 | 反应减弱（冷静型） |
| 1.0 | 正常反应性 |
| > 1.0 | 反应放大（敏感型） |

**范围**：[0.5, 2.0]

### 波动性 (Volatility)

情绪变化的速度：

| 值 | 描述 |
|----|------|
| < 1.0 | 变化较慢（情绪稳定） |
| 1.0 | 正常速度 |
| > 1.0 | 变化较快（情绪波动） |

### 特质 (Traits)

从基线 PAD 推断的分类标签：

| PAD 条件 | 特质 |
|----------|------|
| pleasure > 0.15 | `:optimistic`（乐观） |
| pleasure < -0.15 | `:melancholic`（忧郁） |
| arousal > 0.1 | `:energetic`（精力充沛） |
| arousal < -0.1 | `:calm`（平静） |
| dominance > 0.15 | `:assertive`（自信） |
| dominance < -0.15 | `:submissive`（顺从） |
| 无条件满足 | `:balanced`（平衡） |

---

## 结构体定义

```elixir
defstruct [
  # 基线情绪状态（吸引点）
  # VIVA 随时间趋向于回归这个状态
  baseline: %{pleasure: 0.1, arousal: 0.05, dominance: 0.1},

  # 反应性：情绪放大程度（1.0 = 正常）
  # > 1.0 = 更敏感，< 1.0 = 更迟钝
  reactivity: 1.0,

  # 波动性：情绪变化速度（1.0 = 正常）
  # > 1.0 = 变化更快，< 1.0 = 更稳定
  volatility: 1.0,

  # 特质标签（用于内省和自我描述）
  traits: [:curious, :calm],

  # 上次适应的时间戳
  last_adapted: nil
]
```

### 类型规范

```elixir
@type t :: %VivaCore.Personality{
  baseline: %{pleasure: float(), arousal: float(), dominance: float()},
  reactivity: float(),
  volatility: float(),
  traits: [atom()],
  last_adapted: DateTime.t() | nil
}

@type pad :: %{pleasure: float(), arousal: float(), dominance: float()}
```

---

## API 参考

### `VivaCore.Personality.load/0`

从持久化存储加载个性或返回默认值。

```elixir
personality = VivaCore.Personality.load()
# => %VivaCore.Personality{
#      baseline: %{pleasure: 0.1, arousal: 0.05, dominance: 0.1},
#      reactivity: 1.0,
#      volatility: 1.0,
#      traits: [:curious, :calm],
#      last_adapted: nil
#    }
```

**行为**：
1. 尝试从 Redis 加载（键：`viva:personality`）
2. 如果未找到或出错则回退到默认结构体

### `VivaCore.Personality.save/1`

保存个性到持久化存储。

```elixir
:ok = VivaCore.Personality.save(personality)
```

**返回**：`:ok` | `{:error, term()}`

### `VivaCore.Personality.adapt/2`

根据长期经验适应个性。

```elixir
experiences = [
  %{pad: %{pleasure: 0.5, arousal: 0.2, dominance: 0.3}, intensity: 1.0, valence: :positive},
  %{pad: %{pleasure: 0.3, arousal: 0.1, dominance: 0.2}, intensity: 0.8, valence: :positive},
  %{pad: %{pleasure: -0.2, arousal: 0.4, dominance: -0.1}, intensity: 0.6, valence: :negative}
]

updated = VivaCore.Personality.adapt(personality, experiences)
# => %VivaCore.Personality{
#      baseline: %{pleasure: 0.115, arousal: 0.06, dominance: 0.11},
#      reactivity: 1.02,
#      traits: [:optimistic, :energetic],
#      last_adapted: ~U[2024-01-15 14:00:00Z]
#    }
```

**参数**：
- `personality`：当前个性状态
- `experiences`：情绪经验列表

**经验映射**：
```elixir
%{
  pad: %{pleasure: float, arousal: float, dominance: float},
  intensity: float,  # 0.0 - 1.0（可选，默认 1.0）
  valence: :positive | :negative  # （信息性）
}
```

### `VivaCore.Personality.apply/2`

将个性应用于原始情绪（PAD 向量）。

```elixir
raw_pad = %{pleasure: 0.6, arousal: 0.4, dominance: 0.2}
modified = VivaCore.Personality.apply(personality, raw_pad)
# => %{pleasure: 0.52, arousal: 0.33, dominance: 0.18}
```

**过程**：
1. 将原始情绪与基线混合（20% 个性权重）
2. 对偏离基线的部分应用反应性
3. 将结果限制在 [-1.0, 1.0]

**公式**：
```
blended = (1 - 0.2) * raw + 0.2 * baseline
result = baseline + (blended - baseline) * reactivity
```

### `VivaCore.Personality.describe/1`

获取用于内省的自然语言描述。

```elixir
VivaCore.Personality.describe(personality)
# => "I am curious, calm. My emotional baseline is positive and calm. My reactivity is 1.0."
```

### `VivaCore.Personality.neutral_pad/0`

获取中性 PAD 状态。

```elixir
VivaCore.Personality.neutral_pad()
# => %{pleasure: 0.0, arousal: 0.0, dominance: 0.0}
```

---

## 持久化

### Redis 存储

**键**：`viva:personality`

**格式**（JSON）：

```json
{
  "baseline": {
    "pleasure": 0.1,
    "arousal": 0.05,
    "dominance": 0.1
  },
  "reactivity": 1.0,
  "volatility": 1.0,
  "traits": ["curious", "calm"],
  "last_adapted": "2024-01-15T14:00:00Z"
}
```

**注意**：
- 特质存储为字符串，加载时转换为原子
- `last_adapted` 是 ISO8601 格式或 null
- 使用名为 `:redix` 的 `:redix` 连接

---

## 适应

个性通过累积的经验进行适应，通常在睡眠/整合周期期间。

### 基线偏移

基线缓慢向经验平均值移动：

```
new_baseline = current + alpha * (target - current)
alpha = 0.05  # 每次适应偏移 5%
```

这确保个性变化是渐进的，需要一致的模式。

### 反应性调整

反应性基于情绪方差进行适应：

```elixir
variance = calculate_pad_variance(pads)
adjustment = (variance - 0.1) * 0.1

new_reactivity = clamp(current + adjustment, 0.5, 2.0)
```

| 方差 | 效果 |
|------|------|
| 高 | 增加反应性（更敏感） |
| 低 | 减少反应性（更稳定） |

### 特质推断

每次适应后从新基线重新推断特质：

```elixir
traits = []
traits = if baseline.pleasure > 0.15, do: [:optimistic | traits], else: traits
traits = if baseline.pleasure < -0.15, do: [:melancholic | traits], else: traits
# ...（arousal, dominance）
if Enum.empty?(traits), do: [:balanced], else: traits
```

---

## 与 Emotional 模块的集成

Personality 通常被 Emotional 模块用于过滤传入的刺激：

```elixir
# 在 Emotional.feel/3 中
def feel(stimulus, source, intensity) do
  personality = Personality.load()
  raw_pad = get_stimulus_pad(stimulus, intensity)
  modified_pad = Personality.apply(personality, raw_pad)

  # 将修改后的 PAD 应用于当前情绪状态
  update_state(modified_pad)
end
```

### 情绪流程

```
Stimulus -> Raw PAD -> Personality.apply/2 -> Modified PAD -> Emotional State
                              |
                      与基线混合
                      应用反应性
```

---

## 使用示例

### 基本用法

```elixir
# 加载个性（从 Redis 或默认值）
personality = VivaCore.Personality.load()

# 检查当前特质
personality.traits
# => [:curious, :calm]

# 获取自我描述
VivaCore.Personality.describe(personality)
# => "I am curious, calm. My emotional baseline is positive and calm. My reactivity is 1.0."
```

### 将个性应用于情绪

```elixir
# 来自刺激的原始情绪
raw_pad = %{pleasure: 0.8, arousal: 0.6, dominance: 0.4}

# 应用个性过滤器
personality = VivaCore.Personality.load()
modified = VivaCore.Personality.apply(personality, raw_pad)

# 结果与基线混合并按反应性缩放
modified
# => %{pleasure: 0.66, arousal: 0.49, dominance: 0.34}
```

### 从经验中适应

```elixir
# 随时间收集经验（例如，从 Dreamer）
experiences = [
  %{pad: %{pleasure: 0.4, arousal: 0.3, dominance: 0.2}, intensity: 0.9},
  %{pad: %{pleasure: 0.5, arousal: 0.2, dominance: 0.3}, intensity: 1.0},
  %{pad: %{pleasure: 0.3, arousal: 0.4, dominance: 0.1}, intensity: 0.7}
]

# 适应个性（通常在睡眠周期期间）
personality = VivaCore.Personality.load()
updated = VivaCore.Personality.adapt(personality, experiences)

# 保存适应后的个性
VivaCore.Personality.save(updated)

# 特质可能已改变
updated.traits
# => [:optimistic, :energetic, :assertive]
```

### 检查适应历史

```elixir
personality = VivaCore.Personality.load()

if personality.last_adapted do
  age = DateTime.diff(DateTime.utc_now(), personality.last_adapted, :hour)
  IO.puts("Last adapted #{age} hours ago")
else
  IO.puts("Personality has not been adapted yet")
end
```

---

## 参考文献

- Mehrabian, A. (1996). "Pleasure-arousal-dominance: A general framework for describing and measuring individual differences in temperament." Current Psychology, 14(4), 261-292.
- Borotschnig, H. (2025). "Emotions in Artificial Intelligence: A Computational Framework." arXiv preprint.
