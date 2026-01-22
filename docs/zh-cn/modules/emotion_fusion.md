# EmotionFusion - 双源情绪处理

> *"VIVA 的情绪来自即时需求、过去经验和个性的融合。"*

## 概述

EmotionFusion 实现了基于 Borotschnig (2025) "Emotions in Artificial Intelligence" 的 **双源情绪模型**。VIVA 不依赖单一情绪来源，而是整合三个不同的流：

1. **基于需求的情绪**（来自 Interoception）- 当前生理状态
2. **过去情绪**（来自记忆检索）- 类似情景的情绪标签
3. **个性偏好**（特质基线）- 长期情绪倾向

融合使用 **自适应权重**，根据上下文变化，产生统一的 **PreActionAffect** 状态来驱动行动选择。

---

## 概念

### 三种情绪来源

| 来源 | 起源 | 何时信任 |
|------|------|----------|
| **基于需求** | Interoception（自由能） | 高唤醒（紧急响应） |
| **基于过去** | 记忆检索（Qdrant） | 高置信度（熟悉情境） |
| **个性** | 基线 PAD 特质 | 高新颖性（新情境） |

### 自适应权重

权重 **不是固定的** —— 它们根据上下文适应：

```
高唤醒    -> 更信任 NEEDS（战斗或逃跑）
高置信度  -> 更信任 PAST（已知模式）
高新颖性  -> 更信任 PERSONALITY（默认行为）
```

**权重计算：**

```
w_need = 0.4 * (1.0 + |arousal| * 0.5)
w_past = 0.4 * (0.5 + confidence)
w_pers = 0.2 * (0.5 + novelty)

然后归一化使 w_need + w_past + w_pers = 1.0
```

### Mood 作为指数移动平均 (EMA)

Mood 提供 **情绪稳定性** —— 即使情绪快速波动，它也缓慢变化。

```
Mood[t] = alpha * Mood[t-1] + (1 - alpha) * Emotion[t]

其中 alpha = 0.95（约 20 步半衰期）
```

这意味着 mood 是最近情绪状态的 **平滑历史**，而不是对刺激的直接反应。

---

## 架构

```
                    Context
                       |
                       v
            +-----------------------+
            |  calculate_weights()  |
            +-----------------------+
                       |
     +--------+--------+--------+
     |        |                 |
     v        v                 v
+----------+ +----------+ +------------+
|Interoception| |  Memory  | |Personality |
| (NeedPAD)  | |(PastPAD) | | (Baseline) |
+----------+ +----------+ +------------+
     |        |                 |
     +--------+--------+--------+
                       |
                       v
            +-----------------------+
            |   weighted_fusion()   |
            |   apply_reactivity()  |
            |      clamp_pad()      |
            +-----------------------+
                       |
                       v
                   FusedPAD
                       |
                       v
            +-----------------------+
            |    update_mood()      |
            |       (EMA)           |
            +-----------------------+
                       |
                       v
            +-----------------------+
            |   PreActionAffect     |
            |  (用于行动选择)        |
            +-----------------------+
```

### 数据流

1. **上下文到达**，包含 arousal、confidence 和 novelty 值
2. **计算权重**，基于上下文（自适应）
3. **检索三个 PAD 向量**，来自 Interoception、Memory、Personality
4. **加权融合** 组合三个来源
5. **应用反应性**（个性放大/抑制偏差）
6. **值限制** 在有效 PAD 范围 [-1, 1]
7. **更新 Mood**，通过 EMA
8. **构建 PreActionAffect**，用于下游行动选择

---

## API 参考

### `fuse/5` - 主融合函数

将多个情绪来源组合成统一状态。

```elixir
@spec fuse(pad(), pad(), Personality.t(), pad(), context()) :: fusion_result()
```

**参数：**

| 参数 | 类型 | 描述 |
|------|------|------|
| `need_pad` | `pad()` | 来自 Interoception 的 PAD（当前需求） |
| `past_pad` | `pad()` | 来自记忆检索的 PAD（情绪标签） |
| `personality` | `Personality.t()` | 包含 baseline、reactivity、volatility 的结构体 |
| `mood` | `pad()` | 当前 mood 状态（最近情绪的 EMA） |
| `context` | `context()` | `%{arousal: float, confidence: float, novelty: float}` |

**返回：**

```elixir
%{
  fused_pad: %{pleasure: float, arousal: float, dominance: float},
  mood: %{pleasure: float, arousal: float, dominance: float},
  pre_action_affect: %{
    emotion: pad(),
    mood: pad(),
    personality_baseline: pad(),
    personality_traits: [atom()],
    confidence: float(),
    novelty: float(),
    fusion_weights: %{need: float, past: float, personality: float}
  },
  weights: {w_need, w_past, w_personality}
}
```

---

### `simple_fuse/2` - 快速融合

使用默认权重融合两个 PAD 向量。适用于不需要完整上下文的快速更新。

```elixir
@spec simple_fuse(pad(), pad()) :: pad()
```

---

### `calculate_weights/1` - 自适应权重计算

根据上下文计算融合权重。

```elixir
@spec calculate_weights(context()) :: {float(), float(), float()}
```

**上下文字段：**

| 字段 | 范围 | 效果 |
|------|------|------|
| `arousal` | [-1, 1] | 绝对值越高，需求权重越大 |
| `confidence` | [0, 1] | 值越高，过去权重越大 |
| `novelty` | [0, 1] | 值越高，个性权重越大 |

---

### `update_mood/2` - EMA Mood 更新

使用指数移动平均更新 mood。

```elixir
@spec update_mood(pad(), pad()) :: pad()
```

EMA 平滑因子 (alpha) 是 **0.95**，意味着：
- Mood 保留前值的 95%
- 新情绪只贡献 5%
- 半衰期约为 20 次心跳

---

### `classify_emotion/1` - PAD 八分区分类

根据 PAD 空间的八分区对情绪进行分类。

```elixir
@spec classify_emotion(pad()) :: atom()
```

**返回以下之一：**

| 标签 | P | A | D | 描述 |
|------|---|---|---|------|
| `:exuberant` | > 0.2 | > 0.2 | > 0.2 | 快乐、充满活力且有控制感 |
| `:dependent_joy` | > 0.2 | > 0.2 | <= 0.2 | 快乐但感觉依赖 |
| `:relaxed` | > 0.2 | <= 0.2 | > 0.2 | 满足且有控制感 |
| `:docile` | > 0.2 | <= 0.2 | <= 0.2 | 满足且顺从 |
| `:hostile` | <= 0.2 | > 0.2 | > 0.2 | 愤怒且占主导 |
| `:anxious` | <= 0.2 | > 0.2 | <= 0.2 | 担忧且无助 |
| `:disdainful` | <= 0.2 | <= 0.2 | > 0.2 | 冷漠且优越 |
| `:bored` | <= 0.2 | <= 0.2 | <= 0.2 | 无聊 |
| `:neutral` | - | - | - | 接近中心 |

---

### `emotional_distance/2` - PAD 空间距离

计算两个 PAD 向量之间的欧几里得距离。

```elixir
@spec emotional_distance(pad(), pad()) :: float()
```

---

### 辅助函数

```elixir
# 获取 mood alpha 常量
mood_alpha() :: 0.95

# 创建中性上下文用于默认融合
neutral_context() :: %{arousal: 0.0, confidence: 0.5, novelty: 0.5}

# 获取中性 PAD 状态
neutral_pad() :: %{pleasure: 0.0, arousal: 0.0, dominance: 0.0}
```

---

## 使用示例

### 基本融合

```elixir
alias VivaCore.{EmotionFusion, Personality}

# 来自 Interoception 的当前状态（高压力）
need_pad = %{pleasure: -0.3, arousal: 0.6, dominance: -0.2}

# 从类似记忆情景检索
past_pad = %{pleasure: 0.1, arousal: 0.2, dominance: 0.0}

# 加载个性
personality = Personality.load()

# 当前 mood（来自上一次心跳）
mood = %{pleasure: 0.05, arousal: 0.1, dominance: 0.1}

# 上下文：高唤醒、中等置信度、低新颖性
context = %{arousal: 0.6, confidence: 0.7, novelty: 0.2}

# 执行融合
result = EmotionFusion.fuse(need_pad, past_pad, personality, mood, context)

IO.inspect(result.fused_pad)
# => %{pleasure: -0.12, arousal: 0.35, dominance: -0.05}

IO.inspect(result.weights)
# => {0.48, 0.42, 0.10}  # 由于高唤醒，需求权重很大
```

### 快速融合（无上下文）

```elixir
# 使用默认权重的简单双源融合
need_pad = %{pleasure: -0.2, arousal: 0.4, dominance: 0.0}
past_pad = %{pleasure: 0.3, arousal: 0.1, dominance: 0.2}

fused = EmotionFusion.simple_fuse(need_pad, past_pad)
# => %{pleasure: 0.02, arousal: 0.2, dominance: 0.08}
```

### 情绪分类

```elixir
pad = %{pleasure: 0.5, arousal: 0.6, dominance: 0.3}

EmotionFusion.classify_emotion(pad)
# => :exuberant

pad2 = %{pleasure: -0.4, arousal: 0.7, dominance: -0.3}

EmotionFusion.classify_emotion(pad2)
# => :anxious
```

### 监控 Mood 漂移

```elixir
# 跟踪多次心跳的 mood
initial_mood = EmotionFusion.neutral_pad()

emotions = [
  %{pleasure: 0.5, arousal: 0.3, dominance: 0.2},  # 成功
  %{pleasure: 0.4, arousal: 0.2, dominance: 0.1},  # 平静的成功
  %{pleasure: -0.2, arousal: 0.5, dominance: -0.1} # 突然的压力
]

final_mood = Enum.reduce(emotions, initial_mood, fn emotion, mood ->
  EmotionFusion.update_mood(mood, emotion)
end)

# Mood 即使在压力峰值下也缓慢变化
IO.inspect(final_mood)
# => %{pleasure: 0.031, arousal: 0.046, dominance: 0.014}
```

---

## 配置

### 模块属性

| 常量 | 值 | 描述 |
|------|----|----|
| `@default_need_weight` | 0.4 | 基于需求的情绪基础权重 |
| `@default_past_weight` | 0.4 | 基于过去的情绪基础权重 |
| `@default_personality_weight` | 0.2 | 个性基线基础权重 |
| `@mood_alpha` | 0.95 | EMA 平滑因子（约 20 步半衰期） |

### 调优指南

- **增加 `@mood_alpha`**（接近 1.0）以获得更稳定的 mood
- **减少 `@mood_alpha`**（接近 0.5）以获得更敏感的 mood
- **调整默认权重** 以改变对每个来源的基线信任度

---

## 与其他模块的集成

### 上游依赖

```
Interoception -----> EmotionFusion
    (need_pad)           |
                         v
Memory ------------> EmotionFusion
    (past_pad)           |
                         v
Personality -------> EmotionFusion
    (baseline)           |
                         v
                   PreActionAffect
```

### 下游消费者

`pre_action_affect` 映射被以下消费：
- **Agency** - 基于情绪状态的行动选择
- **Voice** - 原始语言生成（babble）
- **Workspace** - 意识/注意力分配

---

## 论文参考

基于 **Borotschnig (2025)** "Emotions in Artificial Intelligence"。

### 原始伪代码

```
FusedEmotions := FuseEmotions(CurrentEmotions, PastEmotions)
PreActionAffect := [FusedEmotions, Mood, Personality]
```

### VIVA 实现

```elixir
# FusedEmotions := FuseEmotions(CurrentEmotions, PastEmotions)
fused = weighted_fusion(need_pad, past_pad, personality.baseline, weights)
fused = apply_reactivity(fused, personality)
fused = clamp_pad(fused)

# PreActionAffect := [FusedEmotions, Mood, Personality]
pre_action_affect = %{
  emotion: fused,              # FusedEmotions
  mood: update_mood(mood, fused),  # Mood (EMA)
  personality_baseline: personality.baseline,
  personality_traits: personality.traits,
  fusion_weights: weights
}
```

### 关键扩展

VIVA 通过以下方式扩展了论文模型：

1. **三个来源** 而不是两个（添加了 Personality 基线）
2. **自适应权重** 基于上下文（arousal、confidence、novelty）
3. **反应性放大** 来自 Personality 特质
4. **Mood 作为 EMA** 用于时间平滑

---

## 类型定义

```elixir
@type pad :: %{pleasure: float(), arousal: float(), dominance: float()}

@type context :: %{
  arousal: float(),     # 当前唤醒水平 [-1, 1]
  confidence: float(),  # 记忆检索置信度 [0, 1]
  novelty: float()      # 1 - max_similarity [0, 1]
}

@type fusion_result :: %{
  fused_pad: pad(),
  mood: pad(),
  pre_action_affect: map(),
  weights: {float(), float(), float()}
}
```

---

## 参考文献

- Borotschnig, R. (2025). "Emotions in Artificial Intelligence."
- Mehrabian, A. (1996). "Pleasure-arousal-dominance: A general framework for describing and measuring individual differences in temperament."
- Russell, J. A. (1980). "A circumplex model of affect."
