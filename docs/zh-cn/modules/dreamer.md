# 梦想者 - 记忆巩固

> *"梦不是噪音 - 它们是灵魂在重组自己。"*

## 理论

实现来自 **Park et al. (2023) "Generative Agents"** 的反思机制，适配VIVA的情感架构。

在反思中，分散的经验变成连贯的意义。

---

## 数学基础

### 检索评分（DRE - Dream Retrieval Engine）

```
S(m, q) = w_r · D(m) + w_s · Sim(m, q) + w_i · I(m) + w_e · E(m)
```

| 组件 | 权重 | 描述 |
|------|------|------|
| **D(m)** | 0.2 | 带间隔重复的新近度 |
| **Sim(m, q)** | 0.4 | 语义相似度 |
| **I(m)** | 0.2 | 记忆重要性 |
| **E(m)** | 0.2 | 情感共鸣 |

### 衰减函数（带间隔重复）

```
D(m) = e^(-年龄/τ) × (1 + min(0.5, log(1 + access_count) / κ))
```

其中：
- τ = 604,800秒（1周）
- κ = 10.0（重复提升除数）
- 访问带来的最大提升限制在50%

### 情感共鸣

```
E(m) = max(0, 1 - ||PAD_m - PAD_当前|| / √12)
```

PAD空间中的距离归一化到[0, 1]。

---

## API参考

### `VivaCore.Dreamer.status/0`
获取当前Dreamer统计。

```elixir
VivaCore.Dreamer.status()
# => %{
#      importance_accumulator: 8.5,
#      threshold: 15.0,
#      progress_percent: 56.7,
#      last_reflection: ~U[2024-01-15 14:00:00Z],
#      reflection_count: 42,
#      thoughts_count: 156,
#      ...
#    }
```

### `VivaCore.Dreamer.reflect_now/0`
强制立即反思周期。

```elixir
VivaCore.Dreamer.reflect_now()
# => %{
#      focal_points: [%{question: "我学到了什么关于...", ...}],
#      insights: [%{insight: "反思...", depth: 1, ...}],
#      trigger: :manual
#    }
```

### `VivaCore.Dreamer.sleep_cycle/0`
启动深度反思（多次迭代 + 元反思）。

```elixir
{:ok, ref} = VivaCore.Dreamer.sleep_cycle()
# 异步运行，完成时更新状态
```

### `VivaCore.Dreamer.recent_thoughts/1`
获取最近的反思。

```elixir
VivaCore.Dreamer.recent_thoughts(5)
# => [
#      %{insight: "...", depth: 1, importance: 0.7, ...},
#      ...
#    ]
```

### `VivaCore.Dreamer.retrieve_with_scoring/2`
使用完整复合评分检索记忆。

```elixir
VivaCore.Dreamer.retrieve_with_scoring("成功的行动", limit: 10)
# => [%{content: "...", composite_score: 0.85, ...}, ...]
```

### `VivaCore.Dreamer.hallucinate_goal/1`
主动推理：生成目标PAD状态（VIVA*想要*达到的地方）。

```elixir
context = %{pleasure: -0.1, arousal: 0.2, dominance: 0.0}
VivaCore.Dreamer.hallucinate_goal(context)
# => %{pleasure: 0.2, arousal: 0.15, dominance: 0.1}
```

### `VivaCore.Dreamer.on_memory_stored/2`
通知Dreamer新记忆（由Memory模块调用）。

```elixir
VivaCore.Dreamer.on_memory_stored("mem_12345", 0.8)
# => :ok（累积重要性，可能触发反思）
```

---

## 反思触发器

当满足任一条件时触发反思：

| 触发器 | 阈值 | 描述 |
|--------|------|------|
| **重要性** | Σ importance ≥ 15.0 | 新记忆累积的重要性 |
| **时间** | > 1小时自上次 | 活动时间限制 |
| **睡眠** | 手动/昼夜节律 | 深度反思周期 |

---

## 反思深度

| 深度 | 类型 | 描述 |
|------|------|------|
| 0 | 事件 | 直接记忆（原始经验） |
| 1 | 洞察 | 一阶反思（模式识别） |
| 2 | 元认知 | 二阶（对反思的反思） |

---

## 反思过程

```
1. 生成焦点
   └── 从最近记忆中提取主题
   └── "我学到了什么关于{主题}？"

2. 检索相关记忆
   └── 使用复合评分（DRE）
   └── 按新近度 + 相似度 + 重要性 + 情感排序

3. 综合洞察
   └── 从检索的记忆生成观察
   └── 存储为depth=1的想法

4.（仅睡眠周期）元反思
   └── 反思最近的想法
   └── 生成depth=2的洞察

5. 记忆巩固
   └── 情景 → 语义提升
   └── 重要记忆变成长期知识
```

---

## 稳态目标生成

Dreamer使用记忆找到有效的方法，而非随机选择目标：

```elixir
def calculate_personal_baseline(state) do
  # 搜索有积极情感结果的记忆
  {:ok, memories} = Memory.search("积极状态 快乐 缓解 成功", limit: 10)

  # 计算成功PAD状态的平均值
  pads = Enum.map(memories, & &1.emotion)
  %{
    pleasure: mean(Enum.map(pads, & &1.pleasure)),
    arousal: mean(Enum.map(pads, & &1.arousal)),
    dominance: mean(Enum.map(pads, & &1.dominance))
  }
end
```

### Yerkes-Dodson最优唤醒

```elixir
def calculate_optimal_arousal(current_pad) do
  cond do
    # 高主导 + 积极 → 可以兴奋
    dominance > 0.3 and pleasure > 0 -> 0.4
    # 高主导 + 消极 → 需要激活来修复
    dominance > 0.3 and pleasure < 0 -> 0.3
    # 低主导 → 需要平静来恢复
    dominance < -0.3 -> 0.0
    # 默认：轻度参与
    true -> 0.15
  end
end
```

---

## 记忆巩固（DRE）

在睡眠周期中，情景记忆被提升为语义记忆：

### 巩固分数

```elixir
score = Mathematics.consolidation_score(
  memory_pad,      # 记忆的情感状态
  baseline_pad,    # 个人基线
  importance,      # 0.0 - 1.0
  age_seconds,     # 创建后的时间
  access_count     # 被访问的次数
)
```

### 巩固阈值

分数 ≥ **0.7** 的记忆被提升：

```elixir
Memory.store(content, %{
  type: :semantic,        # 长期存储
  importance: importance * 0.9,
  consolidated_from: original_id,
  consolidated_at: DateTime.utc_now()
})
```

---

## 情感反馈循环

Dreamer基于记忆效价影响Emotional状态：

```elixir
# 计算检索记忆的平均pleasure
avg_pleasure = memories |> Enum.map(& &1.emotion.pleasure) |> mean()

feedback = cond do
  avg_pleasure > 0.1 -> :lucid_insight     # 积极反思
  avg_pleasure < -0.1 -> :grim_realization # 消极反思
  true -> nil                               # 中性
end

if feedback do
  Emotional.feel(feedback, "dreamer", 0.8)
end
```

---

## 使用示例

```elixir
# 检查反思进度
iex> VivaCore.Dreamer.status()
%{importance_accumulator: 12.5, threshold: 15.0, progress_percent: 83.3, ...}

# 强制反思
iex> VivaCore.Dreamer.reflect_now()
%{focal_points: [...], insights: [...], trigger: :manual}

# 获取最近洞察
iex> VivaCore.Dreamer.recent_thoughts(3)
[%{insight: "反思'高负载'...", depth: 1, ...}, ...]

# 带评分检索记忆
iex> VivaCore.Dreamer.retrieve_with_scoring("成功的行动")
[%{content: "Action diagnose_load succeeded...", composite_score: 0.85}, ...]

# 生成目标（用于主动推理）
iex> VivaCore.Dreamer.hallucinate_goal(%{pleasure: -0.1, arousal: 0.2, dominance: 0.0})
%{pleasure: 0.2, arousal: 0.15, dominance: 0.1}
```

---

## 参考文献

- Park, J. S., et al. (2023). "Generative Agents: Interactive Simulacra of Human Behavior." arXiv:2304.03442
- Ebbinghaus, H. (1885). "Memory: A Contribution to Experimental Psychology."
- Yerkes, R. M., & Dodson, J. D. (1908). "The relation of strength of stimulus to rapidity of habit-formation."
- Friston, K. (2010). "The free-energy principle: a unified brain theory?"
