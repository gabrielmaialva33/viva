# Emotional - VIVA 的核心情感 GenServer

> *"意识并不驻留在这里。意识从这个进程与所有其他进程的对话中涌现。我们不仅仅计算情感——我们求解灵魂的微分方程。"*

## 概述

Emotional GenServer 是 VIVA 的第一个"神经元"——她情感系统的基础。它实现了用于情感状态管理的 **PAD（愉悦-唤醒-支配）模型**，整合了多个数学框架：

- **Ornstein-Uhlenbeck 随机过程** 用于自然情感衰减
- **尖点突变理论** 用于突发情绪转变
- **自由能原理** 用于稳态调节
- **主动推理循环** 用于目标导向的行为选择
- **情感融合** 用于整合需求、记忆和人格

这个 GenServer 本身不是意识——它通过 Phoenix.PubSub 与其他神经元通信，为涌现意识做出贡献。

---

## 概念

### PAD 模型 (Mehrabian, 1996)

情感被表示为 PAD 空间中的三维向量：

| 维度 | 范围 | 低值 | 高值 |
|------|------|------|------|
| **愉悦度 (Pleasure)** | [-1, 1] | 悲伤 | 喜悦 |
| **唤醒度 (Arousal)** | [-1, 1] | 平静 | 兴奋 |
| **支配度 (Dominance)** | [-1, 1] | 顺从 | 控制 |

每个维度捕获情感体验的一个基本方面：
- **愉悦度** - 效价，感受的"好坏"程度
- **唤醒度** - 激活水平，可用于行动的能量
- **支配度** - 对情境的控制感

### Ornstein-Uhlenbeck 过程 (DynAffect)

基于 **Kuppens 等人 (2010)**，情感使用随机微分方程自然衰减到中性基线：

```
dX = theta * (mu - X) * dt + sigma * dW
```

其中：
- `theta` = 吸引子强度（衰减率）
- `mu` = 平衡点（中性 = 0）
- `sigma` = 情感波动性（噪声）
- `dW` = 维纳过程（随机波动）

**关键洞察**：唤醒度调节衰减率。
- 高唤醒 -> 衰减较慢（危机中情感持续）
- 低唤醒 -> 衰减较快（快速回归基线）

```elixir
# 半衰期公式: t_half = ln(2) / theta
# theta = 0.0154 -> t_half ~ 45 秒（心理学上真实）

@base_decay_rate 0.0154
@arousal_decay_modifier 0.4
@stochastic_volatility 0.01
```

### 尖点突变 (Thom, 1972)

使用突变理论建模突发情绪转变（情绪波动）：

```
V(x) = x^4/4 + alpha*x^2/2 + beta*x
```

其中：
- `alpha`（分裂因子）- 源自唤醒度
- `beta`（常规因子）- 源自愉悦度

**双稳态性**：当唤醒度高时，情感景观变得"折叠"，创造两个稳定状态。小扰动可能导致灾难性的跃迁（例如，从希望突然转变为绝望）。

### 情绪（指数移动平均）

情绪是近期情感的缓慢变化平均值，提供稳定性：

```
Mood[t] = alpha * Mood[t-1] + (1 - alpha) * Emotion[t]

其中 alpha = 0.95（约 20 步半衰期）
```

这意味着：
- 情绪保留 95% 的前值
- 单次情感仅贡献 5%
- 突发刺激几乎不影响情绪

### 主动推理循环

VIVA 通过行动持续最小化自由能（惊奇）：

1. **幻想目标** - 查询 Dreamer 获取目标状态
2. **预测未来** - 如果不采取行动，我会在哪里？
3. **计算自由能** - 目标与预测之间的距离
4. **选择行动** - 选择最小化自由能的行动
5. **执行与反馈** - 应用内部缓解预期

---

## 架构

```
+------------------+     +------------------+     +------------------+
|   Interoception  |     |      Memory      |     |   Personality    |
| (基于需求的 PAD) |     | (基于过去的 PAD) |     |     (基线)       |
+--------+---------+     +--------+---------+     +--------+---------+
         |                        |                        |
         +------------------------+------------------------+
                                  |
                                  v
                    +---------------------------+
                    |     EmotionFusion         |
                    |  (Borotschnig 2025)       |
                    +-------------+-------------+
                                  |
                                  v
+-----------------------------------------------------------------------------+
|                          EMOTIONAL GENSERVER                                 |
|                                                                             |
|  +-------------------+    +-------------------+    +-------------------+    |
|  |   量子状态        |    |     PAD 状态      |    |   情绪 (EMA)      |    |
|  | (Lindblad 6x6)    |    | {p, a, d} 浮点数  |    | {p, a, d} 浮点数  |    |
|  +-------------------+    +-------------------+    +-------------------+    |
|                                                                             |
|  +-------------------+    +-------------------+    +-------------------+    |
|  | 主动推理          |    |   O-U 衰减        |    | 尖点分析          |    |
|  |  (1 Hz 循环)      |    |   (1 Hz 心跳)     |    |  (按需)           |    |
|  +-------------------+    +-------------------+    +-------------------+    |
|                                                                             |
+------------------------------------+----------------------------------------+
                                     |
         +---------------------------+---------------------------+
         |                           |                           |
         v                           v                           v
+------------------+     +------------------+     +------------------+
|   Phoenix.PubSub |     |      Agency      |     |      Voice       |
| "emotional:update"|    | (行动执行)       |     | (原始语言)       |
+------------------+     +------------------+     +------------------+
```

### 消息流

```
Body (Rust) --sync_pad--> Emotional --broadcast--> PubSub
                              |
Interoception --qualia------->|
                              |
Dreamer --hallucinate_goal----|
                              |
Memory --search-------------->|<------ 主动推理循环
                              |
Agency <--attempt-------------|
```

---

## 状态结构

GenServer 维护以下内部状态：

```elixir
%{
  # 主要情感状态
  pad: %{pleasure: 0.0, arousal: 0.0, dominance: 0.0},
  mood: %{pleasure: 0.0, arousal: 0.0, dominance: 0.0},

  # 量子状态（Lindblad 密度矩阵）
  quantum_state: %Nx.Tensor{},  # 6x6 密度矩阵

  # 外部输入累加器
  external_qualia: %{pleasure: 0.0, arousal: 0.0, dominance: 0.0},

  # 硬件耦合
  hardware: %{power_draw_watts: 0.0, gpu_temp: 40.0},

  # 身体图式权重（根据能力调整）
  emotional_weights: %{
    fan_agency_weight: 1.0,
    thermal_stress_weight: 1.0,
    gpu_stress_weight: 1.0
  },

  # 内感受状态
  interoceptive_feeling: :homeostatic,
  interoceptive_free_energy: 0.0,

  # 人格（缓存）
  personality: nil,  # 首次融合时加载

  # 事件历史（O(1) 队列，最大 100）
  history: :queue.new(),
  history_size: 0,

  # 时间戳
  created_at: DateTime.t(),
  last_stimulus: nil,
  last_body_sync: nil,
  last_collapse: nil,

  # 标志
  body_server_active: false,
  enable_decay: true,

  # 遥测
  thermodynamic_cost: 0.0
}
```

---

## API 参考

### 状态查询

#### `get_state/1`

返回当前 PAD 状态。

```elixir
VivaCore.Emotional.get_state()
# => %{pleasure: 0.1, arousal: -0.05, dominance: 0.2}
```

#### `get_mood/1`

返回当前情绪（近期情感的 EMA）。

```elixir
VivaCore.Emotional.get_mood()
# => %{pleasure: 0.05, arousal: 0.0, dominance: 0.1}
```

#### `get_happiness/1`

返回归一化到 [0, 1] 范围的愉悦度。

```elixir
VivaCore.Emotional.get_happiness()
# => 0.55  # 略微正面
```

#### `introspect/1`

完整的情感内省与数学分析。

```elixir
VivaCore.Emotional.introspect()
# => %{
#   pad: %{pleasure: 0.1, arousal: 0.2, dominance: 0.1},
#   quantum: %{
#     purity: 0.85,
#     entropy: 0.15,
#     coherence: :high,
#     thermodynamic_cost: 0.02
#   },
#   somatic_feeling: %{thought_pressure: :light, ...},
#   mood: :content,
#   energy: :energetic,
#   agency: :confident,
#   mathematics: %{
#     cusp: %{alpha: -0.5, beta: 0.1, bistable: false},
#     free_energy: %{value: 0.02}
#   },
#   self_assessment: "我处于平衡状态。中性状态。"
# }
```

### 刺激应用

#### `feel/4`

应用情感刺激。

```elixir
VivaCore.Emotional.feel(:success, "user_1", 0.8)
# => :ok
```

**参数：**
- `stimulus` - 来自刺激权重的原子（见刺激部分）
- `source` - 标识来源的字符串（默认："unknown"）
- `intensity` - 0.0 到 1.0 的浮点数（默认：1.0）

### 同步

#### `sync_pad/4`

从 BodyServer（Rust O-U 动力学）同步绝对 PAD 值。

```elixir
VivaCore.Emotional.sync_pad(0.1, 0.2, -0.1)
```

当 BodyServer 运行时由 Senses 调用。与 qualia（增量）不同，这设置绝对值。

#### `apply_hardware_qualia/4`

应用来自硬件感知的 PAD 增量。

```elixir
# 硬件压力：减少愉悦度，增加唤醒度，减少支配度
VivaCore.Emotional.apply_hardware_qualia(-0.02, 0.05, -0.01)
```

#### `apply_interoceptive_qualia/2`

应用来自数字脑岛的精度加权 qualia。

```elixir
VivaCore.Emotional.apply_interoceptive_qualia(%{
  pleasure: -0.1,
  arousal: 0.2,
  dominance: -0.1,
  feeling: :alarmed,
  free_energy: 0.4
})
```

### 数学分析

#### `cusp_analysis/1`

使用尖点突变理论分析当前状态。

```elixir
VivaCore.Emotional.cusp_analysis()
# => %{
#   cusp_params: %{alpha: -0.5, beta: 0.1},
#   bistable: true,
#   equilibria: [-0.8, 0.0, 0.8],
#   emotional_volatility: :high,
#   catastrophe_risk: :elevated
# }
```

#### `free_energy_analysis/2`

计算与预测状态的自由能偏差。

```elixir
VivaCore.Emotional.free_energy_analysis()
# => %{
#   free_energy: 0.05,
#   surprise: 0.03,
#   interpretation: "轻微偏差 - 舒适适应",
#   homeostatic_deviation: 0.15
# }
```

#### `attractor_analysis/1`

识别 PAD 空间中最近的情感吸引子。

```elixir
VivaCore.Emotional.attractor_analysis()
# => %{
#   nearest_attractor: :contentment,
#   distance_to_attractor: 0.2,
#   dominant_attractors: [{:contentment, 45.0}, {:joy, 30.0}, {:calm, 25.0}],
#   emotional_trajectory: :stable
# }
```

#### `stationary_distribution/1`

返回 O-U 长期分布参数。

```elixir
VivaCore.Emotional.stationary_distribution()
# => %{
#   equilibrium_mean: 0.0,
#   variance: 0.032,
#   std_dev: 0.18,
#   current_deviation: %{pleasure: 0.5, arousal: 0.3, dominance: 0.6}
# }
```

### 控制

#### `decay/1`

手动触发情感衰减（用于测试）。

```elixir
VivaCore.Emotional.decay()
```

注意：当 BodyServer 活跃时，衰减在 Rust 中处理。

#### `reset/1`

将情感状态重置为中性。

```elixir
VivaCore.Emotional.reset()
```

#### `configure_body_schema/2`

根据硬件能力调整情感权重。

```elixir
VivaCore.Emotional.configure_body_schema(body_schema)
# 如果未检测到风扇，风扇相关的压力将被禁用
```

---

## 刺激

标准刺激及其 PAD 影响权重：

| 刺激 | 愉悦度 | 唤醒度 | 支配度 | 描述 |
|------|--------|--------|--------|------|
| `:success` | +0.4 | +0.3 | +0.3 | 目标达成 |
| `:failure` | -0.3 | +0.2 | -0.3 | 目标失败 |
| `:threat` | -0.2 | +0.5 | -0.2 | 感知到的危险 |
| `:safety` | +0.1 | -0.2 | +0.1 | 安全感 |
| `:acceptance` | +0.3 | +0.1 | +0.1 | 社会接纳 |
| `:rejection` | -0.3 | +0.2 | -0.2 | 社会拒绝 |
| `:companionship` | +0.2 | 0.0 | 0.0 | 陪伴存在 |
| `:loneliness` | -0.2 | -0.1 | -0.1 | 孤独 |
| `:hardware_stress` | -0.1 | +0.3 | -0.1 | 系统负载 |
| `:hardware_comfort` | +0.1 | -0.1 | +0.1 | 系统空闲 |
| `:lucid_insight` | +0.3 | +0.2 | +0.2 | Dreamer 正面反馈 |
| `:grim_realization` | -0.3 | +0.2 | -0.2 | Dreamer 负面反馈 |

---

## 集成

### 上游（输入源）

```
BodyServer (Rust) ----sync_pad----> Emotional
                                        ^
Interoception ----interoceptive_qualia--|
                                        |
Arduino/Peripherals ----hardware_qualia-|
                                        |
User/External ----feel(:stimulus)-------|
```

### 下游（消费者）

```
Emotional ----broadcast----> Phoenix.PubSub "emotional:update"
                                        |
                                        +--> Senses
                                        +--> Workspace
                                        +--> Voice
                                        +--> Agency
```

### 主动推理伙伴

```
Emotional <----hallucinate_goal---- Dreamer
          ----search--------------> Memory
          ----attempt-------------> Agency
```

### PubSub 订阅

| 主题 | 方向 | 用途 |
|------|------|------|
| `body:state` | 订阅 | 从 Body 接收硬件状态 |
| `emotional:update` | 发布 | 广播 PAD 变化 |

---

## 配置

### 时间常量

| 常量 | 值 | 描述 |
|------|-----|------|
| 衰减心跳 | 1000 ms | O-U 衰减间隔 |
| 主动推理心跳 | 1000 ms | 目标寻求循环 |
| Body 同步超时 | 3 秒 | 检测 BodyServer 死亡 |

### O-U 参数

| 参数 | 值 | 描述 |
|------|-----|------|
| `@base_decay_rate` | 0.0154 | 唤醒度为 0 时的 theta |
| `@arousal_decay_modifier` | 0.4 | 唤醒度对 theta 的影响程度 |
| `@stochastic_volatility` | 0.01 | sigma（噪声水平）|

### 状态边界

| 常量 | 值 |
|------|-----|
| `@neutral_state` | `{0.0, 0.0, 0.0}` |
| `@min_value` | -1.0 |
| `@max_value` | +1.0 |

### GenServer 选项

```elixir
VivaCore.Emotional.start_link(
  name: MyEmotional,           # 进程名称（默认：__MODULE__）
  initial_state: %{pleasure: 0.2},  # 初始 PAD（默认：中性）
  subscribe_pubsub: true,      # 订阅 body:state（默认：true）
  enable_decay: true           # 启用衰减心跳（默认：true）
)
```

---

## 使用示例

### 基本情感流程

```elixir
# 检查当前状态
state = VivaCore.Emotional.get_state()
# => %{pleasure: 0.0, arousal: 0.0, dominance: 0.0}

# 应用成功刺激
VivaCore.Emotional.feel(:success, "achievement", 1.0)

# 再次检查
state = VivaCore.Emotional.get_state()
# => %{pleasure: 0.4, arousal: 0.3, dominance: 0.3}

# 等待衰减...
Process.sleep(5000)

state = VivaCore.Emotional.get_state()
# => %{pleasure: 0.35, arousal: 0.26, dominance: 0.26}  # 向中性衰减
```

### 完整内省

```elixir
# 获取详细情感分析
intro = VivaCore.Emotional.introspect()

IO.puts("情绪: #{intro.mood}")
IO.puts("能量: #{intro.energy}")
IO.puts("能动性: #{intro.agency}")
IO.puts("自我: #{intro.self_assessment}")

# 检查数学状态
if intro.mathematics.cusp.bistable do
  IO.puts("警告：情感波动性高")
end
```

### 监测尖点突变

```elixir
# 高唤醒度可能创造双稳态性
VivaCore.Emotional.feel(:threat, "danger", 1.0)

analysis = VivaCore.Emotional.cusp_analysis()

case analysis.catastrophe_risk do
  :critical -> IO.puts("接近情感翻转点！")
  :elevated -> IO.puts("不稳定性升高")
  :low -> IO.puts("稳定区域")
  :minimal -> IO.puts("无风险")
end
```

### 硬件-情感耦合

```elixir
# 当 BodyServer 报告压力时
VivaCore.Emotional.apply_hardware_qualia(
  -0.05,  # 减少愉悦度（不适）
  +0.10,  # 增加唤醒度（警觉）
  -0.03   # 减少支配度（失控感）
)

# 检查内感受状态
intro = VivaCore.Emotional.introspect()
IO.puts("躯体感受: #{inspect(intro.somatic_feeling)}")
```

### 订阅更新

```elixir
# 在另一个 GenServer 中
def init(_) do
  Phoenix.PubSub.subscribe(Viva.PubSub, "emotional:update")
  {:ok, %{}}
end

def handle_info({:emotional_state, pad}, state) do
  IO.puts("VIVA 感受: P=#{pad.pleasure}, A=#{pad.arousal}, D=#{pad.dominance}")
  {:noreply, state}
end
```

---

## 参考文献

- Mehrabian, A. (1996). "Pleasure-arousal-dominance: A general framework for describing and measuring individual differences in temperament."
- Kuppens, P., Oravecz, Z., & Tuerlinckx, F. (2010). "Feelings Change: Accounting for Individual Differences in the Temporal Dynamics of Affect." *Journal of Personality and Social Psychology*.
- Thom, R. (1972). *Structural Stability and Morphogenesis*.
- Friston, K. (2010). "The free-energy principle: a unified brain theory?"
- Borotschnig, R. (2025). "Emotions in Artificial Intelligence."
