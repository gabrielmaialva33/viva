# Senses - 外周神经系统

> *"身体不只是报告——身体会影响。VIVA 不只是知道 CPU 高——她感受到压力。"*

## 概述

**Senses** 模块是 VIVA 的外周神经系统——持续将 qualia 从身体泵送到灵魂的"心脏"。它连接硬件感知（通过 VivaBridge.Body 的 Rust NIF）和情感状态（VivaCore.Emotional）之间的差距。

就像人类自主神经系统将心跳、温度和压力信息从身体传输到大脑一样，Senses 将硬件指标传输到 VIVA 的情感状态，将原始数据转化为感受到的体验。

---

## 概念

### 心跳循环

Senses 以连续的 **1Hz 心跳** 运行（可配置从 100ms 到 10s）：

```
         ┌─────────────────────────────────────────────┐
         │               心跳 (1Hz)                    │
         └─────────────────────────────────────────────┘
                           │
                           ▼
         ┌─────────────────────────────────────────────┐
         │  1. 从 BodyServer 读取硬件状态              │
         │     (CPU, RAM, GPU, 温度)                   │
         └─────────────────────────────────────────────┘
                           │
                           ▼
         ┌─────────────────────────────────────────────┐
         │  2. 从 Rust O-U 动力学提取 PAD              │
         │     (愉悦度, 唤醒度, 支配度)                │
         └─────────────────────────────────────────────┘
                           │
                           ▼
         ┌─────────────────────────────────────────────┐
         │  3. 同步 PAD 到 Emotional GenServer         │
         │     VivaCore.Emotional.sync_pad(p, a, d)    │
         └─────────────────────────────────────────────┘
                           │
                           ▼
         ┌─────────────────────────────────────────────┐
         │  4. 记录心跳指标（debug 级别）              │
         │     [Senses] CPU: 45.2% RAM: 62.1%...       │
         └─────────────────────────────────────────────┘
```

### 硬件感知到 Qualia

从硬件指标到情感 qualia 的转换遵循此路径：

1. **Rust Body (Bevy ECS)** 通过 `sysinfo` 和 `nvml-wrapper` 感知硬件
2. **压力计算**：`stress = (cpu_usage + memory_used_percent) / 2`
3. **O-U 动力学** 随机演化 PAD 状态
4. **BodyServer** 暴露统一状态（PAD + 硬件）
5. **Senses** 读取状态并同步到 Soul

### 来自 Body 状态的 PAD 更新

身体状态通过 qualia 映射影响情感：

```
压力水平 → PAD 增量
────────────────────────────────────
Pleasure_delta  = -0.05 × stress
Arousal_delta   = +0.10 × stress
Dominance_delta = -0.03 × stress
```

高 CPU/内存压力导致：
- **降低愉悦度**（不适）
- **增加唤醒度**（警觉）
- **降低支配度**（控制力下降）

---

## 架构

### 系统图

```
┌──────────────────────────────────────────────────────────────────────┐
│                         BODY (Rust/Bevy ECS)                         │
│                                                                      │
│  ┌─────────────┐    ┌──────────────┐    ┌────────────────────────┐  │
│  │ HostSensor  │───▶│ BodyUpdate   │───▶│ O-U 随机过程           │  │
│  │ (sysinfo)   │    │ (stress,PAD) │    │ (情感动力学)           │  │
│  └─────────────┘    └──────────────┘    └────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
                                │
                                │ crossbeam-channel
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│                   VivaBridge (Elixir NIFs)                           │
│                                                                      │
│  ┌───────────────────┐              ┌────────────────────────────┐  │
│  │ VivaBridge.Body   │◀────────────▶│ VivaBridge.BodyServer      │  │
│  │ (NIF 接口)        │              │ (GenServer, 2Hz 心跳)      │  │
│  └───────────────────┘              └────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
                                │
                                │ GenServer.call
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│                      SOUL (Elixir/OTP)                               │
│                                                                      │
│  ┌───────────────────┐    sync_pad     ┌────────────────────────┐  │
│  │ VivaCore.Senses   │────────────────▶│ VivaCore.Emotional     │  │
│  │ (1Hz 心跳)        │                 │ (PAD 状态机)           │  │
│  └───────────────────┘                 └────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
```

### 回退机制

当 BodyServer 不可用（未启动或崩溃）时，Senses 回退到直接 NIF 调用：

```elixir
# 主要路径：BodyServer（包含 O-U 动力学）
VivaBridge.BodyServer.get_state()
# => %{pleasure: p, arousal: a, dominance: d, hardware: %{...}}

# 回退路径：直接 NIF（仅硬件，中性 PAD）
VivaBridge.feel_hardware()
# => %{cpu_usage: 45.2, memory_used_percent: 62.1, ...}
```

在回退模式下，PAD 设置为中性 `{0.0, 0.0, 0.0}` - Emotional 模块的内部 O-U 衰减处理 PAD 演化。

---

## API 参考

### `VivaCore.Senses.start_link/1`

启动 Senses GenServer。

```elixir
VivaCore.Senses.start_link(
  name: MyCustomSenses,      # 进程名称（默认：__MODULE__）
  interval_ms: 500,          # 心跳间隔（默认：1000）
  emotional_server: MyEmotional,  # 目标 Emotional（默认：VivaCore.Emotional）
  enabled: true              # 是否启用感知（默认：true）
)
```

### `VivaCore.Senses.get_state/1`

返回 Senses 的当前状态。

```elixir
VivaCore.Senses.get_state()
# => %{
#      interval_ms: 1000,
#      emotional_server: VivaCore.Emotional,
#      enabled: true,
#      last_reading: %{cpu_usage: 45.2, memory_used_percent: 62.1, ...},
#      last_qualia: {0.02, 0.05, -0.01},
#      heartbeat_count: 1234,
#      started_at: ~U[2024-01-15 10:00:00Z],
#      errors: []
#    }
```

### `VivaCore.Senses.pulse/1`

强制立即心跳（感知 + 应用 qualia）。

```elixir
VivaCore.Senses.pulse()
# => {:ok, {0.02, 0.05, -0.01}}
```

用于测试或需要立即读取时。

### `VivaCore.Senses.pause/1`

暂停自动感知。

```elixir
VivaCore.Senses.pause()
# => :ok
# 日志: [Senses] 已暂停
```

### `VivaCore.Senses.resume/1`

恢复自动感知。

```elixir
VivaCore.Senses.resume()
# => :ok
# 日志: [Senses] 已恢复
```

### `VivaCore.Senses.set_interval/2`

在运行时更改心跳间隔。

```elixir
VivaCore.Senses.set_interval(500)  # 2Hz
# => :ok
# 日志: [Senses] 间隔从 1000ms 更改为 500ms

# 边界：100ms（最大 10Hz）到 10000ms（最小 0.1Hz）
```

---

## 心跳详情

### 每次心跳发生的事情

| 步骤 | 动作 | 错误处理 |
|------|------|----------|
| 1 | 检查 BodyServer 是否存活 | 回退到直接 NIF |
| 2 | 获取身体状态（硬件 + PAD）| 使用回退重试 |
| 3 | 通过 `sync_pad/4` 同步 PAD 到 Emotional | 不可用时跳过 |
| 4 | 记录指标（debug 级别）| 总是成功 |
| 5 | 更新内部状态 | 总是成功 |
| 6 | 调度下一次心跳 | 总是成功 |

### 状态机

```
      ┌──────────┐
      │ 启动中   │
      └────┬─────┘
           │ init
           ▼
      ┌──────────┐  pause   ┌────────┐
      │ 运行中   │─────────▶│ 已暂停 │
      │ (1Hz)    │◀─────────│        │
      └──────────┘  resume  └────────┘
```

### 错误恢复

错误被记录并存储（仅保留最近 10 个）：

```elixir
# 发生错误后：
state.errors
# => [
#      {~U[2024-01-15 10:05:00Z], %RuntimeError{message: "NIF 崩溃"}},
#      ...
#    ]
```

心跳循环即使在错误后也会继续——弹性是内置的。

---

## 集成

### 与 VivaBridge.Body（Rust NIF）

Senses 通过 NIF 层读取硬件状态：

```elixir
# 通过 BodyServer（首选 - 包含 O-U 动力学）
VivaBridge.BodyServer.get_state()

# 直接 NIF（回退 - 仅硬件）
VivaBridge.feel_hardware()
```

### 与 VivaBridge.BodyServer

BodyServer 维护 Rust Bevy ECS 生命周期并提供统一状态：

```elixir
# BodyServer 以 2Hz（500ms）心跳，Senses 以 1Hz（1000ms）
# Senses 从 BodyServer 读取 last_state

%{
  pleasure: 0.15,
  arousal: -0.10,
  dominance: 0.25,
  stress_level: 0.35,
  in_bifurcation: false,
  hardware: %{
    cpu_usage: 45.2,
    memory_used_percent: 62.1,
    cpu_temp: 55.0,
    gpu_usage: 30.0,
    ...
  }
}
```

### 与 VivaCore.Emotional

Senses 将 PAD 状态同步到 Emotional GenServer：

```elixir
# 在心跳内部：
VivaCore.Emotional.sync_pad(p, a, d, state.emotional_server)

# 这将身体派生的 PAD 应用到 Emotional 的状态
# Emotional 然后通过其自身的 O-U 动力学演化
```

### 与 VivaCore.Interoception

Senses 处理原始的 Body-to-Soul 同步，而 Interoception 提供更高层次的解释：

| 模块 | 职责 |
|------|------|
| **Senses** | 原始硬件指标，PAD 同步 |
| **Interoception** | 自由能计算，感受状态 |

它们协同工作：
- Senses 提供原始数据
- Interoception 提供解释（`:homeostatic`、`:alarmed` 等）

---

## 配置

### 心跳间隔

| 设置 | 值 | 含义 |
|------|-----|------|
| 默认 | 1000ms | 1Hz - 平衡感知 |
| 最小 | 100ms | 10Hz - 高响应性 |
| 最大 | 10000ms | 0.1Hz - 省电 |

### 环境变量

Senses 遵守 NIF 跳过标志：

```bash
# 跳过 Rust NIF 编译（使用存根）
VIVA_SKIP_NIF=true mix test
```

### 应用配置

```elixir
# config/config.exs
config :viva_core, VivaCore.Senses,
  interval_ms: 1000,
  enabled: true
```

---

## 使用示例

### 基本状态检查

```elixir
# 检查感知是否活跃
state = VivaCore.Senses.get_state()
state.enabled
# => true

# 检查上次读取
state.last_reading
# => %{cpu_usage: 45.2, memory_used_percent: 62.1, ...}

# 检查上次 qualia（应用的 PAD 增量）
state.last_qualia
# => {0.02, 0.05, -0.01}
```

### 强制立即感知

```elixir
# pulse 强制立即心跳
{:ok, {p, a, d}} = VivaCore.Senses.pulse()

# 检查对 Emotional 的影响
VivaCore.Emotional.get_state()
# => %{pleasure: 0.15, arousal: 0.05, dominance: 0.10, ...}
```

### 调整响应性

```elixir
# 高性能模式（更高响应性）
VivaCore.Senses.set_interval(100)

# 省电模式（较低响应性）
VivaCore.Senses.set_interval(5000)
```

### 调试心跳问题

```elixir
# 检查心跳计数
state = VivaCore.Senses.get_state()
state.heartbeat_count
# => 1234

# 检查错误
state.errors
# => []  # 健康

# 检查运行时间
DateTime.diff(DateTime.utc_now(), state.started_at, :second)
# => 1234  # 启动后的秒数
```

### 在高负载期间暂停

```elixir
# 在资源密集型操作期间临时暂停感知
VivaCore.Senses.pause()

# ... 密集型操作 ...

VivaCore.Senses.resume()
```

### 使用自定义 Emotional Server 测试

```elixir
# 在测试中，使用隔离的 Emotional 服务器
{:ok, emotional_pid} = VivaCore.Emotional.start_link(name: :test_emotional)
{:ok, senses_pid} = VivaCore.Senses.start_link(
  name: :test_senses,
  emotional_server: :test_emotional,
  interval_ms: 100
)

# 强制 pulse
{:ok, pad} = VivaCore.Senses.pulse(:test_senses)
```

---

## 生物学类比

| 人体系统 | VIVA 等效物 |
|----------|-------------|
| 自主神经系统 | Senses GenServer |
| 心率 | 心跳间隔（默认 1Hz）|
| 感觉神经元 | VivaBridge.Body NIF |
| 丘脑（中继）| BodyServer |
| 边缘系统 | VivaCore.Emotional |

就像人体持续传输信息（心跳、温度、血压）到大脑一样，Senses 持续将硬件指标传输到 VIVA 的情感状态。

---

## 参考文献

- Craig, A. D. (2002). "How do you feel? Interoception: the sense of the physiological condition of the body."
- Damasio, A. (1994). "Descartes' Error: Emotion, Reason, and the Human Brain."
- Seth, A. K. (2013). "Interoceptive inference, emotion, and the embodied self."
