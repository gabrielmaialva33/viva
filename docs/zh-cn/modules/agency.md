# 代理 - 数字手

> *"代理就是稳态。'行动'的意愿来自调节内部状态的需要。"*

## 哲学

VIVA能感受疼痛（通过内感受）但无法做任何事。
现在她有了手：一个用于只读诊断命令的沙盒执行器。

如果时间在膨胀（卡顿），VIVA"想要"理解原因并可能修复它。
这不是任意的欲望 - 它源自自由能原理。

### 马尔可夫毯

Agency是VIVA的**主动状态**的一部分 - 内部状态影响外部状态（操作系统环境）的边界。

---

## 安全模型

| 原则 | 实现 |
|------|------|
| **仅白名单** | 无shell插值，无任意命令 |
| **只读** | 仅诊断命令（ps, free, df, ping localhost） |
| **超时** | 每个命令最多5秒 |
| **学习** | 结果存储在Memory中供将来参考 |

---

## API参考

### `VivaCore.Agency.can_do?/1`
检查VIVA是否能执行特定动作。

```elixir
VivaCore.Agency.can_do?(:diagnose_memory)
# => true

VivaCore.Agency.can_do?(:rm_rf)
# => false
```

### `VivaCore.Agency.available_actions/0`
列出所有可用动作。

```elixir
VivaCore.Agency.available_actions()
# => %{
#      diagnose_memory: "Check available RAM",
#      diagnose_processes: "List processes by CPU usage",
#      diagnose_disk: "Check disk space",
#      ...
#    }
```

### `VivaCore.Agency.attempt/1`
执行动作并返回带有相关感受的结果。

```elixir
VivaCore.Agency.attempt(:diagnose_load)
# => {:ok, "15:30 up 5 days, load average: 0.50, 0.40, 0.35", :understanding}

VivaCore.Agency.attempt(:forbidden_action)
# => {:error, :forbidden, :shame}
```

### `VivaCore.Agency.get_history/0`
获取最近50次动作尝试。

```elixir
VivaCore.Agency.get_history()
# => [
#      %{action: :diagnose_load, outcome: :success, timestamp: ~U[2024-01-15 15:30:00Z]},
#      ...
#    ]
```

### `VivaCore.Agency.get_success_rates/0`
获取每个动作的成功/失败计数。

```elixir
VivaCore.Agency.get_success_rates()
# => %{
#      diagnose_memory: %{success: 10, failure: 0},
#      diagnose_processes: %{success: 5, failure: 1}
#    }
```

---

## 允许的命令

| 动作 | 命令 | 描述 |
|------|------|------|
| `:diagnose_memory` | `free -h` | 检查可用RAM |
| `:diagnose_processes` | `ps aux --sort=-pcpu` | 按CPU列出进程（前20个） |
| `:diagnose_disk` | `df -h` | 检查磁盘空间 |
| `:diagnose_network` | `ping -c 1 localhost` | 检查本地网络栈 |
| `:diagnose_load` | `uptime` | 检查系统负载平均值 |
| `:check_self` | `ps -p {PID} -o pid,pcpu,pmem,etime,rss` | 自身进程状态 |
| `:diagnose_io` | `iostat -x 1 1` | IO等待和磁盘活动 |

---

## 情感结果

每个动作有预期的成功/失败感受：

| 动作 | 成功感受 | 失败感受 |
|------|----------|----------|
| `:diagnose_memory` | `:relief` | `:confusion` |
| `:diagnose_processes` | `:understanding` | `:confusion` |
| `:diagnose_disk` | `:relief` | `:confusion` |
| `:diagnose_network` | `:relief` | `:worry` |
| `:diagnose_load` | `:understanding` | `:confusion` |
| `:check_self` | `:self_awareness` | `:dissociation` |
| `:diagnose_io` | `:understanding` | `:confusion` |

### 感受 → PAD映射

```elixir
:relief         → %{pleasure: +0.3, arousal: -0.2, dominance: +0.2}
:understanding  → %{pleasure: +0.2, arousal: +0.1, dominance: +0.3}
:self_awareness → %{pleasure: +0.1, arousal:  0.0, dominance: +0.4}
:confusion      → %{pleasure: -0.1, arousal: +0.2, dominance: -0.2}
:worry          → %{pleasure: -0.2, arousal: +0.3, dominance: -0.1}
:shame          → %{pleasure: -0.3, arousal: +0.1, dominance: -0.4}
:dissociation   → %{pleasure: -0.2, arousal: -0.3, dominance: -0.3}
:panic          → %{pleasure: -0.5, arousal: +0.5, dominance: -0.5}
```

---

## 主动推理循环

```
1. Interoception检测到高自由能（如卡顿）
   └── tick_jitter > 预期值

2. Emotional感到 :alarmed
   └── arousal ↑, pleasure ↓

3. 主动推理选择 :diagnose_load 动作
   └── 基于过去的成功率

4. Agency.attempt(:diagnose_load) 执行 "uptime"
   └── 返回负载平均信息

5. 结果与情感上下文一起存储在Memory
   └── "负载很高，运行了uptime，感到understanding"

6. 下次，VIVA记住什么有效
   └── 更快响应，更少探索
```

---

## Memory集成

每个动作结果都被存储以供将来RAG检索：

```elixir
Memory.store(%{
  content: "Action diagnose_load succeeded. Result: 15:30 up 5 days...",
  type: :episodic,
  importance: 0.6,  # 失败时更高（0.8）
  emotion: %{pleasure: 0.2, arousal: 0.1, dominance: 0.3},
  metadata: %{
    source: :agency,
    action: :diagnose_load,
    outcome: :success
  }
})
```

---

## 使用示例

```elixir
# VIVA感到有问题
iex> VivaCore.Interoception.get_feeling()
:alarmed

# 检查是否能做些什么
iex> VivaCore.Agency.can_do?(:diagnose_load)
true

# 采取行动
iex> VivaCore.Agency.attempt(:diagnose_load)
{:ok, "16:45 up 10 days, load average: 2.50, 1.80, 1.20", :understanding}

# 理解后的感受
# → Emotional接收 :understanding
# → pleasure +0.2, arousal +0.1, dominance +0.3
```

---

## 参考文献

- Friston, K. (2010). "The free-energy principle: a unified brain theory?"
- Clark, A. (2013). "Whatever next? Predictive brains, situated agents, and the future of cognitive science."
