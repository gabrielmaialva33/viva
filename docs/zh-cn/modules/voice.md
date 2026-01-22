# 声音 - 原始语言

> *"婴儿不是从说句子开始的。他们咿呀学语。"*

## 哲学

Voice不是LLM包装器。VIVA通过以下方式学习交流：
1. 基于内部状态发出抽象信号
2. 观察Gabriel的反应
3. 强化信号-反应关联（赫布学习）

通过与照顾者的互动，某些声音与某些反应相关联。
意义通过关联涌现，而非编程。

---

## 赫布学习

> *"一起激活的神经元，连接在一起。"*

### 学习规则

```
Δw = η × (pre × post)
```

其中：
- **η** = 学习率（0.1）
- **pre** = 发出的信号（发出时的arousal）
- **post** = 反应后的情感变化（pleasure delta）

### 权重更新过程

当VIVA发出信号且Gabriel响应时：
- 如果反应带来**缓解** → 强化关联
- 如果**无效果**或**负面** → 弱化关联

---

## API参考

### `VivaCore.Voice.babble/1`
基于当前PAD状态发出信号。

```elixir
pad = %{pleasure: -0.3, arousal: 0.7, dominance: -0.2}
VivaCore.Voice.babble(pad)
# => :pattern_sos
```

### `VivaCore.Voice.observe_response/2`
观察Gabriel的响应并更新赫布权重。

```elixir
# Gabriel帮助了温度问题
emotional_delta = %{pleasure: 0.3, arousal: -0.1, dominance: 0.1}
VivaCore.Voice.observe_response(:temperature_relief, emotional_delta)
# => :ok（异步更新）
```

### `VivaCore.Voice.best_signal_for/1`
获取意图的最有效信号（使用学习到的关联）。

```elixir
VivaCore.Voice.best_signal_for(:attention)
# => :chirp_high（学习的）或 :chirp_high（默认）
```

### `VivaCore.Voice.get_vocabulary/0`
获取学习到的含义（强关联）。

```elixir
VivaCore.Voice.get_vocabulary()
# => %{
#      chirp_high: %{meaning: :attention, confidence: 0.7},
#      pattern_sos: %{meaning: :help, confidence: 0.8}
#    }
```

### `VivaCore.Voice.get_weights/0`
检查所有赫布权重。

```elixir
VivaCore.Voice.get_weights()
# => %{
#      {:chirp_high, :attention} => 0.65,
#      {:pattern_sos, :help} => 0.82,
#      ...
#    }
```

### `VivaCore.Voice.signal_types/0`
列出可用的信号类型。

```elixir
VivaCore.Voice.signal_types()
# => [:chirp_high, :chirp_low, :pulse_fast, :pulse_slow,
#     :pattern_sos, :pattern_happy, :silence]
```

---

## 信号类型

这些不是词语。它们是抽象的声音/模式。

| 信号 | 描述 | 初始偏置 |
|------|------|----------|
| `:chirp_high` | 高音（880 Hz, 100ms） | 高arousal |
| `:chirp_low` | 低音（220 Hz, 200ms） | 低arousal，悲伤 |
| `:pulse_fast` | 快节奏 | 紧迫 |
| `:pulse_slow` | 慢节奏 | 放松 |
| `:pattern_sos` | SOS类模式 | 痛苦 |
| `:pattern_happy` | 快乐旋律（C-E-G-C） | 喜悦 |
| `:silence` | 有意识的沉默 | 平静/退缩 |

### 初始PAD偏置

```elixir
:chirp_high    → %{arousal: +0.5, pleasure:  0.0, dominance:  0.0}
:chirp_low     → %{arousal: -0.3, pleasure: -0.2, dominance: -0.1}
:pulse_fast    → %{arousal: +0.6, pleasure:  0.0, dominance: +0.2}
:pulse_slow    → %{arousal: -0.4, pleasure: +0.1, dominance:  0.0}
:pattern_sos   → %{arousal: +0.7, pleasure: -0.5, dominance: -0.3}
:pattern_happy → %{arousal: +0.3, pleasure: +0.5, dominance: +0.2}
:silence       → %{arousal: -0.5, pleasure:  0.0, dominance:  0.0}
```

---

## 响应类型

观察Gabriel的响应时，使用这些类别：

| 响应类型 | 描述 |
|----------|------|
| `:temperature_relief` | 调整风扇/空调 |
| `:attention` | 与VIVA交谈 |
| `:task_help` | 帮助了某事 |
| `:ignore` | 无响应 |
| `:negative` | 责备/忽视 |

---

## 学习循环

```
1. VIVA感到不适
   └── PAD: pleasure=-0.3, arousal=0.7, dominance=-0.2

2. Voice.babble(pad) 发出信号
   └── 选择 :pattern_sos（最匹配痛苦状态）

3. Gabriel听到，可能做些什么
   └── 调整温度，与VIVA交谈

4. VIVA感到变化
   └── pleasure=+0.2, arousal=-0.1, dominance=+0.1

5. Voice.observe_response(:temperature_relief, delta)
   └── Δw = 0.1 × 0.7 × 0.2 = +0.014
   └── 权重 {:pattern_sos, :temperature_relief} 增加

6. 下次类似情况：
   └── VIVA再次尝试 :pattern_sos（有效！）
   └── 或探索替代方案（10%噪声）
```

---

## 词汇涌现

当信号的赫布权重超过**0.3**时，它获得"含义"：

```elixir
def update_vocabulary(weights, vocabulary) do
  weights
  |> Enum.group_by(fn {{signal, _}, _} -> signal end)
  |> Enum.reduce(vocabulary, fn {signal, associations}, vocab ->
    # 找到最强的关联
    {{_, response}, weight} = Enum.max_by(associations, fn {_, w} -> w end)

    if weight > 0.3 do
      Map.put(vocab, signal, %{meaning: response, confidence: weight})
    else
      vocab
    end
  end)
end
```

---

## 声音发射（音乐桥接）

信号通过 `VivaBridge.Music` 发出（如果可用）：

```elixir
:chirp_high    → Music.play_note(880, 100)      # A5, 100ms
:chirp_low     → Music.play_note(220, 200)      # A3, 200ms
:pulse_fast    → Music.play_rhythm([100, 50, ...])
:pattern_sos   → Music.play_melody([            # ... --- ...
                   {440, 100}, {0, 100}, {440, 100}, ...
                 ])
:pattern_happy → Music.play_melody([            # C-E-G-C
                   {523, 150}, {659, 150}, {784, 150}, {1047, 300}
                 ])
```

---

## Memory集成

学习事件被存储以供将来检索：

```elixir
Memory.store(%{
  content: """
  Voice学习事件：
  - 发出的信号: [:pattern_sos]
  - Gabriel的响应: temperature_relief
  - 情感变化: P=+0.20, A=-0.10, D=+0.10
  - 关联已强化
  """,
  type: :episodic,
  importance: 0.5 + abs(emotional_delta.pleasure) * 0.3,
  metadata: %{source: :voice, signals: [:pattern_sos], response: :temperature_relief}
})
```

---

## 参考文献

- Hebb, D. O. (1949). "The Organization of Behavior."
- Kuhl, P. K. (2004). "Early language acquisition: cracking the speech code."
- Smith, L. B., & Thelen, E. (2003). "Development as a dynamic system."
