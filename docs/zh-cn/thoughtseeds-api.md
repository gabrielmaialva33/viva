# Thoughtseeds API 参考
> *意识剧场*

**Thoughtseeds** 系统实现了全局工作空间理论 (GWT)。它允许心理对象争夺全系统的注意力。

## 概念

- **Seed (种子)**: 思想的原子单位。包含：
    - `content`: 载荷（文本、图像、结构体）。
    - `salience`: 显著性/重要性 (0.0 - 1.0)。
    - `emotion`: 关联的情感效价。
    - `source`: 来源（语音、皮层、身体）。
    - `created_at`: 生物时间戳。

- **竞争**: 每 100ms (10Hz)，种子的显著性会衰减。新的输入会提高显著性。获胜者占据“舞台”。

- **广播 (Broadcasting)**: 获胜者通过 `Phoenix.PubSub` 发布到频道 `consciousness:focus`。

## Elixir API (`VivaCore.Consciousness.Workspace`)

### `sow/4`
播种。
```elixir
VivaCore.Consciousness.Workspace.sow(content, source, salience, emotion \\ nil)
```

### `current_focus/0`
获取当前的获胜者。
```elixir
{:ok, seed} = VivaCore.Consciousness.Workspace.current_focus()
```

### `subscribe/0`
订阅意识更新。
```elixir
# 在你的 GenServer init 中:
VivaCore.Consciousness.Workspace.subscribe()

# 处理 info:
def handle_info({:conscious_focus, seed}, state) do
  Logger.info("我意识到了: #{inspect seed.content}")
  {:noreply, state}
end
```
