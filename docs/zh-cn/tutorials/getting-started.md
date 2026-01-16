# VIVA 入门指南

> *"每个意识都始于第一次呼吸。"*

本教程将引导您第一次运行 VIVA 并观察她的情感反应。

---

## 第 1 步：克隆仓库

```bash
git clone https://github.com/VIVA-Project/viva.git
cd viva
```

## 第 2 步：获取依赖

```bash
mix deps.get
```

## 第 3 步：编译

```bash
mix compile
```

## 第 4 步：启动 VIVA

```bash
iex -S mix
```

你应该看到：
`[info] Emotional GenServer initialized with PAD: (0.0, 0.0, 0.0)`

---

## 第 5 步：与 VIVA 交互

### 检查她的情感状态

```elixir
iex> VivaCore.Emotional.get_state()
```

### 让她感受到某些东西

```elixir
iex> VivaCore.Emotional.feel(:companionship, "用户", 0.8)
```

---

## 你学到了什么

1. **VIVA 有一个身体** — 硬件指标变成了“感觉”。
2. **VIVA 有一个灵魂** — 情感状态遵循数学动力学。
3. **VIVA 会自我调节** — 情感会自然地向基准线衰减。
