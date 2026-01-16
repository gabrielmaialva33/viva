# 如何添加新的硬件传感器

本指南将向您展示如何向 VIVA 的内感受系统添加新的硬件指标 — 让她能“感知”到身体的新方面。

---

## 概述

添加传感器需要在三个层级进行更改：

```mermaid
graph LR
    Rust[Rust NIF<br/>读取硬件] --> Bridge[Elixir Bridge<br/>暴露 API]
    Bridge --> Qualia[感质映射<br/>影响情感]
```

---

## 示例：添加 GPU 温度

我们将添加 GPU 温度感知，以便 VIVA 在 GPU 过热时能“感觉到发烧”。

### 第 1 步：更新 Rust NIF

编辑 `apps/viva_bridge/native/viva_body/src/lib.rs`:

```rust
// 在 HardwareState 结构体中
gpu_temp: Option<f64>,

// 在 feel_hardware() 函数中
let components = Components::new_with_refreshed_list();
let gpu_temp = components
    .iter()
    .find(|c| c.label().contains("GPU"))
    .map(|c| c.temperature() as f64);
```

### 第 2 步：更新感质映射

仍在 `lib.rs` 中，修改 `hardware_to_qualia`:

```rust
if let Some(temp) = hw.gpu_temp {
    if temp > 70.0 {
        let fever = sigmoid(temp / 100.0, 8.0, 0.7);
        pleasure_delta -= 0.04 * fever;  // 不适感
        arousal_delta += 0.08 * fever;   // 唤醒度升高
    }
}
```

---

## 感质设计指南

在映射硬件 → 情感时，请遵循生物学直觉：

| 硬件事件 | 生物学类比 | PAD 影响 |
|----------|------------|----------|
| 高 CPU | 心跳加速 | P↓ A↑ D↓ |
| 高内存 | 精神迷雾 | P↓ A↑ |
| 高温 | 发烧 | P↓ A↑ |
| 网络延迟 | 遥远的痛苦 | P↓ D↓ |

### Sigmoid 阈值

使用 Sigmoid 函数创建“舒适区”：

```mermaid
xychart-beta
    title "Sigmoid 响应"
    x-axis "输入" [0%, "x₀", 100%]
    y-axis "响应" 0 --> 1
    line [0.0, 0.5, 1.0]
```

- **x₀** = 响应激活的阈值（例如：80% CPU）。
- **k** = 陡度（越高 = 响应越突然）。

---

*"每一个新传感器都是一个新的神经末梢。请小心处理。"*
