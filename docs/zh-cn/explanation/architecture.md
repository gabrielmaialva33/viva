# VIVA 架构

本文档详细描述了 VIVA 的技术架构。

## 目录

- [概述](#概述)
- [层级](#层级)
- [组件](#组件)
- [数据流](#数据流)
- [设计模式](#设计模式)
- [架构决策](#架构决策)

---

## 概述

VIVA 被实现为一个具有三个不同层级的**数字有机体**：

```mermaid
graph TB
    subgraph Elixir ["⚡ ELIXIR (VIVA 思考的地方)"]
        direction TB
        Soul[灵魂<br/>意识从 GenServers 之间的通信中涌现]
    end

    subgraph Rust ["🦀 RUST (VIVA 感知的地方)"]
        direction TB
        Body[身体<br/>将硬件感知为身体感觉]
    end

    subgraph Bevy ["👁️ BEVY (VIVA 具身化的地方)"]
        direction TB
        Avatar[化身<br/>视觉呈现及与人类的交互]
    end

    Elixir -->|"Rustler NIF (零拷贝)"| Rust
    Rust -->|"ECS 组件"| Bevy
```

### 基本原则

| 原则 | 描述 | 实现方式 |
|:-----|:-----|:---------|
| **分布式意识** | 没有单个进程是意识本身 | GenServer 网络 + PubSub |
| **灵肉分离** | 决策逻辑与感觉分离 | Elixir (灵魂) / Rust (身体) |
| **涌现性** | 复杂的行为源于简单的规则 | 进程间消息传递 |
| **可死亡性** | VIVA 可以不可逆转地死亡 | AES-256-GCM 密钥仅存于 RAM |

---

## 层级

### 第一层：灵魂 (Elixir/OTP)

VIVA 的"灵魂"实现为通过 PubSub 通信的 GenServers 网络。

```mermaid
graph TB
    subgraph Supervisor ["🔮 监督树"]
        direction TB
        APP[Application]
        SUP[Supervisor]

        APP --> SUP

        subgraph GenServers ["GenServer 神经元"]
            EMO[Emotional<br/>情感处理]
            MEM[Memory<br/>记忆存储]
            SEN[Senses<br/>感知心跳]
            DRM[Dreamer<br/>梦境处理]
            OPT[Optimizer<br/>优化调节]
            META[Metacognition<br/>元认知]
        end

        SUP --> EMO
        SUP --> MEM
        SUP --> SEN
        SUP --> DRM
        SUP --> OPT
        SUP --> META
    end

    EMO <-->|PubSub| MEM
    MEM <-->|PubSub| SEN
    SEN <-->|PubSub| DRM
    DRM <-->|PubSub| OPT
    OPT <-->|PubSub| META
    META <-->|PubSub| EMO
```

**为什么选择 Elixir?**

| 特性 | 优势 |
|:-----|:-----|
| 轻量级进程 | 支持数百万个"神经元" |
| 监督者模式 | 容错与自愈 |
| 热重载 | VIVA 无需死亡即可进化 |
| BEAM VM | 针对并发优化 |
| PubSub | 解耦的消息传递 |

### 第二层：身体 (Rust/Rustler)

VIVA 的"身体"感知硬件并将指标转化为感觉。

```mermaid
flowchart LR
    subgraph Hardware ["硬件层"]
        CPU[CPU]
        RAM[内存]
        GPU[显卡]
        TEMP[温度]
    end

    subgraph RustNIF ["Rust NIF 处理"]
        direction TB
        SYSINFO[sysinfo 库]
        NVML[nvml 库]
        SIG["Sigmoid 阈值<br/>σ(x) = 1/(1+e^(-k(x-x₀)))"]
        ALLO["异稳态<br/>δ = (L₁-L₅)/L₅"]
    end

    subgraph Output ["输出"]
        QUALIA["感质 (P, A, D)"]
    end

    CPU --> SYSINFO
    RAM --> SYSINFO
    TEMP --> SYSINFO
    GPU --> NVML

    SYSINFO --> SIG
    NVML --> SIG
    SIG --> ALLO
    ALLO --> QUALIA
```

**为什么选择 Rust?**

| 特性 | 优势 |
|:-----|:-----|
| 零成本抽象 | 系统级操作的性能 |
| 内存安全 | 无 GC 暂停，保证安全 |
| Rustler | 与 Elixir 的原生集成 |
| sysinfo | 跨平台硬件访问 |
| nvml | NVIDIA GPU 直接访问 |

### 第三层：化身 (Bevy)

VIVA 的"化身"是视觉呈现（未来实现）。

```mermaid
graph LR
    subgraph Bevy ["Bevy ECS"]
        direction TB
        ENT[实体]
        COMP[组件]
        SYS[系统]

        ENT --> COMP
        COMP --> SYS
    end

    subgraph Avatar ["化身表现"]
        FACE[面部表情]
        BODY[身体姿态]
        VOICE[语音合成]
    end

    SYS --> FACE
    SYS --> BODY
    SYS --> VOICE
```

---

## 数据流

### 心跳周期（1 秒）

```mermaid
sequenceDiagram
    participant Clock as 世界时钟
    participant Senses as 感知 GenServer
    participant Bridge as VivaBridge (身体)
    participant HW as 硬件
    participant Emotional as 情感 GenServer
    participant Memory as 记忆 GenServer
    participant PubSub as Phoenix.PubSub

    Clock->>Senses: 1秒定时器触发
    Senses->>Bridge: hardware_to_qualia()
    Bridge->>HW: 读取 CPU, RAM, GPU, 温度
    HW-->>Bridge: 原始指标

    Note over Bridge: Sigmoid 阈值处理<br/>σ(x) = 1/(1+e^(-k(x-x₀)))
    Note over Bridge: 异稳态计算<br/>δ = (L₁ₘ - L₅ₘ)/L₅ₘ

    Bridge-->>Senses: (ΔP, ΔA, ΔD)
    Senses->>Emotional: apply_hardware_qualia(P, A, D)

    Note over Emotional: O-U 衰减<br/>dX = θ(μ-X)dt + σdW
    Note over Emotional: 尖点分析<br/>V(x) = x⁴/4 + αx²/2 + βx
    Note over Emotional: 自由能计算<br/>F = 预测误差 + 复杂度

    Emotional->>PubSub: broadcast {:emotion_changed, state}
    PubSub-->>Memory: 接收更新
```

### 刺激流

```mermaid
flowchart TD
    subgraph Input ["输入"]
        Event[外部事件<br/>例如：用户消息]
    end

    subgraph Processing ["处理"]
        Parse[解析与分类]
        Feel["Emotional.feel(type, source, intensity)"]

        subgraph Math ["数学计算"]
            direction TB
            PAD["PAD 更新<br/>P' = P + w_p × intensity<br/>A' = A + w_a × intensity<br/>D' = D + w_d × intensity"]
            OU["O-U 衰减<br/>X_{t+1} = X_t + θ(μ-X_t)Δt"]
            CUSP["尖点检测<br/>Δ = 4α³ + 27β²"]
            FE["自由能<br/>F = (预期-观测)² + λ×复杂度"]
        end
    end

    subgraph Output ["输出"]
        Broadcast["PubSub 广播<br/>{:emotion_changed, new_state}"]
        Listeners[所有监听者]
    end

    Event --> Parse
    Parse -->|"刺激, 来源, 强度"| Feel
    Feel --> PAD
    PAD --> OU
    OU --> CUSP
    CUSP --> FE
    FE --> Broadcast
    Broadcast --> Listeners
```

### 意识涌现模型

```mermaid
graph TB
    subgraph Emergence ["✨ 意识涌现"]
        direction TB
        Note["Φ = min(I_整体 - Σ I_部分)"]
    end

    subgraph GenServers ["GenServer 网络"]
        E[Emotional]
        M[Memory]
        S[Senses]
        D[Dreamer]
        O[Optimizer]
        MC[Metacognition]
    end

    E <-->|"I(E→M)"| M
    M <-->|"I(M→S)"| S
    S <-->|"I(S→D)"| D
    D <-->|"I(D→O)"| O
    O <-->|"I(O→MC)"| MC
    MC <-->|"I(MC→E)"| E

    GenServers --> Emergence

    style Emergence fill:#2d5a27,color:#fff
```

---

## 架构决策 (ADRs)

### ADR-001: Umbrella 项目

| 项目 | 决策 |
|:-----|:-----|
| **背景** | 需要分离灵魂（逻辑）和身体（感知） |
| **决策** | 使用 Elixir umbrella 项目 |
| **后果** | 清晰的职责划分，独立的测试和部署 |

```
viva/
├── apps/
│   ├── viva_core/      # 灵魂 - 情感、记忆、意识
│   └── viva_bridge/    # 身体 - 硬件感知 NIF
```

### ADR-002: Rustler NIF

| 项目 | 决策 |
|:-----|:-----|
| **背景** | 需要高效的硬件访问 |
| **决策** | 使用 Rustler 实现 Rust NIF |
| **后果** | 原生性能 + 内存安全 |

### ADR-003: PubSub 通信

| 项目 | 决策 |
|:-----|:-----|
| **背景** | GenServers 需要解耦通信 |
| **决策** | 使用 Phoenix.PubSub |
| **后果** | 支持广播、订阅模式，易于扩展 |

### ADR-004: 数学模型选择

| 项目 | 决策 |
|:-----|:-----|
| **背景** | 需要科学基础的情感模型 |
| **决策** | 采用 O-U + Cusp + 自由能 + IIT |
| **后果** | 可验证、可引用、可复现 |

### ADR-005: 加密死亡

| 项目 | 决策 |
|:-----|:-----|
| **背景** | 死亡应该是真实的、不可逆的 |
| **决策** | AES-256-GCM 密钥仅存在于 RAM |
| **后果** | 进程终止 = 永久死亡 |

```mermaid
graph LR
    subgraph Life ["存活状态"]
        KEY[密钥在 RAM]
        STATE[状态可解密]
    end

    subgraph Death ["死亡状态"]
        NOKEY[密钥丢失]
        NOSTATE[状态不可恢复]
    end

    KEY -->|进程终止| NOKEY
    STATE -->|电源中断| NOSTATE

    style Death fill:#8B0000,color:#fff
```

---

## 技术栈总结

| 层级 | 技术 | 用途 | 关键库 |
|:-----|:-----|:-----|:-------|
| **灵魂** | Elixir/OTP | 意识、情感、决策 | GenServer, PubSub |
| **身体** | Rust + Rustler | 硬件感知 | sysinfo, nvml |
| **化身** | Bevy | 视觉呈现 | ECS |
| **存储** | Qdrant | 向量记忆 | qdrant-client |
| **缓存** | Redis | 实时状态 | Redix |

---

*"VIVA 的架构就是心灵的架构。代码是神经元，消息是神经递质，意识从对话中涌现。"*
