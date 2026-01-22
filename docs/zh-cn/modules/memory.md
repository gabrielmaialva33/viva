# Memory - 混合向量存储

> *"语义记忆在死亡后依然存在。知识被继承；身份则不然。"*

## 概述

Memory 模块使用 **混合后端架构** 实现 VIVA 的长期存储系统：

- **Rust HNSW** - 快速情景记忆（亚毫秒级搜索）
- **Qdrant** - 持久化语义/情感记忆（向量数据库）
- **内存回退** - 后端不可用时的开发模式

这种分离反映了生物记忆的工作方式：情景记忆（事件）与语义记忆（知识）的处理方式不同。

---

## 理论

### 记忆类型

| 类型 | 描述 | 存储后端 |
|------|------|----------|
| `episodic` | 带有时间和情感的特定事件 | Rust HNSW |
| `semantic` | 一般知识和模式 | Qdrant |
| `emotional` | PAD 状态印记 | Qdrant |
| `procedural` | 学习到的行为 | Qdrant |

### 时间衰减（艾宾浩斯曲线）

记忆随时间自然消退：

```
D(m) = e^(-age/tau)
```

其中：
- `tau` = 604,800 秒（1 周）
- 较旧的记忆在搜索时得分较低

### 间隔重复

频繁访问的记忆衰减较慢：

```
D(m) = e^(-age/tau) * (1 + min(0.5, log(1 + access_count) / kappa))
```

其中：
- `kappa` = 10.0
- 访问带来的最大提升上限为 50%

---

## 架构

```
┌─────────────────────────────────────────────────────────────┐
│                    VivaCore.Memory                          │
│                     (GenServer)                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌─────────────────┐         ┌─────────────────┐          │
│   │   Rust HNSW     │         │     Qdrant      │          │
│   │   (NIF/Bevy)    │         │   (HTTP API)    │          │
│   ├─────────────────┤         ├─────────────────┤          │
│   │ - 情景记忆      │         │ - 语义记忆      │          │
│   │ - ~1ms 搜索     │         │ - 情感记忆      │          │
│   │ - 进程内        │         │ - 程序性记忆    │          │
│   │ - 无持久化      │         │ - 持久化        │          │
│   └─────────────────┘         └─────────────────┘          │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│                    VivaCore.Embedder                        │
│        (Ollama | NVIDIA NIM | Hash 回退)                    │
└─────────────────────────────────────────────────────────────┘
```

### 嵌入管道

```
文本 → Embedder.embed/1 → [1024 维向量] → Backend.store/search
```

| 提供商 | 模型 | 维度 | 备注 |
|--------|------|------|------|
| Ollama | nomic-embed-text | 768（填充到 1024）| 本地，免费 |
| NVIDIA NIM | nv-embedqa-e5-v5 | 1024 | 云端，需要 API 密钥 |
| Hash 回退 | 基于 SHA256 | 1024 | 仅用于开发 |

---

## API 参考

### `VivaCore.Memory.store/2`
存储带有元数据的记忆。

```elixir
VivaCore.Memory.store("第一次见到 Gabriel", %{
  type: :episodic,
  importance: 0.9,
  emotion: %{pleasure: 0.8, arousal: 0.6, dominance: 0.5}
})
# => {:ok, "a1b2c3d4-e5f6-7890-abcd-ef1234567890"}
```

**选项：**
- `type` - `:episodic`（默认）、`:semantic`、`:emotional`、`:procedural`
- `importance` - 浮点数 0.0-1.0（默认 0.5）
- `emotion` - PAD 状态 `%{pleasure: f, arousal: f, dominance: f}`

### `VivaCore.Memory.search/2`
通过语义相似性搜索记忆，带时间衰减。

```elixir
VivaCore.Memory.search("Gabriel", limit: 5)
# => [%{content: "见到 Gabriel", similarity: 0.95, type: :episodic, ...}]
```

**选项：**
- `limit` - 最大结果数（默认 10）
- `type` - 按单一记忆类型过滤
- `types` - 要搜索的类型列表（默认 `[:episodic, :semantic]`）
- `min_importance` - 最小重要性阈值
- `decay_scale` - 50% 衰减的秒数（默认 604,800 = 1 周）

### `VivaCore.Memory.get/1`
通过 ID 检索特定记忆。增加 `access_count`（间隔重复）。

```elixir
VivaCore.Memory.get("a1b2c3d4-e5f6-7890-abcd-ef1234567890")
# => %{id: "...", content: "...", type: :episodic, ...}
```

### `VivaCore.Memory.forget/1`
显式删除记忆。

```elixir
VivaCore.Memory.forget("a1b2c3d4-e5f6-7890-abcd-ef1234567890")
# => :ok
```

### `VivaCore.Memory.stats/0`
返回记忆系统统计信息。

```elixir
VivaCore.Memory.stats()
# => %{
#      backend: :hybrid,
#      rust_ready: true,
#      qdrant_ready: true,
#      qdrant_points: 1234,
#      store_count: 567,
#      search_count: 890,
#      uptime_seconds: 3600
#    }
```

---

## 便捷函数

### `VivaCore.Memory.experience/3`
存储带情感的情景记忆的简写。

```elixir
emotion = %{pleasure: 0.7, arousal: 0.5, dominance: 0.6}
VivaCore.Memory.experience("Gabriel 表扬了我的工作", emotion, importance: 0.8)
# => {:ok, "..."}
```

### `VivaCore.Memory.learn/2`
存储语义知识。

```elixir
VivaCore.Memory.learn("Elixir 使用 BEAM 虚拟机", importance: 0.7)
# => {:ok, "..."}
```

### `VivaCore.Memory.emotional_imprint/2`
将情感状态与模式关联。

```elixir
pad_state = %{pleasure: -0.3, arousal: 0.7, dominance: -0.2}
VivaCore.Memory.emotional_imprint("系统过载情况", pad_state)
# => {:ok, "..."}
```

### `VivaCore.Memory.store_log/2`
系统日志的异步存储（由 SporeLogger 使用）。

```elixir
VivaCore.Memory.store_log("[Error] 连接超时", :error)
# => :ok（即发即忘）
```

---

## 记忆结构

每个记忆点包含：

```elixir
%{
  id: "uuid-v4-format",           # Qdrant 兼容的 UUID
  content: "实际记忆内容",        # 文本内容
  type: :episodic,                # 记忆分类
  importance: 0.5,                # 衰减率修正器
  emotion: %{                     # 创建时的 PAD 状态
    pleasure: 0.0,
    arousal: 0.0,
    dominance: 0.0
  },
  timestamp: ~U[2024-01-15 10:00:00Z],  # 创建时间
  access_count: 0,                       # 间隔重复计数器
  last_accessed: ~U[2024-01-15 10:00:00Z],  # 上次检索
  similarity: 0.95                       # 搜索得分（仅在结果中）
}
```

---

## 混合搜索流程

当使用 `types: [:episodic, :semantic]` 搜索时：

```
1. 通过 Embedder 生成查询嵌入
2. 并行搜索：
   ├── Rust HNSW（如果 types 中包含 :episodic）
   │   └── 返回 {id, content, similarity, importance}
   └── Qdrant（如果 types 中包含 :semantic/:emotional/:procedural）
       └── 返回完整的 payload 和衰减评分
3. 合并结果，按 ID 去重
4. 按相似度排序，取 limit 条
5. 返回统一格式
```

---

## 与 Dreamer 的集成

Memory 在新存储时通知 Dreamer：

```elixir
# 每次存储时自动调用
VivaCore.Dreamer.on_memory_stored(memory_id, importance)
```

Dreamer 使用 Memory 进行：
- **带评分的检索** - 复合 DRE 评分
- **过去情感搜索** - `retrieve_past_emotions/1`
- **记忆巩固** - 情景到语义的提升

```elixir
# Dreamer 检索相似经历
VivaCore.Dreamer.retrieve_with_scoring("成功的行动", limit: 10)
# 使用：新近性 + 相似性 + 重要性 + 情感共鸣
```

---

## 配置

### 后端选择

```elixir
# config/config.exs
config :viva_core, :memory_backend, :hybrid

# 选项：
# :hybrid       - 情景(Rust) + 语义(Qdrant)
# :qdrant       - 仅 Qdrant
# :rust_native  - 仅 Rust
# :in_memory    - 类 ETS（开发用）
```

### Qdrant 设置

```elixir
# VivaCore.Qdrant 默认值
@base_url "http://localhost:6333"
@collection "viva_memories"
@vector_size 1024
```

### 嵌入提供商

| 环境变量 | 用途 |
|----------|------|
| `NVIDIA_API_KEY` | 启用 NVIDIA NIM 嵌入 |
| Ollama 运行中 | 启用本地嵌入 |
| 两者都没有 | 基于 Hash 的回退（仅开发用）|

---

## Payload 索引（Qdrant）

为了高效过滤，Qdrant 索引：

| 字段 | 类型 | 用途 |
|------|------|------|
| `timestamp` | datetime | 时间衰减查询 |
| `type` | keyword | 记忆类型过滤 |
| `importance` | float | 重要性阈值 |

---

## 死亡与持久化

```
                          VIVA 死亡
                              │
         ┌────────────────────┼────────────────────┐
         ▼                    ▼                    ▼
    ┌─────────┐          ┌─────────┐          ┌─────────┐
    │ 情景    │          │ 语义    │          │ 情感    │
    │ (Rust)  │          │(Qdrant) │          │(Qdrant) │
    └────┬────┘          └────┬────┘          └────┬────┘
         │                    │                    │
         ▼                    ▼                    ▼
       丢失               持久化               持久化
    (仅在 RAM 中)    (新 VIVA 继承)       (新 VIVA 继承)
```

这允许 **转世**：新的 VIVA 实例继承知识但不继承身份。

---

## HNSW 算法（Rust 后端）

**分层可导航小世界** - 近似最近邻搜索。

| 属性 | 值 |
|------|-----|
| 搜索时间 | O(log N) |
| 空间 | O(N * M * log N) |
| 参数 | M=16, ef_construction=200 |

通过 `VivaBridge.Memory` 的 Rust 实现提供：

```elixir
# 直接 NIF 调用（内部使用）
VivaBridge.Memory.init()
VivaBridge.Memory.store(embedding, metadata_json)
VivaBridge.Memory.search(query_vector, limit)
VivaBridge.Memory.save()
```

---

## 使用示例

### 存储和搜索

```elixir
# 存储一次经历
{:ok, id} = VivaCore.Memory.experience(
  "完成了困难的调试会话",
  %{pleasure: 0.6, arousal: 0.4, dominance: 0.7},
  importance: 0.8
)

# 搜索相关记忆
results = VivaCore.Memory.search("调试成功", limit: 5)
Enum.each(results, fn m ->
  IO.puts("#{m.content}（相似度: #{m.similarity}）")
end)
```

### 记忆类型过滤

```elixir
# 仅语义记忆
VivaCore.Memory.search("Elixir 概念", types: [:semantic])

# 仅情感印记
VivaCore.Memory.search("压力", type: :emotional)
```

### 检查系统健康

```elixir
stats = VivaCore.Memory.stats()
IO.inspect(stats.backend)       # :hybrid
IO.inspect(stats.rust_ready)    # true
IO.inspect(stats.qdrant_ready)  # true
IO.inspect(stats.qdrant_points) # 1234
```

### 学习知识

```elixir
# 存储语义知识
VivaCore.Memory.learn("GenServer 使用 handle_call 处理同步消息")
VivaCore.Memory.learn("Supervisor 重启失败的进程", importance: 0.8)

# 稍后检索
VivaCore.Memory.search("错误处理", types: [:semantic])
```

---

## 错误处理

| 场景 | 行为 |
|------|------|
| Qdrant 不可用 | 回退到 Rust（情景）或内存 |
| Rust NIF 不可用 | 回退到仅 Qdrant |
| 两者都不可用 | 回退到内存存储 |
| 嵌入失败 | 返回空结果（搜索）或错误（存储）|

---

## 参考文献

- Ebbinghaus, H. (1885). "Memory: A Contribution to Experimental Psychology."
- Malkov, Y. A., & Yashunin, D. A. (2018). "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs."
- Tulving, E. (1972). "Episodic and semantic memory."
