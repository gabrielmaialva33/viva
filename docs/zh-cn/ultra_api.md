# Ultra 推理 API 参考
> *知识图谱与深度推断*

**Ultra** 是"推理引擎"。它使用图神经网络 (GNN) 来推断 VIVA 记忆中缺失的链接并预测因果关系。

## 特性
- **零样本链接预测 (Zero-Shot Link Prediction)**: 可以在没有针对特定事实进行明确训练的情况下猜测 `(主体, 关系, ?)`。
- **叙事嵌入**: 将文本转换为与液态皮层兼容的语义向量。
- **CogGNN**: 认知图神经网络，用于情感推理并与全局工作空间集成。
- **EWC 记忆保护**: 弹性权重巩固，防止灾难性遗忘。
- **Mamba-2 时序处理**: 用于记忆历史的线性时间序列建模。
- **DoRA 微调**: 权重分解低秩适配，用于情感嵌入。

---

## Elixir API (`VivaBridge.Ultra`)

### 核心函数

#### `ping/0`
检查服务可用性。
```elixir
VivaBridge.Ultra.ping()
# 返回: %{"status" => "pong", "loaded" => true}
```

#### `infer_relations/2`
从文本中提取/推断关系。
```elixir
{:ok, relations} = VivaBridge.Ultra.infer_relations("Gabriel", "风扇")
# 返回: [%{head: "Gabriel", relation: "repair", tail: "风扇"}]
```

#### `predict_links/3`
预测三元组的尾部。
```elixir
{:ok, predictions} = VivaBridge.Ultra.predict_links("VIVA", "feels", 10)
# 返回: %{"triples" => [%{head: "VIVA", relation: "feels", tail: "Happy", score: 0.95}, ...]}
```

#### `embed/1`
获取文本的向量嵌入 (384维 MiniLM)。
```elixir
{:ok, vector} = VivaBridge.Ultra.embed("我感觉活着。")
# 返回: {:ok, [0.123, -0.456, ...]} (384 维)
```

#### `find_path/3`
在实体之间查找多跳推理路径。
```elixir
{:ok, path} = VivaBridge.Ultra.find_path("VIVA", "Gabriel", 3)
# 返回: %{"path" => [%{head: "VIVA", relation: "knows", tail: "Gabriel"}]}
```

#### `build_graph/1`
用新记忆更新知识图谱。
```elixir
{:ok, stats} = VivaBridge.Ultra.build_graph(memories)
# 返回: %{"stats" => %{nodes: 150, edges: 300}}
```

---

## CogGNN (认知图神经网络)

受 NeuCFlow (arXiv:1905.13049) 启发的3层 GNN 架构，将意识建模为通过知识图谱的消息传递。

### 架构
```
第1层 (无意识): 通过 GAT 进行背景感知融合 (4个注意力头)
第2层 (有意识): 带情感调制的主动推理 (2个注意力头)
第3层 (注意力): 全局工作空间广播的焦点选择
```

该网络将 PAD 情感状态整合到所有节点表示中，允许情感上下文调制图注意力模式。

### `init_cog_gnn/2`
初始化认知 GNN 用于情感图推理。

**参数:**
- `in_dim` - 输入嵌入维度 (默认: 384 用于 MiniLM)
- `hidden_dim` - 隐藏层维度 (默认: 64)

```elixir
{:ok, true} = VivaBridge.Ultra.init_cog_gnn()
# 使用自定义维度:
{:ok, true} = VivaBridge.Ultra.init_cog_gnn(768, 128)
```

### `propagate/3`
使用情感上下文 (PAD 状态) 运行 GNN 消息传递。

通过知识图谱传播一个思想，使用 PAD 情感状态来调制注意力。返回注意力最高的节点，代表"意识焦点"。

**参数:**
- `concept` - 要传播的概念/思想 (字符串)
- `pad` - PAD 情感状态映射，包含键 `:pleasure`, `:arousal`, `:dominance`
- `top_k` - 返回的最高注意力节点数 (默认: 5)

```elixir
{:ok, result} = VivaBridge.Ultra.propagate(
  "恐惧",
  %{pleasure: -0.3, arousal: 0.7, dominance: -0.2}
)
# 返回:
# {:ok, %{
#   "attended_nodes" => ["mem_fear_1", "mem_anxiety_2"],
#   "attention_scores" => [0.85, 0.72],
#   "updated_concept" => "mem_fear_1"
# }}
```

### `propagate_query/3`
通过知识图谱的查询条件传播。

结合 GNN 注意力和查询相似度进行聚焦检索。在情感上下文中搜索特定概念时很有用。

**参数:**
- `query` - 用于查找相关节点的查询字符串
- `pad` - PAD 情感状态映射
- `top_k` - 结果数量 (默认: 5)

```elixir
{:ok, result} = VivaBridge.Ultra.propagate_query(
  "什么让我快乐?",
  %{pleasure: 0.2, arousal: 0.3, dominance: 0.1}
)
# 返回:
# {:ok, %{
#   "query" => "什么让我快乐?",
#   "results" => [
#     %{"entity" => "音乐", "combined_score" => 0.89, "attention" => 0.75, "similarity" => 0.92},
#     ...
#   ]
# }}
```

### `conscious_focus/0`
获取 GNN 注意力的当前意识焦点。

返回最后一次传播中注意力最高的节点，代表全局工作空间中的当前"意识焦点"。

```elixir
{:ok, focus} = VivaBridge.Ultra.conscious_focus()
# 返回: {:ok, ["mem_fear_1", "mem_anxiety_2", "emotion_sad"]}
```

---

## EWC (弹性权重巩固)

实现记忆保护以防止持续学习过程中的灾难性遗忘 (Kirkpatrick et al. 2017)。

### 核心概念
- **Fisher 信息**: 测量每个嵌入维度的重要性
- **巩固分数**: 记忆的重要程度 (来自 Dreamer 的 DRE 评分)
- **EWC 惩罚**: `L_ewc = lambda/2 * SUM(F_i * (theta_i - theta*_i)^2)`

### 配置
| 参数 | 默认值 | 描述 |
|------|--------|------|
| `lambda_ewc` | 0.4 | 正则化强度 |
| `min_consolidation_score` | 0.7 | 保护所需的最小 DRE 分数 |
| `max_protected_memories` | 1000 | 最大受保护记忆数量 |
| `decay_rate` | 0.01 | 每周期 Fisher 信息衰减 |

### `protect_memory/4`
使用 EWC 保护已巩固的记忆。

在记忆巩固后由 Dreamer 调用。使用 Fisher 信息识别重要的嵌入维度并保护它们。

**参数:**
- `memory_id` - Qdrant 点 ID
- `embedding` - 记忆嵌入 (384个浮点数的列表)
- `related_embeddings` - 相关记忆的嵌入 (列表的列表)
- `consolidation_score` - 来自 Dreamer 的 DRE 分数 (0.0 - 1.0)

```elixir
{:ok, result} = VivaBridge.Ultra.protect_memory(
  "mem_abc123",
  embedding,          # [0.1, -0.2, ...] (384 维)
  related_embeddings, # [[0.1, ...], [0.2, ...]]
  0.85                # 高巩固分数
)
# 如果受保护则返回:
# {:ok, %{
#   "protected" => true,
#   "qdrant_payload" => %{
#     "ewc_fisher_info" => [...],
#     "ewc_baseline_embedding" => [...],
#     "ewc_consolidation_score" => 0.85,
#     "ewc_consolidated_at" => 1705936800.0
#   }
# }}
# 如果未受保护则返回:
# {:ok, %{"protected" => false, "reason" => "score 0.50 < min 0.7"}}
```

### `ewc_penalty/2`
计算新/修改嵌入的 EWC 惩罚。

用于评估新嵌入对受保护记忆的影响程度。

**参数:**
- `embedding` - 要评估的新嵌入 (浮点数列表)
- `affected_memory_ids` - 要检查的特定记忆 (nil = 全部)

```elixir
{:ok, result} = VivaBridge.Ultra.ewc_penalty(new_embedding)
# 返回:
# {:ok, %{
#   "penalty" => 0.0234,
#   "details" => %{
#     "total_memories_checked" => 15,
#     "top_contributions" => [
#       %{"memory_id" => "mem_xyz", "penalty" => 0.012, "score" => 0.9},
#       ...
#     ]
#   }
# }}
```

### `ewc_stats/0`
获取 EWC 管理器统计信息。

```elixir
{:ok, stats} = VivaBridge.Ultra.ewc_stats()
# 返回:
# {:ok, %{
#   "protected_count" => 42,
#   "avg_consolidation_score" => 0.82,
#   "max_consolidation_score" => 0.98,
#   "total_fisher_mean" => 0.45,
#   "lambda_ewc" => 0.4
# }}
```

### `ewc_decay/0`
应用 Fisher 衰减以允许旧记忆的一定可塑性。

应定期调用 (例如在睡眠/梦境周期中)。

```elixir
:ok = VivaBridge.Ultra.ewc_decay()
```

---

## Mamba-2 (时序记忆处理)

使用状态空间模型 (SSM) 实现 O(n) 线性时间序列处理用于记忆历史。是 Transformer 注意力 (O(n^2)) 的替代方案。

### 主要优势
- **线性复杂度**: 可以处理 100+ 个记忆而不会 VRAM 爆炸
- **隐式记忆**: 隐藏状态捕获时间模式
- **高效推理**: 单次通过，无需 KV 缓存

### 架构
```
记忆嵌入 [t-100:t] -> Mamba-2 -> 上下文[60] -> Cortex
```

### 配置
| 参数 | 默认值 | 描述 |
|------|--------|------|
| `d_model` | 384 | 输入维度 (MiniLM) |
| `d_state` | 64 | SSM 状态维度 |
| `n_layers` | 2 | Mamba 层数 |
| `output_dim` | 60 | 上下文向量维度 |
| `max_seq_len` | 128 | 最大序列长度 |

### `init_mamba/3`
初始化 Mamba 时序处理器。

**参数:**
- `d_model` - 输入嵌入维度 (默认: 384)
- `n_layers` - Mamba 层数 (默认: 2)
- `output_dim` - 输出上下文维度 (默认: 60)

```elixir
{:ok, %{"success" => true}} = VivaBridge.Ultra.init_mamba()
# 使用自定义配置:
{:ok, %{"success" => true}} = VivaBridge.Ultra.init_mamba(384, 4, 128)
```

### `process_sequence/2`
通过 Mamba 处理记忆嵌入序列。

接收记忆嵌入列表并返回捕获时间模式的紧凑上下文向量。

**参数:**
- `embeddings` - 嵌入向量列表 `[[e1], [e2], ...]`
- `timestamps` - 可选的时间戳列表用于时间排序

```elixir
embeddings = [
  [0.1, -0.2, ...],  # t-2 时刻的记忆
  [0.3, 0.1, ...],   # t-1 时刻的记忆
  [0.2, 0.0, ...]    # t 时刻的记忆
]

{:ok, result} = VivaBridge.Ultra.process_sequence(embeddings)
# 返回:
# {:ok, %{
#   "context" => [0.5, -0.1, ...],  # 60维上下文向量
#   "metadata" => %{
#     "seq_len" => 3,
#     "d_model" => 384,
#     "output_dim" => 60,
#     "has_timestamps" => false
#   }
# }}
```

### `mamba_stats/0`
获取 Mamba 处理器统计信息。

```elixir
{:ok, stats} = VivaBridge.Ultra.mamba_stats()
# 返回:
# {:ok, %{
#   "available" => true,
#   "d_model" => 384,
#   "n_layers" => 2,
#   "sequences_processed" => 150
# }}
```

**注意:** 如果未安装 `mamba-ssm`，将自动使用指数加权平均的回退实现。

---

## DoRA (权重分解微调)

实现权重分解低秩适配 (DoRA) 用于在 VIVA 的情感语义空间上微调 MiniLM 嵌入模型 (Liu et al., 2024)。

### 核心概念
- **DoRA = LoRA + 权重分解**: 将权重分解为幅度和方向分量
- 比普通 LoRA **训练更稳定**
- **更好地保留**预训练特征
- **约9%可训练参数** (2M / 22M)

### 使用场景
- 将 MiniLM 嵌入适配到 VIVA 的情感词汇
- 对比学习: 相似情感 -> 相似嵌入

### 配置
| 参数 | 默认值 | 描述 |
|------|--------|------|
| `model_name` | `all-MiniLM-L6-v2` | 基础模型 |
| `r` | 8 | LoRA 秩 |
| `lora_alpha` | 16 | LoRA 缩放因子 |
| `lora_dropout` | 0.1 | LoRA 层的 Dropout |
| `use_dora` | true | 启用权重分解 |
| `learning_rate` | 2e-4 | 训练学习率 |
| `temperature` | 0.07 | InfoNCE 温度 |

### `dora_setup/0`
设置 DoRA 微调器并使用适配器初始化模型。

```elixir
{:ok, %{"success" => true}} = VivaBridge.Ultra.dora_setup()
```

### `dora_train/1`
使用对比学习训练模型的情感样本。

**参数:**
- `samples` - 训练样本列表，每个包含:
  - `text` - 输入文本
  - `pad` - PAD 情感状态 `[pleasure, arousal, dominance]`
  - `label` - 可选的分类标签

```elixir
samples = [
  %{text: "今天我好开心!", pad: [0.8, 0.6, 0.4]},
  %{text: "这太令人沮丧了", pad: [-0.5, 0.7, -0.3]},
  %{text: "宁静的早晨", pad: [0.4, -0.2, 0.3], label: "平静"}
]

{:ok, result} = VivaBridge.Ultra.dora_train(samples)
# 返回:
# {:ok, %{
#   "epochs" => 3,
#   "final_loss" => 0.234,
#   "best_loss" => 0.198,
#   "total_steps" => 150
# }}
```

### `dora_encode/1`
使用微调后的模型编码文本。

**参数:**
- `texts` - 要编码的文本列表

```elixir
{:ok, result} = VivaBridge.Ultra.dora_encode(["我很开心", "我很难过"])
# 返回:
# {:ok, %{
#   "embeddings" => [
#     [0.12, -0.34, ...],  # 384 维
#     [0.45, 0.23, ...]
#   ]
# }}
```

### `dora_save/1`
将 DoRA 适配器权重保存到磁盘。

**参数:**
- `path` - 保存权重的目录路径

```elixir
{:ok, %{"success" => true}} = VivaBridge.Ultra.dora_save("/path/to/checkpoints")
```

### `dora_load/1`
从磁盘加载 DoRA 适配器权重。

**参数:**
- `path` - 包含已保存权重的目录路径

```elixir
{:ok, %{"success" => true}} = VivaBridge.Ultra.dora_load("/path/to/checkpoints")
```

### `dora_stats/0`
获取 DoRA 微调统计信息。

```elixir
{:ok, stats} = VivaBridge.Ultra.dora_stats()
# 返回:
# {:ok, %{
#   "model" => "sentence-transformers/all-MiniLM-L6-v2",
#   "use_dora" => true,
#   "rank" => 8,
#   "alpha" => 16,
#   "training" => %{
#     "epochs_completed" => 3,
#     "total_steps" => 150,
#     "best_loss" => 0.198
#   },
#   "model_initialized" => true
# }}
```

---

## 依赖项

### 必需
- **Python 3.9+**
- **sentence-transformers** - 嵌入
- **torch** - PyTorch 后端
- **torch-geometric** - CogGNN 图神经网络
- **numpy** - 数值运算

### 可选 (增强功能)
| 包 | 功能 | 安装 |
|----|------|------|
| `mamba-ssm>=2.0.0` | Mamba-2 时序处理 | `pip install mamba-ssm causal-conv1d>=1.2.0` |
| `peft>=0.10.0` | DoRA 微调 | `pip install peft` |

**注意:** 没有可选包时，将自动使用回退实现。

---

## 参考文献

- **ULTRA**: arXiv:2310.04562 - 知识图谱推理
- **NeuCFlow**: arXiv:1905.13049 - 神经电路架构
- **EWC**: Kirkpatrick et al. 2017 - 克服灾难性遗忘
- **Mamba-2**: Gu & Dao, 2024 - 线性时间序列建模
- **DoRA**: Liu et al., 2024 - 权重分解低秩适配
- **LoRA**: Hu et al., 2021 - LLM 的低秩适配
