# Ultra 推理 API 参考
> *知识图谱与深度推断*

**Ultra** 是“推理引擎”。它使用图神经网络 (GNN) 来推断 VIVA 记忆中缺失的链接并预测因果关系。

## 特性
- **零样本链接预测 (Zero-Shot Link Prediction)**: 可以在没有针对特定事实进行明确训练的情况下猜测 `(主体, 关系, ?)`。
- **叙事嵌入**: 将文本转换为与液态皮层兼容的语义向量。

## Elixir API (`VivaBridge.Ultra`)

### `infer_relations/2`
从文本中提取/推断关系。
```elixir
{:ok, relations} = VivaBridge.Ultra.infer_relations("Gabriel 修理了风扇。", ["Gabriel", "风扇"])
# 返回: [%{head: "Gabriel", relation: "repair", tail: "风扇"}]
```

### `predict_link/3`
预测三元组的尾部。
```elixir
{:ok, predictions} = VivaBridge.Ultra.predict_link("VIVA", "feels", ?)
# 返回: ["Happy", "Curious", ...]
```

### `embed/1`
获取文本的向量嵌入 (768 维)。
```elixir
{:ok, vector} = VivaBridge.Ultra.embed("我感觉活着。")
```
