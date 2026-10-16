# TinyGPT

从零训练一个小型 Decoder-only 语言模型，用于学习 GPT 架构的完整训练和推理流程。

## 模型配置

| 参数 | 值 |
|------|----|
| 架构 | Decoder-only Transformer（仿 GPT-2）|
| hidden_size | 384 |
| 层数 | 6 |
| attention heads | 8 |
| head_dim | 48 |
| context_len | 256 |
| vocab_size | 50257（tiktoken GPT-2 编码器）|
| 非 Embedding 参数量 | ~10.6M |
| 训练语料 | TinyStories（取前 10 万个故事，约 20M tokens）|
| 训练设备 | Apple M5 Pro，48GB 统一内存，MPS 后端 |

## 文件结构

```
TinyGPT/
├── model.py       # CausalSelfAttention, Block, TinyGPT 类定义
├── train.py       # 数据加载 + 训练循环
├── generate.py    # 加载训练好的模型生成文本
├── requirements.txt
├── myquestions.md # 学习过程中的问答记录
└── tiny_gpt.pt    # 训练好的权重（训练后生成）
```

## 快速开始

```bash
pip install -r requirements.txt

# 训练（首次运行会自动下载数据集，约 2GB）
python train.py

# 生成文本
python generate.py
```

## 架构说明

```
输入 token 序列 (batch, seq=256)
  ↓ Token Embedding (50257, 384) + Position Embedding (256, 384)
  ↓ × 6 层 Block：
  │   Pre-Norm → CausalSelfAttention (8 heads) → 残差
  │   Pre-Norm → FFN (384→1536→384, GELU) → 残差
  ↓ Final LayerNorm
  ↓ LM Head (384→50257)，与 Token Embedding 共享权重
输出 logits (batch, seq, 50257)
```

训练：与右移一位的标签算 Cross-Entropy Loss，反向传播更新全部参数。  
推理：逐步生成，每步取最后位置的 logits 采样下一个 token（top-k + temperature）。

## 训练耗时参考

硬件：Apple M5 Pro，48GB 统一内存，MPS 后端（tok/s ≈ 21,000）

| 配置 | hidden | layers | heads | 语料 | epochs | 总参数 | 非 Embedding 参数 | 训练耗时 |
|------|--------|--------|-------|------|--------|--------|-------------------|----------|
| 理想配置 | 384 | 6 | 8 | 10 万故事（~20M tokens）| 3 | ~30M | ~10.7M | ~10 小时 |
| 演示配置 | 128 | 1 | 4 | 1 万故事（~2M tokens） | 1 | ~6.7M | ~0.23M | ~45 分钟 |

理想配置生成质量更好，演示配置用于快速验证训练流程。

## 训练记录

### 2026-10-15 演示配置首次训练

| 项目 | 值 |
|------|----|
| 日期 | 2026-10-15 |
| 硬件 | Apple M5 Pro，48GB 统一内存，MPS 后端 |
| 开始时间 | 19:10 |
| 结束时间 | ~23:50 |
| 实际耗时 | ~4 小时 40 分钟 |
| 平均 tok/s | ~37,000 |
| 总 steps | 33,778（完成 1 epoch）|
| 最终 loss | 2.33 |

**为什么比预估的 45 分钟长？**

原始估算用的是数据集大小（2M tokens ÷ tok/s），这是错误的。训练的实际计算量远大于数据集大小：

- ChunkDataset 滑动窗口生成了 2,161,822 个训练样本
- 每个样本长 256 tokens，每个 token 都要做 forward + backward
- 实际计算量：33,778 batches × 64 × 256 = **5.53 亿 token 计算量**
- 5.53 亿 ÷ 37,000 tok/s ≈ **4.15 小时**

结论：训练时间应按"总 token 计算量 ÷ tok/s"估算，而不是"数据集大小 ÷ tok/s"。

**tok/s 对硬件对比的价值**

tok/s 直接反映硬件在该模型规模下的实际吞吐，可用于横向对比：

| 硬件 | 估算 tok/s | 同等训练任务耗时 |
|------|-----------|----------------|
| M5 Pro (MPS) | ~37,000 | ~4.7 小时 |
| M5 Pro (CPU) | ~21,000 | ~8.2 小时 |
| RTX 4090 | ~200,000+ | ~45 分钟 |
| A100 80G | ~400,000+ | ~20 分钟 |

M5 Pro CPU 数据为实测值（训练同日短暂切换 CPU 观察所得），MPS 约为 CPU 的 1.76 倍。GPU 估算为同量级小模型的参考值，实际取决于 batch size、精度（fp16/bf16）等设置。
