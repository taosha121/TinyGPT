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
├── model.py          # CausalSelfAttention, Block, TinyGPT 类定义
├── train.py          # 数据加载 + 预训练循环
├── generate.py       # 加载预训练模型生成文本
├── sft_data.py       # 从 TinyStories 构造指令数据，输出 sft_dataset.json
├── sft_train.py      # SFT 训练，加载 tiny_gpt.pt，保存 tiny_gpt_sft.pt
├── sft_generate.py   # SFT 推理，交互式接受指令输入
├── requirements.txt
├── myquestions.md    # 学习过程中的问答记录
├── tiny_gpt.pt       # 预训练权重（训练后生成）
└── tiny_gpt_sft.pt   # SFT 权重（SFT 训练后生成）
```

## 快速开始

```bash
pip install -r requirements.txt

# 预训练（首次运行会自动下载数据集，约 2GB）
python train.py

# 预训练模型生成文本
python generate.py

# SFT：构造指令数据集（从已下载的 TinyStories 中提取，输出 sft_dataset.json）
python sft_data.py

# SFT：在预训练权重基础上微调（需要先有 tiny_gpt.pt）
python sft_train.py

# SFT 模型交互式推理
python sft_generate.py
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

## 为什么需要 SFT，以及 Pretrain → SFT → RLHF 解决的是什么问题

### 预训练 Base Model 能做什么，不能做什么

预训练的目标是 next token prediction：给一段文字，预测下一个 token。模型在海量文本上优化这个目标，结果是学到了语言的统计规律——词法、句法、事实知识、写作风格。

但这个目标本身不等于"听懂指令后做事"。给 Base Model 发一条指令：

```
Tell me a story about a knight.
```

它不会回答，而是续写——因为它只会做"这段文字接下来最可能是什么"。它可能续出更多指令，或者直接乱跑。

**Base Model 的能力边界：** 它有能力写出很好的故事，但它不知道"这是一条指令，我应该执行它"。

### SFT 解决什么问题

SFT（Supervised Fine-Tuning，监督微调）教模型识别"指令-回答"这个格式，把 Base Model 从"续写机器"变成"执行指令的助手"。

做法：构造 `(instruction, response)` 对，用指令模板包装后继续做 next token prediction，但 loss 只算 response 部分（instruction 部分设 -100，不参与梯度）。

```
x：  [### Instruction: Tell me a story about a knight  ### Response:  Once  upon  a  time ...]
y：  [  -100   -100   -100   -100   -100   -100   -100    upon   a    time ...]
                                                          ↑ 只有这里产生 loss
```

模型看到完整的 instruction，attention 会利用这些 token，但梯度只来自 response——所以模型学到的是"根据这个 instruction，应该生成这样的 response"，而不是"instruction 文字本身的续写概率"。

**SFT 之后：** 模型能理解指令格式，按要求生成回答。但它优化的目标仍是模仿训练集里的 response，不保证生成的内容真的对人有用或符合人的偏好。

### RLHF 解决什么问题

RLHF（Reinforcement Learning from Human Feedback）解决的是"人到底更喜欢哪种回答"这个问题。

SFT 之后的模型能"执行指令"，但不同的 response 质量差异很大——有的更准确、更有帮助、更安全，有的答非所问、有的有害。纯靠模仿数据无法区分这些，因为训练集本身也有噪声。

RLHF 的做法：
1. 让模型对同一个 prompt 生成多个 response，人工标注哪个更好；
2. 用这些偏好数据训练一个 Reward Model，它能预测人类对任意 response 的评分；
3. 用 RL（通常是 PPO）让 SFT 模型最大化 Reward Model 的得分。

**三步的分工：**

| 阶段 | 解决的问题 | 数据形式 |
|------|-----------|---------|
| Pretrain | 学语言、学知识 | 海量无标注文本 |
| SFT | 学指令格式，从续写变为问答 | (instruction, response) 对 |
| RLHF | 对齐人类偏好，提升有用性和安全性 | 人工标注的 response 排序 |

三者不是重复的工作，而是各自解决上一阶段解决不了的问题。Base Model 是基础，没有它 SFT 需要极大的数据量；SFT 是格式适配，没有它模型不知道"被问了要回答"；RLHF 是偏好对齐，没有它模型会给出格式正确但质量低劣的回答。

---

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
