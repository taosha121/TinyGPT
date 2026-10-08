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
| vocab_size | 50257（GPT-2 tokenizer）|
| 非 Embedding 参数量 | ~10.6M |
| 训练语料 | TinyStories（取前 10 万个故事，约 20M tokens）|
| 训练设备 | Apple Silicon MPS |

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
