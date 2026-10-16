# 学习笔记：从零训练 TinyGPT

记录学习过程中提出的问题和答案，按主题分类。

---

## Tokenizer

**Q: `GPT2Tokenizer.from_pretrained('AI-ModelScope/gpt2')` 加载的是 tokenizer.json 吗？**

不是。`GPT2Tokenizer` 是"慢速"tokenizer 类，加载的是 `vocab.json` + `merges.txt`。
`tokenizer.json` 是给 `AutoTokenizer`（快速版）用的。两者编码结果完全一样，快速版用 Rust 实现，速度快 5-10 倍。

---

**Q: tokenizer.encode 的输出是什么？**

一个整数列表，每个整数是一个 token 的 id：
```python
tokenizer.encode("Once upon a time")
# → [7454, 2402, 257, 640]
```

---

**Q: tokenizer 都是 BPE 工作原理吗？中文 tokenizer 也一样？**

是的，现代主流 tokenizer 都是 BPE 或其变体。中文 tokenizer（如 Qwen）用 byte-level BPE，每个汉字先转成 UTF-8 字节（每字 3 字节），再做合并操作，因此 merges.txt 里看不到汉字，只有字节的 Unicode 映射符号。

---

**Q: vocab.json、merges.txt、tokenizer.json、tokenizer_config.json 四个文件各自干什么？**

| 文件 | 作用 |
|------|------|
| `vocab.json` | token 字符串 → id 的对照表（老格式） |
| `merges.txt` | BPE 合并规则，按顺序执行（老格式） |
| `tokenizer.json` | 以上两者的打包版 + 分词算法参数，现代格式 |
| `tokenizer_config.json` | 用哪个 Python 类、特殊 token、chat template、最大长度 |

`tokenizer.json` + `tokenizer_config.json` 两个文件已经足够，`vocab.json` 和 `merges.txt` 是向后兼容的冗余备份。

---

## 数据集与预处理

**Q: parquet 是什么格式？**

列式存储的二进制文件格式。和 CSV 相比文件更小（内置压缩）、读取更快（只加载需要的列）。HuggingFace / ModelScope 上大部分数据集用这个格式存储，用 `pd.read_parquet()` 读取。发音：par-KAY（法语，结尾 t 不发音）。

---

**Q: TinyStories 三个文件的关系？**

- `TinyStories_all_data`：train + validation 合并的完整数据
- `TinyStories-train`：只有训练集，是 all_data 的子集
- `TinyStories-v2-gpt4`：GPT-4 重新生成的更高质量版本

训练只需要 train 文件。

---

**Q: all_ids 最后是什么？**

把 10 万个故事的 token id 全部首尾相接，中间用 `eos` 分隔，变成一个约 20M 长的一维 tensor：
```
[故事1 token ids] [eos] [故事2 token ids] [eos] ...
```

---

**Q: ChunkDataset 是干什么的？第一个块为什么不是 256 到 512？**

ChunkDataset 用**滑动窗口**把长序列切成训练块，不是不重叠的切块：
- 索引 0 → token `[0, 255]`
- 索引 1 → token `[1, 256]`
- 索引 2 → token `[2, 257]`

20M tokens 能产生 20M 个训练起点，比不重叠切块（只有 ~85K 块）多 250 倍数据。DataLoader 用 `shuffle=True` 随机采样，不会连续取相邻的重叠块。

---

**Q: `return chunk[:-1], chunk[1:]` 返回了什么？**

chunk 是长度 257 的序列，拆成两个 256 长的序列：
```
chunk[:-1] = [t0, t1, ..., t255]   ← 输入 x
chunk[1:]  = [t1, t2, ..., t256]   ← 标签 y（右移一位）
```
每个位置的对应：输入 t0 → 预测 t1，输入 t0~t1 → 预测 t2，以此类推。

---

**Q: `__init__`、`__len__`、`__getitem__` 这三个方法没人调用，为什么要定义？**

PyTorch `Dataset` 的约定接口，`DataLoader` 内部会调用：
- `__init__`：`ChunkDataset(all_ids, CONTEXT_LEN)` 这行就在调用它
- `__len__`：DataLoader 调用来知道总样本数，控制 epoch 边界
- `__getitem__(i)`：DataLoader 每次采样时调用，传入随机索引 i

这是 Python 的鸭子类型——只要实现了这三个方法，DataLoader 就认它是合法数据集。

---

## 模型架构

**Q: hidden 代表的是隐藏层的数量吗？embedding 的维度需要一致吗？**

`hidden` 不是层数，是**隐向量的维度**（每个 token 在模型里被表示成多少维的向量）。层数是 `layers=6`。叫 hidden 是历史遗留，来自早期神经网络的"隐藏层"概念，也叫 `d_model`、`hidden_size`、`n_embd`，含义相同。

Embedding 的维度必须和 `hidden` 一致，因为 Embedding 查出来的向量要直接输入后续所有层，维度必须贯穿全程。

---

**Q: head 是什么概念？**

把 embedding 维度切成几份，每份独立做一次 Attention。我们 `hidden=384`，`heads=8`，每个 head 分到 `head_dim=48` 维。

好处：每个 head 用不同的 W_Q/W_K/W_V 投影，学到不同维度的关系（句法依存、指代、相邻搭配等），最后 concat 合并所有视角，表达能力比单一大 head 更强。计算量基本不变。

---

**Q: Pre-Norm 是什么？LM Head 是什么？**

**Pre-Norm**：LayerNorm 放在 Attention/FFN 之前（原始 Transformer 是之后）。训练更稳定，现代大模型普遍采用。
```python
x = x + self.attn(self.ln1(x))   # 先归一化再 attention
x = x + self.mlp(self.ln2(x))    # 先归一化再 FFN
```

**LM Head**：最后一层，把 384 维隐向量映射到 50257 个词表 logits，softmax 后得到概率分布。训练时用于算 loss，推理时取概率最大的词。本项目与 Token Embedding 共享权重，节省 ~19M 参数。

---

**Q: LayerNorm 是处理数据还是实现了一个层？为什么要传入 hidden？**

是一个**可学习的层**，不只是纯计算函数。需要传入 `hidden=384` 是因为它要为每个维度学一对参数 γ 和 β（各 384 个）。

计算过程：先标准化（均值=0，方差=1），再逐维度乘 γ 加 β，让模型把分布调整到它需要的形态。

---

**Q: Block 里的两个 LayerNorm 是为了什么？**

防止深层网络里各维度数值差异过大，导致梯度不稳定。Attention 和 FFN 是两个独立子层，各自的输入分布独立控制，所以各用一个。

---

**Q: FFN 是什么结构？只有两层吗？**

就是最普通的全连接网络，两层 Linear + GELU 激活：
```
384 → 1536 → 384
```
升维到 1536 再降回来是为了给模型更大的中间表达空间，4 倍是原始 Transformer 论文的惯例。每个 token 独立经过 FFN，不跨位置交互。

---

**Q: 为什么要有 FFN？**

Attention 本质是加权求和，是线性操作。FFN 里的 GELU 激活引入非线性，让模型能表达更复杂的函数关系。分工上：Attention 负责 token 间的信息交换，FFN 负责每个 token 自身的知识提炼。研究发现 FFN 里存储了大量事实性知识。

---

**Q: Dropout 是干什么的？**

训练时随机把 10% 的神经元输出清零，防止模型过度依赖固定的神经元组合（过拟合）。推理时（`model.eval()`）自动关闭，所有神经元正常工作。

---

**Q: Block 和 CausalSelfAttention 为什么都需要 forward 方法？**

PyTorch `nn.Module` 的规定：继承它的类必须实现 `forward`，定义数据怎么流过这个层。调用时不直接写 `.forward()`，而是 `self.attn(x)`，PyTorch 的 `__call__` 自动触发 `forward`，同时处理 hook、梯度记录等内部逻辑。

---

**Q: 训练好的模型是个 .pt 文件，开源模型为什么文件结构不同？**

`.pt` 存的是 `state_dict`，所有参数的字典（key=参数名，value=张量）。开源模型多了：

| 文件 | 作用 |
|------|------|
| `config.json` | 模型结构超参数，加载时不需要原始代码 |
| `.safetensors` | 更安全更快的权重格式，等价于 `.pt` |
| `generation_config.json` | 生成默认参数 |

关键差别：没有 `config.json`，加载时必须有 `TinyGPT` 类的代码。大模型还会把权重拆成多个分片（单文件太大）。

---

## 数学基础

**Q: 归一化是什么数学概念？**

把一组数缩放到统一的范围或分布。常见两种：

**Min-Max**：压缩到 [0, 1]：`(x - min) / (max - min)`

**标准化（Z-score）**：均值=0，方差=1：`(x - μ) / σ`
- 减均值：把数据平移到以 0 为中心
- 除标准差：把离散程度压缩到统一尺度

LayerNorm 用的是标准化，之后再乘可学习的 γ、加 β。

---

**Q: 方差和标准差怎么算？**

```
数据：[10, 20, 30, 40, 50]，均值 = 30

方差 = [(10-30)² + (20-30)² + (30-30)² + (40-30)² + (50-30)²] / 5
     = [400 + 100 + 0 + 100 + 400] / 5 = 200

标准差 = √200 ≈ 14.14
```

方差衡量离散程度，标准差开根号是为了让单位和原始数据一致。

---

## 训练与推理

**Q: 一个 batch 是 64 个 256 窗口吗？**

对。shape：
```
x: (64, 256)   ← 64 个样本，每个 256 个 token
y: (64, 256)   ← 对应标签，右移一位
```
一次前向传播产生 64×256 = 16,384 个预测，算 16,384 个 Cross-Entropy 取平均。

---

**Q: ChunkDataset 和 Causal Mask 的关系是什么？**

ChunkDataset 负责构造训练数据（输入+右移标签），Causal Mask 负责在模型内部防止偷看答案。两者配合实现了"用一次前向传播同时完成 256 个预测任务"：

```
位置 0：看到 [t0]         → 预测 t1
位置 1：看到 [t0, t1]     → 预测 t2
...
位置 255：看到 [t0~t255]  → 预测 t256
```

---

**Q: 如果输入中文，模型会怎么处理？**

GPT-2 tokenizer 是 byte-level BPE，能处理任意字节，所以中文不会报错（每个汉字变成约 3 个 byte token）。但模型只在英文数据上训练过，这些 byte token 对应的 embedding 从没被更新过（还是随机初始值），所以输出会是不相关的英文或乱码。训练中文模型需要中文语料 + 中文 tokenizer 配套。

---

**Q: Token Embedding 的查找表 (50257, 384) 是训练出来的吗？**

对，从随机矩阵（`std=0.02` 正态分布）开始，和模型一起训练。每次反向传播，被这个 batch 用到的 token 对应的行会被更新。训练完后，语义相近的词在 384 维空间里距离自然变近，不是人工设计的，是通过"预测下一个词"这个任务被迫学出来的。

---

## 神经网络基础

**Q: 神经网络里的"隐藏层"和"hidden size"是什么？**

hidden size 是每一层的节点数，不是层数。层数叫 `num_layers`，两者是独立的超参数。叫 hidden 是历史遗留，来自早期神经网络的"隐藏层"概念。在 Transformer 里更准确的叫法是 `d_model`、`hidden_size`、`n_embd`，含义相同。

---

**Q: 神经网络实际上没有"神经元"对象，隐藏层就是个矩阵，对吗？**

对。"神经元"只是借用生物学比喻，代码里什么神经元对象都没有，就是矩阵乘法：`x @ W + b`。图里每条连线对应矩阵里的一个权重，隐藏层就是一个权重矩阵。输入层的节点不是神经元，只是数据的入口，没有权重，不做计算。

---

**Q: 隐藏层为什么要设计多层？**

单层只能学一种组合关系，多层可以逐层提取更抽象的特征——第一层学边缘，第二层学形状，第三层学五官，第四层学人脸。每一层在上一层基础上组合出更高级的概念，这是单层网络做不到的。

---

## 多头注意力

**Q: 为什么要把 384 维切成 8 个 head？切了就能学到不同模式吗？**

切分本身不能保证学到不同模式，关键在于每个 head 有自己独立的 Q/K/V 权重矩阵。8 个 head 的权重从随机初始化开始各自独立，训练路径不同，最终自然学到不同的关注模式（句法、指代、相邻搭配等）。计算量几乎不变（384 切成 8×48），但表达能力更强。

---

**Q: Q @ K.T 算出来的是什么？为什么还要乘 V？**

`Q @ K.T` 得到 256×256 的相似度矩阵，表示每对 token 之间的关注程度，但不携带任何内容信息，只是一张权重表。

- Q 和 K 决定"关注谁"
- V 决定"关注了之后取什么内容"

`scores @ V` 才是真正的信息提取——每个 token 的输出是所有 token 的 V 按关注程度加权求和，融合了上下文信息。

---

**Q: K 为什么要 transpose(-2, -1)？**

为了让矩阵乘法维度对上。q 和 k 形状都是 `(64, 8, 256, 48)`，直接乘内维不匹配。k 转置最后两维变成 `(64, 8, 48, 256)`，乘完得到 `(64, 8, 256, 256)`，即 256 个 token 两两之间的得分矩阵。

---

**Q: proj（输出投影）是干什么的？**

8 个 head concat 后各自的 48 维仍是割裂的，proj 是一个 384×384 的矩阵乘法，输出的每一维都是输入全部 384 维的加权组合，让 8 个 head 的结果第一次互相混合。

---

## Block 结构

**Q: attention 之后为什么还要接 FFN（MLP）？**

attention 本质是加权求和，是线性操作，只能表达线性关系。FFN 里的 GELU 激活引入非线性，让模型能学到更复杂的变换。分工：attention 负责收集信息（跨 token 交互），FFN 负责消化信息（单 token 非线性变换）。研究发现 FFN 还存储了大量事实性知识（如"Paris 是 France 的首都"）。

---

## 生成（推理）

**Q: generate 里为什么用 multinomial 随机采样，而不是直接取概率最高的词？**

直接取最高的词叫贪心采样，每次输出完全一样，容易陷入重复循环。`multinomial` 按概率分布随机采样，概率高的词被选中机会大，但低概率词也有机会，生成结果更自然多样。

top-k 先过滤掉概率最低的词，只保留前 40 个候选，避免采到完全不合理的词。temperature 控制概率分布的尖锐程度，低温更保守，高温更随机。

**采样流程举例**（词表简化为 5 个词，top-k=3，temperature=0.8）：
```
logits:        [the:3.0,  a:2.0,  cat:1.0,  dog:0.5,  xyz:-2.0]
÷ temperature: [the:3.75, a:2.50, cat:1.25, dog:0.63, xyz:-2.50]
top-3 后:      [the:3.75, a:2.50, cat:1.25, dog:-inf, xyz:-inf]
softmax 概率:  [the:0.67, a:0.24, cat:0.09, dog:0.00, xyz:0.00]
multinomial → 67% 选 "the"，24% 选 "a"，9% 选 "cat"
```

---

## 参数统计

**Q: train.py 打印的"总参数"和"非 Embedding 参数"有什么区别，为什么要单独列非 Embedding？**

演示配置的数字：总参数 6.7M，非 Embedding 0.2M。

差值（约 6.5M）几乎全是 Token Embedding 表：50257 个词 × 128 维 = 6.43M。

Embedding 训练完之后确实能编码语义关系（"king - man + woman ≈ queen" 就是从 embedding 向量空间里算出来的），说它是"查找表"只是描述结构，不是说它没有语义信息。

单独列非 Embedding 的原因：Embedding 的参数量由 tokenizer 词表大小决定（50257），不是可以调节的旋钮。想让模型更强，调的是 hidden、layers、heads，这些改动全体现在非 Embedding 参数上。

对比两种配置：
- 演示配置（layers=1）：非 Embedding 0.2M
- 理想配置（layers=6）：非 Embedding ~10.7M

两者 Embedding 参数几乎一样大，差别全在非 Embedding，所以它更能反映"用了多少计算资源来理解上下文"。

另外，Embedding 只负责把 token id 映射成初始向量，token 之间的关系是后面 Attention 层计算出来的，不是 Embedding 直接算的。

---

## 模型能力边界

**Q: 训练完的 TinyGPT，给它中文 prompt 或英文指令会怎样？**

两种情况都不会理解意思。

**中文**：GPT-2 tokenizer 能处理中文字节，不会报错，但这些 token 对应的 embedding 从没被训练过（还是随机初始值），输出会是不相关的英文或乱码。

**英文指令**（如 "Please give me a story about..."）：这个模型只做了一件事——预测下一个词，完全不知道"指令"是什么概念。它会接着续写，但续写的内容是和训练数据（TinyStories 儿童故事）分布最像的内容，和指令没有任何关系。

能理解并遵循指令，需要预训练之后额外的两个阶段：
1. **SFT（监督微调）**：用指令-回答对继续训练
2. **RLHF（人类反馈强化学习）**：用人类评分优化输出质量

ChatGPT / Claude 都是在预训练模型基础上加了这两步，才有了"理解指令"的能力。
