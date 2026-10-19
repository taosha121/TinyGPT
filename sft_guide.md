# TinyGPT SFT 指南

用已训练好的 TinyGPT（儿童故事续写模型）做一次 SFT，让它学会理解 "Tell me a story about..." 这类指令。

**目标**：从 `tiny_gpt.pt`（只会续写）出发，微调后可以执行指令，保存为 `tiny_gpt_sft.pt`。

**需要新建的三个文件**：

| 文件 | 作用 |
|------|------|
| `sft_data.py` | 从 TinyStories 构造指令数据，输出 `sft_dataset.json` |
| `sft_train.py` | SFT 训练，三处关键改动 vs `train.py` |
| `sft_generate.py` | 推理脚本，接受用户指令输入 |

`model.py` 不需要改动：架构不变，`F.cross_entropy` 默认 `ignore_index=-100`，天然支持 loss mask。

---

## 为什么要加载预训练权重而不是从头训练？

从头训练：模型随机初始化，需要先学会语言，再学会指令格式，数据量需求极大。

加载预训练权重：模型已经学会了写儿童故事，SFT 只需要教它"指令格式"这一件事，数据量少得多（几千条足够）。

---

## Loss Mask 的原理

SFT 和预训练的训练目标相同（next token prediction），区别只在于 instruction 部分不参与 loss：

```
输入序列 x：  [### Instruction:  Tell me about a knight  ### Response:  Once  upon  a  time ...]
                ←──────── instruction tokens ────────→                 ←── response tokens ──→

targets y：   [  -100      -100      -100      -100      -100          upon   a    time  ...]
                ← cross_entropy 自动跳过 ignore_index=-100 的位置 →    ← 只有这里产生 loss →
```

instruction tokens 依然在 x 里，response 的每个 token 预测时都会 attend 到它们。loss 只来自 response 部分，但梯度通过 attention 机制间接影响处理 instruction 的权重——模型因此学会了"根据 instruction 内容生成对应 response"。

---

## 三处关键改动（vs `train.py`）

```
train.py                          sft_train.py
────────────────────────────────────────────────────────
随机初始化                    →   加载 tiny_gpt.pt 预训练权重
纯文本 chunk，全部计算 loss   →   指令模板格式，instruction 部分 -100
LR = 3e-4                     →   LR = 3e-5（低一个数量级，防止破坏已有权重）
```

---

## sft_data.py

从 TinyStories 构造指令数据集，输出 `sft_dataset.json`。

运行：`python sft_data.py`

```python
import re
import json
import glob
import os
import random
import pandas as pd
from modelscope.hub.snapshot_download import snapshot_download

TEMPLATES = [
    "Tell me a story about {topic}.",
    "Write a short story about {topic}.",
    "I want to hear a story about {topic}.",
    "Make up a story where the main character is {topic}.",
    "Can you write a bedtime story about {topic}?",
    "Create a story featuring {topic}.",
    "Write a children's story about {topic}.",
    "Once there was {topic}. Tell me their story.",
]

N_STORIES = 5000
OUTPUT    = 'sft_dataset.json'


def extract_topic(text):
    """从故事第一句提取主角名字，或回落到 a/an + 名词 模式。"""
    first_sent = text.split('.')[0]
    names = re.findall(r'\b[A-Z][a-z]+\b', first_sent)
    if names:
        return names[0]
    nouns = re.findall(r'\b(?:a|an)\s+(\w+)\b', first_sent, re.IGNORECASE)
    if nouns:
        return nouns[0]
    return None


def main():
    dataset_dir = snapshot_download(
        'AI-ModelScope/TinyStories',
        repo_type='dataset',
        allow_file_pattern='*train*.parquet',
    )
    files = sorted(glob.glob(os.path.join(dataset_dir, '**/*.parquet'), recursive=True))

    data = []
    for f in files:
        if len(data) >= N_STORIES:
            break
        df = pd.read_parquet(f, columns=['text'])
        for text in df['text']:
            if len(data) >= N_STORIES:
                break
            topic = extract_topic(str(text))
            if not topic:
                continue
            instruction = random.choice(TEMPLATES).format(topic=topic)
            data.append({'instruction': instruction, 'response': str(text)})

    with open(OUTPUT, 'w') as f:
        json.dump(data, f, indent=2)
    print(f'生成 {len(data)} 条指令数据 → {OUTPUT}')


if __name__ == '__main__':
    main()
```

---

## sft_train.py

在 `tiny_gpt.pt` 基础上做 SFT。

运行：`python sft_train.py`

```python
import json
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import tiktoken
from model import TinyGPT

with open('config.json') as f:
    cfg = json.load(f)

HIDDEN      = cfg['hidden']
LAYERS      = cfg['layers']
HEADS       = cfg['heads']
CONTEXT_LEN = cfg['context_len']
DROPOUT     = cfg['dropout']
BATCH_SIZE  = cfg['batch_size']
EPOCHS      = 3
LR          = 3e-5           # ← 改动 3：比预训练低一个数量级
LOG_EVERY   = cfg['log_every']
CHECKPOINT  = 'tiny_gpt_sft.pt'  # 保存新文件，不覆盖预训练权重

PROMPT_TEMPLATE = "### Instruction:\n{instruction}\n\n### Response:\n"


def get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


class SFTDataset(Dataset):
    """
    每个样本是一条指令-回答对，格式：
        x       = [instr tokens..., resp tokens...]  (去掉最后一个 token)
        targets = [-100, ..., -100, resp tokens...]  (instruction 部分设 -100)

    loss 只在 targets != -100 的位置计算（F.cross_entropy 默认 ignore_index=-100）。
    """
    def __init__(self, path, tokenizer, context_len):
        with open(path) as f:
            raw = json.load(f)

        self.samples = []
        eos = tokenizer.eot_token

        for item in raw:
            instr_text = PROMPT_TEMPLATE.format(instruction=item['instruction'])
            resp_text  = item['response']

            instr_ids = tokenizer.encode(instr_text)
            resp_ids  = tokenizer.encode(resp_text) + [eos]

            ids = instr_ids + resp_ids
            k   = len(instr_ids)

            x        = ids[:-1]           # 输入：去掉最后一个 token
            y        = ids[1:]            # 目标：右移一位

            # ← 改动 2：instruction 部分的位置设 -100，不参与 loss
            y_masked = [-100] * k + y[k:]

            # 截断到 context_len
            x        = x[:context_len]
            y_masked = y_masked[:context_len]

            # 过滤掉 response 部分被截断后全为 -100 的样本
            if all(t == -100 for t in y_masked):
                continue

            self.samples.append((
                torch.tensor(x,        dtype=torch.long),
                torch.tensor(y_masked, dtype=torch.long),
            ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]


def collate(batch):
    """把不同长度的样本 pad 到同一长度，padding 位置的 target 设 -100。"""
    xs, ys  = zip(*batch)
    max_len = max(x.size(0) for x in xs)
    x_pad   = torch.zeros(len(xs), max_len, dtype=torch.long)
    y_pad   = torch.full((len(xs), max_len), -100, dtype=torch.long)
    for i, (x, y) in enumerate(zip(xs, ys)):
        n = x.size(0)
        x_pad[i, :n] = x
        y_pad[i, :n] = y
    return x_pad, y_pad


def train():
    device    = get_device()
    tokenizer = tiktoken.get_encoding('gpt2')
    print(f'设备：{device}')

    model = TinyGPT(
        vocab_size  = tokenizer.n_vocab,
        hidden      = HIDDEN,
        layers      = LAYERS,
        heads       = HEADS,
        context_len = CONTEXT_LEN,
        dropout     = DROPOUT,
    ).to(device)

    # ← 改动 1：加载预训练权重，在已有语言能力上继续微调
    model.load_state_dict(torch.load('tiny_gpt.pt', map_location=device))
    print('已加载预训练权重 tiny_gpt.pt')

    dataset = SFTDataset('sft_dataset.json', tokenizer, CONTEXT_LEN)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                         collate_fn=collate, drop_last=False)
    print(f'SFT 数据集：{len(dataset)} 条，{len(loader)} 个 batch/epoch')

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.1)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        t0 = time.time()

        for step, (x, y) in enumerate(loader, 1):
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            if step % LOG_EVERY == 0:
                avg = total_loss / LOG_EVERY
                print(f'epoch {epoch+1} step {step:4d} | loss {avg:.4f} | {time.time()-t0:.1f}s')
                total_loss = 0.0
                t0 = time.time()

    torch.save(model.state_dict(), CHECKPOINT)
    print(f'SFT 模型已保存：{CHECKPOINT}')


if __name__ == '__main__':
    train()
```

---

## sft_generate.py

接受用户指令输入，推理并输出故事。

运行：`python sft_generate.py`

```python
import json
import torch
import tiktoken
from model import TinyGPT

PROMPT_TEMPLATE = "### Instruction:\n{instruction}\n\n### Response:\n"

with open('config.json') as f:
    cfg = json.load(f)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def main():
    device    = get_device()
    tokenizer = tiktoken.get_encoding('gpt2')

    model = TinyGPT(
        vocab_size  = tokenizer.n_vocab,
        hidden      = cfg['hidden'],
        layers      = cfg['layers'],
        heads       = cfg['heads'],
        context_len = cfg['context_len'],
    ).to(device)
    model.load_state_dict(torch.load('tiny_gpt_sft.pt', map_location=device))
    model.eval()
    print('SFT 模型已加载，输入 q 退出\n')

    while True:
        instruction = input('Instruction: ').strip()
        if instruction.lower() == 'q':
            break
        prompt  = PROMPT_TEMPLATE.format(instruction=instruction)
        ids     = tokenizer.encode(prompt)
        inp     = torch.tensor([ids], device=device)
        out     = model.generate(inp, max_new_tokens=120, temperature=0.8, top_k=40)
        response_ids = out[0].tolist()[len(ids):]
        print(f'\nResponse: {tokenizer.decode(response_ids)}\n')


if __name__ == '__main__':
    main()
```
