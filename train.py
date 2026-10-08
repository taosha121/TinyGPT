import os
import glob
import time
import math
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from modelscope import GPT2Tokenizer
from modelscope.hub.snapshot_download import snapshot_download

from model import TinyGPT

# ── 超参数 ────────────────────────────────────────────────
HIDDEN      = 384
LAYERS      = 6
HEADS       = 8
CONTEXT_LEN = 256
DROPOUT     = 0.1

BATCH_SIZE  = 64
EPOCHS      = 3
LR          = 3e-4
WARMUP      = 200
LOG_EVERY   = 200
GEN_EVERY   = 1000
N_STORIES   = 100_000      # 取前 10 万个故事

CHECKPOINT  = 'tiny_gpt.pt'


# ── 设备 ─────────────────────────────────────────────────
def get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


# ── 数据集 ────────────────────────────────────────────────
class ChunkDataset(Dataset):
    """
    把一条长 token 序列切成滑动窗口训练块。
    每个样本是 (输入, 标签)，标签是输入右移一位——
    位置 i 的输入对应位置 i+1 的预测目标。
    """
    def __init__(self, data, context_len):
        self.data = data
        self.context_len = context_len

    def __len__(self):
        # DataLoader 调用此方法获取样本总数
        return len(self.data) - self.context_len

    def __getitem__(self, i):
        # DataLoader 每次采样时调用，返回一个训练样本
        chunk = self.data[i : i + self.context_len + 1]
        return chunk[:-1], chunk[1:]


def load_data(tokenizer):
    print('下载数据集（已缓存则跳过）...')
    dataset_dir = snapshot_download(
        'AI-ModelScope/TinyStories',
        repo_type='dataset',
        allow_file_pattern='*train*.parquet',
    )
    files = sorted(glob.glob(os.path.join(dataset_dir, '**/*.parquet'), recursive=True))
    print(f'找到 {len(files)} 个文件')

    eos = tokenizer.eos_token_id
    all_ids = []
    count = 0
    for f in files:
        df = pd.read_parquet(f, columns=['text'])
        for text in df['text']:
            if count >= N_STORIES:
                break
            all_ids.extend(tokenizer.encode(str(text)) + [eos])
            count += 1
        if count >= N_STORIES:
            break
        print(f'  {os.path.basename(f)}: {count:,} 个故事')

    data = torch.tensor(all_ids, dtype=torch.long)
    print(f'共 {count:,} 个故事，{len(data):,} tokens')
    return data


# ── 学习率调度：线性 warmup ───────────────────────────────
def get_lr_scale(step):
    if step < WARMUP:
        return step / WARMUP
    return 1.0


# ── 训练 ──────────────────────────────────────────────────
def train():
    device = get_device()
    print(f'设备：{device}')

    tokenizer = GPT2Tokenizer.from_pretrained('AI-ModelScope/gpt2')
    data = load_data(tokenizer)

    dataset = ChunkDataset(data, CONTEXT_LEN)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    print(f'数据集：{len(dataset):,} 块，{len(loader):,} 个 batch/epoch')

    model = TinyGPT(
        vocab_size  = tokenizer.vocab_size,
        hidden      = HIDDEN,
        layers      = LAYERS,
        heads       = HEADS,
        context_len = CONTEXT_LEN,
        dropout     = DROPOUT,
    ).to(device)

    total  = sum(p.numel() for p in model.parameters())
    no_emb = total - model.tok_emb.weight.numel()
    print(f'总参数：{total/1e6:.1f}M，非 Embedding：{no_emb/1e6:.1f}M')

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr_scale)
    prompt_ids = tokenizer.encode('Once upon a time')

    global_step = 0
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for x, y in loader:
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            global_step += 1

            if global_step % LOG_EVERY == 0:
                elapsed = time.time() - t0
                avg_loss = epoch_loss / LOG_EVERY
                tok_per_sec = LOG_EVERY * BATCH_SIZE * CONTEXT_LEN / elapsed
                print(f'step {global_step:5d} | loss {avg_loss:.4f} | {tok_per_sec:,.0f} tok/s')
                epoch_loss = 0.0
                t0 = time.time()

            if global_step % GEN_EVERY == 0:
                model.eval()
                inp = torch.tensor([prompt_ids], device=device)
                out = model.generate(inp, max_new_tokens=60)
                print('\n[生成样本]', tokenizer.decode(out[0].tolist()), '\n')
                model.train()

    torch.save(model.state_dict(), CHECKPOINT)
    print(f'模型已保存到 {CHECKPOINT}')


if __name__ == '__main__':
    train()
