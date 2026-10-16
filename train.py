import os
import glob
import time
import math
import json
import torch
import torch.nn as nn
import pandas as pd
import tiktoken
from torch.utils.data import Dataset, DataLoader
from modelscope.hub.snapshot_download import snapshot_download

from model import TinyGPT

# ── 超参数（从 config.json 读取）────────────────────────────
# 理想配置 hidden=384/layers=6/heads=8/n_stories=100000/epochs=3，训练约 10 小时
# 当前为演示配置，训练约 45 分钟
with open('config.json') as f:
    cfg = json.load(f)

HIDDEN      = cfg['hidden']
LAYERS      = cfg['layers']
HEADS       = cfg['heads']
CONTEXT_LEN = cfg['context_len']
DROPOUT     = cfg['dropout']
BATCH_SIZE  = cfg['batch_size']
EPOCHS      = cfg['epochs']
LR          = cfg['lr']
WARMUP      = cfg['warmup']
LOG_EVERY   = cfg['log_every']
GEN_EVERY   = cfg['gen_every']
N_STORIES   = cfg['n_stories']
CHECKPOINT  = cfg['checkpoint']


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

    eos = tokenizer.eot_token
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

    tokenizer = tiktoken.get_encoding('gpt2')
    data = load_data(tokenizer)

    dataset = ChunkDataset(data, CONTEXT_LEN)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    print(f'数据集：{len(dataset):,} 块，{len(loader):,} 个 batch/epoch')

    model = TinyGPT(
        vocab_size  = tokenizer.n_vocab,
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
        # model.train() 来自 nn.Module，切换到训练模式：Dropout 随机丢弃神经元（防止过拟合）
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
                # model.eval() 切换到推理模式：关闭 Dropout，所有神经元正常工作
                model.eval()
                inp = torch.tensor([prompt_ids], device=device)
                out = model.generate(inp, max_new_tokens=60)
                print('\n[生成样本]', tokenizer.decode(out[0].tolist()), '\n')
                # 生成完切回训练模式，继续训练
                model.train()

    torch.save(model.state_dict(), CHECKPOINT)
    print(f'模型已保存到 {CHECKPOINT}')


if __name__ == '__main__':
    train()
