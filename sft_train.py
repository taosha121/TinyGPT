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
# 比预训练低一个数量级：预训练权重已经收敛，用大 LR 会把已学到的语言能力破坏掉
LR          = 3e-5
LOG_EVERY   = cfg['log_every']
CHECKPOINT  = 'tiny_gpt_sft.pt'  # 保存新文件，不覆盖预训练权重

# 指令模板：训练和推理必须用同一个模板，否则模型看到的格式对不上
PROMPT_TEMPLATE = "### Instruction:\n{instruction}\n\n### Response:\n"


def get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


class SFTDataset(Dataset):
    """
    把 (instruction, response) 对转换成 (x, y_masked) 训练样本。

    x 是完整序列去掉最后一个 token，y_masked 是右移一位的标签，
    但 instruction 对应的位置全部设为 -100。

    F.cross_entropy 默认 ignore_index=-100，所以 -100 位置不产生 loss，
    梯度只来自 response 部分——模型学的是"根据 instruction 生成 response"，
    而不是在模仿 instruction 的写法。
    """
    def __init__(self, path, tokenizer, context_len):
        with open(path) as f:
            raw = json.load(f)

        self.samples = []
        # eot_token 是 tiktoken GPT-2 编码器的句子结束符，
        # 加在 response 末尾，让模型学会在故事结束时停止生成
        eos = tokenizer.eot_token

        for item in raw:
            instr_text = PROMPT_TEMPLATE.format(instruction=item['instruction'])
            resp_text  = item['response']

            instr_ids = tokenizer.encode(instr_text)
            resp_ids  = tokenizer.encode(resp_text) + [eos]

            # 把 instruction 和 response 拼成一条完整序列
            ids = instr_ids + resp_ids
            # k 是 instruction 部分的 token 数，用来确定 -100 mask 的边界
            k   = len(instr_ids)

            x        = ids[:-1]   # 输入：去掉最后一个 token
            y        = ids[1:]    # 目标：右移一位，x[i] 预测 y[i]

            # instruction 对应的 k 个位置设 -100，不参与 loss
            # response 部分保持真实 token id，产生 loss
            y_masked = [-100] * k + y[k:]

            # 截断到 context_len，防止超出模型最大位置编码长度
            x        = x[:context_len]
            y_masked = y_masked[:context_len]

            # 如果 response 部分被完全截断，这条样本没有任何 loss，直接丢掉
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
    """
    DataLoader 的自定义拼批函数。

    SFT 样本长度不一（不同故事长度不同），默认的 collate 无法处理。
    这里把同一 batch 内的样本都 pad 到最长那条的长度：
    - x_pad 用 0（token id 0 是个普通 token，不影响训练，因为对应位置的 y 是 -100）
    - y_pad 用 -100（pad 位置不参与 loss）
    """
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

    # 加载预训练权重：模型已经学会写故事，SFT 只需要在此基础上学指令格式，
    # 所需数据量远少于从头训练（几千条 vs 几百万条）
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
            # model.forward(x, y) 返回 (logits, loss)
            # loss 由 model 内部的 F.cross_entropy 计算，ignore_index=-100 自动跳过 mask 位置
            _, loss = model(x, y)

            optimizer.zero_grad()
            loss.backward()
            # 梯度裁剪：防止单步更新幅度过大，破坏预训练已经学到的权重
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
