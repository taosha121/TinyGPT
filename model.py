import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# 本项目参数值：
#   hidden=384：每个 token 被表示成 384 维的向量
#   context_len=256：训练文本约 20M tokens，用滑动窗口每次右移 1 个 token 截取 256 个 token，
#                    共约 20M 个训练样本（不是 20M÷256≈78K）。
#                    一个 batch 有 64 个这样的样本，输入形状是 (64, 256, 384)。
#   heads=8：把每个 token 的 384 维向量切成 8 份，每份 48 维，独立做一次 attention。
#            8 个头的权重矩阵各自独立训练，会学到不同的关注模式（如句法、指代、相邻搭配等）。
#            切分后形状从 (64, 256, 384) 变成 (64, 8, 256, 48)，最后再 concat 回 (64, 256, 384)。


# 输入输出形状完全相同：(64, 256, 384) → (64, 256, 384)
# 但内容发生本质变化：
#   输入：每个 token 的向量是孤立的，只包含自己的信息
#   输出：每个 token 的向量已融合了上下文中它关注的其他 token 的信息
class CausalSelfAttention(nn.Module):
    def __init__(self, hidden, heads, context_len, dropout):
        super().__init__()
        self.heads    = heads
        self.head_dim = hidden // heads          # 每个 head 的维度，384/8=48
        # 一个 Linear 同时生成 Q/K/V，比三个独立 Linear 更高效
        self.qkv  = nn.Linear(hidden, 3 * hidden, bias=False)
        self.proj = nn.Linear(hidden, hidden, bias=False)
        self.drop = nn.Dropout(dropout)
        # Causal Mask：创建 256×256 的下三角矩阵，1 表示可见，0 表示遮住。
        # 第 i 行表示第 i 个 token 只能看到它自己和它之前的 token，不能看到后面的。
        mask = torch.tril(torch.ones(context_len, context_len))
        # register_buffer 是 nn.Module 的方法，把 mask 注册到模型内部状态：
        #   - model.to('mps') 时 mask 跟着移动，不会出现 x 在 MPS 而 mask 在 CPU 的报错
        #   - 保存模型时 mask 也会被保存
        #   - 但不算可训练参数，反向传播不会更新它
        self.register_buffer('mask', mask)

    def forward(self, x):
        B, T, C = x.shape                        # batch, seq_len, hidden
        H, D = self.heads, self.head_dim

        # x 的形状：(64个样本, 256个token, 384维向量)
        # self.qkv(x) 做矩阵乘法 (64,256,384) @ (384,1152) = (64,256,1152)
        # .split(384, dim=-1) 沿最后一维切成三份，q/k/v 各自 (64, 256, 384)
        q, k, v = self.qkv(x).split(C, dim=-1)  # 各自 (B, T, C)

        # view 和 transpose 只是改变形状，不做任何计算，数据本身不变。
        # .view(64, 256, 8, 48)：把最后一维 384 拆成 8个头 × 每头48维，形状变成 (64, 256, 8, 48)
        # .transpose(1, 2)：把 head 维度提前，变成 (64, 8, 256, 48)。
        #   目的：让后续 q @ k.T 在 (256, 48) 上做，64个样本和8个head只是并行批量，不参与计算。
        q = q.view(B, T, H, D).transpose(1, 2)
        k = k.view(B, T, H, D).transpose(1, 2)
        v = v.view(B, T, H, D).transpose(1, 2)

        # scores = Q @ K.T：256×256 的相似度矩阵，scores[i][j] 表示 token i 对 token j 的关注程度。
        # Q 和 K 决定"关注谁"，除以 sqrt(48) 防止数值过大导致 softmax 梯度消失。
        scores = q @ k.transpose(-2, -1) / math.sqrt(D)  # (64, 8, 256, 256)
        # causal mask：右上角设为 -inf，softmax 后权重归零，保证 token i 看不到 i 之后的内容
        scores = scores.masked_fill(self.mask[:T, :T] == 0, float('-inf'))
        # softmax 把 scores 变成概率分布，dropout 随机断掉 10% 的注意力连接防止过拟合
        w = self.drop(F.softmax(scores, dim=-1))          # (64, 8, 256, 256)

        # w @ v：用相似度权重对 V 加权求和，token i 的输出 = 所有 token 的 V 按 w[i] 混合。
        # 每个 token 不再孤立，已融合了上下文中它关注的信息。
        # transpose + view 把 (64, 8, 256, 48) 拼回 (64, 256, 384)：
        #   .transpose(1,2)：(64,8,256,48) → (64,256,8,48)，把 head 维度放回原位
        #   .contiguous()：transpose 后内存不连续，view 要求连续，这步整理内存，不做数值计算
        #   .view(64,256,384)：把最后两维 8×48 合并回 384，8 个 head 的结果直接拼在一起
        out = (w @ v).transpose(1, 2).contiguous().view(B, T, C)
        # proj 是 384×384 的矩阵乘法，让 8 个 head 的结果第一次互相混合：
        # concat 后每个 head 的 48 维仍是割裂的，proj 的输出每一维都是全部 384 维的加权组合
        return self.proj(out)                              # (64, 256, 384)


class Block(nn.Module):
    # 一个 Transformer Block = Pre-Norm + Attention + 残差 + Pre-Norm + FFN + 残差
    def __init__(self, hidden, heads, context_len, dropout):
        super().__init__()
        self.ln1  = nn.LayerNorm(hidden)
        self.attn = CausalSelfAttention(hidden, heads, context_len, dropout)
        self.ln2  = nn.LayerNorm(hidden)
        self.mlp  = nn.Sequential(
            nn.Linear(hidden, 4 * hidden),   # 升维到 1536，扩展表达能力
            nn.GELU(),
            nn.Linear(4 * hidden, hidden),   # 降回 384
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # ln1：对每个 token 的 384 维向量标准化，防止数值过大/过小，让 attention 计算稳定
        # attn：token 之间的信息交换，每个 token 融合上下文中它关注的信息（唯一的跨位置操作）
        # 残差：把 attention 输出加回原始输入，保留原有信息，attention 只需学"补充什么"
        x = x + self.attn(self.ln1(x))
        # ln2：进 FFN 前再归一化一次，FFN 的输入同样需要稳定的数值范围
        # mlp：每个 token 独立做非线性变换（384→1536→384），不跨位置。
        #      attention 负责收集信息，FFN 负责消化信息。
        #      GELU 激活引入非线性，attention 的加权求和是线性的，FFN 补充了非线性表达能力。
        #      研究发现 FFN 还存储了大量事实性知识（如"Paris 是 France 的首都"）。
        # 残差：同上，保留原有信息，只叠加 FFN 新增的部分
        x = x + self.mlp(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, vocab_size, hidden, layers, heads, context_len, dropout=0.1):
        super().__init__()
        self.context_len = context_len
        self.tok_emb = nn.Embedding(vocab_size, hidden)   # token id → 384 维向量
        self.pos_emb = nn.Embedding(context_len, hidden)  # 位置序号 → 384 维向量
        self.drop    = nn.Dropout(dropout)
        self.blocks  = nn.ModuleList([
            Block(hidden, heads, context_len, dropout) for _ in range(layers)
        ])
        self.ln_f    = nn.LayerNorm(hidden)               # 最后一层 Block 后的归一化
        # LM Head：最后一个线性层 (384, 50257)，把每个 token 的 384 维向量映射到词表大小的得分。
        # 例如输入 "Once upon a time"，最后一个 token "time" 的 384 维向量经过 LM Head 得到
        # 50257 个得分，softmax 后是"下一个词是每个词的概率"，训练时算 loss，推理时采样下一个 token。
        self.lm_head = nn.Linear(hidden, vocab_size, bias=False)
        # 共享权重：LM Head 和 Token Embedding 用同一个矩阵（转置使用）
        # 节省 50257×384 ≈ 19M 参数，且语义上合理：相似词的 embedding 距离近，预测概率也近
        self.lm_head.weight = self.tok_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, m):
        # self.apply(_init_weights) 递归遍历模型所有子模块，包括 CausalSelfAttention 和 Block
        # 里面所有的 Linear 和 Embedding，在这里统一随机初始化——之前定义时只是声明了矩阵的形状，
        # 到这里才真正填入数据。
        # std=0.02 来自 GPT-2 论文，权重不能太大（梯度爆炸）也不能太小（梯度消失）。
        # 这些权重初始化后随训练反向传播不断更新，最终收敛到能预测下一个词的值。
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.zeros_(m.bias)

    # targets：训练时传入右移一位的标签，用于计算 loss；推理时不传，区分训练和推理两个阶段。
    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos  = torch.arange(T, device=idx.device)
        # token embedding + 位置编码叠加
        x    = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        # 依次流过 6 个 Block，每个 Block 做一次 attention（跨 token 信息交换）+ FFN（非线性变换）
        # 输入输出形状不变，始终是 (64, 256, 384)，每过一层信息融合更充分
        for block in self.blocks:
            x = block(x)
        # LayerNorm 后接 LM Head，把 (64, 256, 384) 映射到 (64, 256, 50257)，得到每个位置的词表得分
        logits = self.lm_head(self.ln_f(x))
        # 训练时计算 loss，推理时 targets=None 跳过
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), targets.view(-1)
        ) if targets is not None else None
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=40):
        for _ in range(max_new_tokens):
            # 超出 context_len 时截断，只保留最近的窗口
            idx_in = idx[:, -self.context_len:]
            logits, _ = self(idx_in)
            # 只取最后一个位置的 logits，除以 temperature 拉大/缩小数值间差距：
            #   temperature 低（如 0.3）→ 差距更大 → 更保守接近贪心
            #   temperature 高（如 1.5）→ 差距更小 → 更随机
            # 例：logits = [the:3.0, a:2.0, cat:1.0, dog:0.5, xyz:-2.0]
            #     ÷ 0.8  = [the:3.75, a:2.50, cat:1.25, dog:0.63, xyz:-2.50]
            logits = logits[:, -1, :] / temperature
            # top-k 截断：只保留最高的 k 个词，其余设为 -inf
            # 例：top-3 后 = [the:3.75, a:2.50, cat:1.25, dog:-inf, xyz:-inf]
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = float('-inf')
            # softmax 变成概率后用 multinomial 按概率随机采样（不是直接取最高的）：
            # 例：softmax 后 = [the:0.67, a:0.24, cat:0.09, dog:0.00, xyz:0.00]
            #     multinomial → 67% 概率选 "the"，24% 选 "a"，9% 选 "cat"
            next_tok = torch.multinomial(F.softmax(logits, dim=-1), 1)
            idx = torch.cat([idx, next_tok], dim=1)
        return idx
