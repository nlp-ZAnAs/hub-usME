"""
字符级语言模型训练脚本，支持 RNN / LSTM 切换，含 PPL 计算。
用法:
    python language_model.py --model lstm --epochs 20
    python language_model.py --model rnn  --epochs 20
"""

import math
import argparse
import glob
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ─────────────────────────── 数据 ───────────────────────────

def load_corpus(pattern="*.txt"):
    texts = []
    for path in glob.glob(pattern):
        with open(path, encoding="utf-8", errors="ignore") as f:
            texts.append(f.read())
    return "".join(texts)


def build_vocab(text, max_vocab=None):
    """
    构建字表：按出现频率取前 max_vocab 个字（None = 全部字符）。
    语料里的生僻字超出字表后会被 CharDataset 过滤掉 —— 字表越小模型越小，
    但覆盖不足会丢信息，这是词表大小的取舍。
    """
    from collections import Counter
    counter = Counter(text)
    chars = [c for c, _ in counter.most_common(max_vocab)] if max_vocab else sorted(set(text))
    char2idx = {c: i for i, c in enumerate(chars)}
    idx2char = {i: c for c, i in char2idx.items()}
    return char2idx, idx2char


class CharDataset(Dataset):
    """
    字符级语言模型的训练样本：输入 x = 连续 seq_len 个字符，目标 y = 往后错一位。

    stride 控制切块间隔：
        stride=1  —— 每个字符位置都当一个样本（样本大量重叠，一期语料能切出上百万个样本）
        stride=seq_len —— 不重叠切块（标准做法，样本数 = 语料长度 / seq_len）
    语料一大，stride=1 会让一轮训练变得极慢，这不是模型慢，而是样本数被放大了 seq_len 倍。
    """

    def __init__(self, text, char2idx, seq_len, stride=1):
        self.seq_len = seq_len
        self.stride = max(1, stride)
        ids = [char2idx[c] for c in text if c in char2idx]
        self.data = torch.tensor(ids, dtype=torch.long)

    def __len__(self):
        return max(0, (len(self.data) - self.seq_len - 1) // self.stride + 1)

    def __getitem__(self, idx):
        start = idx * self.stride
        x = self.data[start: start + self.seq_len]
        y = self.data[start + 1: start + self.seq_len + 1]
        return x, y


# ─────────────────────────── 模型 ───────────────────────────

class TransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model=192, n_head=4, n_layer=4, max_len=512, dropout=0.1, use_pos=True, causal=True):
        super().__init__()
        self.causal = causal
        self.use_pos = use_pos
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model) if use_pos else None
        self.drop = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead = n_head, dim_feedforward = 4 * d_model, 
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=n_layer, norm=nn.LayerNorm(d_model),
            enable_nested_tensor=False
        )

        self.head = nn.linear(d_model,vocab_size, bias=False)
        self.head.weight = self.tok.weight

        nn.init.normal_(self.tok.weight, std=0.02)
        if self.pos is not None:
            nn.init.normal_(self.pos.weight, std=0.02)

    def forward(self, idx):
        x = self.tok(idx)
        if self.use_pos:
            x = x + self.pos(torch.arange(idx.size(1), device=idx.device))
        x = self.drop(x)
        mask = None
        if self.causal:
            mask = nn.Transformer.generate_square_subsequent_mask(
                idx.size(1), device=idx.device)   # 下三角 0、右上 -inf
        x = self.encoder(x, mask=mask)
        return self.head(x)


# ─────────────────────────── 训练 / 评估 ───────────────────────────

@torch.no_grad()
def evaluate(model, loader, device, last_only=False):
    """返回 (loss, ppl)；last_only=True 时只看每段最后一个位置（这个位置抄不到答案）"""
    # model.eval() 和 @torch.no_grad() 是两件事，缺一不可：
    #   model.eval()  —— 把 Dropout 关掉（推理时随即丢弃会让同一个输入的输出每次都变，
    #                    PPL 就没意义了）、把 BatchNorm 换成用运行统计量（本模型没有 BN，但习惯要养成）；
    #   no_grad()     —— 不建计算图，显存/速度都省不少，评测几十个 batch 时差别很明显。
    # 如果忘了 eval()，dropout 会让 PPL 系统性偏高，而且每次跑出来的数都不一样 —— 很多人对不上数就是这个原因。
    model.eval()
    total, tokens = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        if last_only:
            # 只取每段「最后一个位置」的预测：
            # 训练窗口是 [s, s+T)，最后一个位置的输入是 x[s+T-1]，目标是 x[s+T]，而这一个目标字
            # 从来没出现在输入窗口里 —— 所以这个位置永远「抄不到答案」。
            # 这就是识破 --no-mask 泄漏实验的关键指标：无 mask 时全位置 PPL 能到 1.07，
            # 但末位 PPL 高达 98.23（12M 版是 88.50），后者才反映真实语言能力。
            logits, y = logits[:, -1, :], y[:, -1]
        # 用 reduction="sum" 再自己除以 token 数，而不是让 cross_entropy 直接取 mean：
        #   mean 是「每个 batch 内先平均」，最后一个不满的 batch 会和满 batch 权重相同，
        #   算出来的值会和「按 token 加权」有细微差异。PPL 是要写进作业表里横向比的，口径必须干净。
        total += float(F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                       y.reshape(-1), reduction="sum"))
        tokens += y.numel()
    loss = total / tokens
    # 交叉熵是以 e 为底的平均负对数似然，exp 回来就是 PPL（困惑度）：
    # 「模型在每一步平均在多少个候选字之间犹豫」。字表 4000 时，瞎猜的 PPL ≈ 4000，越好越接近 1。
    return loss, math.exp(loss)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    text = load_corpus(args.corpus)
    if args.max_chars and len(text) > args.max_chars:
        # 只在没 GPU 时用来快速跑通流程的近似手段；注意截断后 PPL 不能和全量语料的数字比
        text = text[:args.max_chars]
    char2idx, idx2char = build_vocab(text, args.max_vocab)
    V = len(char2idx)      # 词表大小：Embedding 行数、输出头列数、softmax 类别数，三者必须一致
    print(f"device: {device}  语料 {len(text):,} 字  字表 {V}")

    lines = text.splitlines()
    random.shuffle(lines)
    split = int(len(lines) * (1 - args.val_ratio))     # 默认 5% 留作验证
    tr = CharDataset("\n".join(lines[:split]), char2idx, args.seq_len, args.stride)
    va = CharDataset("\n".join(lines[split:]), char2idx, args.seq_len, args.stride)
    # 训练集 shuffle=True：每个 epoch 的 batch 组成不同，等价于做了一点数据层面的正则；
    # drop_last=True：丢掉最后一个不满的 batch，避免 batch 很小时（比如只剩 1 条）梯度过噪，
    #                 也让每个 batch 的 shape 固定
    train_loader = DataLoader(tr, batch_size=args.batch_size, shuffle=True, drop_last=True)
    # 验证集 shuffle=False：不需要打乱（评测不看顺序），关掉能省一点开销、也让结果可复现
    val_loader = DataLoader(va, batch_size=args.batch_size, shuffle=False, drop_last=True)

    model = TransformerLM(V, args.d_model, args.heads, args.layers, args.seq_len,
                          args.dropout, use_pos=not args.no_pos, causal=not args.no_mask).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"模型: TransformerEncoderLayer × {args.layers}（d={args.d_model} h={args.heads} "
          f"FFN={4*args.d_model} Pre-LN）  位置编码={'有' if not args.no_pos else '无'}  "
          f"因果掩码={'有' if not args.no_mask else '**无（泄漏实验）**'}  参数量 {n_par/1e6:.2f}M")

    # Adam + 固定学习率就够了（小模型，6 个 epoch），没有 warmup / cosine：
    # Pre-LN 的稳定性就是体现在「省掉这些也训得动」上，这点值得在作业里点一句
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    best = float("inf")
    print(f"\n{'Epoch':>6} {'TrainLoss':>10} {'ValPPL(全位置)':>15} {'ValPPL(末位)':>13} {'tok/s':>9}")
    print("-" * 60)
    for ep in range(1, args.epochs + 1):
        model.train()
        st, seen, tot = time.time(), 0, 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
        
            loss = F.cross_entropy(model(x).reshape(-1, V), y.reshape(-1))
            opt.zero_grad()
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            tot += float(loss) * y.numel()
            seen += y.numel()
        dt = time.time() - st
    
        _, v_ppl = evaluate(model, val_loader, device, last_only=False)
        _, v_ppl_last = evaluate(model, val_loader, device, last_only=True)
        mark = ""
        if v_ppl < best:
            best = v_ppl
            mark = " *"

            torch.save({"model_state": model.state_dict(), "char2idx": char2idx,
                        "idx2char": idx2char, "args": vars(args)}, args.save)
        print(f"{ep:>6} {tot/seen:>10.4f} {v_ppl:>15.2f} {v_ppl_last:>13.2f} {seen/dt:>9,.0f}{mark}")

    print(f"\n最佳验证 PPL（全位置）: {best:.2f}  已保存 {args.save}")


@torch.no_grad()
def generate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # map_location：让「在 GPU 上训的 ckpt 能在 CPU 机器上加载」，学生换机器不会卡在这里
    ck = torch.load(args.ckpt, map_location=device)
    c = ck["args"]
    char2idx, idx2char = ck["char2idx"], ck["idx2char"]
    # 用 ckpt 里记下的超参重建同构模型（含 no_mask / no_pos 两个开关）。
    # dropout 传 0.0：生成过程中本来就该关 dropout，这里顺手传 0 双保险
    model = TransformerLM(len(char2idx), c["d_model"], c["heads"], c["layers"], c["seq_len"], 0.0,
                          use_pos=not c.get("no_pos", False),
                          causal=not c.get("no_mask", False)).to(device)
    model.load_state_dict(ck["model_state"])
    model.eval()      # 关 dropout，否则同一个开头每次生成的风格都会漂
    # 开头里不在字表的字直接跳过（否则 KeyError）；如果一个都没有，out 会是空列表
    ids = [char2idx[ch] for ch in args.generate if ch in char2idx]
    out = list(ids)
    for _ in range(args.n_chars):
        # 只喂最近 seq_len 个 token：位置编码最多只有 seq_len 行（max_len 就是按 seq_len 建的），
        # 而且训练时模型也只见过长度 ≤ seq_len 的上下文，超长输入反而会让位置分布跑出训练区间。
        # [0, -1] 取 batch 里第一条、最后一个位置的 logits —— 那是对「下一个字」的打分
        logits = model(torch.tensor([out[-args.seq_len:]], device=device))[0, -1]
        if args.temperature <= 0.01:
            # 温度趋近 0 等价于贪心解码（argmax）：结果完全确定，适合课上反复讲同一个例子。
            # 用阈值而不是 ==0，是为了兼容命令行传 0 时可能出现的浮点误差，也避免下面除零
            nxt = int(logits.argmax())
        else:
            # 温度采样：logits / T 再 softmax。T<1 分布更尖（更保守、更像贪心），
            # T>1 更平（更发散、更容易出语法错误）。0.8 是个常用折中
            nxt = int(torch.multinomial(F.softmax(logits / args.temperature, dim=-1), 1))
        out.append(nxt)
    print(f"输入: {args.generate}")
    # 只打印「新生成的部分」（out 里前 len(ids) 个是用户给的开头）
    print(f"续写: {''.join(idx2char[i] for i in out[len(ids):])}")


def main():
    ap = argparse.ArgumentParser()
    # 默认语料指向脚本同目录的 corpus_sample.txt：80 万字样例语料，学生拿到就能跑通，
    # 不用先去课件目录找 corpus_general.txt
    default_corpus = os.path.join(HERE, "corpus_sample.txt")
    ap.add_argument("--corpus", default=default_corpus, help="语料路径（支持通配符）")
    ap.add_argument("--max_chars", type=int, default=0, help="只用前 N 个字符（CPU 慢时截断）")
    ap.add_argument("--max_vocab", type=int, default=4000)
    ap.add_argument("--seq_len", type=int, default=64)
    # stride 默认等于 seq_len（不重叠）。调小会增加样本量但让相邻窗口高度相关；
    # 和 LSTM 基线对比时必须保持 64，否则切分口径不一致，PPL 不可比
    ap.add_argument("--stride", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-3)
    # 默认这套超参（d=192 / 4 层 / 4 头）参数量约 2.66M，正好和本周 LSTM 基线的 2.46M 对齐在 ±10% 内 ——
    # 作业要求「先对齐参数量再比 PPL」，否则比出来的不是结构差异而是模型大小的差异
    ap.add_argument("--d_model", type=int, default=192)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--val_ratio", type=float, default=0.05)
    # seed 决定「按行打乱」的结果，也就是训练/验证的划分本身。
    # 复算 PPL（demo_continuation.py）必须用同一个 seed 才能还原同一份验证集
    ap.add_argument("--seed", type=int, default=42)
    # ⚠ --no-mask 不是「更快的写法」，而是泄漏实验：模型能看到未来 → 学会抄输入。
    # 打开后请务必同时看「末位 PPL」和续写效果，否则会被 1.07 的全位置 PPL 骗过去
    ap.add_argument("--no-mask", dest="no_mask", action="store_true",
                    help="去掉因果掩码（泄漏实验）")
    # --no-pos 是位置编码消融：自注意力本身对位置是「置换等变」的（打乱输入顺序，输出只会跟着打乱），
    # 没有位置编码它就退化成一个「词袋 + 上下文加权」，连「上一个字是什么」这种最强特征都拿不到。
    # 实测 PPL 从 24.95 掉到 27.42（seq_len 拉长到 256 时差距会更明显）
    ap.add_argument("--no-pos", dest="no_pos", action="store_true", help="去掉位置编码")
    ap.add_argument("--save", default="tf_best.pt")
    ap.add_argument("--generate", default=None, help="给一段开头，续写文本")
    ap.add_argument("--ckpt", default="tf_best.pt")
    ap.add_argument("--n_chars", type=int, default=40)
    ap.add_argument("--temperature", type=float, default=0.8)
    args = ap.parse_args()

    # 一条命令两用：带 --generate 就只做续写（不再训练），否则进入训练。
    # 这样学生不用记两个脚本名；注意 --generate 模式下其余训练参数（--epochs 等）会被忽略
    if args.generate:
        generate(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
