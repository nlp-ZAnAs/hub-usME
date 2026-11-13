"""
字符级语言模型训练脚本，支持 RNN / LSTM / Transformer 切换，含 PPL 计算。
用法:
    python language_model.py --model lstm        --epochs 20
    python language_model.py --model rnn         --epochs 20
    python language_model.py --model transformer --epochs 20 --lr 3e-4 --dropout 0.1
    python language_model.py --sft --seq_len 128 --batch_size 32 --lr 1e-4 --epochs 30
"""

import math
import json
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

class LM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, model_type, dropout):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        rnn_cls = nn.LSTM if model_type == "lstm" else nn.RNN
        self.rnn = rnn_cls(
            embed_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        e = self.drop(self.embed(x))
        out, _ = self.rnn(e)
        logits = self.fc(self.drop(out))   # (B, T, V)
        return logits


class PositionalEncoding(nn.Module):
    """正弦位置编码：无训练参数，max_len 为支持的最大序列长度。"""

    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # 存成 buffer：随模型一起 save/load，但不参与梯度更新
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d_model)
        self.pe: torch.Tensor   # 显式标注类型，便于类型检查器识别

    def forward(self, x):
        # x: (B, T, d_model)，按序列长度截取对应位置编码
        return x + self.pe[:, :x.size(1)]


class TransformerLM(nn.Module):
    """
    字符级 Transformer 语言模型：Embedding + 位置编码 + TransformerEncoder + Linear。
    通过因果掩码实现"只看上文"，每个时间步的输出语义与 RNN 一致。
    """

    def __init__(self, vocab_size, embed_dim, num_layers, nhead, dim_feedforward, dropout, max_len=512):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos = PositionalEncoding(embed_dim, max_len)
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,   # 输入为 (B, T, D)，必须显式指定
            ),
            num_layers=num_layers,
        )
        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        t = x.size(1)
        # 因果掩码：上三角 True 表示禁止注意的位置（只允许看自己及左侧）
        mask = torch.triu(torch.ones(t, t, dtype=torch.bool, device=x.device), diagonal=1)
        e = self.drop(self.pos(self.embed(x)))   # (B, T, d_model)
        out = self.encoder(e, mask=mask)         # (B, T, d_model)
        logits = self.fc(self.drop(out))         # (B, T, V)
        return logits


# ─────────────────────────── 训练 / 评估 ───────────────────────────

def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train(train)
    total_loss = 0.0
    total_tokens = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # -100 是被 mask 的 prompt / padding 位置，不计入 loss 均值与 PPL 分母
        n_tokens = (y != -100).sum().item()
        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens

    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    return avg_loss, ppl


# ─────────────────────────── SFT ───────────────────────────

# 训练、评估、生成共用同一套模板；末尾换行是回答的结束符，也是生成时的停止条件
SFT_TEMPLATE = "用户：{instruction}\n助手：{output}\n"
SFT_PROMPT = "用户：{instruction}\n助手："
# padding 用句号，与结束符换行区分开，避免结束信号被大量填充字符淹没
SFT_PAD_CHAR = "。"


def normalize(text):
    """删除半角空格。训练与推理共用，保证进入模型的字符都在预训练字表内。"""
    return text.replace(" ", "")


def load_sft_samples(path, char2idx):
    """
    读取 jsonl 并做规范化，整条剔除仍含字表外字符的样本。
    返回 [(prompt, answer), ...]，answer 末尾已带结束符换行。
    """
    samples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            instruction = normalize(item["instruction"])
            output = normalize(item["output"])
            # 回答内部出现换行会与结束符混淆，这类样本不参与训练
            if "\n" in instruction or "\n" in output:
                continue
            prompt = SFT_PROMPT.format(instruction=instruction)
            text = SFT_TEMPLATE.format(instruction=instruction, output=output)
            # 字表外字符不做替换，含这类字符的样本整条丢弃
            if any(c not in char2idx for c in text):
                continue
            samples.append((prompt, output + "\n"))
    return samples


class SFTDataset(Dataset):
    """
    每条 SFT 数据独立成块：x = 模板文本去掉最后一字，y = 往后错一位。
    y 中 prompt 部分与右侧 padding 置 -100，只对回答（含结束符）算 loss。
    """

    def __init__(self, samples, char2idx, seq_len, pad_id):
        self.items = []
        for prompt, answer in samples:
            ids = [char2idx[c] for c in prompt + answer]
            x = ids[:-1]
            y = ids[1:]
            labels = [-100] * (len(prompt) - 1) + y[len(prompt) - 1:]
            if len(x) > seq_len:
                raise ValueError(f"样本长度 {len(x)} 超过 seq_len={seq_len}，请调大 --seq_len")
            pad_len = seq_len - len(x)
            x = x + [pad_id] * pad_len
            labels = labels + [-100] * pad_len
            self.items.append((
                torch.tensor(x, dtype=torch.long),
                torch.tensor(labels, dtype=torch.long),
            ))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def generate_answer(model, char2idx, idx2char, instruction, max_new_tokens, device, top_k):
    """
    按模板拼 prompt，整段送入模型取最后一位 logits，top-k 采样自回归到结束符。
    对最近出现过的字符压低 logits，抑制字符级模型贪心解码的重复循环。
    """
    prompt = SFT_PROMPT.format(instruction=normalize(instruction))
    if any(c not in char2idx for c in prompt):
        raise ValueError("指令包含预训练字表外的字符，请检查输入")
    context = [char2idx[c] for c in prompt]
    eos_id = char2idx["\n"]
    chars = []
    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            x = torch.tensor([context], dtype=torch.long, device=device)
            logits = model(x)[0, -1].clone()
            # 最近 12 个字符的 logits 减 1，softmax 后概率约降到原来的 0.37 倍
            recent = [c for c in set(context[-12:]) if c != eos_id]
            if recent:
                logits[recent] = logits[recent] - 1.0
            values, indices = torch.topk(logits, top_k)
            probs = torch.softmax(values, dim=-1)
            next_id = int(indices[torch.multinomial(probs, 1)].item())
            if next_id == eos_id:
                break
            chars.append(idx2char[next_id])
            context.append(next_id)
            # 连续 3 个相同字符视为解码退化，提前停止
            if len(chars) >= 3 and chars[-1] == chars[-2] == chars[-3]:
                break
    return "".join(chars)


def run_sft(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = torch.load(args.model_path, map_location=device, weights_only=False)
    char2idx, idx2char = cfg["char2idx"], cfg["idx2char"]
    pa = cfg["args"]

    # 结构超参一律取自预训练 checkpoint，保证 load_state_dict 形状一致
    model = TransformerLM(
        vocab_size=len(char2idx),
        embed_dim=pa["embed_dim"],
        num_layers=pa["num_layers"],
        nhead=pa["nhead"],
        dim_feedforward=pa["dim_feedforward"],
        dropout=pa["dropout"],
        max_len=pa["max_len"],
    ).to(device)
    model.load_state_dict(cfg["model_state"])

    pad_id = char2idx[SFT_PAD_CHAR]
    train_samples = load_sft_samples(args.sft_train, char2idx)
    val_samples = load_sft_samples(args.sft_val, char2idx)
    print(f"SFT 样本: train {len(train_samples)}  val {len(val_samples)}  词表 {len(char2idx)}")

    train_ds = SFTDataset(train_samples, char2idx, args.seq_len, pad_id)
    val_ds = SFTDataset(val_samples, char2idx, args.seq_len, pad_id)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_val_ppl = float("inf")

    print(f"\n{'Epoch':>6}  {'Train Loss':>10}  {'Train PPL':>10}  {'Val Loss':>10}  {'Val PPL':>10}")
    print("-" * 56)

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_ppl = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        with torch.no_grad():
            va_loss, va_ppl = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

        marker = "  *" if va_ppl < best_val_ppl else ""
        if va_ppl < best_val_ppl:
            best_val_ppl = va_ppl
            save_args = dict(pa)
            save_args.update({
                "sft": True,
                "seq_len": args.seq_len,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "epochs": args.epochs,
                "save": args.sft_save,
            })
            torch.save({
                "model_state": model.state_dict(),
                "char2idx": char2idx,
                "idx2char": idx2char,
                "args": save_args,
            }, args.sft_save)

        print(f"{epoch:>6}  {tr_loss:>10.4f}  {tr_ppl:>10.2f}  {va_loss:>10.4f}  {va_ppl:>10.2f}{marker}")

    print(f"\nSFT 完成。最佳验证 PPL: {best_val_ppl:.2f}  已保存至 {args.sft_save}")

    # 训练结束后抽查验证集的生成结果
    print("\n生成抽查:")
    for prompt, answer in val_samples[:3]:
        instruction = prompt[len("用户："):-len("\n助手：")]
        print(f"  指令: {instruction}")
        print(f"  参考: {answer.strip()}")
        print(f"  生成: {generate_answer(model, char2idx, idx2char, instruction, args.sft_gen_len, device, args.sft_top_k)}")


# ─────────────────────────── 主函数 ───────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      default="lstm", choices=["rnn", "lstm", "transformer"])
    parser.add_argument("--epochs",     type=int,   default=20)
    parser.add_argument("--seq_len",    type=int,   default=64)
    parser.add_argument("--stride",     type=int,   default=1, help="切块间隔：1=密集重叠，seq_len=不重叠")
    parser.add_argument("--batch_size", type=int,   default=128)
    parser.add_argument("--embed_dim",  type=int,   default=128)
    parser.add_argument("--hidden_dim", type=int,   default=256, help="仅 RNN/LSTM 使用")
    parser.add_argument("--nhead",      type=int,   default=4,   help="Transformer 注意力头数，需整除 embed_dim")
    parser.add_argument("--dim_feedforward", type=int, default=512, help="Transformer 前馈层维度")
    parser.add_argument("--max_len",    type=int,   default=512, help="Transformer 位置编码最大序列长度")
    parser.add_argument("--num_layers", type=int,   default=2)
    parser.add_argument("--dropout",    type=float, default=0.3)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--val_ratio",  type=float, default=0.05)
    parser.add_argument("--max_vocab",  type=int,   default=4000, help="字表最大字数，0=不限")
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--corpus",     default="*.txt")
    parser.add_argument("--save",       default="best_model.pt")
    parser.add_argument("--sft",        action="store_true", help="基于预训练权重做指令微调")
    parser.add_argument("--model_path", default="best_model_transformer.pt")
    parser.add_argument("--sft_train",  default="sft_train.jsonl")
    parser.add_argument("--sft_val",    default="sft_val.jsonl")
    parser.add_argument("--sft_save",   default="best_model_sft.pt")
    parser.add_argument("--sft_gen_len", type=int, default=40, help="生成抽查时的最大回答长度")
    parser.add_argument("--sft_top_k", type=int, default=20, help="生成抽查时的 top-k 采样范围，1=贪心")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    if args.sft:
        run_sft(args)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}  model: {args.model.upper()}")

    # 数据准备
    text = load_corpus(args.corpus)
    if not text:
        raise FileNotFoundError("未找到任何 .txt 文件，请确认路径正确。")
    print(f"语料字符数: {len(text):,}")

    char2idx, idx2char = build_vocab(text, args.max_vocab or None)
    vocab_size = len(char2idx)
    print(f"词表大小: {vocab_size}")

    lines = text.splitlines()
    random.shuffle(lines)
    split = int(len(lines) * (1 - args.val_ratio))
    train_text = "\n".join(lines[:split])
    val_text   = "\n".join(lines[split:])

    train_ds = CharDataset(train_text, char2idx, args.seq_len, args.stride)
    val_ds   = CharDataset(val_text,   char2idx, args.seq_len, args.stride)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=True, drop_last=True)

    # 模型
    if args.model == "transformer":
        if args.embed_dim % args.nhead != 0:
            raise ValueError(f"nhead={args.nhead} 必须整除 embed_dim={args.embed_dim}")
        if args.seq_len > args.max_len:
            raise ValueError(f"seq_len={args.seq_len} 不能超过位置编码长度 max_len={args.max_len}")
        model = TransformerLM(
            vocab_size=vocab_size,
            embed_dim=args.embed_dim,
            num_layers=args.num_layers,
            nhead=args.nhead,
            dim_feedforward=args.dim_feedforward,
            dropout=args.dropout,
            max_len=args.max_len,
        ).to(device)
    else:
        model = LM(
            vocab_size=vocab_size,
            embed_dim=args.embed_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            model_type=args.model,
            dropout=args.dropout,
        ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")

    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_ppl = float("inf")

    print(f"\n{'Epoch':>6}  {'Train Loss':>10}  {'Train PPL':>10}  {'Val Loss':>10}  {'Val PPL':>10}")
    print("-" * 56)

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_ppl = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        with torch.no_grad():
            va_loss, va_ppl = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

        marker = "  *" if va_ppl < best_val_ppl else ""
        if va_ppl < best_val_ppl:
            best_val_ppl = va_ppl
            torch.save({
                "model_state": model.state_dict(),
                "char2idx": char2idx,
                "idx2char": idx2char,
                "args": vars(args),
            }, args.save)

        print(f"{epoch:>6}  {tr_loss:>10.4f}  {tr_ppl:>10.2f}  {va_loss:>10.4f}  {va_ppl:>10.2f}{marker}")

    print(f"\n训练完成。最佳验证 PPL: {best_val_ppl:.2f}  已保存至 {args.save}")


if __name__ == "__main__":
    main()
