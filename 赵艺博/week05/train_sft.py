# -*- coding: utf-8 -*-
"""
train_sft.py — 作业：基于语言模型做 SFT（监督微调）

在第四周训好的字符级 Transformer 语言模型上继续训练。
数据是课件里的指令-回答对（data/sft_train.jsonl、data/sft_val.jsonl）。

和「把整段话都拿来预测下一个字」的差别：
    样本拼成「指令：…\\n回答：…」
    只在回答的字上算交叉熵，指令部分的标签写成 -100，不产生损失
    模型学的是按指令作答，而不是把指令再抄一遍

原模型上下文只有 32 个字。这里把位置编码插值扩到 96：新位置是相邻旧位置的
加权平均，而不是把 32 个位置向量复制三份。微调时这些位置继续更新。

第四周词表只有 2000 字，指令里的空格和一部分汉字会被静默丢掉。这里先把
训练集和验证集中没见过的字补进词表，新字的向量用旧词向量的均值加小噪声。

用法：
    python train_sft.py
    python train_sft.py --epochs 16 --lr 3e-4
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import sys
import time
import random
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# ─── 超参数 ───────────────────────────────────────────────────────────
SEED = 42
MAX_LEN = 96          # 输入序列长度。指令+回答的中位长度约 64，96 能盖住大部分样本
BATCH_SIZE = 16
LR = 3e-4
EPOCHS = 24
DROPOUT = 0.0      # 监督微调样本少，沿用预训练的 0.1 会让回答分布更平
EXTRA_LAYERS = 2   # 在第四周的层上面加新层，新层一开始是恒等映射
HEAD_CHARS = 8     # 回答开头这几个字决定「像不像训练集的格式」
HEAD_BOOST = 3.0   # 开头几个字的损失权重大一些，避免只去拟合后面的细节
THREADS = 8
IGNORE = -100
MAX_NEW = 40
REP_PENALTY = 1.0

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(HERE, "data", "sft_train.jsonl")
VAL_PATH = os.path.join(HERE, "data", "sft_val.jsonl")
BASE_PATH = os.path.normpath(os.path.join(HERE, "..", "week04", "transformer_lm.pt"))


# ─── 1. 数据 ──────────────────────────────────────────────────────────
def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            instruction = obj["instruction"].strip()
            extra = (obj.get("input") or "").strip()
            if extra:
                instruction = instruction + "\n" + extra
            output = obj["output"].strip()
            if instruction and output:
                rows.append((instruction, output))
    return rows


def encode(text, char2idx):
    return [char2idx[c] for c in text if c in char2idx]


def build_example(instruction, output, char2idx, max_len):
    """
    拼成一条训练样本。
    x[i] 用来预测 y[i]（即下一个字）。
    下一个字还落在指令里时，y[i] = -100，损失会跳过它。
    """
    prompt = f"指令：{instruction}\n回答："
    prompt_ids = encode(prompt, char2idx)
    output_ids = encode(output, char2idx)
    if not prompt_ids or not output_ids:
        return None

    # x 的长度最多 max_len，所以整段最多 max_len+1 个字
    budget = max_len + 1
    if len(prompt_ids) + len(output_ids) > budget:
        keep_out = min(len(output_ids), max(1, budget // 3))
        keep_prompt = budget - keep_out
        if keep_prompt < 1:
            return None
        prompt_ids = prompt_ids[-keep_prompt:]
        output_ids = output_ids[:keep_out]

    full = prompt_ids + output_ids
    prompt_len = len(prompt_ids)
    x = full[:-1]
    y = []
    for i in range(1, len(full)):
        # full[i] 是被预测的字。它属于回答时才计损失（i >= prompt_len）
        y.append(full[i] if i >= prompt_len else IGNORE)
    if not any(t != IGNORE for t in y):
        return None
    return (
        torch.tensor(x, dtype=torch.long),
        torch.tensor(y, dtype=torch.long),
    )


class SFTDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


def collate(batch):
    width = max(x.numel() for x, _ in batch)
    xs, ys, pads = [], [], []
    for x, y in batch:
        n = x.numel()
        pad_n = width - n
        xs.append(F.pad(x, (0, pad_n), value=0))
        ys.append(F.pad(y, (0, pad_n), value=IGNORE))
        pads.append(torch.tensor([False] * n + [True] * pad_n))
    return torch.stack(xs), torch.stack(ys), torch.stack(pads)


def make_pairs(rows, char2idx, max_len):
    pairs = []
    kept_chars = 0
    raw_chars = 0
    supervised = 0
    for instruction, output in rows:
        raw_chars += len(instruction) + len(output)
        kept_chars += len(encode(instruction + output, char2idx))
        pair = build_example(instruction, output, char2idx, max_len)
        if pair is None:
            continue
        pairs.append(pair)
        supervised += int((pair[1] != IGNORE).sum())
    return pairs, raw_chars, kept_chars, supervised


# ─── 2. 模型（与第四周同一结构，便于直接加载权重）────────────────────
class TransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, max_len, dropout):
        super().__init__()
        self.max_len = max_len
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.drop = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(
            layer, num_layers=n_layers, enable_nested_tensor=False)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x, pad_mask=None):
        _, t = x.shape
        if t > self.max_len:
            raise ValueError(f"序列长度 {t} 超过位置编码上限 {self.max_len}")
        pos = torch.arange(t, device=x.device)
        h = self.drop(self.embed(x) + self.pos(pos)[None, :, :])
        causal = torch.triu(torch.ones(t, t, device=x.device, dtype=torch.bool), diagonal=1)
        h = self.blocks(h, mask=causal, src_key_padding_mask=pad_mask)
        h = self.ln(h)
        return F.linear(h, self.embed.weight)


def extend_vocab(char2idx, idx2char, texts):
    """把指令数据里词表没有的字补进去。空格和生僻字否则会在编码时被删掉。"""
    counts = {}
    for text in texts:
        for ch in text:
            if ch not in char2idx:
                counts[ch] = counts.get(ch, 0) + 1
    for ch, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        idx = len(char2idx)
        char2idx[ch] = idx
        idx2char[idx] = ch
    return len(counts)


def interpolate_pos(old_pos, new_len):
    """按顺序把位置向量插值拉长。复制粘贴会让位置 0、32、64 用同一个向量。"""
    old_len = old_pos.shape[0]
    if old_len == new_len:
        return old_pos.contiguous()
    if new_len < old_len:
        return old_pos[:new_len].contiguous()
    coord = torch.linspace(0, old_len - 1, new_len)
    left = coord.floor().long()
    right = (left + 1).clamp(max=old_len - 1)
    alpha = (coord - left.to(coord.dtype)).unsqueeze(1)
    mixed = old_pos[left] * (1 - alpha) + old_pos[right] * alpha
    return mixed.contiguous()


def same_vocab(saved, current):
    if len(saved) != len(current):
        return False
    for ch, idx in current.items():
        if saved.get(ch) != idx:
            return False
    return True


def zero_layer(layer):
    """输出投影置零后，残差层先按恒等映射工作，再在微调里学新东西。"""
    nn.init.zeros_(layer.self_attn.out_proj.weight)
    nn.init.zeros_(layer.self_attn.out_proj.bias)
    nn.init.zeros_(layer.linear2.weight)
    nn.init.zeros_(layer.linear2.bias)


def load_weights(model, path):
    """按形状能对上的键加载。词表变大时，新字用已有字向量的均值加小噪声。"""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = dict(ckpt["model_state"])
    old_embed = state["embed.weight"]
    new_vocab, dim = model.embed.weight.shape
    if old_embed.shape[0] < new_vocab:
        extra = new_vocab - old_embed.shape[0]
        center = old_embed.mean(dim=0, keepdim=True)
        noise = torch.randn(extra, dim) * 0.02
        state["embed.weight"] = torch.cat([old_embed, center + noise], dim=0)
    elif old_embed.shape[0] > new_vocab:
        state["embed.weight"] = old_embed[:new_vocab]
    if state["pos.weight"].shape[0] != model.pos.weight.shape[0]:
        state["pos.weight"] = interpolate_pos(state["pos.weight"], model.pos.weight.shape[0])
    own = model.state_dict()
    merged = {k: own[k] for k in own}
    loaded_layers = set()
    for key, value in state.items():
        if key not in own or tuple(own[key].shape) != tuple(value.shape):
            continue
        merged[key] = value
        if key.startswith("blocks.layers."):
            loaded_layers.add(int(key.split(".")[2]))
    model.load_state_dict(merged)
    return ckpt, loaded_layers


# ─── 3. 训练 / 生成 ───────────────────────────────────────────────────
def run_epoch(model, loader, optimizer, train):
    """记录的损失仍是回答里每个字的平均交叉熵。训练时把开头几个字的权重加大。"""
    model.train(train)
    total_loss = 0.0
    total_tokens = 0
    correct = 0
    for x, y, pad in loader:
        logits = model(x, pad)
        raw = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
            ignore_index=IGNORE,
            reduction="none",
        ).view_as(y)
        valid = y != IGNORE
        n = int(valid.sum())
        plain = (raw * valid).sum() / max(n, 1)
        if train:
            seen = valid.long().cumsum(dim=1)
            weight = torch.where(valid, torch.ones_like(raw), torch.zeros_like(raw))
            weight = torch.where(
                valid & (seen <= HEAD_CHARS),
                torch.full_like(raw, HEAD_BOOST),
                weight,
            )
            loss = (raw * weight).sum() / weight.sum().clamp(min=1)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        total_loss += float(plain.detach()) * n
        total_tokens += n
        pred = logits.argmax(dim=-1)
        correct += int(((pred == y) & valid).sum())
    avg = total_loss / max(1, total_tokens)
    acc = correct / max(1, total_tokens)
    return avg, acc


def _penalize_repeats(logits, generated, penalty):
    """压低刚生成过的字，并禁止把已经出现过的 8 字片段再接一遍。"""
    if penalty != 1.0:
        for prev in set(generated[-12:]):
            if logits[prev] > 0:
                logits[prev] /= penalty
            else:
                logits[prev] *= penalty
    n = 8
    if len(generated) < n - 1:
        return
    tail = tuple(generated[-(n - 1):])
    for i in range(len(generated) - n + 1):
        if tuple(generated[i:i + n - 1]) == tail:
            logits[generated[i + n - 1]] = -1e9


@torch.no_grad()
def generate_answer(model, char2idx, idx2char, instruction, max_len, max_new=MAX_NEW,
                    repetition_penalty=REP_PENALTY, finish_early=True):
    """从「回答：」后面逐字生成。只惩罚已生成的字，指令里的数字仍可照抄。"""
    model.eval()
    prompt = f"指令：{instruction}\n回答："
    ids = encode(prompt, char2idx)
    if not ids:
        return ""
    prompt_len = len(ids)
    newline = char2idx.get("\n")
    pieces = []
    for _ in range(max_new):
        window = ids[-max_len:]
        x = torch.tensor(window, dtype=torch.long).unsqueeze(0)
        logits = model(x)[0, -1].clone()
        generated = ids[prompt_len:]
        _penalize_repeats(logits, generated, repetition_penalty)
        if newline is not None and len(pieces) < 4:
            logits[newline] = -1e9
        nxt = int(logits.argmax())
        ch = idx2char[nxt]
        if ch == "\n":
            break
        pieces.append(ch)
        ids.append(nxt)
        if not finish_early:
            continue
        if len(pieces) >= 4 and len(set(pieces[-4:])) == 1:
            break
        if ch == "。" and len(pieces) >= 8:
            break
    return "".join(pieces)


def opening_rates(model, rows, char2idx, idx2char, max_len, n=80, seed=1):
    """训练集原句生成后，开头若干字和标准答案一致的比例。只生成开头 8 个字。"""
    rng = random.Random(seed)
    pick = rows if len(rows) <= n else rng.sample(rows, n)
    hits = {1: 0, 4: 0, 8: 0}
    for instruction, gold in pick:
        pred = generate_answer(
            model, char2idx, idx2char, instruction, max_len,
            max_new=8, repetition_penalty=1.0, finish_early=False,
        )
        for k in hits:
            if pred[:k] == gold[:k]:
                hits[k] += 1
    total = max(1, len(pick))
    return {k: hits[k] / total for k in hits}


def find_demos(rows):
    """挑几条格式不一样的训练指令，用来看模型能不能把该有的开头写出来。"""
    keys = ["平均值", "待办", "语法", "相加"]
    found = []
    used = set()
    for key in keys:
        for i, (instruction, output) in enumerate(rows):
            if i in used or key not in instruction:
                continue
            found.append((instruction, output))
            used.add(i)
            break
    for i, pair in enumerate(rows):
        if len(found) >= 4:
            break
        if i not in used:
            found.append(pair)
            used.add(i)
    return found[:4]


def clone_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def plot_curve(history, path, base_loss, base_acc):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(epochs, history["train_loss"], "b-o", label="train loss")
    axes[0].plot(epochs, history["val_loss"], "r-o", label="val loss")
    axes[0].axhline(base_loss, color="gray", ls="--", label=f"微调前 {base_loss:.2f}")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("response cross entropy")
    axes[0].set_title("回答部分的损失")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["train_acc"], "b-o", label="train acc")
    axes[1].plot(epochs, history["val_acc"], "r-o", label="val acc")
    axes[1].axhline(base_acc, color="gray", ls="--", label=f"微调前 {base_acc:.3f}")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("token accuracy")
    hi = max(history["train_acc"] + history["val_acc"] + [base_acc])
    axes[1].set_ylim(0, min(1.0, max(0.45, hi + 0.08)))
    axes[1].set_title("回答部分的字准确率")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("SFT：在语言模型上监督微调")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def short(text, n=48):
    text = text.replace("\n", " ")
    return text if len(text) <= n else text[:n] + "…"


# ─── 4. 主流程 ────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default=TRAIN_PATH)
    parser.add_argument("--val", default=VAL_PATH)
    parser.add_argument("--base", default=BASE_PATH)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max-len", type=int, default=MAX_LEN)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--dropout", type=float, default=DROPOUT)
    parser.add_argument("--extra-layers", type=int, default=EXTRA_LAYERS)
    parser.add_argument("--resume", default=os.path.join(HERE, "sft_lm.pt"))
    parser.add_argument("--threads", type=int, default=THREADS)
    parser.add_argument("--save", default=os.path.join(HERE, "sft_lm.pt"))
    parser.add_argument("--curve", default=os.path.join(HERE, "train_curve.png"))
    parser.add_argument("--report", default=os.path.join(HERE, "result.txt"))
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    random.seed(SEED)
    torch.manual_seed(SEED)

    for path in (args.train, args.val, args.base):
        if not os.path.exists(path):
            raise FileNotFoundError(f"找不到文件：{path}")

    base = torch.load(args.base, map_location="cpu", weights_only=False)
    char2idx = dict(base["char2idx"])
    idx2char = {int(k): v for k, v in base["idx2char"].items()}
    base_vocab = len(char2idx)

    train_rows = load_jsonl(args.train)
    val_rows = load_jsonl(args.val)
    vocab_text = [f"指令：{ins}\n回答：{out}" for ins, out in train_rows + val_rows]
    added = extend_vocab(char2idx, idx2char, vocab_text)
    vocab_size = len(char2idx)
    train_pairs, train_raw, train_kept, train_sup = make_pairs(train_rows, char2idx, args.max_len)
    val_pairs, _, _, val_sup = make_pairs(val_rows, char2idx, args.max_len)
    if not train_pairs or not val_pairs:
        raise RuntimeError("没有可用样本。请检查词表是否盖住了指令数据。")

    train_loader = DataLoader(
        SFTDataset(train_pairs), batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(
        SFTDataset(val_pairs), batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    n_layers = base["n_layers"] + args.extra_layers
    model = TransformerLM(
        vocab_size=vocab_size,
        d_model=base["d_model"],
        n_heads=base["n_heads"],
        n_layers=n_layers,
        d_ff=base["d_ff"],
        max_len=args.max_len,
        dropout=args.dropout,
    )
    _, loaded_base = load_weights(model, args.base)
    fresh_layers = [i for i in range(n_layers) if i not in loaded_base]
    for i in fresh_layers:
        zero_layer(model.blocks.layers[i])
    resumed = False
    if args.resume and os.path.exists(args.resume) and os.path.abspath(args.resume) != os.path.abspath(args.base):
        prev = torch.load(args.resume, map_location="cpu", weights_only=False)
        if same_vocab(prev.get("char2idx", {}), char2idx):
            _, loaded_resume = load_weights(model, args.resume)
            resumed = True
            fresh_layers = [i for i in range(n_layers) if i not in loaded_resume]
            for i in fresh_layers:
                zero_layer(model.blocks.layers[i])
        else:
            print("已有微调权重的词表对不上，改为从第四周继续")
    n_params = sum(p.numel() for p in model.parameters())
    pos_cos = 1.0
    if model.pos.weight.shape[0] > 32:
        pos_cos = F.cosine_similarity(
            model.pos.weight[0], model.pos.weight[32], dim=0).item()

    print(f"device: cpu   threads: {args.threads}")
    print(f"基座模型: {args.base}")
    print(f"继续已有微调: {resumed}    新层（恒等初始化）: {fresh_layers}")
    print(f"词表: {vocab_size}（第四周 {base_vocab}，补充 {added}）    "
          f"参数量: {n_params:,}    层数: {n_layers}    上下文: {args.max_len}")
    print(f"位置编码: 插值扩展，位置 0 与 32 的余弦相似度 {pos_cos:.3f}")
    print(f"训练样本: {len(train_pairs):,} / {len(train_rows):,}    "
          f"验证样本: {len(val_pairs):,} / {len(val_rows):,}")
    cover = train_kept / max(1, train_raw)
    print(f"训练集字符覆盖: {cover:.1%}")
    print(f"计入损失的回答字数: 训练 {train_sup:,}    验证 {val_sup:,}")

    with torch.no_grad():
        base_loss, base_acc = run_epoch(model, val_loader, optimizer=None, train=False)
    before_rates = opening_rates(model, train_rows, char2idx, idx2char, args.max_len)
    print(f"加训前验证: loss={base_loss:.4f}  acc={base_acc:.3f}")
    print(
        f"加训前训练集开头重合: 1字 {before_rates[1]:.1%}  "
        f"4字 {before_rates[4]:.1%}  8字 {before_rates[8]:.1%}"
    )

    demo_rows = val_rows[:4]
    train_demos = find_demos(train_rows)
    before = [generate_answer(model, char2idx, idx2char, ins, args.max_len) for ins, _ in demo_rows]
    train_before = [generate_answer(model, char2idx, idx2char, ins, args.max_len) for ins, _ in train_demos]

    fresh_ids = set()
    for i in fresh_layers:
        for param in model.blocks.layers[i].parameters():
            fresh_ids.add(id(param))
    old_params = [p for p in model.parameters() if id(p) not in fresh_ids]
    new_params = [p for p in model.parameters() if id(p) in fresh_ids]
    groups = [{"params": old_params, "lr": args.lr}]
    if new_params:
        groups.append({"params": new_params, "lr": args.lr * 3})
    optimizer = torch.optim.AdamW(groups, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.3)
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_loss = float("inf")
    best_epoch = 0
    best_acc = 0.0
    best_prefix = before_rates[4]
    best_prefix_epoch = 0
    best_rates = dict(before_rates)
    best_state = clone_state(model)

    def save_ckpt(epoch, prefix, rates):
        torch.save({
            "model_state": model.state_dict(),
            "char2idx": char2idx,
            "idx2char": idx2char,
            "vocab_size": vocab_size,
            "d_model": base["d_model"],
            "n_heads": base["n_heads"],
            "n_layers": n_layers,
            "d_ff": base["d_ff"],
            "dropout": args.dropout,
            "max_len": args.max_len,
            "base_val_loss": base_loss,
            "base_val_acc": base_acc,
            "best_val_loss": best_loss,
            "best_val_acc": best_acc,
            "best_epoch": best_epoch,
            "prefix_epoch": epoch,
            "prefix_rates": rates,
        }, args.save)

    print(f"\n{'Epoch':>6}  {'Train Loss':>10}  {'Train Acc':>10}  {'Val Loss':>10}  {'Val Acc':>10}")
    print("-" * 56)
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, optimizer, train=True)
        with torch.no_grad():
            va_loss, va_acc = run_epoch(model, val_loader, optimizer=None, train=False)
        scheduler.step()
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)
        mark = ""
        if va_loss < best_loss:
            best_loss = va_loss
            best_acc = va_acc
            best_epoch = epoch
            mark = "  *"
        print(f"{epoch:>6}  {tr_loss:>10.4f}  {tr_acc:>10.3f}  {va_loss:>10.4f}  {va_acc:>10.3f}{mark}")
        if epoch % 6 == 0 or epoch == args.epochs:
            rates = opening_rates(model, train_rows, char2idx, idx2char, args.max_len)
            print(
                f"         训练集开头重合  1字 {rates[1]:.1%}  "
                f"4字 {rates[4]:.1%}  8字 {rates[8]:.1%}"
            )
            if rates[4] >= best_prefix:
                best_prefix = rates[4]
                best_prefix_epoch = epoch
                best_rates = rates
                best_state = clone_state(model)
                save_ckpt(epoch, best_prefix, rates)

    elapsed = time.time() - t0
    model.load_state_dict(best_state)
    save_ckpt(best_prefix_epoch, best_prefix, best_rates)
    after = [generate_answer(model, char2idx, idx2char, ins, args.max_len) for ins, _ in demo_rows]
    train_after = [generate_answer(model, char2idx, idx2char, ins, args.max_len) for ins, _ in train_demos]

    print(f"\n微调完成。耗时 {elapsed:.0f} 秒。验证 loss 最低 {best_loss:.4f}（epoch {best_epoch}）")
    print(f"训练集开头 4 字重合最高 {best_prefix:.1%}（epoch {best_prefix_epoch}），用这一轮的权重做生成")
    print(f"模型已保存：{args.save}")
    print("\n--- 训练集原句：加训前 / 加训后 / 参考回答 ---")
    for (ins, gold), pre, post in zip(train_demos, train_before, train_after):
        print(f"指令: {short(ins)}")
        print(f"  加训前: {short(pre, 60)}")
        print(f"  加训后: {short(post, 60)}")
        print(f"  参考:   {short(gold, 60)}")
    print("\n--- 验证集指令：加训前 / 加训后 / 参考回答 ---")
    for (ins, gold), pre, post in zip(demo_rows, before, after):
        print(f"指令: {short(ins)}")
        print(f"  加训前: {short(pre, 60)}")
        print(f"  加训后: {short(post, 60)}")
        print(f"  参考:   {short(gold, 60)}")

    plot_curve(history, args.curve, base_loss, base_acc)
    print(f"训练曲线已保存：{args.curve}")

    lines = [
        "作业：基于语言模型的 SFT 训练",
        "",
        "基座：第四周字符级 Transformer 语言模型（因果自注意力）",
        f"结构：d_model={base['d_model']}  heads={base['n_heads']}  layers={n_layers}  "
        f"ff={base['d_ff']}  max_len={args.max_len}",
        f"参数量：{n_params:,}",
        f"其中 {base['n_layers']} 层来自第四周，另外 {len(fresh_layers)} 层从恒等映射开始学",
        f"词表：第四周 {base_vocab} 字，补充 {added} 个新字，合计 {vocab_size}，训练集覆盖 {cover:.1%}",
        "位置编码：由 32 插值到当前长度，避免 0/32/64 共用同一个位置向量",
        f"dropout：{args.dropout}    权重衰减：0    回答前 {HEAD_CHARS} 个字损失权重 {HEAD_BOOST:g}",
        f"样本：训练 {len(train_pairs)}    验证 {len(val_pairs)}",
        "损失：只计算「回答」部分；指令部分标签为 -100",
        f"训练耗时：{elapsed:.0f} 秒（CPU，{args.epochs} 个 epoch，学习率 {args.lr}）",
        f"是否从上一轮微调继续：{resumed}",
        "",
        f"加训前验证 loss / 字准确率：{base_loss:.4f} / {base_acc:.3f}",
        f"验证 loss 最低 / 字准确率：{best_loss:.4f} / {best_acc:.3f}（epoch {best_epoch}）",
        f"加训前训练集开头重合：1字 {before_rates[1]:.1%}    4字 {before_rates[4]:.1%}    8字 {before_rates[8]:.1%}",
        f"加训后训练集开头重合：1字 {best_rates[1]:.1%}    4字 {best_rates[4]:.1%}    8字 {best_rates[8]:.1%}"
        f"（epoch {best_prefix_epoch}，生成用的是这一轮）",
        "",
        "各轮：",
        f"{'epoch':>5}  {'train_loss':>10}  {'train_acc':>10}  {'val_loss':>10}  {'val_acc':>10}",
    ]
    for i in range(args.epochs):
        lines.append(
            f"{i + 1:>5}  {history['train_loss'][i]:>10.4f}  {history['train_acc'][i]:>10.3f}  "
            f"{history['val_loss'][i]:>10.4f}  {history['val_acc'][i]:>10.3f}"
        )
    lines.append("")
    lines.append("训练集原句的生成对比：")
    for (ins, gold), pre, post in zip(train_demos, train_before, train_after):
        lines.append(f"指令: {short(ins, 80)}")
        lines.append(f"  加训前: {pre}")
        lines.append(f"  加训后: {post}")
        lines.append(f"  参考:   {gold}")
        lines.append("")
    lines.append("验证指令的生成对比：")
    for (ins, gold), pre, post in zip(demo_rows, before, after):
        lines.append(f"指令: {short(ins, 80)}")
        lines.append(f"  加训前: {pre}")
        lines.append(f"  加训后: {post}")
        lines.append(f"  参考:   {gold}")
        lines.append("")
    with open(args.report, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"结果已写入：{args.report}")


if __name__ == "__main__":
    main()
