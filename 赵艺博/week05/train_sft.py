# -*- coding: utf-8 -*-
"""
train_sft.py — 作业：基于语言模型做 SFT（监督微调）

在第四周训好的字符级 Transformer 语言模型上继续训练。
数据是课件里的指令-回答对（data/sft_train.jsonl、data/sft_val.jsonl）。

和「把整段话都拿来预测下一个字」的差别：
    样本拼成「指令：…\\n回答：…」
    只在回答的字上算交叉熵，指令部分的标签写成 -100，不产生损失
    模型学的是按指令作答，而不是把指令再抄一遍

原模型上下文只有 32 个字，这里把位置编码扩到 96，多出来的位置用已学到的
位置向量循环填上，微调时继续更新。

用法：
    python train_sft.py
    python train_sft.py --epochs 6 --lr 3e-4
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
EPOCHS = 6
THREADS = 8
IGNORE = -100
MAX_NEW = 40

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


def load_base(model, path):
    """加载第四周权重。位置编码比原来长时，用已有位置循环补齐。"""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = dict(ckpt["model_state"])
    old_pos = state["pos.weight"]
    new_len = model.pos.weight.shape[0]
    if old_pos.shape[0] != new_len:
        reps = (new_len + old_pos.shape[0] - 1) // old_pos.shape[0]
        state["pos.weight"] = old_pos.repeat(reps, 1)[:new_len].contiguous()
    model.load_state_dict(state)
    return ckpt


# ─── 3. 训练 / 生成 ───────────────────────────────────────────────────
def run_epoch(model, loader, criterion, optimizer, train):
    model.train(train)
    total_loss = 0.0
    total_tokens = 0
    correct = 0
    for x, y, pad in loader:
        logits = model(x, pad)
        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        if train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        valid = y != IGNORE
        n = int(valid.sum())
        total_loss += loss.item() * n
        total_tokens += n
        pred = logits.argmax(dim=-1)
        correct += int(((pred == y) & valid).sum())
    avg = total_loss / max(1, total_tokens)
    acc = correct / max(1, total_tokens)
    return avg, acc


@torch.no_grad()
def generate_answer(model, char2idx, idx2char, instruction, max_len, max_new=MAX_NEW):
    """从「回答：」后面逐字生成，遇到换行或到达长度就停。"""
    model.eval()
    prompt = f"指令：{instruction}\n回答："
    ids = encode(prompt, char2idx)
    if not ids:
        return ""
    pieces = []
    for _ in range(max_new):
        window = ids[-max_len:]
        x = torch.tensor(window, dtype=torch.long).unsqueeze(0)
        nxt = int(model(x)[0, -1].argmax())
        ch = idx2char[nxt]
        if ch == "\n":
            break
        pieces.append(ch)
        ids.append(nxt)
        if len(pieces) >= 4 and len(set(pieces[-4:])) == 1:
            break
    return "".join(pieces)


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
    axes[1].set_ylim(0, 0.45)
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
    char2idx = base["char2idx"]
    idx2char = {int(k): v for k, v in base["idx2char"].items()}
    vocab_size = len(char2idx)

    train_rows = load_jsonl(args.train)
    val_rows = load_jsonl(args.val)
    train_pairs, train_raw, train_kept, train_sup = make_pairs(train_rows, char2idx, args.max_len)
    val_pairs, _, _, val_sup = make_pairs(val_rows, char2idx, args.max_len)
    if not train_pairs or not val_pairs:
        raise RuntimeError("没有可用样本。请检查词表是否盖住了指令数据。")

    train_loader = DataLoader(
        SFTDataset(train_pairs), batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(
        SFTDataset(val_pairs), batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    model = TransformerLM(
        vocab_size=vocab_size,
        d_model=base["d_model"],
        n_heads=base["n_heads"],
        n_layers=base["n_layers"],
        d_ff=base["d_ff"],
        max_len=args.max_len,
        dropout=base["dropout"],
    )
    load_base(model, args.base)
    n_params = sum(p.numel() for p in model.parameters())

    print(f"device: cpu   threads: {args.threads}")
    print(f"基座模型: {args.base}")
    print(f"词表: {vocab_size}    参数量: {n_params:,}    上下文: {args.max_len}")
    print(f"训练样本: {len(train_pairs):,} / {len(train_rows):,}    "
          f"验证样本: {len(val_pairs):,} / {len(val_rows):,}")
    cover = train_kept / max(1, train_raw)
    print(f"训练集汉字覆盖（在第四周词表内）: {cover:.1%}")
    print(f"计入损失的回答字数: 训练 {train_sup:,}    验证 {val_sup:,}")

    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE)
    with torch.no_grad():
        base_loss, base_acc = run_epoch(model, val_loader, criterion, optimizer=None, train=False)
    print(f"微调前验证: loss={base_loss:.4f}  acc={base_acc:.3f}")

    demo_rows = val_rows[:4]
    before = [generate_answer(model, char2idx, idx2char, ins, args.max_len) for ins, _ in demo_rows]

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_loss = float("inf")
    best_epoch = 0
    best_acc = 0.0

    print(f"\n{'Epoch':>6}  {'Train Loss':>10}  {'Train Acc':>10}  {'Val Loss':>10}  {'Val Acc':>10}")
    print("-" * 56)
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, train=True)
        with torch.no_grad():
            va_loss, va_acc = run_epoch(model, val_loader, criterion, optimizer, train=False)
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
            torch.save({
                "model_state": model.state_dict(),
                "char2idx": char2idx,
                "idx2char": idx2char,
                "vocab_size": vocab_size,
                "d_model": base["d_model"],
                "n_heads": base["n_heads"],
                "n_layers": base["n_layers"],
                "d_ff": base["d_ff"],
                "dropout": base["dropout"],
                "max_len": args.max_len,
                "base_val_loss": base_loss,
                "base_val_acc": base_acc,
                "best_val_loss": best_loss,
                "best_val_acc": best_acc,
                "best_epoch": best_epoch,
            }, args.save)
        print(f"{epoch:>6}  {tr_loss:>10.4f}  {tr_acc:>10.3f}  {va_loss:>10.4f}  {va_acc:>10.3f}{mark}")

    elapsed = time.time() - t0
    ckpt = torch.load(args.save, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    after = [generate_answer(model, char2idx, idx2char, ins, args.max_len) for ins, _ in demo_rows]

    print(f"\n微调完成。耗时 {elapsed:.0f} 秒。最佳验证 loss: {best_loss:.4f}（epoch {best_epoch}）")
    print(f"模型已保存：{args.save}")
    print("\n--- 同一条指令，微调前 / 微调后 / 参考回答 ---")
    for (ins, gold), pre, post in zip(demo_rows, before, after):
        print(f"指令: {short(ins)}")
        print(f"  微调前: {short(pre, 60)}")
        print(f"  微调后: {short(post, 60)}")
        print(f"  参考:   {short(gold, 60)}")

    plot_curve(history, args.curve, base_loss, base_acc)
    print(f"训练曲线已保存：{args.curve}")

    lines = [
        "作业：基于语言模型的 SFT 训练",
        "",
        "基座：第四周字符级 Transformer 语言模型（因果自注意力）",
        f"结构：d_model={base['d_model']}  heads={base['n_heads']}  layers={base['n_layers']}  "
        f"ff={base['d_ff']}  max_len={args.max_len}",
        f"参数量：{n_params:,}",
        f"词表：沿用第四周的 {vocab_size} 字，训练集覆盖 {cover:.1%}",
        f"样本：训练 {len(train_pairs)}    验证 {len(val_pairs)}",
        "损失：只计算「回答」部分；指令部分标签为 -100",
        f"训练耗时：{elapsed:.0f} 秒（CPU，{args.epochs} 个 epoch，学习率 {args.lr}）",
        "",
        f"微调前验证 loss / 字准确率：{base_loss:.4f} / {base_acc:.3f}",
        f"微调后最佳验证 loss / 字准确率：{best_loss:.4f} / {best_acc:.3f}（epoch {best_epoch}）",
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
    lines.append("同一条验证指令的生成对比：")
    for (ins, gold), pre, post in zip(demo_rows, before, after):
        lines.append(f"指令: {short(ins, 80)}")
        lines.append(f"  微调前: {pre}")
        lines.append(f"  微调后: {post}")
        lines.append(f"  参考:   {gold}")
        lines.append("")
    with open(args.report, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"结果已写入：{args.report}")


if __name__ == "__main__":
    main()
