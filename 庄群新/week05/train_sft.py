# -*- coding: utf-8 -*-
"""
train_sft.py — 第五周作业：在第四周字符级 Transformer 上做 SFT（GPU）

和第四周 train_transformer_lm.py 的关系：同一套模型、同一个 optimizer，
只改三件事 ——

    1. 数据：从「任意文本切 64 字窗口」→「问：…答：…」的指令对（见 sft_data.py）
    2. loss：从「每个位置都算」→「只在答句段算」（labels == -100 的位置一律忽略）
    3. 起点：从零初始化 → 从第四周 tf_mask.pt 继续训
       （这才是 SFT 的完整含义：Supervised Fine-Tuning 的「前置」是预训练）

跑法：
    python -X utf8 train_sft.py                  # 默认 3 epoch
    python -X utf8 train_sft.py --epochs 5 --lr 3e-4
    python -X utf8 train_sft.py --rebuild        # 强制重建 base_sft.pt
"""

import argparse
import math
import os
import random
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from build_base import DEFAULT_BASE_CKPT, DEFAULT_DATA_DIR, build_base
from model import generate, load_model
from sft_data import (IGNORE_INDEX, SFTDataset, Vocab, make_collate_fn,
                      preview_one, load_jsonl)

HERE = os.path.dirname(os.path.abspath(__file__))


# ─────────────────────────────── loss ───────────────────────────────

def sft_loss(model, batch):
    """
    SFT 的 loss：只在答句段算，问题段和 padding 段全部 ignore。

    logits[:, :-1] 去对 labels[:, 1:] —— 位置 t 的输出预测第 t+1 个字。
    因为答句段第一个字的 label 就落在「答：」后面，所以模型第一步学的是
    「紧跟『答：』之后该说什么」，而不是「答：这两个字本身」（那两个字节已经在 prompt 里给了）。
    """
    logits = model(batch["input_ids"], batch["pad_mask"])
    shift_logits = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
    shift_labels = batch["labels"][:, 1:].contiguous().view(-1)
    return F.cross_entropy(shift_logits, shift_labels, ignore_index=IGNORE_INDEX)


@torch.no_grad()
def evaluate(model, loader, device):
    """验证集：返回 (答段平均 loss, PPL)。只统计有 label 的位置。"""
    model.eval()
    total, n_tokens = 0.0, 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(batch["input_ids"], batch["pad_mask"])
        shift_logits = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
        shift_labels = batch["labels"][:, 1:].contiguous().view(-1)
        total += float(F.cross_entropy(shift_logits, shift_labels,
                                       ignore_index=IGNORE_INDEX, reduction="sum"))
        n_tokens += int((shift_labels != IGNORE_INDEX).sum())
    loss = total / max(n_tokens, 1)
    return loss, math.exp(loss)


# ─────────────────────────────── 主流程 ───────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_ckpt", default=DEFAULT_BASE_CKPT, help="第四周 tf_mask.pt")
    ap.add_argument("--data_dir", default=DEFAULT_DATA_DIR, help="sft_*.jsonl 所在目录")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4,
                    help="比第四周的 1e-3 小一个量级：现在是继续训，不是从零训")
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--rebuild", action="store_true", help="强制重建 base_sft.pt")
    ap.add_argument("--out", default="sft.pt")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}" + (f"  {torch.cuda.get_device_name(0)}" if device.type == "cuda" else ""))

    # ── 1) 准备 base：第四周权重 + 扩好的字表/位置编码 ──
    base_path = os.path.join(HERE, "base_sft.pt")
    if args.rebuild or not os.path.exists(base_path):
        build_base(args.base_ckpt, args.data_dir, args.max_len, verbose=False)
    model, vocab, ck_args = load_model(base_path, device, dropout=0.1)
    model.to(device).train()
    print(f"[vocab] {len(vocab)} 字（含 pad/eos/unk）   max_len={ck_args['max_len']}")

    # ── 2) 数据 ──
    train_rows = load_jsonl(os.path.join(args.data_dir, "sft_train.jsonl"))
    val_rows = load_jsonl(os.path.join(args.data_dir, "sft_val.jsonl"))
    ds_tr = SFTDataset(train_rows, vocab, args.max_len)
    ds_va = SFTDataset(val_rows, vocab, args.max_len, verbose=False)
    collate = make_collate_fn(vocab.pad_id)
    loader_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    loader_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    print(f"[data] train {len(ds_tr)} 条 / {len(loader_tr)} 步每 epoch   val {len(ds_va)} 条")

    # ── 3) 先看一眼 Loss Mask 长什么样（训练前唯一必须做的自检）──
    preview_one(vocab, ds_tr, index=0)

    # 训练前的 base 行为：它还没学过「答」，应该会接着列问题
    print("\n[SFT 之前] base 模型对同一问题的输出：")
    q = "问：" + train_rows[0]["instruction"] + "\n答："
    print(f"  问题：{train_rows[0]['instruction']}")
    print(f"  参考答案：{train_rows[0]['output']}")
    print(f"  base 生成：{generate(model, vocab, q, 30, device=device)!r}")

    # ── 4) 训练 ──
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_val = float("inf")

    print(f"\n{'Epoch':>6} {'TrainLoss':>10} {'ValLoss':>9} {'ValPPL':>8} {'tok/s':>9}  用时")
    print("-" * 56)

    for ep in range(1, args.epochs + 1):
        model.train()
        st, seen, total = time.time(), 0, 0.0
        for batch in loader_tr:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = sft_loss(model, batch)

            opt.zero_grad()
            loss.backward()
            nn_clip(model)          # 梯度裁剪，防共享权重偶发尖刺（沿用第四周做法）
            opt.step()

            n = int((batch["labels"][:, 1:] != IGNORE_INDEX).sum())
            total += float(loss) * n
            seen += n

        dt = time.time() - st
        v_loss, v_ppl = evaluate(model, loader_va, device)

        mark = ""
        if v_loss < best_val:
            best_val = v_loss
            mark = " *"
            torch.save({"model_state": model.state_dict(), "args": ck_args,
                        **vocab.to_ckpt_fields(), "val_loss": v_loss},
                       os.path.join(HERE, args.out))

        print(f"{ep:>6} {total/max(seen,1):>10.4f} {v_loss:>9.4f} {v_ppl:>8.2f} "
              f"{seen/dt:>9,.0f}  {dt:.1f}s{mark}")

        # 每轮抽 2 条训练集问题看行为变化（贪心解码，结果可复现）
        model.eval()
        for r in train_rows[:2]:
            p = "问：" + r["instruction"] + "\n答："
            print(f"    Q: {r['instruction']}")
            print(f"    A: {generate(model, vocab, p, 40, device=device)!r}")
        model.train()

    print(f"\n[sft] 最佳验证 loss {best_val:.4f}，已保存 {args.out}")
    print(f"[next] python -X utf8 eval_sft.py     # base 与 SFT 并排对照")


def nn_clip(model):
    """单独包一层，让训练循环里那行看起来不像魔法。"""
    import torch.nn as nn
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)


if __name__ == "__main__":
    main()
