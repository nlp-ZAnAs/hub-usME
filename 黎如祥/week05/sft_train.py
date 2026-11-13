# -*- coding: utf-8 -*-
"""
Week05 作业：基于语言模型的 SFT（Supervised Fine-Tuning）训练

把「只会接龙」的预训练模型 Qwen2-0.5B，用 (instruction, output) 数据微调成
「看到指令就回答、答完就停」的助手。核心只有三件事：

  1. Chat 模板：把一条样本拼成 <|im_start|>user ... <|im_end|><|im_start|>assistant ... <|im_end|>
  2. Loss Mask：prompt 段 labels 填 -100，只在回答段算交叉熵
  3. 参数高效微调：默认 LoRA（r=16），也可以 --no-lora 做全参数微调

用法：
    python sft_train.py --model D:\\Env\\models\\Qwen2-0.5B
    python sft_train.py --model ... --no-mask          # 消融：不遮 prompt
    python sft_train.py --model ... --no-lora --lr 2e-5  # 全参数 SFT（显存/磁盘要求高）
"""

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

# Anaconda 里 torch 和 matplotlib 同时加载会撞 OpenMP 运行库
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

ROOT = Path(__file__).resolve().parent
DEFAULT_TRAIN = ROOT / "data" / "sft_train.jsonl"
DEFAULT_VAL = ROOT / "data" / "sft_val.jsonl"


# --------------------------------------------------------------------------- #
# 数据
# --------------------------------------------------------------------------- #
def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def encode_example(tok, ex, max_len, mask_prompt=True):
    """一条 (instruction, output) -> input_ids / labels。

    labels 里 prompt 段是 -100（不算 loss），回答段是真实 token id。
    """
    messages = [
        {"role": "user", "content": ex["instruction"]},
        {"role": "assistant", "content": ex["output"]},
    ]
    full_text = tok.apply_chat_template(messages, tokenize=False)
    prompt_text = tok.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=True
    )

    full_ids = tok(full_text, add_special_tokens=False)["input_ids"]
    prompt_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]

    # 全序列必须严格以 prompt 开头，否则 mask 会错位
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("prompt 不是 full 的前缀，chat 模板有问题")

    if len(full_ids) > max_len:
        full_ids = full_ids[:max_len]
    if len(prompt_ids) >= len(full_ids):
        raise ValueError("回答被截断没了：请调大 --max-len")

    if mask_prompt:
        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
    else:
        labels = list(full_ids)  # 消融：连提问一起背

    return {
        "input_ids": full_ids,
        "labels": labels,
        "prompt_len": len(prompt_ids),
        "n_resp": len(full_ids) - len(prompt_ids),
    }


class SFTDataset(Dataset):
    def __init__(self, rows, tok, max_len, mask_prompt=True):
        self.items = [encode_example(tok, r, max_len, mask_prompt) for r in rows]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


class PadCollator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, batch):
        max_len = max(len(x["input_ids"]) for x in batch)
        input_ids = torch.full((len(batch), max_len), self.pad_id, dtype=torch.long)
        attention = torch.zeros((len(batch), max_len), dtype=torch.long)
        labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
        for i, x in enumerate(batch):
            n = len(x["input_ids"])
            input_ids[i, :n] = torch.tensor(x["input_ids"], dtype=torch.long)
            attention[i, :n] = 1
            labels[i, :n] = torch.tensor(x["labels"], dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attention, "labels": labels}


# --------------------------------------------------------------------------- #
# 评估
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate(model, loader, device, dtype):
    """返回 (token 平均 loss, ppl, next-token 准确率)。只在回答段统计。"""
    model.eval()
    total_loss = 0.0
    total_tok = 0
    correct = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast("cuda", dtype=dtype, enabled=(dtype != torch.float32)):
            out = model(**batch)
        n = int((batch["labels"] != -100).sum())
        total_loss += out.loss.item() * n
        total_tok += n

        logits = out.logits[:, :-1, :]
        tgt = batch["labels"][:, 1:]
        mask = tgt != -100
        pred = logits.argmax(-1)
        correct += int(((pred == tgt) & mask).sum())
    mean_loss = total_loss / max(total_tok, 1)
    return mean_loss, math.exp(mean_loss), correct / max(total_tok, 1)


def make_loader(ds, batch_size, shuffle, pad_id):
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=PadCollator(pad_id),
        pin_memory=True,
        drop_last=False,
    )


# --------------------------------------------------------------------------- #
# 杂项
# --------------------------------------------------------------------------- #
def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def dump_encoding(tok, ds, path, k=3):
    """把前 k 条样本的 token / labels 写出来，直观检查 loss mask。"""
    lines = []
    for i in range(min(k, len(ds))):
        item = ds[i]
        toks = [tok.decode([t]) for t in item["input_ids"]]
        lines.append(f"--- sample {i + 1}  (prompt {item['prompt_len']} tok, "
                     f"response {item['n_resp']} tok) ---")
        for t, lab in zip(toks, item["labels"]):
            mark = "  " if lab == -100 else ">>"
            lines.append(f"{mark} {t!r:12} label={lab}")
        lines.append("")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def plot_curve(history, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    steps = [h["step"] for h in history["train"]]
    losses = [h["loss"] for h in history["train"]]
    ax.plot(steps, losses, lw=0.8, alpha=0.45, label="train loss (per step)")
    # 滑动平均
    if len(losses) >= 20:
        window = max(1, len(losses) // 20)
        smooth = [
            sum(losses[max(0, i - window): i + 1]) / len(losses[max(0, i - window): i + 1])
            for i in range(len(losses))
        ]
        ax.plot(steps, smooth, lw=1.8, label=f"train loss (ma{window})")
    val_steps = [h["step"] for h in history["val"]]
    val_losses = [h["loss"] for h in history["val"]]
    ax.plot(val_steps, val_losses, "o-", lw=1.8, label="val loss (response only)")
    ax.set_yscale("log")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("cross-entropy (log)")
    ax.set_title("SFT loss")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    epochs = [h["epoch"] for h in history["val"]]
    ax.plot(epochs, [h["ppl"] for h in history["val"]], "s-", color="tab:red")
    for e, p in zip(epochs, [h["ppl"] for h in history["val"]]):
        ax.annotate(f"{p:.2f}", (e, p), textcoords="offset points", xytext=(0, 6),
                    ha="center", fontsize=8)
    ax.set_xlabel("epoch")
    ax.set_ylabel("val PPL")
    ax.set_title("validation perplexity (response tokens)")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="SFT Qwen2-0.5B on instruction data")
    parser.add_argument("--model", default=os.environ.get("SFT_MODEL", "Qwen/Qwen2-0.5B"),
                        help="基座模型路径或 HF id（默认读 SFT_MODEL 环境变量）")
    parser.add_argument("--train", default=str(DEFAULT_TRAIN))
    parser.add_argument("--val", default=str(DEFAULT_VAL))
    parser.add_argument("--output", default=str(ROOT / "outputs" / "lora_r16"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=None,
                        help="默认 LoRA 2e-4；--no-lora 时 2e-5")
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--no-lora", action="store_true", help="全参数微调（显存/磁盘占用大）")
    parser.add_argument("--no-mask", action="store_true", help="消融：prompt 也参与 loss")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0, help="只取前 N 条，调试用")
    parser.add_argument("--save-last", action="store_true", help="除了 best 再存一份最后权重")
    args = parser.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.txt"

    def log(msg):
        print(msg, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    log("=" * 78)
    log(f"Week05 SFT  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"model={args.model}")
    log(f"mask_prompt={not args.no_mask}  lora={not args.no_lora}  "
        f"epochs={args.epochs} batch={args.batch_size} grad_accum={args.grad_accum}")

    # ---------- tokenizer ----------
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    log(f"tokenizer: vocab={len(tok)} pad={tok.pad_token!r}({tok.pad_token_id}) "
        f"eos={tok.eos_token!r}({tok.eos_token_id})")

    train_rows = read_jsonl(args.train)
    val_rows = read_jsonl(args.val)
    if args.limit:
        train_rows = train_rows[: args.limit]
    train_ds = SFTDataset(train_rows, tok, args.max_len, mask_prompt=not args.no_mask)
    val_ds = SFTDataset(val_rows, tok, args.max_len, mask_prompt=True)

    def stats(ds):
        lens = [len(x["input_ids"]) for x in ds.items]
        p = [x["prompt_len"] for x in ds.items]
        r = [x["n_resp"] for x in ds.items]
        return {
            "n": len(ds), "tokens_mean": sum(lens) / len(lens), "tokens_max": max(lens),
            "prompt_mean": sum(p) / len(p), "resp_mean": sum(r) / len(r),
            "resp_share": sum(r) / sum(lens),
        }

    train_stats = stats(train_ds)
    val_stats = stats(val_ds)
    log(f"train: {train_stats}")
    log(f"val  : {val_stats}")
    dump_encoding(tok, train_ds, out_dir / "example_encoding.txt")

    train_loader = make_loader(train_ds, args.batch_size, True, tok.pad_token_id)
    val_loader = make_loader(val_ds, args.batch_size, False, tok.pad_token_id)

    # ---------- model ----------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32
    log(f"device={device} dtype={dtype}")
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
    model.to(device)
    model.config.use_cache = False
    n_all = sum(p.numel() for p in model.parameters())

    if not args.no_lora:
        from peft import LoraConfig, get_peft_model

        targets = ["q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj"]
        lora_cfg = LoraConfig(
            task_type="CAUSAL_LM",
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=targets,
            bias="none",
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log(f"LoRA r={args.lora_r} alpha={args.lora_alpha} dropout={args.lora_dropout}")
        log(f"trainable {n_train:,} / total {n_all:,} = {100 * n_train / n_all:.2f}%")
    else:
        n_train = n_all
        log(f"全参数微调，trainable {n_train:,}")

    trainable = [p for p in model.parameters() if p.requires_grad]
    lr = args.lr if args.lr is not None else (2e-4 if not args.no_lora else 2e-5)

    steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    warmup = max(1, int(total_steps * 0.03))
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.0, betas=(0.9, 0.999))
    sched = get_cosine_schedule_with_warmup(opt, warmup, total_steps)
    log(f"lr={lr} total_steps={total_steps} warmup={warmup}")

    history = {"train": [], "val": []}

    # ---------- 训练前基线（LoRA 的 B 初始化为 0，此时等价于原模型） ----------
    t0 = time.time()
    base_loss, base_ppl, base_acc = evaluate(model, val_loader, device, dtype)
    log(f"[step 0 / base] val_loss={base_loss:.4f} ppl={base_ppl:.2f} acc={base_acc:.4f}")
    history["val"].append({"epoch": 0, "step": 0, "loss": base_loss,
                           "ppl": base_ppl, "acc": base_acc})

    best_loss = base_loss
    best_epoch = 0
    global_step = 0
    opt.zero_grad(set_to_none=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        ep_start = time.time()
        running = 0.0
        running_tok = 0
        for i, batch in enumerate(train_loader):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=dtype, enabled=(dtype != torch.float32)):
                out = model(**batch)
                loss = out.loss
            (loss / args.grad_accum).backward()

            n_tok = int((batch["labels"] != -100).sum())
            running += loss.item() * n_tok
            running_tok += n_tok

            if (i + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                global_step += 1
                history["train"].append({"epoch": epoch, "step": global_step,
                                         "loss": loss.item()})
                if global_step % 20 == 0 or global_step == 1:
                    lr_now = sched.get_last_lr()[0]
                    log(f"epoch {epoch} step {global_step}/{total_steps} "
                        f"loss={loss.item():.4f} lr={lr_now:.2e}")

        train_loss = running / max(running_tok, 1)
        val_loss, val_ppl, val_acc = evaluate(model, val_loader, device, dtype)
        history["val"].append({"epoch": epoch, "step": global_step,
                               "loss": val_loss, "ppl": val_ppl, "acc": val_acc})
        log(f"[epoch {epoch}] train_loss={train_loss:.4f} | val_loss={val_loss:.4f} "
            f"ppl={val_ppl:.2f} acc={val_acc:.4f} | {time.time() - ep_start:.1f}s")

        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            model.save_pretrained(out_dir / "best")
            log(f"  -> new best (val_loss={best_loss:.4f}) saved to {out_dir / 'best'}")
        if args.save_last and epoch == args.epochs:
            model.save_pretrained(out_dir / "last")

    elapsed = time.time() - t0
    log(f"done in {elapsed:.1f}s, best epoch={best_epoch} val_loss={best_loss:.4f}")

    # ---------- 保存 ----------
    run_config = vars(args).copy()
    run_config.update({
        "n_all_params": n_all, "n_trainable_params": n_train,
        "train_stats": train_stats, "val_stats": val_stats,
        "device": str(device), "dtype": str(dtype),
        "elapsed_sec": round(elapsed, 1),
        "best_epoch": best_epoch, "best_val_loss": best_loss,
        "torch": torch.__version__,
    })
    (out_dir / "run_config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "metrics.json").write_text(
        json.dumps({"history": history, "run": run_config}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    plot_curve(history, out_dir / "loss_curve.png")
    log(f"saved: {out_dir / 'metrics.json'} , loss_curve.png")


if __name__ == "__main__":
    main()
