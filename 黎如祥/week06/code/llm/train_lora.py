# -*- coding: utf-8 -*-
"""
方法五：Qwen3-0.6B + LoRA SFT（生成式分类）

和 BERT 的区别不是「模型大小」，而是**把分类当生成做**：
  输入 prompt → 模型直接生成标签文本（"正面" / "负面"）。
封闭标签任务上这通常不如判别式划算，但它能处理开放标签、能复用同一套 SFT 流程。

关键实现（和 Week05 的 SFT 作业同一套）：
  1. chat 模板训练/推理必须一致（Qwen3 传 enable_thinking=False）；
  2. Loss Mask：prompt 段 labels=-100，只在标签文本上算 loss；
  3. LoRA r=8，只训 0.4% 参数；推理时用同一模板 + greedy decoding。

用法：
  python code/llm/train_lora.py --dataset chnsenticorp --epochs 2
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.config import CKPT_DIR, DATASETS, QWEN_DIR, model_path, result_path
from common.data import eval_samples, label_names, load_dataset
from common.metrics import classification_report, pred_distribution

IM_END = 151645

PROMPTS = {
    "tnews": ("请把下面的新闻标题分类到这些类别之一：{labels}。只输出类别名称，不要解释。\n标题：{text}"),
    "chnsenticorp": "判断下面这条中文评论是正面还是负面，只输出「正面」或「负面」，不要解释。\n评论：{text}",
}


def user_prompt(dataset, text):
    labels = label_names(dataset)
    if dataset == "tnews":
        return PROMPTS["tnews"].format(labels="、".join(labels), text=text)
    return PROMPTS["chnsenticorp"].format(text=text)


def encode(tok, dataset, text, answer, max_len):
    msgs = [{"role": "user", "content": user_prompt(dataset, text)},
            {"role": "assistant", "content": answer}]
    full = tok.apply_chat_template(msgs, tokenize=False, enable_thinking=False)
    prompt = tok.apply_chat_template(msgs[:-1], tokenize=False,
                                     add_generation_prompt=True, enable_thinking=False)
    full_ids = tok(full, add_special_tokens=False)["input_ids"]
    prompt_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    assert full_ids[:len(prompt_ids)] == prompt_ids
    full_ids = full_ids[:max_len]
    if len(prompt_ids) >= len(full_ids):
        return None
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
    return {"input_ids": full_ids, "labels": labels}


class ClsDataset(Dataset):
    def __init__(self, rows, tok, dataset, max_len):
        labels = label_names(dataset)
        self.items = []
        for r in rows:
            item = encode(tok, dataset, r["sentence"], labels[r["label"]], max_len)
            if item:
                self.items.append(item)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def collate(batch, pad_id):
    n = max(len(x["input_ids"]) for x in batch)
    ids = torch.full((len(batch), n), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), n), -100, dtype=torch.long)
    attn = torch.zeros((len(batch), n), dtype=torch.long)
    for i, x in enumerate(batch):
        k = len(x["input_ids"])
        ids[i, :k] = torch.tensor(x["input_ids"])
        labels[i, :k] = torch.tensor(x["labels"])
        attn[i, :k] = 1
    return {"input_ids": ids, "attention_mask": attn, "labels": labels}


@torch.no_grad()
def evaluate_loss(model, loader, device):
    model.eval()
    total, toks = 0.0, 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(**batch)
        n = int((batch["labels"] != -100).sum())
        total += out.loss.item() * n
        toks += n
    return total / max(toks, 1)


def parse_label(dataset, raw):
    labels = label_names(dataset)
    text = raw.strip()
    if dataset == "tnews":
        hits = [x for x in labels if x in text]
        return labels.index(max(hits, key=len)) if hits else -1
    if "负面" in text:
        return labels.index("负面")
    if "正面" in text:
        return labels.index("正面")
    return -1


@torch.no_grad()
def generate_preds(model, tok, dataset, samples, device, batch_size=16, max_new=8):
    tok.padding_side = "left"
    preds, raws = [], []
    for start in range(0, len(samples), batch_size):
        chunk = samples[start:start + batch_size]
        texts = [tok.apply_chat_template(
            [{"role": "user", "content": user_prompt(dataset, s["sentence"])}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False) for s in chunk]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        out = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.pad_token_id,
                             eos_token_id=[IM_END, tok.eos_token_id])
        new = out[:, enc["input_ids"].shape[1]:]
        for row in new:
            raw = tok.decode(row, skip_special_tokens=True).strip()
            raws.append(raw)
            preds.append(parse_label(dataset, raw))
    tok.padding_side = "right"
    return preds, raws


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DATASETS))
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max_len", type=int, default=320)
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out_path = result_path(args.dataset, "qwen3_lora")
    if out_path.exists() and not args.force:
        print(f"[skip] {out_path} 已存在")
        return

    device = "cuda"
    torch.manual_seed(42)
    tok = AutoTokenizer.from_pretrained(str(model_path(QWEN_DIR)))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    train, _ = load_dataset(args.dataset)
    if args.limit:
        train = train[:args.limit]
    samples, meta = eval_samples(args.dataset, 500, 42)
    labels = label_names(args.dataset)

    ds_tr = ClsDataset(train, tok, args.dataset, args.max_len)
    loader = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True,
                        collate_fn=lambda b: collate(b, tok.pad_token_id))
    print(f"[data] train={len(train)} usable={len(ds_tr)} eval={len(samples)} "
          f"labels={len(labels)}", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        str(model_path(QWEN_DIR)), dtype=torch.bfloat16).to(device)
    model.config.use_cache = False
    from peft import LoraConfig, get_peft_model
    lora = LoraConfig(task_type="CAUSAL_LM", r=args.lora_r, lora_alpha=args.lora_r * 2,
                      lora_dropout=0.05, bias="none",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lora)
    n_train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    print(f"[lora] trainable {n_train_p/1e6:.2f}M / {n_all/1e6:.1f}M "
          f"({100*n_train_p/n_all:.2f}%)", flush=True)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=0.0)
    steps_per_epoch = math.ceil(len(loader) / args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    warmup = max(1, int(total_steps * 0.03))
    sched = get_cosine_schedule_with_warmup(opt, warmup, total_steps)

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        run, n = 0.0, 0
        opt.zero_grad(set_to_none=True)
        for i, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(**batch).loss
            (loss / args.grad_accum).backward()
            run += loss.item()
            n += 1
            if (i + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
            if i % 100 == 0:
                print(f"  epoch {epoch} step {i}/{len(loader)} loss={loss.item():.4f}", flush=True)
        print(f"[epoch {epoch}] train_loss={run/max(n,1):.4f} ({time.time()-t0:.0f}s)", flush=True)

    train_seconds = round(time.time() - t0, 1)

    t1 = time.time()
    preds, raws = generate_preds(model, tok, args.dataset, samples, device)
    infer_seconds = (time.time() - t1) / len(samples)
    y_true = [s["label"] for s in samples]
    report = classification_report(y_true, preds, len(labels))
    report["method"] = "qwen3_lora"
    report["pred_distribution"] = pred_distribution(preds, len(labels))

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    adapter_dir = CKPT_DIR / f"{args.dataset}_qwen3_lora"
    model.save_pretrained(adapter_dir)
    print(f"adapter -> {adapter_dir}", flush=True)

    result = {
        "method": "qwen3_lora",
        "display": "Qwen3-0.6B + LoRA SFT",
        "family": "llm_lora",
        "dataset": args.dataset,
        "meta": meta,
        "report": report,
        "cost": {"train_seconds": train_seconds,
                 "per_item_ms": round(infer_seconds * 1000, 2),
                 "params": n_all, "trainable_params": n_train_p, "peak_mem_gb": None},
        "preds": [{"text": s["sentence"], "true": int(s["label"]), "pred": int(p),
                   "prob": None, "raw": r} for s, p, r in zip(samples, preds, raws)],
    }
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[{args.dataset}] LoRA acc={report['acc_incl_unparsed']} "
          f"macro_f1={report['macro_f1']} used={report['pred_distribution']['used_classes']}/"
          f"{len(labels)} train={train_seconds}s")
    print(f"→ {out_path}")


if __name__ == "__main__":
    main()
