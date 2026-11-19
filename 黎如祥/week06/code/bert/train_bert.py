# -*- coding: utf-8 -*-
"""
方法三：预训练编码器微调（本作业的「换模型」主实验）

同一个训练脚本换 `--backbone`：
  bert-base-chinese / chinese-roberta-wwm-ext / chinese-macbert-base
三个都是 12 层 768 维的 BERT 家族，参数预算几乎一样，
差别只在预训练语料与预训练目标（MLM+NSP / 全词掩码去 NSP / 纠错式掩码）。
所以这是一次**控制变量**的「换预训练模型」实验。

其他控制变量（三个模型完全一致）：
  池化 cls、类别加权、分层学习率（BERT 2e-5 / 头 ×5）、warmup 10%、
  3 epoch、batch、max_len、seed=42、同一份训练集与同一批 500 条评测样本。

用法：
  python code/bert/train_bert.py --dataset tnews --backbone bert-base-chinese
  python code/bert/train_bert.py --dataset chnsenticorp --backbone chinese-roberta-wwm-ext
  python code/bert/train_bert.py --dataset tnews --backbone bert-base-chinese --max_steps 20 --epochs 1  # 冒烟
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.config import BACKBONES, CKPT_DIR, DATASETS, model_path, result_path
from common.data import eval_samples, label_names, load_dataset
from common.metrics import classification_report, pred_distribution


class EncoderClassifier(nn.Module):
    def __init__(self, model_dir, num_labels, dropout=0.1):
        super().__init__()
        from transformers import AutoModel
        self.encoder = AutoModel.from_pretrained(str(model_dir))
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_labels)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask,
                           return_dict=True)
        return self.classifier(self.dropout(out.last_hidden_state[:, 0]))


def tokenize_all(rows, tok, max_len, text_key="sentence"):
    enc = tok([r[text_key] for r in rows], truncation=True, max_length=max_len,
              padding="max_length", return_tensors="pt")
    return enc["input_ids"], enc["attention_mask"]


@torch.no_grad()
def predict(model, input_ids, attention_mask, device, batch_size=128):
    model.eval()
    logits_all = []
    for i in range(0, len(input_ids), batch_size):
        ids = input_ids[i:i + batch_size].to(device)
        am = attention_mask[i:i + batch_size].to(device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=(device == "cuda")):
            logits = model(ids, am)
        logits_all.append(logits.float().cpu())
    return torch.cat(logits_all)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DATASETS))
    ap.add_argument("--backbone", required=True, choices=list(BACKBONES))
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=0, help="0=按数据集默认")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--head_lr_mult", type=float, default=5.0)
    ap.add_argument("--max_steps", type=int, default=0, help="每 epoch 最多走几步（冒烟）")
    ap.add_argument("--no_save", action="store_true", help="冒烟：不落盘 checkpoint")
    ap.add_argument("--force", action="store_true", help="已有结果也重跑")
    args = ap.parse_args()

    cfg = DATASETS[args.dataset]
    out_path = result_path(args.dataset, f"bert_{args.backbone}")
    if out_path.exists() and not args.force and not args.max_steps:
        print(f"[skip] {out_path} 已存在（--force 可重跑）")
        return

    from transformers import AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)
    np.random.seed(42)

    model_dir = model_path(args.backbone)
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    train, val = load_dataset(args.dataset)
    samples, meta = eval_samples(args.dataset, 500, 42)
    labels = label_names(args.dataset)
    batch_size = args.batch_size or cfg["batch_size"]

    ids_tr, am_tr = tokenize_all(train, tok, cfg["max_len"])
    y_tr = torch.tensor([r["label"] for r in train])
    ids_ev, am_ev = tokenize_all(samples, tok, cfg["max_len"])
    y_ev = torch.tensor([r["label"] for r in samples])

    train_dl = DataLoader(TensorDataset(ids_tr, am_tr, y_tr), batch_size=batch_size,
                          shuffle=True)
    # 训练中监控：从干净 val 中剔除最终评测的 500 条，避免「用评测集选 ckpt」
    from common.data import clean_val
    clean_rows, _ = clean_val(train, val)
    eval_texts = {s["sentence"] for s in samples}
    monitor = [d for d in clean_rows if d["sentence"] not in eval_texts][:1000]
    ids_mon, am_mon = tokenize_all(monitor, tok, cfg["max_len"])
    y_mon = torch.tensor([r["label"] for r in monitor])

    model = EncoderClassifier(model_dir, len(labels)).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    # 分层学习率：头随机初始化需要快，编码器要小步保护预训练知识
    head = list(model.classifier.parameters()) + list(model.dropout.parameters())
    enc = [p for n, p in model.named_parameters()
           if not n.startswith("classifier") and not n.startswith("dropout")]
    opt = torch.optim.AdamW([
        {"params": enc, "lr": args.lr},
        {"params": head, "lr": args.lr * args.head_lr_mult},
    ], weight_decay=0.01)

    from sklearn.utils.class_weight import compute_class_weight
    cw = compute_class_weight("balanced", classes=np.arange(len(labels)),
                              y=[r["label"] for r in train])
    weight = torch.tensor(cw, dtype=torch.float).to(device)
    crit = nn.CrossEntropyLoss(weight=weight)

    steps_per_epoch = min(len(train_dl), args.max_steps) if args.max_steps else len(train_dl)
    total_steps = steps_per_epoch * args.epochs
    warmup = max(1, int(total_steps * 0.1))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: s / warmup if s < warmup else max(0.0, (total_steps - s) / max(1, total_steps - warmup)))
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    print(f"[cfg] dataset={args.dataset} backbone={args.backbone} "
          f"labels={len(labels)} n_train={len(train)} max_len={cfg['max_len']} "
          f"batch={batch_size} epochs={args.epochs} lr={args.lr} params={n_params/1e6:.1f}M",
          flush=True)

    best_f1, best_state, log = -1.0, None, []
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running, n_step = 0.0, 0
        ep_t0 = time.time()
        for i, (ids, am, y) in enumerate(train_dl):
            ids, am, y = ids.to(device), am.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=(device == "cuda")):
                loss = crit(model(ids, am), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            running += loss.item()
            n_step += 1
            if i % 200 == 0:
                print(f"  epoch {epoch} step {i}/{steps_per_epoch} loss={loss.item():.4f}",
                      flush=True)
            if args.max_steps and n_step >= args.max_steps:
                break

        mon_logits = predict(model, ids_mon, am_mon, device)
        mon_pred = mon_logits.argmax(-1).tolist()
        mon_f1 = classification_report(y_mon.tolist(), mon_pred, len(labels))["macro_f1"]
        log.append({"epoch": epoch, "train_loss": round(running / max(1, n_step), 4),
                    "monitor_macro_f1": mon_f1,
                    "elapsed_s": round(time.time() - ep_t0, 1)})
        print(f"epoch {epoch}: monitor_macro_f1={mon_f1} ({time.time()-ep_t0:.0f}s)", flush=True)

        if mon_f1 > best_f1:
            best_f1 = mon_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    train_seconds = round(time.time() - t0, 1)

    if best_state is not None:
        model.load_state_dict(best_state)

    # 最终评测：固定 500 条干净样本
    t_inf = time.time()
    ev_logits = predict(model, ids_ev, am_ev, device)
    infer_seconds = (time.time() - t_inf) / len(samples)
    probs = torch.softmax(ev_logits, dim=-1).numpy()
    preds = ev_logits.argmax(-1).tolist()
    report = classification_report(y_ev.tolist(), preds, len(labels), probs=probs)
    report["method"] = f"bert_{args.backbone}"
    report["pred_distribution"] = pred_distribution(preds, len(labels))

    if not args.no_save:
        CKPT_DIR.mkdir(parents=True, exist_ok=True)
        ckpt_path = CKPT_DIR / f"{args.dataset}_{args.backbone}.pt"
        torch.save({"state_dict": best_state, "backbone": args.backbone,
                    "dataset": args.dataset, "num_labels": len(labels),
                    "max_len": cfg["max_len"], "monitor_f1": best_f1}, ckpt_path)
        print(f"ckpt -> {ckpt_path}")

    peak_gb = round(torch.cuda.max_memory_allocated() / 1e9, 2) if device == "cuda" else 0
    result = {
        "method": f"bert_{args.backbone}",
        "display": f"{BACKBONES[args.backbone]['display']} 微调",
        "family": "finetune",
        "dataset": args.dataset,
        "meta": meta,
        "report": report,
        "cost": {"train_seconds": train_seconds, "per_item_ms": round(infer_seconds * 1000, 1),
                 "params": n_params, "peak_mem_gb": peak_gb},
        "train_log": log,
        "preds": [{"text": s["sentence"], "true": int(s["label"]), "pred": int(p),
                   "prob": [round(float(x), 5) for x in probs[i]]}
                  for i, (s, p) in enumerate(zip(samples, preds))],
    }
    if not args.no_save:
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"→ {out_path}")
    print(f"acc={report['acc_incl_unparsed']} macro_f1={report['macro_f1']} "
          f"used={report['pred_distribution']['used_classes']}/{len(labels)} "
          f"train={train_seconds}s peak={peak_gb}GB")


if __name__ == "__main__":
    main()
