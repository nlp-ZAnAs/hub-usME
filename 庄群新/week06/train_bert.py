"""方法三 · L2 BERT 微调（本任务的主力）

架构只改两处：池化方式（cls/mean/max）+ 损失函数（类别加权）。
三个旋钮比换模型更值钱：分层学习率（BERT 2e-5 / 头 ×5）、Warmup 10% + 线性衰减、类别加权（23× 不均衡）。
工程加速：AMP 混合精度（显存约降 40%）。

评测口径与另外两路完全一致：同一批 clean val、同 seed、同一个 metrics 函数。
资源口径：训练墙钟时间、GPU 峰值显存、进程峰值 RSS、模型文件大小、单条/批量推理延迟。

用法：
  python train_bert.py                        # 训练 3 epoch + 评测 500 条 + 落盘 models/bert_ckpt.pt
  python train_bert.py --pool mean            # 换池化
  python train_bert.py --n 0                  # 在全部 8,901 条干净样本上评测
  python train_bert.py --epochs 1 --max_steps 30 --no_save   # 冒烟：验证链路通不通
  python train_bert.py --eval_only            # 只加载已有 ckpt 重新评测（不训练）
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
import dataio
import metrics as M
import resources as R
from labels import LABELS

METHOD_PREFIX = "BERT 微调"


class BertClassifier:
    """BERT + 池化 + 线性头（用 nn.Module 包一层，池化前一定要 mask 掉 [PAD]）"""

    @staticmethod
    def build(model_dir, num_labels, pool="cls", dropout=0.1):
        import torch.nn as nn
        from transformers import AutoModel

        class _Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.bert = AutoModel.from_pretrained(str(model_dir))
                self.pool = pool
                self.dropout = nn.Dropout(dropout)
                hidden = self.bert.config.hidden_size
                self.classifier = nn.Linear(hidden, num_labels)

            def forward(self, input_ids, attention_mask):
                out = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
                hs = out.last_hidden_state                     # [B, L, 768]
                if self.pool == "cls":
                    vec = hs[:, 0]
                else:
                    m = attention_mask.unsqueeze(-1).float()
                    if self.pool == "mean":
                        vec = (hs * m).sum(1) / m.sum(1).clamp(min=1e-9)
                    else:                                      # max：PAD 压到 -1e9 再取 max
                        vec = hs.masked_fill(m == 0, -1e9).max(1).values
                return self.classifier(self.dropout(vec))

        return _Net()


def pretokenize(texts, tok, max_len):
    """一次性把全部文本编码成张量：省掉每个 step 的 tokenizer 开销，也避免 DataLoader 里反复拷贝"""
    import torch
    feats = tok(texts, truncation=True, max_length=max_len,
                padding="max_length", return_tensors="pt")
    return feats["input_ids"], feats["attention_mask"]


def evaluate(model, ids, mask, ys, device, batch=64):
    import torch
    from sklearn.metrics import accuracy_score, f1_score
    model.eval()
    logits_all = []
    with torch.no_grad():
        for i in range(0, len(ys), batch):
            with torch.autocast("cuda", dtype=torch.float16, enabled=(device == "cuda")):
                logits = model(ids[i:i + batch].to(device), mask[i:i + batch].to(device))
            logits_all.append(logits.float().cpu())
    logits_all = torch.cat(logits_all)
    pred = logits_all.argmax(-1).numpy()
    return {
        "acc": round(float(accuracy_score(ys, pred)), 4),
        "macro_f1": round(float(f1_score(ys, pred, average="macro")), 4),
        "logits": logits_all,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=config.BERT_POOL, choices=["cls", "mean", "max"])
    ap.add_argument("--epochs", type=int, default=config.BERT_EPOCHS)
    ap.add_argument("--batch", type=int, default=config.BERT_BATCH)
    ap.add_argument("--lr", type=float, default=config.BERT_LR)
    ap.add_argument("--head_lr_mult", type=float, default=config.BERT_HEAD_LR_MULT)
    ap.add_argument("--max_len", type=int, default=config.BERT_MAX_LEN)
    ap.add_argument("--no_class_weight", action="store_true")
    ap.add_argument("--n", type=int, default=config.N_EVAL)
    ap.add_argument("--seed", type=int, default=config.SEED)
    # 冒烟 / 调试
    ap.add_argument("--max_steps", type=int, default=0)
    ap.add_argument("--no_save", action="store_true")
    ap.add_argument("--eval_only", action="store_true")
    ap.add_argument("--tag", default="", help="实验后缀（如 _nw），用于区分 ckpt / 指标 / 预测文件名")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    import numpy as np
    import torch
    import torch.nn as nn
    from torch.optim import AdamW
    from torch.utils.data import TensorDataset, DataLoader
    from transformers import AutoTokenizer

    config.ensure_dirs()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train, _ = dataio.load_splits()
    samples, meta = dataio.load_clean_val(args.n or None)
    y_true = np.array([d["label"] for d in samples])
    tok = AutoTokenizer.from_pretrained(str(config.BERT_PRETRAINED))

    ckpt_path = config.MODEL_DIR / f"bert_ckpt{args.tag}.pt"
    out_name = args.out or f"bert{args.tag}_metrics.json"
    log = []

    if args.eval_only:
        if not ckpt_path.exists():
            raise SystemExit(f"没有找到 {ckpt_path}，先跑一次训练（去掉 --eval_only）")
        blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        net = BertClassifier.build(config.BERT_PRETRAINED, config.NUM_LABELS, blob.get("pool", args.pool))
        net.load_state_dict(blob["state_dict"])
        net.to(device)
        train_seconds = 0.0
        best = blob.get("val_macro_f1", 0.0)
        pool = blob.get("pool", args.pool)
    else:
        pool = args.pool
        ids_te, mask_te = pretokenize([d["sentence"] for d in samples], tok, args.max_len)
        ys = torch.from_numpy(y_true).long()

        attempt = 0
        while True:
            attempt += 1
            bs = max(4, args.batch // (2 ** (attempt - 1)))
            try:
                net = BertClassifier.build(config.BERT_PRETRAINED, config.NUM_LABELS, pool).to(device)
                ids_tr, mask_tr = pretokenize([d["sentence"] for d in train], tok, args.max_len)
                ys_tr = torch.tensor([d["label"] for d in train]).long()
                ds = TensorDataset(ids_tr, mask_tr, ys_tr)
                dl = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=0, drop_last=False)

                head_params = list(net.classifier.parameters()) + list(net.dropout.parameters())
                body_params = [p for n, p in net.named_parameters()
                               if not n.startswith("classifier") and not n.startswith("dropout")]
                opt = AdamW([{"params": body_params, "lr": args.lr},
                             {"params": head_params, "lr": args.lr * args.head_lr_mult}],
                            weight_decay=0.01)

                if args.no_class_weight:
                    weight = None
                else:
                    from sklearn.utils.class_weight import compute_class_weight
                    cw = compute_class_weight("balanced", classes=np.arange(config.NUM_LABELS),
                                              y=[d["label"] for d in train])
                    weight = torch.tensor(cw, dtype=torch.float).to(device)
                crit = nn.CrossEntropyLoss(weight=weight)
                steps_per_epoch = min(len(dl), args.max_steps) if args.max_steps else len(dl)
                total_steps = max(1, steps_per_epoch * args.epochs)
                warmup = int(total_steps * 0.1)
                sched = torch.optim.lr_scheduler.LambdaLR(
                    opt, lambda s: s / max(1, warmup) if s < warmup
                    else max(0.0, (total_steps - s) / max(1, total_steps - warmup)))
                scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
                break
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"[oom] batch={bs} 显存不够，降到 {max(4, bs // 2)} 重试", flush=True)
                if attempt >= 3:
                    raise

        print(f"[cfg] pool={pool} bs={bs} epochs={args.epochs} lr={args.lr} "
              f"head×{args.head_lr_mult} max_len={args.max_len} "
              f"class_weight={'off' if weight is None else 'on'} device={device}", flush=True)

        best, best_state = 0.0, None
        with R.ResourceMonitor(interval=0.5) as mon:
            with R.Timer() as timer:
                for epoch in range(1, args.epochs + 1):
                    net.train()
                    t0, running, n_step = time.time(), 0.0, 0
                    for ids, mask, y in dl:
                        ids, mask, y = ids.to(device), mask.to(device), y.to(device)
                        opt.zero_grad(set_to_none=True)
                        with torch.autocast("cuda", dtype=torch.float16, enabled=(device == "cuda")):
                            loss = crit(net(ids, mask), y)
                        scaler.scale(loss).backward()
                        scaler.unscale_(opt)
                        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                        scaler.step(opt); scaler.update(); sched.step()
                        running += loss.item(); n_step += 1
                        if n_step % 200 == 0:
                            print(f"  epoch {epoch} step {n_step}/{steps_per_epoch} "
                                  f"loss={loss.item():.4f}", flush=True)
                        if args.max_steps and n_step >= args.max_steps:
                            break
                    res = evaluate(net, ids_te, mask_te, ys, device)
                    rec = {"epoch": epoch, "train_loss": round(running / max(1, n_step), 4),
                           "val_acc": res["acc"], "val_macro_f1": res["macro_f1"],
                           "elapsed_s": round(time.time() - t0, 1)}
                    log.append(rec)
                    print(f"epoch {epoch}: val_acc={res['acc']} val_f1={res['macro_f1']} "
                          f"({rec['elapsed_s']}s)", flush=True)
                    if res["macro_f1"] > best:
                        best = res["macro_f1"]
                        best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            train_seconds = timer.seconds

        if best_state is not None:
            net.load_state_dict(best_state)

        if not args.no_save:
            torch.save({"state_dict": net.state_dict(), "pool": pool,
                        "labels": LABELS, "val_macro_f1": best,
                        "max_len": args.max_len, "epochs": args.epochs,
                        "model_dir": str(config.BERT_PRETRAINED)}, ckpt_path)

    # ---------- 统一评测 ----------
    ids_te, mask_te = pretokenize([d["sentence"] for d in samples], tok, args.max_len)
    ys = torch.from_numpy(y_true).long()

    with R.ResourceMonitor(interval=0.2) as mon:
        t0 = time.time()
        res = evaluate(net, ids_te, mask_te, ys, device, batch=32)
        infer_seconds = round(time.time() - t0, 3)

        # 单条延迟（batch=1）：前 30 条逐个前向取中位数
        lat = []
        for i in range(min(30, len(samples))):
            if device == "cuda":
                torch.cuda.synchronize()
            s = time.perf_counter()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=(device == "cuda")):
                net(ids_te[i:i + 1].to(device), mask_te[i:i + 1].to(device))
            if device == "cuda":
                torch.cuda.synchronize()
            lat.append((time.perf_counter() - s) * 1000)
        lat.sort()
        ms_b1 = round(lat[len(lat) // 2], 3)

    pred = res["logits"].argmax(-1).numpy().tolist()
    probs = torch.softmax(res["logits"].float(), dim=-1).numpy()
    report = M.classification_report(y_true.tolist(), pred)
    ece = M.expected_calibration_error(y_true, probs)

    n_params = sum(p.numel() for p in net.parameters())
    result = {
        "method": f"BERT 微调（{pool} 池化）",
        "level": "L2",
        "metrics": {**report, "group_acc": M.group_level_report(y_true.tolist(), pred), "ece": ece},
        "cost": {
            "train_seconds": train_seconds,
            "infer_seconds": infer_seconds,
            "infer_ms_b1": ms_b1,
            "infer_ms_b32": round(infer_seconds * 1000 / len(samples), 3),
            "throughput": round(len(samples) / max(infer_seconds, 1e-6), 1),
            "model_size_mb": R.file_size_mb(ckpt_path) if (ckpt_path.exists() and not args.no_save) else 0.0,
            "peak_rss_mb": round(mon.peak_rss_mb, 1),
            "peak_gpu_mb": round(mon.peak_gpu_mb, 1),
            "n_params": n_params,
        },
        "meta": {**meta, "env": R.env_info(),
                 "hyper": {"pool": pool, "epochs": args.epochs, "batch": args.batch,
                           "lr": args.lr, "max_len": args.max_len,
                           "class_weight": not args.no_class_weight},
                 "train_log": log},
    }

    (config.OUT_DIR / out_name).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (config.OUT_DIR / f"bert{args.tag}_pred.json").write_text(
        json.dumps({"y_true": y_true.tolist(), "y_pred": pred,
                    "sentence": [d["sentence"] for d in samples]}, ensure_ascii=False), encoding="utf-8")
    (config.OUT_DIR / f"bert{args.tag}_train_log.json").write_text(
        json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"评测 {len(samples)} 条（干净集）| {result['method']}")
    print(f"| {result['method']} | acc={report['acc']} | macro-F1={report['macro_f1']} | "
          f"覆盖 {report['coverage']} | ECE={ece} |")
    print(f"训练 {train_seconds}s | 单条 {ms_b1} ms | batch32 {result['cost']['infer_ms_b32']} ms/条 | "
          f"显存峰值 {result['cost']['peak_gpu_mb']} MB | 模型 {result['cost']['model_size_mb']} MB")
    print(f"→ 模型 {ckpt_path if not args.no_save else '(未落盘)'}")
    print(f"→ 指标 {config.OUT_DIR / out_name}")


if __name__ == "__main__":
    main()
