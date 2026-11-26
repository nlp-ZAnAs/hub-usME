"""
L2 主线：BERT 序列标注做事件抽取（一次前向同时抽「事件类型 + 触发词 + 论元」）

教学重点：
  1. 子词对齐（word_ids）：bert-base-chinese 对中文基本一字一 token，但特殊符 / 空格（不产生
     token）/ 换 tokenizer 都会错位。只有首个子词背标签、其余设 -100；否则同一 span 的 loss 被重复算。
  2. 一套脚本两种解码器：--decoder linear|crf
       linear：每 token 独立 softmax，简单快，但会喷非法 BIO 序列
       crf   ：加转移矩阵做全局最优解码，非法序列显著减少（手写，见 common/crf.py）
  3. 非法 BIO 统计：linear vs crf 的核心量化对比点。
  4. joint 标签体系：CEC = 27 个标签（O + 2×(5 类触发词 + 8 个角色)），端到端联合抽取。
  5. 模型选择用内层验证集：从 train 切 10% 选 epoch，dev 只做最终评测（避免用 dev 选点抬虚）。

用法（Mac 自动用 MPS；也可 --device cpu）：
  python bert/train.py                                  # 主线：12 层 BERT + CRF
  python bert/train.py --decoder linear                 # 对照：BERT + Linear
  python bert/train.py --bert_layers 4                  # 消融：只训 4 层（更快）
  python bert/train.py --decoder crf --hard_constraint  # 非法转移硬约束到 0
  python bert/train.py --bert_path /path/to/bert-base-chinese

产物：
  outputs/checkpoints/<run_name>.pt
  outputs/logs/train_<run_name>.json、<run_name>_steps.jsonl
  outputs/disc/pred_<run_name>.jsonl
  outputs/metrics/disc/<run_name>.json
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import BertTokenizer, get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[1]   # week07/
sys.path.insert(0, str(ROOT))

from common.crf import CRFClassifier  # noqa: E402
from common.eval_utils import CKPT_DIR, LOG_DIR, Timer, evaluate_predictions  # noqa: E402
from common.log_utils import TrainLogger, peak_gpu_mb  # noqa: E402
from common.event_data import (  # noqa: E402
    BioDataset, DATA_DIR, build_bio_constraint_mask, build_joint_schema,
    build_start_constraint_mask, collate_bio, decode_bio_spans, illegal_bio_stats,
    load_enums, load_extract_split, load_schema, make_pred_row, row_event, spans_to_event,
)
from common.models import batch_to_device, build_model, predict_char_tags  # noqa: E402
from common.model_paths import BERT_MODEL_DIR, resolve_model_path  # noqa: E402


# ─────────────────────────── 工具 ───────────────────────────
def resolve_bert_path(args) -> str:
    """定位本地 bert-base-chinese：--bert_path > 向上查找 pretrain_models/ > 本机已知位置。"""
    if args.bert_path:
        return args.bert_path
    try:
        return str(resolve_model_path(BERT_MODEL_DIR))
    except FileNotFoundError:
        pass
    env = os.environ.get("BERT_MODEL_PATH")
    cands = [env] if env else []
    cands.append("/Users/may/badouai/week4-LM/bert-base-chinese")  # 本机已有权重
    for c in cands:
        p = Path(c) if c else None
        if p and (p / "config.json").exists() and (
                (p / "pytorch_model.bin").exists() or list(p.glob("*.safetensors"))):
            return str(p)
    raise FileNotFoundError(
        "找不到 bert-base-chinese。请用 --bert_path 指定本地模型目录，"
        "或设置环境变量 BERT_MODEL_PATH。")


def pick_device(spec: str) -> torch.device:
    if spec:
        return torch.device(spec)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")   # Apple GPU：BERT-base 比 CPU 快很多
    return torch.device("cpu")


def gold_path_for(data_dir: Path, split: str) -> Path:
    for name in (f"{split}_extract.jsonl", f"{split}.jsonl"):
        p = data_dir / name
        if p.exists():
            return p
    return data_dir / f"{split}.jsonl"


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def span_f1_from_char_tags(rows: list, char_tags: list) -> float:
    """训练中做模型选择的轻量指标：span 级精确匹配 F1（角色+文本），口径接近最终 argument F1。"""
    tp = fp = fn = 0
    for row, tags in zip(rows, char_tags):
        ge = row_event(row) or {}
        gold = {(a["role"], a["text"]) for a in ge.get("args") or []}
        gold.add(("__trigger__", (ge.get("trigger") or {}).get("text", "")))
        ev = spans_to_event(decode_bio_spans(tags), row["text"])
        pred = {(a["role"], a["text"]) for a in (ev or {}).get("args", [])}
        pred.add(("__trigger__", (ev or {}).get("trigger", {}).get("text", "")))
        pred.discard(("__trigger__", ""))
        gold.discard(("__trigger__", ""))
        tp += len(gold & pred)
        fp += len(pred - gold)
        fn += len(gold - pred)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def evaluate_loss_and_f1(model, loader, id2label: dict, device) -> tuple:
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            batch = batch_to_device(batch, device)
            _, loss = model(batch["input_ids"], batch["attention_mask"],
                            batch["labels"], batch.get("token_type_ids"))
            total += float(loss.item())
            n += 1
    avg_loss = total / max(n, 1)
    tags, _ = predict_char_tags(model, loader, id2label, device)
    return avg_loss, span_f1_from_char_tags(loader.dataset.rows, tags)


def build_optimizer(model, args):
    """分层学习率：BERT 用小 lr（2e-5），新初始化的分类头 / CRF 用 head_lr_mult 倍。"""
    enc_params, head_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (enc_params if name.startswith("encoder.") else head_params).append(p)
    groups = [{"params": enc_params, "lr": args.lr}]
    if head_params:
        groups.append({"params": head_params, "lr": args.lr * args.head_lr_mult})
    return AdamW(groups, weight_decay=args.weight_decay)


def run_epoch(model, loader, optimizer, scheduler, device, args, epoch: int,
              logger=None) -> float:
    model.train()
    total, n = 0.0, 0
    optimizer.zero_grad()
    for step, batch in enumerate(loader):
        batch = batch_to_device(batch, device)
        _, loss = model(batch["input_ids"], batch["attention_mask"],
                        batch["labels"], batch.get("token_type_ids"))
        (loss / args.grad_accum).backward()
        total += float(loss.item())
        n += 1
        if (step + 1) % args.grad_accum == 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        if logger is not None:
            logger.log_step(epoch=epoch, step=step, loss=float(loss.item()),
                            lr=scheduler.get_last_lr()[0])
        if (step + 1) % 50 == 0:
            print(f"  epoch {epoch} step {step + 1}/{len(loader)} loss={total / n:.4f}", flush=True)
    if len(loader) % args.grad_accum != 0:
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    return total / max(n, 1)


# ─────────────────────────── 主流程 ───────────────────────────
def main():
    args = parse_args()
    set_seed(args.seed)
    device = pick_device(args.device)
    print(f"设备：{device} | BERT：{args.bert_path}")

    data_dir = Path(args.data_dir) if args.data_dir else DATA_DIR
    schema = load_schema()      # CEC：data/conf/event_schema.json（列表形态）
    enums = load_enums()        # CEC 无枚举角色 → {}
    label_scheme = build_joint_schema(schema)
    labels, id2label = label_scheme["labels"], label_scheme["id2label"]
    print(f"标签体系：joint，{len(labels)} 个标签（{schema_stats_str(schema)}）")

    tokenizer = BertTokenizer.from_pretrained(args.bert_path)
    train_rows = load_extract_split("train", data_dir)
    dev_rows = load_extract_split("dev", data_dir)

    # 内层验证集：从 train 切 10%，只用于选 epoch；dev 完全不参与训练/选点
    rng = random.Random(args.seed)
    idxs = list(range(len(train_rows)))
    rng.shuffle(idxs)
    n_inner = max(1, int(len(train_rows) * args.inner_val))
    inner_idx = set(idxs[:n_inner])
    fit_rows = [r for i, r in enumerate(train_rows) if i not in inner_idx]
    val_rows = [r for i, r in enumerate(train_rows) if i in inner_idx]
    print(f"训练 {len(fit_rows)} 句 / 内层验证 {len(val_rows)} 句 / 评测 dev {len(dev_rows)} 句")

    fit_ds = BioDataset(fit_rows, tokenizer, label_scheme, args.max_len)
    val_ds = BioDataset(val_rows, tokenizer, label_scheme, args.max_len)
    dev_ds = BioDataset(dev_rows, tokenizer, label_scheme, args.max_len)
    fit_loader = DataLoader(fit_ds, batch_size=args.batch_size, shuffle=True,
                            num_workers=0, collate_fn=collate_bio)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, collate_fn=collate_bio)
    dev_loader = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, collate_fn=collate_bio)

    model = build_model(args, label_scheme).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数：{n_params / 1e6:.1f}M（可训练 {n_trainable / 1e6:.1f}M），"
          f"解码器={args.decoder}，BERT 层数={args.bert_layers or 'full(12)'}")

    optimizer = build_optimizer(model, args)
    total_steps = max(1, len(fit_loader) * args.epochs // max(args.grad_accum, 1))
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    print(f"训练步数 {total_steps}（warmup {warmup_steps} 步，{args.warmup_ratio:.0%}）")

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CKPT_DIR / f"{args.run_name}.pt"

    logger = TrainLogger(method=args.run_name, args=args, root=ROOT, params=n_params,
                         notes=f"BERT 序列标注（{args.decoder}）训练过程")

    log, best_f1, best_epoch = [], -1.0, 0
    train_start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        train_loss = run_epoch(model, fit_loader, optimizer, scheduler, device, args, epoch,
                               logger=logger)
        val_loss, val_f1 = evaluate_loss_and_f1(model, val_loader, id2label, device)
        _, dev_f1_monitor = evaluate_loss_and_f1(model, dev_loader, id2label, device)
        elapsed = time.perf_counter() - t0
        print(f"epoch {epoch}/{args.epochs} | train_loss={train_loss:.4f} | "
              f"inner_loss={val_loss:.4f} inner_spanF1={val_f1:.4f} | "
              f"dev_spanF1(监控)={dev_f1_monitor:.4f} | {elapsed:.0f}s", flush=True)
        logger.log_epoch(epoch=epoch, train_loss=train_loss, elapsed=elapsed,
                         val_metrics={"inner_loss": val_loss, "inner_span_f1": val_f1,
                                      "dev_span_f1_monitor": dev_f1_monitor},
                         extra={"best_so_far": round(best_f1, 6)})
        log.append({"epoch": epoch, "train_loss": round(train_loss, 6),
                    "inner_val_loss": round(val_loss, 6),
                    "inner_val_span_f1": round(val_f1, 6),
                    "dev_span_f1_monitor": round(dev_f1_monitor, 6),
                    "elapsed_s": round(elapsed, 1)})
        if val_f1 > best_f1:
            best_f1, best_epoch = val_f1, epoch
            torch.save({"state_dict": model.state_dict(),
                        "label_scheme": {k: label_scheme[k] for k in ("labels", "label2id", "id2label", "name")},
                        "id2label": id2label, "args": vars(args), "epoch": epoch,
                        "inner_val_span_f1": val_f1}, ckpt_path)
            print(f"  ★ 内层验证最优 → 保存 {ckpt_path.name}", flush=True)

    train_s = time.perf_counter() - train_start
    with open(LOG_DIR / f"train_{args.run_name}.json", "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)

    # ── 用最优 checkpoint 在 dev 上出最终预测 ──
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    print(f"加载最优 checkpoint（epoch {ckpt['epoch']}，内层验证 spanF1={best_f1:.4f}）")

    with Timer() as timer:
        dev_tags, _ = predict_char_tags(model, dev_loader, id2label, device)
    dev_illegal = illegal_bio_stats(dev_tags)

    with Timer() as timer_fit:
        fit_tags, _ = predict_char_tags(model, fit_loader, id2label, device)
    fit_illegal = illegal_bio_stats(fit_tags)

    preds = []
    n_empty = 0
    for row, tags in zip(dev_rows, dev_tags):
        ev = spans_to_event(decode_bio_spans(tags), row["text"], enums=enums)
        if ev is None:
            n_empty += 1
        preds.append(make_pred_row(row, [ev] if ev else []))
    print(f"dev 解码：{len(preds)} 句，其中没抽到触发词 {n_empty} 句")

    lat = timer.summary(len(dev_rows))
    meta = {
        "display_name": f"BERT-{args.bert_layers or 12}L+{args.decoder.upper()}",
        "train_s": round(train_s, 1),
        "params_m": round(n_params / 1e6, 1),
        "ms_per_sentence": lat["ms_per_sentence"],
        "peak_gpu_mb": lat["peak_gpu_mb"],
        "decoder": args.decoder,
        "bert_layers": args.bert_layers or 12,
        "max_len": args.max_len, "epochs": args.epochs, "lr": args.lr,
        "head_lr_mult": args.head_lr_mult, "batch_size": args.batch_size,
        "grad_accum": args.grad_accum, "best_epoch": best_epoch,
        "inner_val_span_f1": round(best_f1, 4), "num_labels": len(labels),
        "illegal_dev": dev_illegal, "illegal_train_pred": fit_illegal,
        "n_truncated": dev_ds.n_truncated, "hard_constraint": bool(args.hard_constraint),
        "device": str(device), "seed": args.seed,
        "selection_note": "按 train 切出的 10% 内层验证集选 epoch；dev 未参与训练与选点",
    }
    res = evaluate_predictions(args.run_name, preds, meta=meta,
                               gold_path=gold_path_for(data_dir, "dev"))

    print("\n【非法 BIO 统计（dev，模型原始输出）】")
    print(f"  序列数 {dev_illegal['total_seqs']}：含非法序列 {dev_illegal['illegal_seqs']} 条"
          f"（{dev_illegal['illegal_seq_rate']:.1%}），其中非法开头 {dev_illegal['illegal_start']} 处、"
          f"非法转移 {dev_illegal['illegal_transition']} 处")
    print(f"  训练集对照：含非法序列 {fit_illegal['illegal_seqs']} 条"
          f"（{fit_illegal['illegal_seq_rate']:.1%}）")
    print(f"\n训练耗时 {train_s:.0f}s | 推理延迟 {lat['ms_per_sentence']} ms/句 | "
          f"峰值显存 {lat['peak_gpu_mb']:.0f} MiB")

    logger.finalize(
        total_s=train_s, best_epoch=best_epoch,
        best_inner_val_span_f1=round(best_f1, 6),
        dev_metrics={k: res[k]["f1"] for k in ("trigger", "argument", "event_exact")},
        inference_latency_ms=lat["ms_per_sentence"], peak_gpu_mb=peak_gpu_mb(),
        illegal_dev=dev_illegal, illegal_train_pred_sample=fit_illegal,
    )


def schema_stats_str(schema: dict) -> str:
    roles = {r for v in schema.values() for r in v}
    return f"{len(schema)} 类 / {sum(len(v) for v in schema.values())} 槽位 / {len(roles)} 去重角色"


def parse_args():
    ap = argparse.ArgumentParser(description="BERT 序列标注做事件抽取（linear / crf，CEC）")
    ap.add_argument("--decoder", choices=["linear", "crf"], default="crf")
    ap.add_argument("--bert_path", default="")
    ap.add_argument("--bert_layers", type=int, default=0, help="0=完整 12 层；4=消融")
    ap.add_argument("--max_len", type=int, default=256)
    # CEC 每 epoch 仅 ~15 步；BERT 微调需要几百步才收敛，epoch 数比 DuEE 的 3 要大。
    # 实测 8 epoch 欠拟合（trigger F1≈0.59 < BiLSTM），15 epoch / lr 3e-5 才稳定超过 BiLSTM。
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--head_lr_mult", type=float, default=5.0)
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--inner_val", type=float, default=0.1)
    ap.add_argument("--hard_constraint", action="store_true")
    ap.add_argument("--device", default="", help="cuda / mps / cpu；留空自动选")
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--run_name", default="")
    ap.add_argument("--data_dir", default="", help="数据目录；留空=data/processed")
    args = ap.parse_args()
    args.bert_path = resolve_bert_path(args)
    if not args.run_name:
        tag = f"bert{args.bert_layers}L" if args.bert_layers else "bert"
        extra = "_hard" if args.hard_constraint else ""
        args.run_name = f"{tag}_{args.decoder}_{args.max_len}{extra}"
    return args


if __name__ == "__main__":
    main()
