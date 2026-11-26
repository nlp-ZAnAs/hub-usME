"""
L3 两段式流水线（pipeline）事件抽取：类型分类 → 类型条件论元抽取（CEC）

和 BERT 联合抽取（bert/train.py，27 个 joint 标签）是同任务、同数据的对照：
  联合：一个模型、一次前向，同时决定「类型 + 触发词 + 论元」，标签空间 27；
  两段：先判 5 类事件类型，再用「该类型的专家模型」抽「触发词 + 该类型角色」，
        每个专家的标签空间只有 1 + 2×(1+角色数)，大约 7~15 个。

为什么 CEC 这里是两段而不是参考项目的三段：
  参考项目有「事件检测器」（句子里到底有没有事件，负样本占多数）。
  但 CEC 的 train/dev/test.jsonl **每条都恰好一个事件**，没有负样本，
  检测器没有训练数据、也没有存在意义 —— 硬加只会教条。
  所以 CEC 的两段 = (b) 类型 5 分类 + (c) 5 个类型条件论元专家。

为什么流水线可能更高：
  (c) 标签空间小一个数量级，每个标签的样本密度高，论元边界学得更准。
为什么流水线也可能更低（课上要讲的反面）：
  误差会累积 —— 类型判错，(c) 一定全错；而且要多跑一个模型，延迟更高。

模型选择：和 bert/train.py 一致，从 train 切 10% 做内层验证挑 epoch，dev 只做最终评测。

用法：
  python bert_pipeline/train.py                     # 训练 + 端到端 dev 评测（约 10 分钟）
  python bert_pipeline/train.py --cls_epochs 8 --arg_epochs 8
  python bert_pipeline/train.py --skip_train        # 复用已有 checkpoint，只重新组合评测
  python bert_pipeline/train.py --only_experts      # 只重训论元专家（复用类型分类器）

产物：
  outputs/checkpoints/pipe_type.pt、pipe_arg_<类型>.pt
  outputs/metrics/disc/pipeline.json               端到端 dev
  outputs/metrics/disc/pipeline_oracle_type.json   类型给金标的上界（定位瓶颈）
  outputs/disc/pred_pipeline*.jsonl、outputs/logs/bert_pipeline_*
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from transformers import logging as hf_logging

hf_logging.set_verbosity_error()   # 屏蔽每个 BERT 权重加载时的大量 UNEXPECTED 报告

ROOT = Path(__file__).resolve().parents[1]   # week07/
sys.path.insert(0, str(ROOT))                 # 让 `import common.*` 生效（脚本在子目录里）
sys.path.insert(0, str(ROOT / "common"))     # 让 common/models.py 里的 `from crf import ...` 生效

from common.eval_utils import CKPT_DIR, LOG_DIR, Timer, evaluate_predictions  # noqa: E402
from common.log_utils import TrainLogger, peak_gpu_mb  # noqa: E402
from common.event_data import (  # noqa: E402
    BioDataset, ClsDataset, DATA_DIR, build_typed_schema, collate_bio, collate_cls,
    decode_bio_spans, load_enums, load_extract_split, load_schema, make_pred_row,
    row_event, spans_to_event,
)
from common.model_paths import BERT_MODEL_DIR, resolve_model_path  # noqa: E402
from common.models import (  # noqa: E402
    BertCls, _layers_of, build_arg_model, predict_char_tags,
)


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
        return torch.device("mps")
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


def split_inner(rows: list, frac: float, seed: int) -> tuple[list, list]:
    """从 rows 里切一份内层验证集（只用来选 epoch，dev 仍留作最终评测）。"""
    idxs = list(range(len(rows)))
    random.Random(seed).shuffle(idxs)
    n_val = max(1, int(len(rows) * frac)) if len(rows) > 4 else 0
    val_idx = set(idxs[:n_val])
    fit = [r for i, r in enumerate(rows) if i not in val_idx]
    val = [r for i, r in enumerate(rows) if i in val_idx]
    return fit, val


def typed_span_f1(rows: list, char_tags: list, etype: str) -> float:
    """类型条件专家的内层指标：角色+文本（含触发词）精确匹配 F1。"""
    tp = fp = fn = 0
    for row, tags in zip(rows, char_tags):
        ge = row_event(row) or {}
        gold = {(a["role"], a["text"]) for a in ge.get("args") or []}
        if (ge.get("trigger") or {}).get("text"):
            gold.add(("__trig__", ge["trigger"]["text"]))
        ev = spans_to_event(decode_bio_spans(tags), row["text"], typed_type=etype)
        pred = {(a["role"], a["text"]) for a in (ev or {}).get("args", [])}
        if (ev or {}).get("trigger", {}).get("text"):
            pred.add(("__trig__", ev["trigger"]["text"]))
        tp += len(gold & pred)
        fp += len(pred - gold)
        fn += len(gold - pred)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


# ─────────────────────────── (b) 类型分类器 ───────────────────────────
def train_type_classifier(rows, type2id, num_labels, tokenizer, args, device,
                          logger=None) -> dict:
    """BERT 5 分类（事件类型），内层验证挑准确率最高的 epoch。"""
    fit_rows, val_rows = split_inner(rows, args.inner_val, args.seed)
    fit_ds = ClsDataset(fit_rows, tokenizer, args.cls_max_len, task="type", type2id=type2id)
    val_ds = ClsDataset(val_rows, tokenizer, args.cls_max_len, task="type", type2id=type2id)
    fit_loader = DataLoader(fit_ds, batch_size=args.cls_batch_size, shuffle=True,
                            num_workers=0, collate_fn=collate_cls)
    val_loader = DataLoader(val_ds, batch_size=args.cls_batch_size, shuffle=False,
                            num_workers=0, collate_fn=collate_cls)

    model = BertCls(str(args.bert_path), num_labels, args.dropout).to(device)
    opt = AdamW(model.parameters(), lr=args.cls_lr, weight_decay=0.01)
    total = max(1, len(fit_loader) * args.cls_epochs)
    sched = get_linear_schedule_with_warmup(opt, int(total * 0.1), total)

    best_acc, best_state, best_epoch = -1.0, None, 0
    t0 = time.perf_counter()
    for epoch in range(1, args.cls_epochs + 1):
        model.train()
        tot, n = 0.0, 0
        for step, batch in enumerate(fit_loader):
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            tti = batch["token_type_ids"].to(device)
            lab = batch["labels"].to(device)
            _, loss = model(ids, mask, tti, lab)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += float(loss.item())
            n += 1
            if logger is not None:
                logger.log_step(epoch=epoch, step=step, loss=float(loss.item()),
                                lr=sched.get_last_lr()[0], extra={"stage": "type"})
        acc = _cls_accuracy(model, val_loader, device)
        print(f"  [type] epoch {epoch}/{args.cls_epochs} loss={tot / max(n, 1):.4f} "
              f"inner_acc={acc:.4f}", flush=True)
        if acc > best_acc:
            best_acc, best_epoch = acc, epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    train_s = time.perf_counter() - t0
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "type2id": type2id,
                "args": vars(args), "inner_acc": best_acc, "epoch": best_epoch},
               CKPT_DIR / "pipe_type.pt")
    return {"train_s": round(train_s, 1), "epochs": args.cls_epochs,
            "best_epoch": best_epoch, "inner_acc": round(best_acc, 4)}


@torch.no_grad()
def _cls_accuracy(model, loader, device) -> float:
    model.eval()
    correct = total = 0
    for batch in loader:
        logits, _ = model(batch["input_ids"].to(device),
                          batch["attention_mask"].to(device),
                          batch["token_type_ids"].to(device))
        pred = logits.argmax(-1)
        correct += int((pred == batch["labels"].to(device)).sum().item())
        total += int(batch["labels"].numel())
    return correct / max(total, 1)


@torch.no_grad()
def predict_type(model, rows, tokenizer, args, device, type2id) -> list[int]:
    ds = ClsDataset(rows, tokenizer, args.cls_max_len, task="type", type2id=type2id)
    loader = DataLoader(ds, batch_size=args.cls_batch_size, shuffle=False,
                        num_workers=0, collate_fn=collate_cls)
    model.eval()
    out = []
    for batch in loader:
        logits, _ = model(batch["input_ids"].to(device),
                          batch["attention_mask"].to(device),
                          batch["token_type_ids"].to(device))
        out.extend(logits.argmax(-1).tolist())
    return out


# ─────────────────────────── (c) 类型条件论元专家 ───────────────────────────
def train_arg_experts(extract_rows, schema, tokenizer, args, device, logger=None) -> dict:
    """
    为每个事件类型训一个「该类型的专家」序列标注模型。

    数据：只用该类型的句子；标签空间：TRG + roles(t)（每个类型十几类）。
    逐个训练、训完即释放 —— 5 个 BERT 同时驻留显存没必要，也不利于内存。
    每个专家用该类型的内层验证 span F1 挑 epoch。
    """
    by_type = defaultdict(list)
    for r in extract_rows:
        ev = row_event(r)
        if ev:
            by_type[ev["type"]].append(r)

    infos = {}
    t_all = time.perf_counter()
    for etype in sorted(schema):
        rows = by_type.get(etype, [])
        if not rows:
            print(f"  [arg:{etype}] 没有训练样本，跳过")
            continue
        scheme = build_typed_schema(schema, etype)
        fit_rows, val_rows = split_inner(rows, args.inner_val, args.seed)
        fit_ds = BioDataset(fit_rows, tokenizer, scheme, args.max_len, typed=True)
        fit_loader = DataLoader(fit_ds, batch_size=args.arg_batch_size, shuffle=True,
                                num_workers=0, collate_fn=collate_bio)
        val_loader = None
        if val_rows:
            val_ds = BioDataset(val_rows, tokenizer, scheme, args.max_len, typed=True)
            val_loader = DataLoader(val_ds, batch_size=args.arg_batch_size, shuffle=False,
                                    num_workers=0, collate_fn=collate_bio)

        model = build_arg_model(str(args.bert_path), scheme, args).to(device)
        opt = AdamW(model.parameters(), lr=args.arg_lr, weight_decay=0.01)
        total = max(1, len(fit_loader) * args.arg_epochs)
        sched = get_linear_schedule_with_warmup(opt, int(total * 0.1), total)

        best_f1, best_state, best_epoch = -1.0, None, args.arg_epochs
        t0 = time.perf_counter()
        for epoch in range(1, args.arg_epochs + 1):
            model.train()
            tot, n = 0.0, 0
            for step, batch in enumerate(fit_loader):
                inp = batch["input_ids"].to(device)
                msk = batch["attention_mask"].to(device)
                tti = batch.get("token_type_ids")
                tti = tti.to(device) if tti is not None else None
                lab = batch["labels"].to(device)
                _, loss = model(inp, msk, lab, tti)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                tot += float(loss.item())
                n += 1
                if logger is not None:
                    logger.log_step(epoch=epoch, step=step, loss=float(loss.item()),
                                    lr=sched.get_last_lr()[0], extra={"stage": f"arg:{etype}"})
            f1 = -1.0
            if val_loader is not None:
                tags, _ = predict_char_tags(model, val_loader, scheme["id2label"], device)
                f1 = typed_span_f1(val_rows, tags, etype)
            print(f"  [arg:{etype}] epoch {epoch}/{args.arg_epochs} "
                  f"loss={tot / max(n, 1):.4f} inner_spanF1={f1:.4f}", flush=True)
            # >= 而不是 >：若内层 F1 一直并列（小数据下常见 0），取更晚、训得更久的那个 epoch，
            # 否则会误存第 1 个几乎没训的模型。
            if f1 >= best_f1:
                best_f1, best_epoch = f1, epoch
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if best_state is not None:
            model.load_state_dict(best_state)
        torch.save({"state_dict": model.state_dict(), "labels": scheme["labels"],
                    "etype": etype, "args": vars(args), "inner_span_f1": best_f1,
                    "epoch": best_epoch}, CKPT_DIR / f"pipe_arg_{etype}.pt")
        infos[etype] = {"n_rows": len(rows), "n_fit": len(fit_rows), "n_val": len(val_rows),
                        "num_labels": len(scheme["labels"]), "best_epoch": best_epoch,
                        "inner_span_f1": round(best_f1, 4),
                        "train_s": round(time.perf_counter() - t0, 1)}
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    infos["_total_train_s"] = round(time.perf_counter() - t_all, 1)
    return infos


# ─────────────────────────── 组合：类型 → 论元 ───────────────────────────
def compose(dev_rows, tokenizer, args, device, type_model, type2id,
            oracle_type: bool = False) -> tuple[list[dict], dict]:
    """
    把两段串起来，对 dev 出端到端预测。

    oracle_type=True 时用**金标类型**代替类型分类器的输出 —— 标准的上界分析：
    类型给对能到多少，就能判断端到端损失主要来自「类型判错」还是「论元抽不准」。
    """
    id2type = {i: t for t, i in type2id.items()}
    type_pred = predict_type(type_model, dev_rows, tokenizer, args, device, type2id)

    final_types, n_type_wrong = [], 0
    for row, tp in zip(dev_rows, type_pred):
        gold_type = (row_event(row) or {}).get("type", "")
        t = gold_type if oracle_type else id2type[tp]
        final_types.append(t)
        if t != gold_type:
            n_type_wrong += 1

    preds: list[dict] = [None] * len(dev_rows)
    idx_by_type = defaultdict(list)
    for i, t in enumerate(final_types):
        idx_by_type[t].append(i)

    enums = load_enums()   # CEC 无枚举角色 → 空 dict
    n_arg_trunc = 0
    for etype, idxs in idx_by_type.items():
        ckpt_path = CKPT_DIR / f"pipe_arg_{etype}.pt"
        if not ckpt_path.exists():
            for i in idxs:
                preds[i] = make_pred_row(dev_rows[i], [])
            continue
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        labels = ckpt["labels"]
        scheme = {"labels": labels,
                  "label2id": {l: k for k, l in enumerate(labels)},
                  "id2label": {k: l for k, l in enumerate(labels)}}
        model = build_arg_model(str(args.bert_path), scheme,
                                _layers_of(ckpt, args.dropout)).to(device)
        model.load_state_dict(ckpt["state_dict"])
        rows = [dev_rows[i] for i in idxs]
        ds = BioDataset(rows, tokenizer, scheme, args.max_len, typed=True)
        n_arg_trunc += ds.n_truncated
        loader = DataLoader(ds, batch_size=args.arg_batch_size, shuffle=False,
                            num_workers=0, collate_fn=collate_bio)
        tags, _ = predict_char_tags(model, loader, scheme["id2label"], device)
        for i, row, tg in zip(idxs, rows, tags):
            ev = spans_to_event(decode_bio_spans(tg), row["text"], typed_type=etype,
                                enums=enums)
            preds[i] = make_pred_row(row, [ev] if ev else [])
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    for i, p in enumerate(preds):
        if p is None:   # 保险：绝不丢行（丢行会让评测分母变小、F1 虚高）
            preds[i] = make_pred_row(dev_rows[i], [])

    diag = {
        "n_type_wrong": n_type_wrong,
        "type_accuracy_on_dev": round(1 - n_type_wrong / max(len(dev_rows), 1), 4),
        "n_arg_truncated": n_arg_trunc,
    }
    return preds, diag


# ─────────────────────────── 主流程 ───────────────────────────
def main():
    args = parse_args()
    set_seed(args.seed)
    device = pick_device(args.device)
    data_dir = Path(args.data_dir) if args.data_dir else DATA_DIR
    print(f"设备：{device} | 数据：{data_dir}")

    args.bert_path = resolve_bert_path(args)
    print(f"BERT：{args.bert_path}")

    schema = load_schema()                    # data/conf/event_schema.json（CEC 列表格式）
    types_sorted = sorted(schema)
    type2id = {t: i for i, t in enumerate(types_sorted)}
    tokenizer = BertTokenizer.from_pretrained(str(args.bert_path))

    train_rows = load_extract_split("train", data_dir)
    dev_rows = load_extract_split("dev", data_dir)
    print(f"数据：train {len(train_rows)} 句 / dev {len(dev_rows)} 句 | 事件类型 {len(types_sorted)} 类")
    print(f"  type2id：{type2id}")

    logger = TrainLogger(method="bert_pipeline", args=args, root=ROOT,
                         notes="两段式：类型 5 分类 + 5 个类型条件论元专家（CEC）")

    t_all = time.perf_counter()

    # ── (b) 类型分类 ──
    type_info = {"train_s": 0.0, "skipped": True}
    if not args.skip_train and not args.only_experts:
        print("\n(b) 训练事件类型分类器（5 分类）...")
        type_info = train_type_classifier(train_rows, type2id, len(types_sorted),
                                          tokenizer, args, device, logger=logger)
    elif args.only_experts:
        print("  [only_experts] 跳过类型分类训练，复用 pipe_type.pt")

    # ── (c) 5 个论元专家 ──
    arg_infos = {}
    if not args.skip_train:
        print("\n(c) 训练类型条件论元专家（逐个训练，训完即释放）...")
        arg_infos = train_arg_experts(train_rows, schema, tokenizer, args, device,
                                      logger=logger)
    else:
        print("  [skip_train] 跳过论元专家训练，复用 pipe_arg_*.pt")

    # ── 组合评测 ──
    if args.skip_train or args.only_experts:
        ckpt = torch.load(CKPT_DIR / "pipe_type.pt", map_location=device, weights_only=False)
        type_model = BertCls(str(args.bert_path), len(types_sorted), args.dropout).to(device)
        type_model.load_state_dict(ckpt["state_dict"])
    else:
        type_model = BertCls(str(args.bert_path), len(types_sorted), args.dropout).to(device)
        type_model.load_state_dict(
            torch.load(CKPT_DIR / "pipe_type.pt", map_location=device,
                       weights_only=False)["state_dict"])

    print("\n组合两段 → dev 端到端评测...")
    with Timer() as timer:
        preds, diag = compose(dev_rows, tokenizer, args, device, type_model, type2id)
    lat = timer.summary(len(dev_rows))
    print(f"   诊断：类型判错 {diag['n_type_wrong']} 句 / "
          f"类型准确率 {diag['type_accuracy_on_dev']:.4f}")

    meta = {
        "display_name": "两段式流水线（类型分类 → 类型条件论元）",
        "train_s": round(time.perf_counter() - t_all, 1),
        "ms_per_sentence": lat["ms_per_sentence"],
        "peak_gpu_mb": lat["peak_gpu_mb"],
        "max_len": args.max_len,
        "cls_epochs": args.cls_epochs, "arg_epochs": args.arg_epochs,
        "arg_layers": args.arg_layers,
        "pipeline_stages": 2,
        "num_models": 1 + len([k for k in arg_infos if k != "_total_train_s"]),
        "type_classifier": type_info,
        "arg_experts": arg_infos,
        "diag": diag,
        "pretrained": True,
        "seed": args.seed,
        "note": "两段各跑一次；类型判错会直接让该句论元全错（误差累积）",
    }
    res = evaluate_predictions("pipeline", preds, meta=meta,
                               gold_path=gold_path_for(data_dir, "dev"))

    # ── 上界分析：类型给金标 ──
    print("\n上界分析：类型用金标（隔离「类型判错」造成的损失）...")
    preds_o, diag_o = compose(dev_rows, tokenizer, args, device, type_model, type2id,
                              oracle_type=True)
    meta_o = dict(meta)
    meta_o["display_name"] = "两段式（类型给金标的上界）"
    meta_o["note"] = "类型用金标，只保留论元段的错误 → 定位瓶颈在类型还是论元"
    meta_o["diag"] = diag_o
    res_o = evaluate_predictions("pipeline_oracle_type", preds_o, meta=meta_o,
                                 gold_path=gold_path_for(data_dir, "dev"))

    print("\n" + "=" * 70)
    print(f"{'方法':<30}{'trigger F1':>12}{'arg F1':>10}{'event F1':>10}")
    print("-" * 70)
    for tag, r in (("流水线（端到端）", res), ("流水线（类型金标上界）", res_o)):
        print(f"{tag:<30}{r['trigger']['f1']:>12.4f}{r['argument']['f1']:>10.4f}"
              f"{r['event_exact']['f1']:>10.4f}")
    print("=" * 70)
    print(f"总耗时 {meta['train_s']:.0f}s，推理延迟 {lat['ms_per_sentence']} ms/句")

    logger.finalize(
        total_s=meta["train_s"],
        type_inner_acc=type_info.get("inner_acc"),
        type_accuracy_on_dev=diag["type_accuracy_on_dev"],
        dev_metrics={k: res[k]["f1"] for k in ("trigger", "argument", "event_exact")},
        dev_metrics_oracle={k: res_o[k]["f1"] for k in ("trigger", "argument", "event_exact")},
        inference_latency_ms=lat["ms_per_sentence"],
        peak_gpu_mb=peak_gpu_mb(),
        num_models=meta["num_models"],
    )


def parse_args():
    ap = argparse.ArgumentParser(description="两段式流水线事件抽取（CEC）")
    ap.add_argument("--bert_path", default="")
    ap.add_argument("--cls_epochs", type=int, default=6)
    ap.add_argument("--cls_lr", type=float, default=2e-5)
    ap.add_argument("--cls_batch_size", type=int, default=16)
    ap.add_argument("--cls_max_len", type=int, default=192)
    # CEC 每类型仅 59~143 句、batch 32 → 每 epoch 约 3~4 步，epoch 数要给够（BERT 学论元边界并不快）。
    ap.add_argument("--arg_epochs", type=int, default=30)
    ap.add_argument("--arg_lr", type=float, default=3e-5)
    ap.add_argument("--arg_batch_size", type=int, default=32)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--arg_layers", type=int, default=12,
                    help="论元专家保留几层 BERT（默认 12=全模型；CEC 只有 5 类、每类样本少，"
                         "层数给足才学得动。改成 4 可省 ~2GB 磁盘，但实测 F1 明显下降）")
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--inner_val", type=float, default=0.1)
    ap.add_argument("--device", default="", help="cuda / mps / cpu；留空自动选")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip_train", action="store_true",
                    help="复用已训好的 checkpoint，只重新组合评测")
    ap.add_argument("--only_experts", action="store_true",
                    help="只重训论元专家，复用类型分类器")
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--data_dir", default="", help="数据目录；留空=data/processed")
    args = ap.parse_args()
    return args


if __name__ == "__main__":
    main()
