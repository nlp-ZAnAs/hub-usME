"""共同的数据、标签、评测与结果协议。

所有方法都把预测写成同一种结构：每个测试句一行 {id, entities:[{type,start,end,text}]}，
end 为左闭右开；指标一律由 evaluate() 计算，解析失败的句子以空预测留在分母里。
"""

import hashlib
import importlib.metadata
import json
import platform
import random
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
TYPES = ["address", "book", "company", "game", "government",
         "movie", "name", "organization", "position", "scene"]
TYPE_ZH = {"address": "地址", "book": "书名", "company": "公司", "game": "游戏",
           "government": "政府机构", "movie": "影视作品", "name": "人名",
           "organization": "组织机构", "position": "职位", "scene": "景点"}
LABELS = ["O"] + [f"{p}-{t}" for t in TYPES for p in ("B", "I")]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}

METHODS = {
    "rule": "训练集词典匹配（最长优先）",
    "bilstm_crf": "字符 BiLSTM-CRF（无预训练）",
    "roberta_linear": "RoBERTa-wwm + Linear",
    "roberta_linear_constrained": "RoBERTa-wwm + Linear + 约束解码",
    "roberta_crf": "RoBERTa-wwm + CRF",
    "roberta_crf_constrained": "RoBERTa-wwm + CRF + 约束解码",
    "qwen_zero": "Qwen2.5-1.5B 零样本",
    "qwen_few": "Qwen2.5-1.5B 少样本",
    "qwen_lora": "Qwen2.5-1.5B LoRA",
    "deepseek_zero": "DeepSeek API 零样本",
    "deepseek_few": "DeepSeek API 少样本",
}
API_METHODS = {"deepseek_zero", "deepseek_few"}


# ───────────────────────────── IO ─────────────────────────────

def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                    encoding="utf-8")


def config():
    return read_json(ROOT / "config.json")


def fingerprint(value):
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_path(model_id):
    return ROOT / ".cache" / model_id.split("/")[-1]


def checkpoint_path(name):
    return ROOT / "checkpoints" / name


# ───────────────────────────── 数据 ─────────────────────────────

def load_data():
    splits = {name: read_jsonl(ROOT / "data" / f"{name}.jsonl") for name in ["train", "val", "test"]}
    validate_splits(splits)
    manifest = read_json(ROOT / "data" / "manifest.json")
    if {k: fingerprint(v) for k, v in splits.items()} != manifest["split_sha256"]:
        raise ValueError("数据文件与 manifest 不一致，请重新 prepare_data.py 并重跑所有方法")
    cfg = config()
    if any(manifest["split_config"][k] != cfg[k] for k in manifest["split_config"]):
        raise ValueError("切分配置已变化，请先重新运行 prepare_data.py")
    return splits


def api_subset_ids():
    return read_json(ROOT / "data" / "api_subset_ids.json")["ids"]


def validate_entities(text, entities):
    last_end = 0
    for e in entities:
        if e["type"] not in TYPES or not 0 <= e["start"] < e["end"] <= len(text):
            raise ValueError(f"非法实体 {e}")
        if text[e["start"]:e["end"]] != e["text"]:
            raise ValueError(f"实体文本与 offset 不符：{e}")
        if e["start"] < last_end:
            raise ValueError("实体必须按起点排序且互不重叠")
        last_end = e["end"]


def validate_splits(splits):
    seen_ids, seen_texts = set(), set()
    for name, rows in splits.items():
        if not rows:
            raise ValueError(f"{name}: 数据为空")
        for row in rows:
            if not row["text"] or row["id"] in seen_ids or row["text"] in seen_texts:
                raise ValueError(f"{name}: 检测到空文本、重复 ID 或跨集合文本泄漏")
            validate_entities(row["text"], row["entities"])
            seen_ids.add(row["id"])
            seen_texts.add(row["text"])


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(6)
    torch.backends.cudnn.benchmark = False


# ───────────────────────────── BIO ─────────────────────────────

def spans_to_bio(text, entities):
    tags = ["O"] * len(text)
    for e in entities:
        tags[e["start"]] = "B-" + e["type"]
        for i in range(e["start"] + 1, e["end"]):
            tags[i] = "I-" + e["type"]
    return tags


def is_allowed(prev, cur):
    """BIO 合法性：I-x 只能接在 B-x / I-x 之后；prev=None 表示句首。"""
    if not cur.startswith("I-"):
        return True
    return prev is not None and prev != "O" and prev[2:] == cur[2:]


def count_illegal(tags):
    prev, n = None, 0
    for tag in tags:
        n += not is_allowed(prev, tag)
        prev = tag
    return n


def bio_to_spans(text, tags):
    """conlleval 式宽松解码：非法的 I-x（句首、接 O、接其他类型）视为新实体的开头。

    这样「不做约束」的解码仍然产生确定的输出；非法转移的数量另由 count_illegal 统计。
    """
    if len(tags) != len(text):
        raise ValueError("标签数必须等于字符数")
    spans, start, kind = [], None, None
    for i, tag in enumerate(list(tags) + ["O"]):
        boundary = tag == "O" or tag.startswith("B-") or (tag.startswith("I-") and tag[2:] != kind)
        if boundary and start is not None:
            spans.append({"type": kind, "start": start, "end": i, "text": text[start:i]})
            start, kind = None, None
        if tag != "O" and start is None:
            start, kind = i, tag[2:]
    return spans


def transition_constraints():
    """返回 (allowed_start[C], allowed[C,C]) 两个布尔矩阵，供约束维特比使用。"""
    allowed_start = np.array([is_allowed(None, b) for b in LABELS])
    allowed = np.array([[is_allowed(a, b) for b in LABELS] for a in LABELS])
    return allowed_start, allowed


# ───────────────────────────── 评测 ─────────────────────────────

def _prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def sentence_counts(gold_entities, pred_entities):
    """单句的 TP/FP/FN；可加总后算 micro，也供配对 bootstrap 重采样使用。"""
    out = {}
    views = {
        "strict": lambda e: (e["type"], e["start"], e["end"]),
        "boundary": lambda e: (e["start"], e["end"]),
        "text": lambda e: (e["type"], e["text"]),
    }
    for name, key in views.items():
        g, p = {key(e) for e in gold_entities}, {key(e) for e in pred_entities}
        out[name] = (len(g & p), len(p - g), len(g - p))
    for t in TYPES:
        g = {(e["start"], e["end"]) for e in gold_entities if e["type"] == t}
        p = {(e["start"], e["end"]) for e in pred_entities if e["type"] == t}
        out[t] = (len(g & p), len(p - g), len(g - p))
    return out


def evaluate(gold_rows, pred_rows):
    """strict：类型+起止完全一致（主指标）；boundary：只看起止；text：(类型, 实体文本) 集合。"""
    if [r["id"] for r in gold_rows] != [r["id"] for r in pred_rows]:
        raise ValueError("预测与金标的 ID 或顺序不一致")
    totals = {}
    for gold, pred in zip(gold_rows, pred_rows, strict=True):
        validate_entities(gold["text"], gold["entities"])
        for e in pred["entities"]:
            if e["type"] not in TYPES or gold["text"][e["start"]:e["end"]] != e["text"]:
                raise ValueError(f"{pred['id']}: 预测实体与原文不符 {e}")
        for k, (tp, fp, fn) in sentence_counts(gold["entities"], pred["entities"]).items():
            acc = totals.setdefault(k, [0, 0, 0])
            acc[0] += tp
            acc[1] += fp
            acc[2] += fn
    result = {"n": len(gold_rows)}
    for view in ["strict", "boundary", "text"]:
        p, r, f = _prf(*totals[view])
        result[view] = {"precision": p, "recall": r, "f1": f,
                        "tp": totals[view][0], "fp": totals[view][1], "fn": totals[view][2]}
    per_type = {}
    for t in TYPES:
        tp, fp, fn = totals[t]
        p, r, f = _prf(tp, fp, fn)
        per_type[t] = {"precision": p, "recall": r, "f1": f, "support": tp + fn}
    result["per_type"] = per_type
    result["macro_f1"] = float(np.mean([per_type[t]["f1"] for t in TYPES]))
    return result


def error_categories(gold_entities, pred_entities):
    """把单句错误归成四类：类型错 / 边界错 / 漏抽 / 多抽（strict 口径）。"""
    gold = {(e["start"], e["end"]): e for e in gold_entities}
    pred = {(e["start"], e["end"]): e for e in pred_entities}
    out = {"type_error": [], "boundary_error": [], "missed": [], "spurious": []}
    matched_gold = set()
    for key, p in pred.items():
        if key in gold:
            if gold[key]["type"] != p["type"]:
                out["type_error"].append({"gold": gold[key], "pred": p})
            matched_gold.add(key)
            continue
        overlaps = [g for gk, g in gold.items()
                    if gk not in pred and g["start"] < p["end"] and p["start"] < g["end"]]
        if overlaps:
            out["boundary_error"].append({"gold": overlaps[0], "pred": p})
            matched_gold.update((g["start"], g["end"]) for g in overlaps)
        else:
            out["spurious"].append(p)
    out["missed"] = [g for k, g in gold.items() if k not in matched_gold]
    return out


# ───────────────────────────── 结果 ─────────────────────────────

def environment():
    import torch
    packages = ["torch", "numpy", "transformers", "peft", "accelerate",
                "huggingface-hub", "tokenizers", "matplotlib"]
    versions = {}
    for p in packages:
        try:
            versions[p] = importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError:
            versions[p] = None
    return {
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": versions,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "cuda": torch.version.cuda,
    }


def save_result(method, splits, predictions, *, eval_ids=None, train_seconds=0.0,
                predict_seconds=0.0, settings=None, history=None, extra=None,
                trainable_params=0, total_params=0, peak_gpu_gib=0.0, selected_epoch=None):
    """predictions: 与评测行一一对应的 {id, entities, [raw], [status]} 列表。

    eval_ids 为空时评测完整测试集；API 方法传入固定子集 ID。
    """
    test = splits["test"]
    if eval_ids is not None:
        index = {r["id"]: r for r in test}
        test = [index[i] for i in eval_ids]
    if [r["id"] for r in test] != [p["id"] for p in predictions]:
        raise ValueError("预测 ID/顺序与评测集不一致")
    for row, pred in zip(test, predictions, strict=True):
        pred["entities"] = sorted(pred["entities"], key=lambda e: (e["start"], e["end"]))
        pred["illegal_transitions"] = pred.get("illegal_transitions", 0)
    metric = evaluate(test, predictions)
    metric["illegal_transitions"] = sum(p["illegal_transitions"] for p in predictions)
    result = {
        "method": method, "name": METHODS[method],
        "config": config(), "settings": settings or {},
        "split_sha256": {k: fingerprint(v) for k, v in splits.items()},
        "eval_ids": [r["id"] for r in test],
        "environment": environment(),
        "metrics": metric,
        "cost": {"train_seconds": train_seconds, "predict_seconds": predict_seconds,
                 "milliseconds_per_item": predict_seconds * 1000 / len(test),
                 "trainable_params": trainable_params, "total_params": total_params,
                 "peak_gpu_allocated_gib": peak_gpu_gib},
        "selected_epoch": selected_epoch, "history": history or [],
        "extra": extra or {},
        "predictions": predictions,
    }
    write_json(ROOT / "results" / f"{method}.json", result)
    s = metric["strict"]
    print(f"{method}: n={metric['n']} strict P={s['precision']:.4f} R={s['recall']:.4f} "
          f"F1={s['f1']:.4f} macro={metric['macro_f1']:.4f} "
          f"illegal={metric['illegal_transitions']}", flush=True)
    return result
