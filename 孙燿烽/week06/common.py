"""共同的数据、评测和结果协议；失败预测必须计入 Accuracy 和 Macro-F1。"""

import hashlib
import importlib.metadata
import json
import platform
import random
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parent
LABELS = ["负面", "正面"]
METHODS = {
    "majority": "多数类基线",
    "rule": "情感词规则",
    "tfidf_nb": "TF-IDF + MultinomialNB",
    "tfidf_lr": "TF-IDF + LogisticRegression",
    "tfidf_svm": "TF-IDF + LinearSVC",
    "rbt3": "RBT3 全量微调",
    "qwen_zero": "Qwen2.5-0.5B 零样本",
    "qwen_lora": "Qwen2.5-0.5B LoRA",
    "qwen_full": "Qwen2.5-0.5B 全量微调",
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


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


def load_data():
    splits = {}
    for name in ["train", "val", "test"]:
        path = ROOT / "data" / (name + ".jsonl")
        splits[name] = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    validate_splits(splits)
    manifest = read_json(ROOT / "data" / "manifest.json")
    if {k: fingerprint(v) for k, v in splits.items()} != manifest["split_sha256"]:
        raise ValueError("数据文件与 manifest 不一致，请重新 prepare_data.py 并重跑所有方法")
    cfg = config()
    if any(manifest["split_config"][k] != cfg[k] for k in manifest["split_config"]):
        raise ValueError("切分配置已变化，请先重新运行 prepare_data.py")
    return splits


def validate_splits(splits):
    seen_ids, seen_texts = set(), set()
    for name, rows in splits.items():
        if not rows or {r["label"] for r in rows} != {0, 1}:
            raise ValueError(f"{name}: 数据为空或缺少类别")
        for row in rows:
            if not row["text"] or row["id"] in seen_ids or row["text"] in seen_texts:
                raise ValueError(f"{name}: 检测到空文本、重复或跨集合泄漏")
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


def metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if len(y_true) == 0 or y_true.shape != y_pred.shape:
        raise ValueError("标签和预测必须非空且长度一致")
    if not np.isin(y_true, [0, 1]).all() or not np.isin(y_pred, [-1, 0, 1]).all():
        raise ValueError("标签范围错误")
    # 显式固定 labels，-1 预测仍是对应真实类别的 FN，不得先滤掉。
    p, r, f, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    matrix = [[int(((y_true == i) & (y_pred == j)).sum()) for j in [0, 1, -1]]
              for i in [0, 1]]
    return {
        "n": len(y_true),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f.mean()),
        "weighted_f1": float(np.average(f, weights=support)),
        "invalid": int((y_pred == -1).sum()),
        "used_classes": int(len(set(y_pred) & {0, 1})),
        "per_class": {LABELS[i]: {"precision": float(p[i]), "recall": float(r[i]),
                                  "f1": float(f[i]), "support": int(support[i])}
                      for i in range(2)},
        "confusion_matrix": matrix,
        "confusion_columns": ["负面", "正面", "解析失败"],
    }


def environment():
    import torch
    packages = ["torch", "numpy", "scipy", "scikit-learn", "transformers",
                "peft", "accelerate", "huggingface-hub", "tokenizers", "matplotlib"]
    return {
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": {p: importlib.metadata.version(p) for p in packages},
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "cuda": torch.version.cuda,
    }


def save_result(method, splits, predictions, *, train_seconds, predict_seconds,
                settings, history=None, raw=None, trainable_params=0, total_params=0,
                peak_gpu_gib=0, selected_epoch=None):
    test = splits["test"]
    if len(test) != len(predictions):
        raise ValueError("预测数与测试集不一致")
    rows = [{"id": d["id"], "true": d["label"], "pred": int(p)}
            for d, p in zip(test, predictions, strict=True)]
    if raw is not None:
        for row, text in zip(rows, raw, strict=True):
            row["raw"] = text
    result = {
        "method": method, "name": METHODS[method],
        "config": config(), "settings": settings,
        "split_sha256": {k: fingerprint(v) for k, v in splits.items()},
        "environment": environment(),
        "metrics": metrics([d["label"] for d in test], predictions),
        "cost": {"train_seconds": train_seconds, "predict_seconds": predict_seconds,
                 "milliseconds_per_item": predict_seconds * 1000 / len(test),
                 "trainable_params": trainable_params, "total_params": total_params,
                 "peak_gpu_allocated_gib": peak_gpu_gib},
        "selected_epoch": selected_epoch, "history": history or [], "predictions": rows,
    }
    write_json(ROOT / "results" / (method + ".json"), result)
    print(f"{method}: accuracy={result['metrics']['accuracy']:.4f}, "
          f"macro_f1={result['metrics']['macro_f1']:.4f}, "
          f"invalid={result['metrics']['invalid']}", flush=True)
    return result
