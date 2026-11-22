# -*- coding: utf-8 -*-
"""
数据读取：两个数据集走同一套「去重 + 固定 seed 抽样」协议

教学重点：
  1. TNEWS 的 val 里有 11% 与 train 句子完全相同（数据泄漏）。不做处理时，
     靠字面匹配的 TF-IDF 会被抬高 3 个点。两个数据集都统一去重后再评测。
  2. 所有方法必须在**同一批 500 条样本**上评测（seed=42），否则「差 2 个点」
     可能只是抽样波动。这个模块就是那个唯一入口。
"""

import json
import random
from pathlib import Path

from .config import DATASETS, dataset_paths


def load_dataset(name: str):
    """返回 (train, val)，每条 {"sentence": str, "label": int}"""
    train_p, val_p = dataset_paths(name)
    train = json.loads(train_p.read_text(encoding="utf-8"))
    val = json.loads(val_p.read_text(encoding="utf-8"))
    for row in train + val:
        row["label"] = int(row["label"])
    return train, val


def clean_val(train, val):
    """剔除与训练集句子完全相同的验证样本，返回 (clean, n_dup)。"""
    seen = {d["sentence"] for d in train}
    clean = [d for d in val if d["sentence"] not in seen]
    return clean, len(val) - len(clean)


def eval_samples(name: str, n: int = 500, seed: int = 42):
    """最常用入口：读数据 → 去重 → 固定 seed 抽 n 条，返回 (samples, meta)。"""
    cfg = DATASETS[name]
    train, val = load_dataset(name)
    clean, n_dup = clean_val(train, val)
    if n and n < len(clean):
        samples = random.Random(seed).sample(clean, n)
    else:
        samples = list(clean)
    meta = {
        "dataset": name,
        "dataset_display": cfg["display"],
        "n_train": len(train),
        "n_val": len(val),
        "n_dup": n_dup,
        "dup_rate": round(n_dup / len(val), 4) if val else 0.0,
        "n_clean": len(clean),
        "n_eval": len(samples),
        "seed": seed,
    }
    return samples, meta


def label_names(name: str):
    return DATASETS[name]["labels"]


def majority_label(train):
    from collections import Counter
    return Counter(d["label"] for d in train).most_common(1)[0][0]


if __name__ == "__main__":
    for ds in DATASETS:
        samples, meta = eval_samples(ds, 500, 42)
        print(ds, json.dumps(meta, ensure_ascii=False))
