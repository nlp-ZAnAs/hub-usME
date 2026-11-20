"""数据读取与「干净验证集」构建

教学点：TNEWS 这份切分里，验证集有 1,099/10,000 条（11%）与训练集句子完全相同。
       不去重的话，靠词面匹配的 TF-IDF 会「背答案」，指标虚高约 3 个点。
       所以三种方法统一在 clean val 上评测，并且用同一个 seed 抽同一批样本。
"""

import json
import random

import config

_DATA_CACHE = {}


def _read(name: str):
    if name not in _DATA_CACHE:
        with open(config.DATA_DIR / name, encoding="utf-8") as f:
            _DATA_CACHE[name] = json.load(f)
    return _DATA_CACHE[name]


def load_splits():
    return _read("train.json"), _read("val.json")


def clean_val(train=None, val=None):
    if train is None or val is None:
        train, val = load_splits()
    seen = {d["sentence"] for d in train}
    clean = [d for d in val if d["sentence"] not in seen]
    return clean, len(val) - len(clean)


def sample_subset(items, n, seed=config.SEED):
    if not n or n >= len(items):
        return list(items)
    rng = random.Random(seed)
    return rng.sample(items, n)


def load_clean_val(n=config.N_EVAL, seed=config.SEED):
    train, val = load_splits()
    clean, n_dup = clean_val(train, val)
    samples = sample_subset(clean, n, seed)
    meta = {
        "n_train": len(train),
        "n_val": len(val),
        "n_dup": n_dup,
        "dup_rate": round(n_dup / len(val), 4),
        "n_clean": len(clean),
        "n_eval": len(samples),
        "seed": seed,
    }
    return samples, meta


def dataset_meta():
    train, val = load_splits()
    clean, n_dup = clean_val(train, val)
    from collections import Counter
    cnt = Counter(d["label"] for d in train)
    return {
        "n_train": len(train),
        "n_val": len(val),
        "n_dup": n_dup,
        "dup_rate": round(n_dup / len(val), 4),
        "n_clean": len(clean),
        "class_counts": dict(sorted(cnt.items())),
        "imbalance": round(max(cnt.values()) / min(cnt.values()), 1),
        "len_median": sorted(len(d["sentence"]) for d in train)[len(train) // 2],
    }
