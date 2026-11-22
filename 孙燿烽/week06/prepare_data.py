"""先去重和剔除冲突标签，再按原类别比例抽取固定的 train/val/test。"""

import csv
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict

from sklearn.model_selection import train_test_split

from common import ROOT, config, file_sha256, fingerprint, read_json, validate_splits, write_json


def normalize(text):
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text)).strip()


def clean_rows(raw_rows, max_chars):
    groups = defaultdict(list)
    missing = invalid_labels = truncated = 0
    full_texts = set()
    for index, raw in enumerate(raw_rows):
        text = normalize(raw.get("review") or "")
        if not text:
            missing += 1
            continue
        if str(raw.get("label")) not in {"0", "1"}:
            invalid_labels += 1
            continue
        full_texts.add(text)
        truncated += len(text) > max_chars
        # 所有方法只看到统一的前 max_chars 字，按实际模型输入再次去重。
        groups[text[:max_chars]].append({"source_row": index, "label": int(raw["label"])})
    clean, conflicts, duplicate_rows = [], 0, 0
    for text, group in groups.items():
        if len({r["label"] for r in group}) > 1:
            conflicts += len(group)
            continue
        duplicate_rows += len(group) - 1
        clean.append({"id": hashlib.sha256(text.encode()).hexdigest()[:20],
                      "source_row": group[0]["source_row"], "text": text,
                      "label": group[0]["label"]})
    return clean, {
        "raw_rows": len(raw_rows), "empty_rows_removed": missing,
        "invalid_labels_removed": invalid_labels, "conflicting_rows_removed": conflicts,
        "duplicate_input_rows_removed": duplicate_rows, "unique_full_texts": len(full_texts),
        "raw_rows_over_character_limit": truncated, "clean_rows": len(clean),
    }


def main():
    cfg = config()
    source = ROOT / ".cache" / "ChnSentiCorp_htl_all.csv"
    assets = read_json(ROOT / "assets_manifest.json")
    if file_sha256(source) != assets["dataset"]["sha256"]:
        raise ValueError("原始数据校验失败")
    with source.open(encoding="utf-8-sig", newline="") as f:
        raw_rows = list(csv.DictReader(f))
    rows, audit = clean_rows(raw_rows, cfg["max_chars"])
    remaining, test = train_test_split(rows, test_size=cfg["test_size"],
        stratify=[r["label"] for r in rows], random_state=cfg["seed"])
    remaining, val = train_test_split(remaining, test_size=cfg["val_size"],
        stratify=[r["label"] for r in remaining], random_state=cfg["seed"])
    train, unused = train_test_split(remaining, train_size=cfg["train_size"],
        stratify=[r["label"] for r in remaining], random_state=cfg["seed"])
    splits = {"train": train, "val": val, "test": test}
    validate_splits(splits)
    (ROOT / "data").mkdir(exist_ok=True)
    for name, items in splits.items():
        (ROOT / "data" / (name + ".jsonl")).write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in items), encoding="utf-8")
    audit.update({
        "unused_rows": len(unused), "cross_split_duplicate_inputs": 0,
        "split_counts": {k: {"total": len(v), "labels": dict(Counter(r["label"] for r in v))}
                         for k, v in splits.items()},
        "split_config": {k: cfg[k] for k in ["seed", "train_size", "val_size", "test_size", "max_chars"]},
        "split_sha256": {k: fingerprint(v) for k, v in splits.items()},
        "source": assets["dataset"],
        "limitations": ["未做语义近重复聚类或按酒店分组；无可靠酒店 ID。",
                        "公开数据可能出现在预训练语料中，无法排除预训练污染。",
                        "这是自建固定子集，不是原数据集官方 benchmark 切分。"],
    })
    write_json(ROOT / "data" / "manifest.json", audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
