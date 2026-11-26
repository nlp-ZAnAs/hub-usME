"""CLUENER2020 → 统一格式；官方 train 切出 val，官方 dev 作为测试集（官方 test 无公开标注）。

输出 data/{train,val,test}.jsonl、data/api_subset_ids.json、data/manifest.json。
offset 统一为左闭右开；重叠实体按「起点靠前、同起点取更长」保留一个，其余记入审计。
"""

import json
import random
from collections import Counter

from common import (ROOT, TYPES, config, fingerprint, read_json, spans_to_bio, bio_to_spans,
                    validate_splits, write_json, write_jsonl)
from llm_common import gold_mentions, mentions_to_spans

SOURCE = ROOT / ".cache" / "cluener" / "extracted"


def convert(path, prefix, audit):
    rows = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        raw = json.loads(line)
        text = raw["text"]
        entities = []
        for kind, items in raw["label"].items():
            if kind not in TYPES:
                raise ValueError(f"未知类型 {kind}")
            for mention, positions in items.items():
                for start, end in positions:  # CLUENER 的 end 是闭区间
                    if text[start:end + 1] != mention:
                        raise ValueError(f"{prefix}-{index}: offset 与文本不符")
                    entities.append({"type": kind, "start": start, "end": end + 1, "text": mention})
                occurrences = text.count(mention)
                if occurrences > len(positions):
                    audit["mentions_with_unlabeled_repeats"] += 1
        entities.sort(key=lambda e: (e["start"], -(e["end"] - e["start"])))
        kept, last_end = [], 0
        for e in entities:
            if e["start"] < last_end:
                audit["overlap_dropped"].append({"id": f"{prefix}-{index:05d}", "entity": e})
                continue
            kept.append(e)
            last_end = e["end"]
        rows.append({"id": f"{prefix}-{index:05d}", "text": text, "entities": kept})
    return rows


def stats(rows):
    lengths = sorted(len(r["text"]) for r in rows)
    return {
        "sentences": len(rows),
        "entities": sum(len(r["entities"]) for r in rows),
        "sentences_without_entity": sum(not r["entities"] for r in rows),
        "per_type": dict(Counter(e["type"] for r in rows for e in r["entities"])),
        "max_chars": lengths[-1], "mean_chars": sum(lengths) / len(lengths),
    }


def mapping_ceiling(rows, policy):
    """金标 → (类型, 文本) → 回填 span，再与金标比 strict F1：生成式链路在该口径下的理论上限。"""
    tp = fp = fn = 0
    for row in rows:
        spans, missing = mentions_to_spans(row["text"], gold_mentions(row), policy)
        if missing:
            raise ValueError("金标文本必然在原文中")
        g = {(e["type"], e["start"], e["end"]) for e in row["entities"]}
        p = {(e["type"], e["start"], e["end"]) for e in spans}
        tp, fp, fn = tp + len(g & p), fp + len(p - g), fn + len(g - p)
    precision, recall = tp / (tp + fp), tp / (tp + fn)
    return {"precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall), "fp": fp, "fn": fn}


def main():
    cfg = config()
    audit = {"overlap_dropped": [], "mentions_with_unlabeled_repeats": 0}
    train_all = convert(SOURCE / "train.json", "train", audit)
    test = convert(SOURCE / "dev.json", "dev", audit)
    for row in train_all + test:
        if bio_to_spans(row["text"], spans_to_bio(row["text"], row["entities"])) != row["entities"]:
            raise ValueError(f"{row['id']}: BIO 往返不一致")
    shuffled = list(train_all)
    random.Random(cfg["seed"]).shuffle(shuffled)
    val_ids = {r["id"] for r in shuffled[:cfg["val_size"]]}
    splits = {"train": [r for r in train_all if r["id"] not in val_ids],
              "val": [r for r in train_all if r["id"] in val_ids],
              "test": test}
    validate_splits(splits)
    for name, rows in splits.items():
        write_jsonl(ROOT / "data" / f"{name}.jsonl", rows)

    subset = set(r["id"] for r in random.Random(cfg["seed"]).sample(test, cfg["api_subset_size"]))
    write_json(ROOT / "data" / "api_subset_ids.json",
               {"seed": cfg["seed"], "size": cfg["api_subset_size"],
                "ids": [r["id"] for r in test if r["id"] in subset]})

    # 回填策略只按验证集选择；测试集上限只作报告，不参与选择。
    ceilings = {policy: {"val": mapping_ceiling(splits["val"], policy),
                         "test": mapping_ceiling(splits["test"], policy)}
                for policy in ["first", "all"]}
    policy = max(["first", "all"], key=lambda p: ceilings[p]["val"]["f1"])
    assets = read_json(ROOT / "assets_manifest.json")
    write_json(ROOT / "data" / "manifest.json", {
        "source": {"dataset": "CLUENER2020", **assets["dataset"]},
        "split_config": {k: cfg[k] for k in ["seed", "val_size", "api_subset_size"]},
        "split_rule": "官方 train 按 seed 打乱后取 val_size 条作验证集；官方 dev 全部作测试集",
        "split_sha256": {k: fingerprint(v) for k, v in splits.items()},
        "stats": {k: stats(v) for k, v in splits.items()},
        "audit": {"overlap_dropped": audit["overlap_dropped"],
                  "mentions_with_unlabeled_repeats": audit["mentions_with_unlabeled_repeats"],
                  "cross_split_text_overlap": 0},
        "llm_span_policy": {"selected": policy, "selected_by": "validation ceiling F1",
                            "ceilings": ceilings},
    })
    for name, rows in splits.items():
        print(name, stats(rows))
    print("overlap dropped:", len(audit["overlap_dropped"]),
          "| unlabeled repeats:", audit["mentions_with_unlabeled_repeats"])
    print("LLM span policy:", policy, json.dumps(ceilings, ensure_ascii=False))


if __name__ == "__main__":
    main()
