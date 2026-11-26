# -*- coding: utf-8 -*-
"""
显著性检验：换编码器带来的 1~2 个点，是真实提升还是 930 条 dev 的噪声？

做法：按句配对 bootstrap（重采样 930 句，2000 次），每次重算 trigger/argument 的
micro-F1 并取差，给出 95% 置信区间。区间跨 0 → 不能宣称更好。

用法：python code/eval/significance.py
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]          # week07
CODE = ROOT / "code"
sys.path.insert(0, str(CODE / "common"))
from evaluate_events import as_event_list, compare_events, load_jsonl, row_key  # noqa: E402

GOLD = CODE / "data" / "processed" / "dev_extract.jsonl"
OURS = CODE / "outputs" / "disc"
BASE = ROOT / "baseline" / "pred"
RESULTS = ROOT / "results"


def pred_path(name):
    for p in (OURS / f"pred_{name}.jsonl", BASE / f"pred_{name}.jsonl"):
        if p.exists():
            return p
    return None


def row_counts(name, gold):
    p = pred_path(name)
    if p is None:
        return None
    preds = {row_key(r): r for r in load_jsonl(str(p))}
    counts = []
    for g in gold:
        pr = preds.get(row_key(g))
        t, a, e = compare_events(as_event_list(g), as_event_list(pr) if pr else [])
        counts.append((t, a, e))
    return counts  # list of ((tp,fp,fn) x3)


def f1(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def bootstrap_pair(ca, cb, n_boot=2000, seed=42):
    """返回 (argument diff mean/lo/hi, trigger diff mean/lo/hi)"""
    a = np.array([[x[1][0], x[1][1], x[1][2], x[0][0], x[0][1], x[0][2]] for x in ca], dtype=float)
    b = np.array([[x[1][0], x[1][1], x[1][2], x[0][0], x[0][1], x[0][2]] for x in cb], dtype=float)
    rng = np.random.default_rng(seed)
    n = len(a)
    out = {"argument": [], "trigger": []}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        sa, sb = a[idx].sum(0), b[idx].sum(0)
        out["argument"].append(f1(sa[0], sa[1], sa[2]) - f1(sb[0], sb[1], sb[2]))
        out["trigger"].append(f1(sa[3], sa[4], sa[5]) - f1(sb[3], sb[4], sb[5]))
    res = {}
    for k, v in out.items():
        v = np.array(v)
        res[k] = (float(v.mean()), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    return res


def main():
    gold = load_jsonl(str(GOLD))
    pairs = [
        ("roberta_linear_256", "bert_base_linear_256", "RoBERTa vs BERT（同 decoder）"),
        ("macbert_linear_256", "bert_base_linear_256", "MacBERT vs BERT（同 decoder）"),
        ("macbert_linear_256", "roberta_linear_256", "MacBERT vs RoBERTa"),
        ("bert_crf_256", "bert_base_linear_256", "BERT+CRF vs BERT+Linear（课程基线）"),
        ("bilstm_crf", "bert_base_linear_256", "BiLSTM-CRF vs BERT+Linear"),
    ]
    lines = ["# 配对 bootstrap（dev 930 句，2000 次重采样）", "",
             "> 差值 = A − B；95% CI 跨 0 表示不显著。", ""]
    for a, b, label in pairs:
        ca, cb = row_counts(a, gold), row_counts(b, gold)
        if ca is None or cb is None:
            lines.append(f"- {label}：缺少预测文件，跳过")
            continue
        res = bootstrap_pair(ca, cb)
        def cell(k):
            m, lo, hi = res[k]
            sig = "显著" if (lo > 0 or hi < 0) else "不显著"
            return f"{m:+.4f} [{lo:+.4f}, {hi:+.4f}] {sig}"
        lines += [f"## {label}", "",
                  "| 指标 | 差值 | 95% CI | 结论 |", "|---|---|---|---|"]
        for k, name in [("argument", "论元 F1"), ("trigger", "触发词 F1")]:
            m, lo, hi = res[k]
            lines.append(f"| {name} | {m:+.4f} | [{lo:+.4f}, {hi:+.4f}] | "
                         f"{'显著' if (lo > 0 or hi < 0) else '不显著'} |")
        lines.append("")
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "significance.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"→ {RESULTS / 'significance.md'}")


if __name__ == "__main__":
    main()
