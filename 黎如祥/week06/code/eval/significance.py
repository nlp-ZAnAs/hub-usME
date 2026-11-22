# -*- coding: utf-8 -*-
"""
显著性检验：macro-F1 差 1 个点，到底是「真提升」还是「500 条抽样的噪声」？

做法：配对 bootstrap（对同一批样本有放回重采样 2000 次），
每次同时算两个模型的 macro-F1 并取差，得到差值的 95% 置信区间。
区间跨 0 → 不能宣称谁更好；区间整体 >0 → 提升可信。

这是本次作业最容易被忽略、但面试最容易被追问的一点：
「RoBERTa 比 BERT 高 1.2 个点」在 500 条评测集上可能根本不显著。

用法：
  python code/eval/significance.py
"""

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.config import DATASETS, RESULTS_DIR


def load(ds):
    out = {}
    for p in RESULTS_DIR.glob(f"pred_{ds}_*.json"):
        d = json.loads(p.read_text(encoding="utf-8"))
        if d.get("method", "").endswith("_leak"):
            continue
        out[d["method"]] = d
    return out


def arrays(d):
    y = np.array([x["true"] for x in d["preds"]])
    p = np.array([x["pred"] for x in d["preds"]])
    return y, p


def macro_f1(y, p):
    return f1_score(y, p, average="macro")


def paired_bootstrap(y, p1, p2, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        s = rng.choice(idx, len(idx), replace=True)
        diffs[b] = macro_f1(y[s], p1[s]) - macro_f1(y[s], p2[s])
    mean = float(diffs.mean())
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    # 单侧 p：差值 <= 0 的比例（越小越说明 p1 更好）
    p_win = float((diffs <= 0).mean())
    return mean, float(lo), float(hi), p_win


def main():
    lines = ["# Macro-F1 配对 bootstrap 检验（2000 次重采样）", "",
             "> 差值 = 模型 A − 模型 B；95% CI 跨 0 表示差异不显著。", ""]
    for ds in DATASETS:
        runs = load(ds)
        if "bert_bert-base-chinese" not in runs:
            continue
        y, p_base = arrays(runs["bert_bert-base-chinese"])
        lines += [f"## {DATASETS[ds]['display']}", "",
                  "| 对比（A vs B） | A Macro-F1 | B Macro-F1 | 差值 | 95% CI | 结论 |",
                  "|---|---|---|---|---|---|"]
        pairs = []
        for m in ["bert_chinese-roberta-wwm-ext", "bert_chinese-macbert-base",
                  "tfidf_svm", "qwen3_lora"]:
            if m in runs:
                pairs.append((m, "bert_bert-base-chinese"))
        # 最好的微调模型 vs TF-IDF
        fts = [m for m in runs if m.startswith("bert_")]
        best_ft = max(fts, key=lambda m: runs[m]["report"]["macro_f1"])
        if "tfidf_svm" in runs and best_ft != "bert_bert-base-chinese":
            pairs.append((best_ft, "tfidf_svm"))
        for a, b in pairs:
            ya, pa = arrays(runs[a])
            yb, pb = arrays(runs[b])
            assert (ya == yb).all()
            mean, lo, hi, p_win = paired_bootstrap(ya, pa, pb)
            sig = "显著" if (lo > 0 or hi < 0) else "不显著"
            lines.append(
                f"| {runs[a]['display']} vs {runs[b]['display']} | "
                f"{runs[a]['report']['macro_f1']:.3f} | {runs[b]['report']['macro_f1']:.3f} | "
                f"{mean:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {sig} |")
        lines.append("")
    (RESULTS_DIR / "significance.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"→ {RESULTS_DIR / 'significance.md'}")


if __name__ == "__main__":
    main()
