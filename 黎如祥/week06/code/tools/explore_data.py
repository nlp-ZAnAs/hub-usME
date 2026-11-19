# -*- coding: utf-8 -*-
"""
数据体检：两个数据集的类别分布、文本长度、重复率

产出 results/fig_data_overview.png，并打印一份数据卡。
「换数据集」的第一步不是训练，而是先看它和原数据集差在哪：
这里最关键的差异是——TNEWS 15 类、最大类 11%（23× 不均衡）、标题 22 字；
ChnSentiCorp 2 类均衡、评论 109 字、验证集 28% 与训练集重复。

用法：
  python code/tools/explore_data.py
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.config import DATASETS, RESULTS_DIR
from common.data import clean_val, label_names, load_dataset

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def main():
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))
    for row, ds in enumerate(DATASETS):
        train, val = load_dataset(ds)
        labels = label_names(ds)
        clean, n_dup = clean_val(train, val)

        ax = axes[row][0]
        counts = [sum(1 for d in train if d["label"] == i) for i in range(len(labels))]
        ax.bar(range(len(labels)), counts, color="#1565c0")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_title(f"{DATASETS[ds]['display']}：训练集类别分布"
                     f"（max/min={max(counts)/max(min(counts),1):.1f}×）", fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        for i, c in enumerate(counts):
            if c == max(counts):
                ax.text(i, c, str(c), ha="center", va="bottom", fontsize=7)

        ax = axes[row][1]
        lens = sorted(len(d["sentence"]) for d in train)
        ax.hist(lens, bins=40, color="#ff9800", log=True)
        ax.axvline(lens[int(len(lens) * 0.95)], color="red", linestyle="--", linewidth=1)
        ax.set_title(f"{DATASETS[ds]['display']}：训练集长度分布"
                     f"（均值 {sum(lens)/len(lens):.0f}，P95 {lens[int(len(lens)*0.95)]}）", fontsize=9)
        ax.set_xlabel("字符数")
        ax.set_ylabel("样本数（log）")
        ax.grid(alpha=0.3)

        print(f"[{ds}] train={len(train)} val={len(val)} clean_val={len(clean)} "
              f"dup={n_dup} ({n_dup/len(val):.1%}) labels={len(labels)}")
    fig.suptitle("两个数据集的数据体检（换数据集前先看分布）", fontsize=12)
    fig.tight_layout()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(RESULTS_DIR / "fig_data_overview.png", dpi=150)
    print(f"→ {RESULTS_DIR / 'fig_data_overview.png'}")


if __name__ == "__main__":
    main()
