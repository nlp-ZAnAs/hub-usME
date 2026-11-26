# -*- coding: utf-8 -*-
"""
week07 作业 1：四路（+两条中间态）同口径对比图

左图：本机实跑、dev 930 全量口径
右图：n=200 固定子集口径（本机实跑 + 课件已有的 LLM 三行，后者标注来源）

数据来源：
- 本机：code/outputs/metrics/disc/*.json（本次跑出来的）
- 课件：code/outputs/llm/method_compare.csv（未在本机复现，仅作参照）
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

METRIC_DIR = r"D:\BaiduNetdiskDownload\week7-信息抽取\code\outputs\metrics\disc"

for name in ["Microsoft YaHei", "SimHei"]:
    if name in {f.name for f in font_manager.fontManager.ttflist}:
        plt.rcParams["font.sans-serif"] = [name]
        plt.rcParams["axes.unicode_minus"] = False
        break


def load(name):
    with open(os.path.join(METRIC_DIR, name), encoding="utf-8") as f:
        j = json.load(f)
    return (j["trigger"]["f1"], j["argument"]["f1"], j["event_exact"]["f1"])


# 左：dev 930
LEFT = [
    ("规则", load("rule.json"), "#94a3b8"),
    ("BERT+Linear", load("bert_linear_full.json"), "#3b82f6"),
    ("BERT+CRF", load("bert_crf_full.json"), "#6366f1"),
    ("CRF+硬约束", load("bert_crf_full_hard.json"), "#16a34a"),
    ("两段式", load("pipeline.json"), "#f59e0b"),
]

# 右：n=200（本机三行 + 课件 LLM 三行）
RIGHT = [
    ("规则", load("rule_n200.json"), "#94a3b8"),
    ("Linear", load("bert_linear_full_n200.json"), "#3b82f6"),
    ("CRF", load("bert_crf_full_n200.json"), "#6366f1"),
    ("CRF+硬约束", load("bert_crf_full_hard_n200.json"), "#16a34a"),
    ("两段式", load("pipeline_n200.json"), "#f59e0b"),
    ("LLM zero", (0.779, 0.525, 0.040), "#a78bfa"),
    ("LLM few", (0.829, 0.535, 0.091), "#8b5cf6"),
    ("LoRA SFT", (0.900, 0.697, 0.185), "#7c3aed"),
]

fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), dpi=150)
for ax, data, title in [
    (axes[0], LEFT, "dev 930 全量（本机实跑）"),
    (axes[1], RIGHT, "n=200 固定子集（后三行为课件结果，非本机实跑）"),
]:
    n = len(data)
    w = 0.26
    for i, (label, (t, a, e), color) in enumerate(data):
        xs = [i - w, i, i + w]
        ax.bar([xs[0]], [t], width=w * 0.92, color=color, alpha=0.55)
        ax.bar([xs[1]], [a], width=w * 0.92, color=color, alpha=1.0)
        ax.bar([xs[2]], [e], width=w * 0.92, color=color, alpha=0.75)
        for x, v in zip(xs, [t, a, e]):
            ax.text(x, v + 0.008, f"{v:.2f}", ha="center", va="bottom", fontsize=6.6)
    ax.set_xticks(range(n))
    ax.set_xticklabels([d[0] for d in data], fontsize=8, rotation=15)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("F1")
    ax.set_title(title, fontsize=10)
    ax.grid(axis="y", alpha=0.3)

handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=a)
           for c, a in [("#94a3b8", 0.55), ("#94a3b8", 1.0), ("#94a3b8", 0.75)]]
fig.legend(handles, ["trigger（浅）", "argument（实，主指标）", "event_exact（中）"],
           loc="lower center", ncol=3, frameon=False, fontsize=9)
fig.suptitle("第七周作业：六路同口径对比（argument F1 是主指标）", fontsize=12)
fig.tight_layout(rect=[0, 0.05, 1, 1])
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "compare_methods.png")
fig.savefig(out, bbox_inches="tight")
print("saved", out)
