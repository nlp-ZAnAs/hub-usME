# -*- coding: utf-8 -*-
"""
汇总：把两个数据集、所有方法的预测结果拉到同一张表 + 画对比图

报告口径（和课程一致）：
  - Accuracy 都含解析失败（zero-shot 解析不出来的按答错算）；
  - 主指标 Macro-F1，因为它对 15 类里的长尾类更敏感；
  - 所有方法都在同一批 500 条去重样本上评测（seed=42）。

用法：
  python code/eval/summarize.py
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.config import DATASETS, RESULTS_DIR
from common.data import label_names

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

FAMILY_ORDER = {"rule": 0, "linear": 1, "finetune": 2, "llm_zero_shot": 3, "llm_lora": 4}
FAMILY_COLOR = {"rule": "#9e9e9e", "linear": "#ff9800", "finetune": "#2196f3",
                "llm_zero_shot": "#9c27b0", "llm_lora": "#e91e63"}


def load_results():
    out = {}
    leak = {}
    for p in sorted(RESULTS_DIR.glob("pred_*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[warn] {p} 解析失败: {e}")
            continue
        if data.get("method", "").endswith("_leak"):
            leak[data["dataset"]] = data
            continue
        out.setdefault(data["dataset"], []).append(data)
    for rows in out.values():
        rows.sort(key=lambda r: (FAMILY_ORDER.get(r.get("family"), 9), r["display"]))
    return out, leak


def fmt(v, nd=3):
    return "—" if v is None else f"{v:.{nd}f}"


def table(rows, labels):
    lines = ["| 方法 | 家族 | 参数 | Acc（含失败） | Macro-F1 | 覆盖类别 | 训练耗时 | 推理 ms/条(批量) |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        rep, cost = r["report"], r.get("cost", {})
        params = cost.get("params")
        params_s = "—" if not params else (f"{params/1e6:.1f}M" if params >= 1e6 else str(params))
        t = cost.get("train_seconds")
        t_s = "—" if t is None else (f"{t:.0f}s" if t < 120 else f"{t/60:.1f}min")
        ms = cost.get("per_item_ms")
        ms_s = "—" if ms is None else ("<0.1ms" if ms < 0.1 else f"{ms:.1f}ms")
        used = rep.get("pred_distribution", {}).get("used_classes", "—")
        lines.append(
            f"| {r['display']} | {r.get('family','-')} | {params_s} | "
            f"{rep.get('acc_incl_unparsed', rep.get('acc','—')):.3f} | {rep.get('macro_f1','—')} | "
            f"{used}/{len(labels)} | {t_s} | {ms_s} |")
    return "\n".join(lines)


def bar_compare(data, path):
    datasets = list(data)
    fig, axes = plt.subplots(1, len(datasets), figsize=(6.2 * len(datasets), 4.6), squeeze=False)
    for ax, ds in zip(axes[0], datasets):
        rows = data[ds]
        names = [r["display"].replace(" 微调", "") for r in rows]
        accs = [r["report"].get("acc_incl_unparsed", 0) for r in rows]
        f1s = [r["report"].get("macro_f1", 0) for r in rows]
        x = np.arange(len(rows))
        w = 0.38
        ax.bar(x - w / 2, accs, w, label="Accuracy", color="#90caf9")
        ax.bar(x + w / 2, f1s, w, label="Macro-F1", color="#1565c0")
        for i, (a, f) in enumerate(zip(accs, f1s)):
            ax.text(i - w / 2, a + 0.01, f"{a:.2f}", ha="center", fontsize=7)
            ax.text(i + w / 2, f + 0.01, f"{f:.2f}", ha="center", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=25, ha="right", fontsize=8)
        ax.set_ylim(0, 1.0)
        ax.set_title(DATASETS[ds]["display"])
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("分类方法对比（同一批 500 条去重验证样本）", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def cost_scatter(data, path):
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    markers = {"tnews": "o", "chnsenticorp": "s"}
    offsets = {
        "BERT-base-chinese 微调": (8, -14),
        "RoBERTa-wwm-ext 微调": (8, 0),
        "MacBERT-base 微调": (8, 14),
        "Qwen3-0.6B + LoRA SFT": (-78, 8),
        "Qwen3-0.6B zero-shot": (8, 8),
        "规则 / 关键词": (8, -14),
        "TF-IDF + LinearSVC": (8, 6),
    }
    for ds, rows in data.items():
        for r in rows:
            t = r.get("cost", {}).get("train_seconds") or 0.1
            f1 = r["report"].get("macro_f1", 0)
            ax.scatter(t, f1, marker=markers.get(ds, "o"), s=70,
                       color=FAMILY_COLOR.get(r.get("family"), "gray"), alpha=0.85)
            dx, dy = offsets.get(r["display"], (6, 6))
            ax.annotate(r["display"].replace(" 微调", ""), (t, f1),
                        textcoords="offset points", xytext=(dx, dy), fontsize=6.5)
    ax.set_xscale("log")
    ax.set_xlim(0.07, 3000)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("训练耗时（秒，log）")
    ax.set_ylabel("Macro-F1")
    ax.set_title("性价比：训练成本 vs Macro-F1（圆=TNEWS，方=ChnSentiCorp）")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def confusion_fig(result, path):
    labels = label_names(result["dataset"])
    cm = np.array(result["report"]["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if cm[i, j]:
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=6,
                        color="white" if cm[i, j] > cm.max() * 0.5 else "black")
    ax.set_xlabel("预测")
    ax.set_ylabel("真实")
    ax.set_title(f"混淆矩阵：{result['display']} / {DATASETS[result['dataset']]['display']}")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def per_class_fig(data, path):
    fig, axes = plt.subplots(1, len(data), figsize=(6.4 * len(data), 4.2), squeeze=False)
    for ax, (ds, rows) in zip(axes[0], data.items()):
        labels = label_names(ds)
        best_ft = max([r for r in rows if r.get("family") == "finetune"],
                      key=lambda r: r["report"].get("macro_f1", 0), default=None)
        tfidf = next((r for r in rows if r.get("family") == "linear"), None)
        x = np.arange(len(labels))
        if best_ft:
            ax.bar(x - 0.2, [best_ft["report"]["per_class_recall"].get(str(i),
                            best_ft["report"]["per_class_recall"].get(i)) or 0
                            for i in range(len(labels))],
                   0.4, label=best_ft["display"].replace(" 微调", ""), color="#1565c0")
        if tfidf:
            ax.bar(x + 0.2, [tfidf["report"]["per_class_recall"].get(str(i),
                            tfidf["report"]["per_class_recall"].get(i)) or 0
                            for i in range(len(labels))],
                   0.4, label="TF-IDF", color="#ff9800")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Recall")
        ax.set_title(f"每类召回：{DATASETS[ds]['display']}")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    data, leak = load_results()
    if not data:
        print("没有找到任何 pred_*.json，先跑各路实验")
        return

    sections = ["# Week06 分类方法对比：结果汇总", ""]
    for ds, rows in data.items():
        labels = label_names(ds)
        sections += [f"## {DATASETS[ds]['display']}", "", table(rows, labels), ""]
        meta = rows[0]["meta"]
        sections += [f"> 训练 {meta['n_train']} 条 / 验证 {meta['n_val']} 条，"
                     f"去重 {meta['n_dup']} 条（{meta['dup_rate']:.1%}），"
                     f"评测 {meta['n_eval']} 条，seed={meta['seed']}。", ""]
        best_ft = max([r for r in rows if r.get("family") == "finetune"],
                      key=lambda r: r["report"].get("macro_f1", 0), default=None)
        if best_ft:
            cm = np.array(best_ft["report"]["confusion_matrix"])
            pairs = []
            for i in range(len(labels)):
                for j in range(len(labels)):
                    if i != j and cm[i, j] > 0:
                        pairs.append((int(cm[i, j]), labels[i], labels[j]))
            pairs.sort(reverse=True)
            sections += [f"**{best_ft['display']} 最常混淆的类对**：" +
                         "；".join(f"{a}→{b}（{n} 条）" for n, a, b in pairs[:5]), ""]

    # 跨数据集：每类方法的最好成绩
    sections += ["## 跨数据集对比（每族最好成绩）", "",
                 "| 方法族 | TNEWS Macro-F1 | ChnSentiCorp Macro-F1 |", "|---|---|---|"]
    for fam in ["rule", "linear", "finetune", "llm_zero_shot"]:
        cells = []
        for ds in DATASETS:
            rows = [r for r in data.get(ds, []) if r.get("family") == fam]
            cells.append(f"{max(r['report']['macro_f1'] for r in rows):.3f}" if rows else "—")
        sections.append(f"| {fam} | {cells[0]} | {cells[1]} |")

    if leak:
        sections += ["", "## 数据泄漏检查：TF-IDF 在「不去重」的 val 上会虚高多少", "",
                     "| 数据集 | 干净集 Macro-F1 | 含重复 val Macro-F1 | 虚高 |",
                     "|---|---|---|---|"]
        for ds, lk in leak.items():
            clean = [r for r in data.get(ds, []) if r.get("method") == "tfidf_svm"]
            if not clean:
                continue
            c = clean[0]["report"]["macro_f1"]
            l = lk["report"]["macro_f1"]
            sections.append(f"| {DATASETS[ds]['display']} | {c:.3f} | {l:.3f} | +{l-c:.3f} |")

    (RESULTS_DIR / "summary.md").write_text("\n".join(sections) + "\n", encoding="utf-8")

    bar_compare(data, RESULTS_DIR / "fig_compare.png")
    cost_scatter(data, RESULTS_DIR / "fig_cost.png")
    per_class_fig(data, RESULTS_DIR / "fig_per_class.png")
    for ds, rows in data.items():
        best_ft = max([r for r in rows if r.get("family") == "finetune"],
                      key=lambda r: r["report"].get("macro_f1", 0), default=None)
        if best_ft:
            confusion_fig(best_ft, RESULTS_DIR / f"fig_confusion_{ds}.png")
    print((RESULTS_DIR / "summary.md").read_text(encoding="utf-8"))
    print(f"→ {RESULTS_DIR / 'summary.md'} + 4 张图")


if __name__ == "__main__":
    main()
