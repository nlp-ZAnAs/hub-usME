"""类别加权 A/B 对比：同一份数据、同一 seed、同 3 epoch，只改 loss 里的一个开关

对照组：
  A  加权（weight = balanced，默认）：outputs/bert_metrics.json
  B  不加权（--no_class_weight）：   outputs/bert_nw_metrics.json

为什么只看 per-class recall：Macro-F1 是 15 类平均，加权的收益藏在尾部类里，
总分经常打平（甚至 Accuracy 略降），但「整类不漏」在业务上比总分更重要。

产出：
  figures/fig7_class_weight_recall.png   每类 recall 并排 + Δrecall
  figures/fig8_class_weight_summary.png  总分 / 头部类 / 尾部类 的汇总对比
  outputs/class_weight_compare.md/.csv   可直接贴进报告的表
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config
import dataio
from labels import LABELS

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130

C_W, C_NW = "#2F5597", "#BFBFBF"     # 加权 / 不加权


def load_pair():
    a = config.OUT_DIR / "bert_metrics.json"        # 加权
    b = config.OUT_DIR / "bert_nw_metrics.json"     # 不加权
    if not a.exists() or not b.exists():
        raise SystemExit("缺少 bert_metrics.json 或 bert_nw_metrics.json：\n"
                         "  加权版：python train_bert.py --n 500 --epochs 3\n"
                         "  对照版：python train_bert.py --n 500 --epochs 3 --no_class_weight --tag _nw")
    return json.loads(a.read_text(encoding="utf-8")), json.loads(b.read_text(encoding="utf-8"))


def recalls(res):
    pc = res["metrics"]["per_class_recall"]
    return np.array([pc[str(i)] if pc.get(str(i)) is not None else 0.0 for i in range(len(LABELS))])


def head_tail_sets():
    """按训练集频次切头部/尾部：取最多的 5 类与最少的 5 类"""
    meta = dataio.dataset_meta()
    counts = {int(k): v for k, v in meta["class_counts"].items()}
    order = sorted(counts, key=lambda k: -counts[k])
    head, tail = order[:5], order[-5:]
    return head, tail, counts


def fig_recall(rw, rnw):
    d = rw - rnw
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 4.8),
                                   gridspec_kw={"width_ratios": [2.1, 1]})
    x = np.arange(len(LABELS)); w = 0.4
    ax1.bar(x - w / 2, rnw, w, label="不加权", color=C_NW, edgecolor="white")
    ax1.bar(x + w / 2, rw, w, label="类别加权", color=C_W, edgecolor="white")
    ax1.set_xticks(x); ax1.set_xticklabels(LABELS, rotation=0, fontsize=9)
    ax1.set_ylim(0, 1.0); ax1.set_ylabel("Recall")
    ax1.set_title("每类召回率：加权 vs 不加权（同一批 500 条）")
    ax1.legend(frameon=False, fontsize=9); ax1.grid(axis="y", ls=":", alpha=.45)
    ax1.spines[["top", "right"]].set_visible(False)

    order = np.argsort(d)
    names = [LABELS[i] for i in order]
    vals = d[order]
    colors = ["#C00000" if v < 0 else "#2E7D32" for v in vals]
    ax2.barh(range(len(vals)), vals, color=colors, edgecolor="white")
    ax2.set_yticks(range(len(vals))); ax2.set_yticklabels(names, fontsize=9)
    ax2.axvline(0, color="#333333", lw=1)
    ax2.set_title("Δ Recall（加权 − 不加权）")
    ax2.set_xlabel("召回率变化")
    pad = max(abs(vals).max() * 0.10, 0.02)
    ax2.set_xlim(vals.min() - pad * 3, vals.max() + pad * 3)
    for i, v in enumerate(vals):
        ax2.text(v + (pad if v >= 0 else -pad), i, f"{v:+.2f}", va="center",
                 ha="left" if v >= 0 else "right", fontsize=8)
    ax2.grid(axis="x", ls=":", alpha=.45)
    ax2.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(config.FIG_DIR / "fig7_class_weight_recall.png"); plt.close(fig)


def fig_summary(rw, rnw, head, tail):
    mw, mn = rw["metrics"], rnw["metrics"]
    acc_w, acc_n = mw["acc"], mn["acc"]
    f1_w, f1_n = mw["macro_f1"], mn["macro_f1"]
    grp_w, grp_n = mw.get("group_acc", 0), mn.get("group_acc", 0)
    ece_w, ece_n = mw.get("ece", 0), mn.get("ece", 0)
    rec_w, rec_n = recalls(rw), recalls(rnw)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6))
    labels = ["Accuracy", "Macro-F1", "主题组准确率", "ECE"]
    vw = [acc_w, f1_w, grp_w, ece_w]
    vn = [acc_n, f1_n, grp_n, ece_n]
    x = np.arange(len(labels))
    ax1.bar(x - .2, vn, .38, label="不加权", color=C_NW, edgecolor="white")
    ax1.bar(x + .2, vw, .38, label="类别加权", color=C_W, edgecolor="white")
    for i in range(len(labels)):
        ax1.text(x[i] - .2, vn[i] + .008, f"{vn[i]:.3f}", ha="center", fontsize=8)
        ax1.text(x[i] + .2, vw[i] + .008, f"{vw[i]:.3f}", ha="center", fontsize=8)
    ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_title("总分口径（注意 ECE 越小越好）")
    ax1.legend(frameon=False, fontsize=9); ax1.grid(axis="y", ls=":", alpha=.45)
    ax1.spines[["top", "right"]].set_visible(False)

    groups = [("头部 5 类", head), ("尾部 5 类", tail)]
    xs = np.arange(len(groups))
    gw = [rec_w[g].mean() for _, g in groups]
    gn = [rec_n[g].mean() for _, g in groups]
    ax2.bar(xs - .18, gn, .34, label="不加权", color=C_NW, edgecolor="white")
    ax2.bar(xs + .18, gw, .34, label="类别加权", color=C_W, edgecolor="white")
    for i in range(len(groups)):
        ax2.text(xs[i] - .18, gn[i] + .012, f"{gn[i]:.2f}", ha="center", fontsize=9)
        ax2.text(xs[i] + .18, gw[i] + .012, f"{gw[i]:.2f}", ha="center", fontsize=9)
    ax2.set_xticks(xs); ax2.set_xticklabels([g for g, _ in groups], fontsize=10)
    ax2.set_ylim(0, 1.05); ax2.set_ylabel("平均 Recall")
    ax2.set_title("头部 vs 尾部：加权的收益与代价")
    ax2.legend(frameon=False, fontsize=9); ax2.grid(axis="y", ls=":", alpha=.45)
    ax2.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(config.FIG_DIR / "fig8_class_weight_summary.png"); plt.close(fig)


def export_table(rw, rnw, head, tail):
    rec_w, rec_n = recalls(rw), recalls(rnw)
    rows = []
    for i, name in enumerate(LABELS):
        rows.append({"类别": name, "不加权": round(float(rec_n[i]), 3),
                     "类别加权": round(float(rec_w[i]), 3),
                     "Δ": round(float(rec_w[i] - rec_n[i]), 3)})
    md = ["# 类别加权 A/B（同数据 / 同 seed / 同 3 epoch，只改 loss 权重）", "",
          "## 总分", "",
          "| 口径 | 不加权 | 类别加权 | Δ |", "|---|---|---|---|"]
    for k, lab in [("acc", "Accuracy"), ("macro_f1", "Macro-F1"),
                   ("group_acc", "主题组准确率"), ("ece", "ECE（越小越好）")]:
        a = rnw["metrics"].get(k); b = rw["metrics"].get(k)
        if a is None or b is None:
            continue
        md.append(f"| {lab} | {a} | {b} | {round(b - a, 4):+} |")
    md += ["", "## 每类召回", "", "| 类别 | 不加权 | 类别加权 | Δ |", "|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['类别']} | {r['不加权']} | {r['类别加权']} | {r['Δ']:+} |")

    hw, hn = rec_w[head].mean(), rec_n[head].mean()
    tw, tn = rec_w[tail].mean(), rec_n[tail].mean()
    md += ["", "## 头部 vs 尾部", "",
           f"- 头部 5 类（{', '.join(LABELS[i] for i in head)}）：{hn:.3f} → {hw:.3f}（{hw-hn:+.3f}）",
           f"- 尾部 5 类（{', '.join(LABELS[i] for i in tail)}）：{tn:.3f} → {tw:.3f}（{tw-tn:+.3f}）"]

    (config.OUT_DIR / "class_weight_compare.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    import csv
    with open(config.OUT_DIR / "class_weight_compare.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["类别", "不加权", "类别加权", "Δ"])
        w.writeheader(); w.writerows(rows)
    return rows, (hw, hn, tw, tn)


def main():
    config.ensure_dirs()
    rw, rnw = load_pair()
    head, tail, counts = head_tail_sets()

    fig_recall(recalls(rw), recalls(rnw))
    fig_summary(rw, rnw, head, tail)
    rows, (hw, hn, tw, tn) = export_table(rw, rnw, head, tail)

    mw, mn = rw["metrics"], rnw["metrics"]
    d = recalls(rw) - recalls(rnw)
    top_up = sorted(range(len(LABELS)), key=lambda i: -d[i])[:3]
    top_dn = sorted(range(len(LABELS)), key=lambda i: d[i])[:3]

    print(f"加权 vs 不加权（{rw['metrics']['n']} 条同一批样本）")
    print(f"  Accuracy   {mn['acc']:.4f} → {mw['acc']:.4f} ({mw['acc']-mn['acc']:+.4f})")
    print(f"  Macro-F1   {mn['macro_f1']:.4f} → {mw['macro_f1']:.4f} ({mw['macro_f1']-mn['macro_f1']:+.4f})")
    print(f"  主题组acc  {mn.get('group_acc',0):.4f} → {mw.get('group_acc',0):.4f} "
          f"({mw.get('group_acc',0)-mn.get('group_acc',0):+.4f})")
    print(f"  ECE        {mn.get('ece',0):.4f} → {mw.get('ece',0):.4f} "
          f"({mw.get('ece',0)-mn.get('ece',0):+.4f})")
    print(f"  头部 5 类 recall  {hn:.3f} → {hw:.3f} ({hw-hn:+.3f})")
    print(f"  尾部 5 类 recall  {tn:.3f} → {tw:.3f} ({tw-tn:+.3f})")
    print("  涨最多：" + "、".join(f"{LABELS[i]} {d[i]:+.2f}" for i in top_up))
    print("  跌最多：" + "、".join(f"{LABELS[i]} {d[i]:+.2f}" for i in top_dn))
    print(f"→ {config.FIG_DIR / 'fig7_class_weight_recall.png'}")
    print(f"→ {config.FIG_DIR / 'fig8_class_weight_summary.png'}")
    print(f"→ {config.OUT_DIR / 'class_weight_compare.md'}")


if __name__ == "__main__":
    main()
