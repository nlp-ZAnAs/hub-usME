"""可视化对比：准确率 / 耗时 / 资源占用 / 效率

读 outputs/ 下三份 *_metrics.json（规则、TF-IDF、BERT），画 6 张图 + 1 张总览大图，
并导出 summary.csv / summary.md，方便直接贴进报告。

四类指标分开画，因为它们回答的是不同问题：
  fig1 准确率 —— 效果值不值
  fig2 耗时   —— 训练+推理要等多久（效率的时间面）
  fig3 资源   —— 模型多大、内存/显存吃多少（部署可行不可行）
  fig4 权衡   —— 效果 vs 成本的散点图（选型的核心一张图）
  fig5 长尾   —— 每类 recall 热力图（总分看不出长尾塌陷）
  fig6 混淆   —— 最好那一路的错误长什么样

用法：
  python plot_compare.py
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
from labels import LABELS

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130

C_RULE, C_TFIDF, C_BERT = "#8C8C8C", "#3C78D8", "#E8710A"
ORDER = ["rule", "tfidf", "bert"]
SHORT = {"rule": "规则/关键词\n(L0)", "tfidf": "TF-IDF+SVM\n(L1)", "bert": "BERT 微调\n(L2)"}
COLOR = {"rule": C_RULE, "tfidf": C_TFIDF, "bert": C_BERT}
NAME_CN = {"rule": "规则", "tfidf": "TF-IDF", "bert": "BERT"}


def short_name(k):
    return NAME_CN[k]


def load_results():
    out = {}
    for key in ORDER:
        p = config.OUT_DIR / f"{key}_metrics.json"
        if p.exists():
            out[key] = json.loads(p.read_text(encoding="utf-8"))
    return out


def fig_accuracy(res):
    keys = [k for k in ORDER if k in res]
    acc = [res[k]["metrics"]["acc"] for k in keys]
    f1 = [res[k]["metrics"]["macro_f1"] for k in keys]
    grp = [res[k]["metrics"].get("group_acc", 0) for k in keys]
    x = np.arange(len(keys)); w = 0.26
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    for i, (vals, lab) in enumerate([(acc, "Accuracy"), (f1, "Macro-F1"), (grp, "主题组准确率")]):
        b = ax.bar(x + (i - 1) * w, vals, w, label=lab, color=["#BFBFBF", "#2F5597", "#9DC3E6"][i],
                   edgecolor="white")
        for rect, v in zip(b, vals):
            ax.text(rect.get_x() + rect.get_width() / 2, v + 0.012, f"{v:.3f}",
                    ha="center", fontsize=8.5)
    ax.set_xticks(x); ax.set_xticklabels([SHORT[k] for k in keys], fontsize=9)
    ax.set_ylim(0, max(acc + f1 + grp) * 1.22 + 0.02)
    ax.set_ylabel("分数"); ax.set_title("准确率对比（同一批干净验证集）")
    ax.legend(frameon=False, fontsize=9); ax.grid(axis="y", ls=":", alpha=.5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(config.FIG_DIR / "fig1_accuracy.png"); plt.close(fig)


def fig_time(res):
    keys = [k for k in ORDER if k in res]
    tr = [res[k]["cost"]["train_seconds"] for k in keys]
    inf = [res[k]["cost"]["infer_ms_b32"] for k in keys]
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    x = np.arange(len(keys))
    b1 = ax.bar(x - 0.2, [max(t, 0.01) for t in tr], 0.38, label="训练耗时（秒，对数轴）",
                color="#E8710A", alpha=.85, edgecolor="white")
    ax.set_yscale("log"); ax.set_ylabel("训练耗时 (s, log)")
    for rect, v in zip(b1, tr):
        ax.text(rect.get_x() + rect.get_width() / 2, max(v, 0.01) * 1.12,
                "0（不训练）" if v == 0 else (f"{v:.0f}s" if v < 600 else f"{v/60:.1f}min"),
                ha="center", fontsize=8.5)
    ax2 = ax.twinx()
    b2 = ax2.bar(x + 0.2, inf, 0.38, label="推理延迟（ms/条, batch32）", color="#3C78D8",
                 alpha=.85, edgecolor="white")
    for rect, v in zip(b2, inf):
        ax2.text(rect.get_x() + rect.get_width() / 2, v * 1.06, f"{v:.2f}", ha="center", fontsize=8.5)
    ax2.set_ylabel("推理延迟 (ms/条)")
    ax.set_xticks(x); ax.set_xticklabels([SHORT[k] for k in keys], fontsize=9)
    ax.set_title("耗时：训练成本 vs 推理延迟")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=9, loc="upper left")
    ax.grid(axis="y", ls=":", alpha=.4)
    ax.spines[["top"]].set_visible(False); ax2.spines[["top"]].set_visible(False)
    fig.tight_layout(); fig.savefig(config.FIG_DIR / "fig2_time.png"); plt.close(fig)


def fig_resource(res):
    keys = [k for k in ORDER if k in res]
    size = [res[k]["cost"]["model_size_mb"] for k in keys]
    rss = [res[k]["cost"]["peak_rss_mb"] for k in keys]
    gpu = [res[k]["cost"]["peak_gpu_mb"] for k in keys]
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    x = np.arange(len(keys)); w = 0.26
    for i, (vals, lab, c) in enumerate([(size, "模型文件 (MB)", "#7F7F7F"),
                                        (rss, "进程峰值内存 (MB)", "#2F5597"),
                                        (gpu, "GPU 峰值显存 (MB)", "#E8710A")]):
        b = ax.bar(x + (i - 1) * w, vals, w, label=lab, color=c, edgecolor="white")
        for rect, v in zip(b, vals):
            ax.text(rect.get_x() + rect.get_width() / 2, v + max(size + rss + gpu) * 0.02,
                    "0" if v == 0 else f"{v:.0f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([SHORT[k] for k in keys], fontsize=9)
    ax.set_ylabel("占用（MB）"); ax.set_title("资源占用：磁盘 / 内存 / 显存")
    ax.legend(frameon=False, fontsize=9); ax.grid(axis="y", ls=":", alpha=.5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(config.FIG_DIR / "fig3_resource.png"); plt.close(fig)


def fig_tradeoff(res):
    keys = [k for k in ORDER if k in res]
    max_size = max(res[k]["cost"]["model_size_mb"] for k in keys) or 1
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    for k in keys:
        c = res[k]["cost"]
        f1 = res[k]["metrics"]["macro_f1"]
        t = max(c["train_seconds"], 0.3)         # 规则法 0 秒画在 0.3 处
        s = 40 + 420 * (c["model_size_mb"] / max_size) ** 0.5
        ax.scatter(t, f1, s=s, color=COLOR[k], alpha=.75, edgecolor="white", linewidth=1.4,
                   zorder=3)
        ax.annotate(f"{res[k]['method'].split('（')[0]}\nF1={f1:.3f} · {c['model_size_mb']:.0f}MB",
                    (t, f1), textcoords="offset points", xytext=(0, 20),
                    ha="center", fontsize=8.5)
    ax.set_xscale("log")
    ax.set_xlim(0.2, 3000)
    ax.set_ylim(min(res[k]["metrics"]["macro_f1"] for k in keys) - 0.08,
                max(res[k]["metrics"]["macro_f1"] for k in keys) + 0.10)
    ax.set_xlabel("训练耗时（秒，log；规则法不训练，画在 0.3 处）")
    ax.set_ylabel("Macro-F1")
    ax.set_title("效果 vs 成本：气泡大小 = 模型文件大小")
    ax.grid(ls=":", alpha=.5); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(config.FIG_DIR / "fig4_tradeoff.png"); plt.close(fig)


def fig_per_class(res):
    keys = [k for k in ORDER if k in res]
    mat = []
    for k in keys:
        pc = res[k]["metrics"]["per_class_recall"]
        mat.append([pc[str(i)] if pc.get(str(i)) is not None else 0.0 for i in range(len(LABELS))])
    mat = np.array(mat, dtype=float)
    fig, ax = plt.subplots(figsize=(10.5, 3.4))
    im = ax.imshow(mat, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(LABELS))); ax.set_xticklabels(LABELS, fontsize=9)
    ax.set_yticks(range(len(keys))); ax.set_yticklabels([SHORT[k].replace("\n", " ") for k in keys], fontsize=9)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=7.5,
                    color="white" if mat[i, j] > 0.6 else "#333333")
    ax.set_title("每类召回率（长尾类塌陷在这里最明显）")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout(); fig.savefig(config.FIG_DIR / "fig5_per_class.png"); plt.close(fig)


def fig_confusion(res):
    keys = [k for k in ORDER if k in res]
    best = max(keys, key=lambda k: res[k]["metrics"]["macro_f1"])
    cm = np.array(res[best]["metrics"]["confusion_matrix"], dtype=float)
    row = cm.sum(axis=1, keepdims=True); norm = np.divide(cm, row, out=np.zeros_like(cm), where=row > 0)
    fig, ax = plt.subplots(figsize=(8.6, 7.2))
    im = ax.imshow(norm, cmap="OrRd", vmin=0, vmax=1)
    ax.set_xticks(range(len(LABELS))); ax.set_xticklabels(LABELS, rotation=45, ha="right", fontsize=8.5)
    ax.set_yticks(range(len(LABELS))); ax.set_yticklabels(LABELS, fontsize=8.5)
    ax.set_xlabel("预测"); ax.set_ylabel("真实")
    ax.set_title(f"混淆矩阵（{res[best]['method']}，行归一化 = 召回比例）")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    fig.tight_layout(); fig.savefig(config.FIG_DIR / "fig6_confusion_best.png"); plt.close(fig)


def fig_dashboard(res):
    keys = [k for k in ORDER if k in res]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.4))
    acc = [res[k]["metrics"]["acc"] for k in keys]
    f1 = [res[k]["metrics"]["macro_f1"] for k in keys]
    tr = [res[k]["cost"]["train_seconds"] for k in keys]
    inf = [res[k]["cost"]["infer_ms_b32"] for k in keys]
    size = [res[k]["cost"]["model_size_mb"] for k in keys]
    rss = [res[k]["cost"]["peak_rss_mb"] for k in keys]
    gpu = [res[k]["cost"]["peak_gpu_mb"] for k in keys]
    x = np.arange(len(keys)); sh = [SHORT[k].replace("\n", " ") for k in keys]

    ax = axes[0][0]
    ax.bar(x - .18, acc, .34, label="Accuracy", color="#BFBFBF", edgecolor="white")
    ax.bar(x + .18, f1, .34, label="Macro-F1", color="#2F5597", edgecolor="white")
    for i in range(len(keys)):
        ax.text(x[i] - .18, acc[i] + .01, f"{acc[i]:.3f}", ha="center", fontsize=8)
        ax.text(x[i] + .18, f1[i] + .01, f"{f1[i]:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(sh, fontsize=8); ax.set_title("准确率"); ax.legend(frameon=False, fontsize=8)

    ax = axes[0][1]
    ax.bar(x, [max(t, .01) for t in tr], .5, color="#E8710A", edgecolor="white")
    ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels(sh, fontsize=8)
    ax.set_title("训练耗时 (s, log)")
    for i, v in enumerate(tr):
        ax.text(x[i], max(v, .01) * 1.15, "0" if v == 0 else f"{v:.0f}", ha="center", fontsize=8)

    ax = axes[0][2]
    ax.bar(x, inf, .5, color="#3C78D8", edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(sh, fontsize=8); ax.set_title("推理延迟 (ms/条, batch32)")
    for i, v in enumerate(inf):
        ax.text(x[i], v * 1.08, f"{v:.2f}", ha="center", fontsize=8)

    ax = axes[1][0]
    ax.bar(x - .22, size, .3, label="模型 MB", color="#7F7F7F", edgecolor="white")
    ax.bar(x, rss, .3, label="峰值内存 MB", color="#2F5597", edgecolor="white")
    ax.bar(x + .22, gpu, .3, label="GPU 显存 MB", color="#E8710A", edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(sh, fontsize=8); ax.set_title("资源占用"); ax.legend(frameon=False, fontsize=7.5)

    ax = axes[1][1]
    max_size = max(res[k]["cost"]["model_size_mb"] for k in keys) or 1
    for k in keys:
        c = res[k]["cost"]; f1 = res[k]["metrics"]["macro_f1"]
        t = max(c["train_seconds"], .3)
        ax.scatter(t, f1, s=40 + 200 * (c["model_size_mb"] / max_size) ** 0.5,
                   color=COLOR[k], alpha=.7, edgecolor="white")
        ax.annotate(short_name(k), (t, f1), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=7)
    ax.set_xscale("log"); ax.set_xlim(.2, 3000)
    ax.set_title("效果 vs 训练成本"); ax.set_xlabel("训练秒 (log)"); ax.set_ylabel("Macro-F1")

    ax = axes[1][2]
    best = max(keys, key=lambda k: res[k]["metrics"]["macro_f1"])
    pc = res[best]["metrics"]["per_class_recall"]
    vals = [pc[str(i)] if pc.get(str(i)) is not None else 0 for i in range(len(LABELS))]
    ax.barh(range(len(LABELS)), vals, color=[COLOR[best]] * len(LABELS), edgecolor="white")
    ax.set_yticks(range(len(LABELS))); ax.set_yticklabels(LABELS, fontsize=7.5)
    ax.invert_yaxis(); ax.set_title(f"每类召回（{res[best]['method']}）")
    for i, v in enumerate(vals):
        ax.text(v + .01, i, f"{v:.2f}", va="center", fontsize=7)

    for row in axes:
        for a in row:
            a.grid(axis="x", ls=":", alpha=.4)
            a.spines[["top", "right"]].set_visible(False)
    fig.suptitle("第 6 周作业 · 三种分类方式对比（同一批干净验证集）", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(config.FIG_DIR / "fig0_dashboard.png"); plt.close(fig)


def export_tables(res):
    keys = [k for k in ORDER if k in res]
    rows = []
    for k in keys:
        r, c = res[k]["metrics"], res[k]["cost"]
        rows.append({
            "方法": res[k]["method"], "层级": res[k]["level"],
            "Accuracy": r["acc"], "Macro-F1": r["macro_f1"], "主题组准确率": r.get("group_acc"),
            "覆盖类别": r["coverage"], "ECE": r.get("ece", "-"),
            "训练耗时(s)": c["train_seconds"], "单条延迟(ms)": c["infer_ms_b1"],
            "批量延迟(ms/条)": c["infer_ms_b32"], "吞吐(条/s)": c["throughput"],
            "模型大小(MB)": c["model_size_mb"], "峰值内存(MB)": c["peak_rss_mb"],
            "GPU峰值(MB)": c["peak_gpu_mb"], "参数量": c["n_params"],
        })
    csv_path = config.OUT_DIR / "summary.csv"
    import csv
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    md = ["| 方法 | 层级 | Accuracy | Macro-F1 | 主题组准确率 | 覆盖 | ECE | 训练(s) | 单条(ms) | 批量(ms/条) | 模型(MB) | 峰值内存(MB) | GPU(MB) |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['方法']} | {r['层级']} | {r['Accuracy']} | {r['Macro-F1']} | {r['主题组准确率']} | "
                  f"{r['覆盖类别']} | {r['ECE']} | {r['训练耗时(s)']} | {r['单条延迟(ms)']} | "
                  f"{r['批量延迟(ms/条)']} | {r['模型大小(MB)']} | {r['峰值内存(MB)']} | {r['GPU峰值(MB)']} |")
    (config.OUT_DIR / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return rows


def main():
    config.ensure_dirs()
    res = load_results()
    if not res:
        raise SystemExit("outputs/ 下没有 *_metrics.json，先跑 classify_rule.py / train_tfidf.py / train_bert.py")
    fig_accuracy(res); fig_time(res); fig_resource(res)
    fig_tradeoff(res); fig_per_class(res); fig_confusion(res); fig_dashboard(res)
    rows = export_tables(res)
    print("已生成：")
    for p in sorted(config.FIG_DIR.glob("*.png")):
        print(f"  {p.name}  ({p.stat().st_size/1024:.0f} KB)")
    print(f"  summary.csv / summary.md → {config.OUT_DIR}")
    print("\n" + (config.OUT_DIR / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
