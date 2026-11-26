# -*- coding: utf-8 -*-
"""
week07 作业 5：用 outputs/logs/<name>_steps.jsonl 画 loss 曲线

用法：
    python plot_loss.py --logs <logs目录> --names bert_linear_full bert_crf_full --out loss_curve.png

说明：
- steps.jsonl 每 50 步一行（--log_every 可调），字段 epoch/step/loss/lr/elapsed_s
- 横向按“全局步数”对齐：epoch * steps_per_epoch + step
- 中文标签用 Microsoft YaHei（Windows 自带）；没有中文字形时自动退回英文
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

CANDIDATE_FONTS = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC"]


def setup_font():
    names = {f.name for f in font_manager.fontManager.ttflist}
    for name in CANDIDATE_FONTS:
        if name in names:
            plt.rcParams["font.sans-serif"] = [name]
            plt.rcParams["axes.unicode_minus"] = False
            return name
    return None


def load_steps(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def global_curve(rows):
    """把 (epoch, step) 映射成全局步数"""
    epochs = {}
    for r in rows:
        epochs.setdefault(r["epoch"], set()).add(r["step"])
    steps_per_epoch = max(max(s) for s in epochs.values()) + 50  # 最后一行不一定整除
    xs, ys = [], []
    for r in rows:
        if r["loss"] is None:
            continue
        xs.append((r["epoch"] - 1) * steps_per_epoch + r["step"])
        ys.append(r["loss"])
    bounds = [k * steps_per_epoch for k in range(1, len(epochs))]
    return xs, ys, bounds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", required=True)
    ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="")
    args = ap.parse_args()

    zh = setup_font()
    use_zh = zh is not None
    colors = ["#2563eb", "#dc2626", "#16a34a", "#9333ea"]

    curves = []
    for name in args.names:
        path = os.path.join(args.logs, f"{name}_steps.jsonl")
        if not os.path.exists(path):
            print(f"[skip] {path} 不存在")
            continue
        rows = load_steps(path)
        xs, ys, bounds = global_curve(rows)
        curves.append((name, xs, ys, bounds))
        print(f"{name}: {len(ys)} 个点, loss {ys[0]:.3f} → {min(ys):.3f}, 末点 {ys[-1]:.3f}")
    if not curves:
        raise SystemExit("没有可画的曲线")

    # 量级差 >10 倍就拆面板（CRF 的 loss = logZ − score，量级和 Linear 不可同轴）
    peaks = [max(c[2]) for c in curves]
    separate = len(curves) > 1 and (max(peaks) / max(min(peaks), 1e-9)) > 10
    nrow = len(curves) if separate else 1
    fig, axes = plt.subplots(nrow, 1, figsize=(9, 3.6 * nrow), dpi=150, sharex=True,
                             squeeze=False)
    for i, (name, xs, ys, bounds) in enumerate(curves):
        ax = axes[i][0]
        ax.plot(xs, ys, color=colors[i % len(colors)], linewidth=1.6, label=name)
        for b in bounds:
            ax.axvline(b, color="#9ca3af", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.set_ylabel(("训练 loss" if use_zh else "train loss") + (f"\n({name})" if separate else ""))
        ax.grid(alpha=0.3)
        if separate:
            ax.set_title(name, fontsize=10)
    if not separate:
        axes[0][0].legend()
        axes[0][0].set_title(args.title or ("训练 loss 曲线（虚线=epoch 边界）" if use_zh
                                            else "Train loss (dashed = epoch boundary)"))
    axes[-1][0].set_xlabel("步数 (global step)" if use_zh else "global step")
    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight")
    print(f"已保存 {args.out}（字体：{zh or 'default'}）")


if __name__ == "__main__":
    main()
