# -*- coding: utf-8 -*-
"""
统一评测指标（所有方法、两个数据集共用）

为什么只用 Accuracy 不够：
  15 类 TNEWS 里最大类占 11%，最小类占 0.5%（23× 不均衡）。
  只报 Acc 会被头部类骗，所以主指标是 Macro-F1（每类等权），
  再配 per-class recall 和混淆矩阵定位「哪一类被吃掉了」。
  生成式方法（zero-shot）解析失败的样本按答错计入分母，不允许丢样本虚高。
"""

import numpy as np


def classification_report(y_true, y_pred, num_classes, probs=None):
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    valid = y_pred >= 0                       # 解析失败用 -1 标记
    unparsed = int((~valid).sum())
    report = {
        "n": int(len(y_true)),
        "num_classes": int(num_classes),
        "unparsed": unparsed,
        "unparsed_rate": round(unparsed / len(y_true), 4) if len(y_true) else 0.0,
        "acc_incl_unparsed": round(float((y_true[valid] == y_pred[valid]).sum()) / len(y_true), 4)
        if len(y_true) else 0.0,
    }
    if valid.sum() == 0:
        return report

    yt, yp = y_true[valid], y_pred[valid]
    report["acc"] = round(float(accuracy_score(yt, yp)), 4)
    report["macro_f1"] = round(float(f1_score(yt, yp, average="macro")), 4)
    report["weighted_f1"] = round(float(f1_score(yt, yp, average="weighted")), 4)
    report["macro_precision"] = round(
        float(precision_score(yt, yp, average="macro", zero_division=0)), 4)

    labels = list(range(num_classes))
    per_class_recall = {}
    for c in labels:
        mask = yt == c
        per_class_recall[c] = (round(float((yp[mask] == c).mean()), 4)
                               if mask.sum() else None)
    report["per_class_recall"] = per_class_recall
    report["confusion_matrix"] = confusion_matrix(yt, yp, labels=labels).tolist()

    if probs is not None:
        p = np.asarray(probs)[valid]
        report["ece"] = expected_calibration_error(yt, p)
    return report


def expected_calibration_error(y_true, probs, n_bins=10):
    """ECE：置信度分桶后 |平均置信度 - 实际准确率| 的加权和。"""
    y_true = np.asarray(y_true)
    probs = np.asarray(probs)
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y_true).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (conf > bins[i]) & (conf <= bins[i + 1])
        if mask.sum() == 0:
            continue
        ece += abs(correct[mask].mean() - conf[mask].mean()) * mask.sum() / len(conf)
    return round(float(ece), 4)


def pred_distribution(preds, num_classes):
    """预测分布：zero-shot 塌缩时一眼就能看出来。"""
    counts = {}
    for p in preds:
        counts[int(p)] = counts.get(int(p), 0) + 1
    used = sum(1 for k in counts if k >= 0)
    return {"used_classes": used, "counts": counts}


def print_table(rows, headers):
    widths = [max(len(str(r[i])) for r in [headers] + rows) for i in range(len(headers))]
    line = "  ".join(str(h).ljust(widths[i]) for i, h in enumerate(headers))
    print(line)
    print("-" * len(line))
    for r in rows:
        print("  ".join(str(v).ljust(widths[i]) for i, v in enumerate(r)))
