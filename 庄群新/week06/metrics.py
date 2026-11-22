"""统一评测指标：三路方法共用同一个函数，保证数字可比

主指标 Macro-F1（15 类等权）——Accuracy 会被头部类掩盖，尾部类全错也看不出来。
"""

import numpy as np

import config


def classification_report(y_true, y_pred, num_classes=config.NUM_LABELS):
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    valid = y_pred >= 0                     # 解析失败用 -1 标记，计入分母
    unparsed = int((~valid).sum())

    report = {"n": int(len(y_true)), "unparsed": unparsed,
              "unparsed_rate": round(unparsed / len(y_true), 4) if len(y_true) else 0.0}
    if valid.sum() == 0:
        return report

    yt, yp = y_true[valid], y_pred[valid]
    n_correct = int((yt == yp).sum())
    report["acc"] = round(float(accuracy_score(yt, yp)), 4)
    report["acc_incl_unparsed"] = round(n_correct / len(y_true), 4) if len(y_true) else 0.0
    report["macro_f1"] = round(float(f1_score(yt, yp, average="macro")), 4)
    report["weighted_f1"] = round(float(f1_score(yt, yp, average="weighted")), 4)

    per_class = {}
    for c in range(num_classes):
        mask = yt == c
        per_class[c] = None if not mask.sum() else round(float((yp[mask] == c).mean()), 4)
    report["per_class_recall"] = per_class
    report["confusion_matrix"] = confusion_matrix(yt, yp, labels=list(range(num_classes))).tolist()

    used = sorted({int(p) for p in yp if 0 <= p < num_classes})
    report["coverage"] = f"{len(used)}/{num_classes}"
    return report


def expected_calibration_error(y_true, probs, n_bins=10):
    """ECE：模型说 0.9 时，实际命中率是不是 0.9。>0.1 就别拿置信度当阈值。"""
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


def group_level_report(y_true_ids, y_pred_ids):
    """15 类 id → 4 组主题口径的准确率（业务视角：主题分错才是真错）"""
    from labels import to_group
    yt = [to_group(i) for i in y_true_ids]
    yp = [to_group(i) if 0 <= i < config.NUM_LABELS else "未知" for i in y_pred_ids]
    correct = sum(1 for a, b in zip(yt, yp) if a == b)
    return round(correct / max(1, len(yt)), 4)
