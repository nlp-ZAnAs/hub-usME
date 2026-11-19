"""逐条核对样本 ID 和标签，重算指标，再输出结果表、配对区间和图。"""

import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import (METHODS, ROOT, config, fingerprint, load_data, metrics,
                    read_json, write_json)

SHORT_NAMES = {
    "majority": "Majority", "rule": "Keywords", "tfidf_nb": "TF-IDF + NB",
    "tfidf_lr": "TF-IDF + LR", "tfidf_svm": "TF-IDF + SVM",
    "rbt3": "RBT3 fine-tuning", "qwen_zero": "Qwen zero-shot",
    "qwen_lora": "Qwen LoRA", "qwen_full": "Qwen full fine-tuning",
}


def validate_result(result, splits):
    expected_hashes = {k: fingerprint(v) for k, v in splits.items()}
    if result["split_sha256"] != expected_hashes or result["config"] != config():
        raise ValueError(f"{result['method']}: 数据或配置不同，不能直接比较")
    expected = [(r["id"], r["label"]) for r in splits["test"]]
    actual = [(r["id"], r["true"]) for r in result["predictions"]]
    if expected != actual:
        raise ValueError(f"{result['method']}: 测试样本 ID、顺序或标签不一致")
    recomputed = metrics([r["true"] for r in result["predictions"]],
                         [r["pred"] for r in result["predictions"]])
    if recomputed != result["metrics"]:
        raise ValueError(f"{result['method']}: 已保存指标与逐条预测不一致")
    return recomputed


def vector_scores(truth, pred):
    accuracy = (truth == pred).mean(axis=-1)
    f1s = []
    for label in [0, 1]:
        tp = ((truth == label) & (pred == label)).sum(axis=-1)
        fp = ((truth != label) & (pred == label)).sum(axis=-1)
        fn = ((truth == label) & (pred != label)).sum(axis=-1)
        denominator = 2 * tp + fp + fn
        f1s.append(np.divide(2 * tp, denominator, out=np.zeros_like(tp, dtype=float),
                             where=denominator > 0))
    return accuracy, np.mean(f1s, axis=0)


def paired_bootstrap(truth, a, b, repetitions=2000, seed=42):
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(truth), size=(repetitions, len(truth)))
    scores_a = vector_scores(truth[indices], a[indices])
    scores_b = vector_scores(truth[indices], b[indices])
    observed_a, observed_b = vector_scores(truth, a), vector_scores(truth, b)
    return {metric: {"difference": float(observed_a[i] - observed_b[i]),
                     "ci95": np.quantile(scores_a[i] - scores_b[i], [0.025, 0.975]).tolist()}
            for i, metric in enumerate(["accuracy", "macro_f1"])}


def main():
    splits = load_data()
    results = {}
    for method in METHODS:
        result = read_json(ROOT / "results" / (method + ".json"))
        validate_result(result, splits)
        results[method] = result
    rows = []
    for method, result in results.items():
        m, c = result["metrics"], result["cost"]
        rows.append({
            "method": method, "name": METHODS[method], "accuracy": m["accuracy"],
            "macro_f1": m["macro_f1"], "negative_recall": m["per_class"]["负面"]["recall"],
            "positive_recall": m["per_class"]["正面"]["recall"], "invalid": m["invalid"],
            **c,
        })
    with (ROOT / "results" / "summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    write_json(ROOT / "results" / "summary.json", rows)

    lines = [
        "| 方法 | Accuracy | Macro-F1 | 负面 Recall | 训练/选模秒 | 测试总秒 | 峰值显存 GiB | 失败数 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(f"| {r['name']} | {r['accuracy']:.4f} | {r['macro_f1']:.4f} | "
                     f"{r['negative_recall']:.4f} | {r['train_seconds']:.2f} | "
                     f"{r['predict_seconds']:.2f} | {r['peak_gpu_allocated_gib']:.2f} | {r['invalid']} |")
    lines.extend([
        "", "口径：800 条相同测试输入；失败预测同时计入 Accuracy 和 Macro-F1。",
        "CPU：规则和 TF-IDF；GPU：RBT3 和 Qwen。显存为 PyTorch allocated 峰值，不是系统总占用。",
        "耗时为单次实测：训练列包含验证选模与最佳权重保存；TF-IDF 包含三组候选。",
        "推理不含权重加载，RBT3 使用预先编码的输入，Qwen 包含 tokenizer 和解析；"
        "批大小也不同，因此这些时间只代表当前实现，不能作为严格的模型吞吐基准。",
        "",
    ])
    truth = np.array([r["label"] for r in splits["test"]])
    pairs = {}
    for a, b in [("qwen_full", "qwen_lora"), ("rbt3", "tfidf_svm"), ("qwen_lora", "qwen_zero")]:
        pairs[a + "_minus_" + b] = paired_bootstrap(
            truth, np.array([r["pred"] for r in results[a]["predictions"]]),
            np.array([r["pred"] for r in results[b]["predictions"]]))
    write_json(ROOT / "results" / "paired_bootstrap.json",
               {"repetitions": 2000, "seed": 42, "pairs": pairs,
                "limitation": "只反映固定已训练模型下的测试样本抽样不确定性，不含训练 seed 或酒店聚类差异。"})
    lines.extend(["| 配对比较（前者减后者） | Macro-F1 差 | 95% bootstrap 区间 |",
                  "|---|---:|---:|"])
    for key, pair in pairs.items():
        delta = pair["macro_f1"]
        lo, hi = delta["ci95"]
        a, b = key.split("_minus_")
        lines.append(f"| {METHODS[a]} − {METHODS[b]} | {delta['difference']:+.4f} | [{lo:+.4f}, {hi:+.4f}] |")
    lines.extend(["", "配对 bootstrap 2,000 次，固定 seed=42；此区间不包含训练随机性。", ""])
    (ROOT / "results" / "summary.md").write_text("\n".join(lines), encoding="utf-8")

    # 科研比较图使用标准绘图库，导出可直接放入作业的独立图片。
    figure, ax = plt.subplots(figsize=(10, 6.2), layout="constrained")
    y = np.arange(len(rows))
    ax.barh(y - 0.17, [r["accuracy"] for r in rows], 0.32, label="Accuracy", color="#507aa8")
    ax.barh(y + 0.17, [r["macro_f1"] for r in rows], 0.32, label="Macro-F1", color="#e09547")
    for i, row in enumerate(rows):
        ax.text(row["macro_f1"] + .01, i + .17, f"{row['macro_f1']:.3f}", va="center", fontsize=9)
    ax.set(yticks=y, yticklabels=[SHORT_NAMES[r["method"]] for r in rows],
           xlim=(0, 1.06), xlabel="Score (0 to 1)", title="Hotel sentiment classification: nine methods")
    ax.invert_yaxis()
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.legend(loc="upper right", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.15)
    ax.set_axisbelow(True)
    figure.savefig(ROOT / "results" / "comparison.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(2, 2, figsize=(9, 7), layout="constrained")
    max_count = max(sum(r["label"] == label for r in splits["test"]) for label in [0, 1])
    for ax, method in zip(axes.flat, ["tfidf_svm", "rbt3", "qwen_lora", "qwen_full"], strict=True):
        matrix = np.array(results[method]["metrics"]["confusion_matrix"])
        ax.imshow(matrix, cmap="Blues", vmin=0, vmax=max_count)
        for i in range(2):
            for j in range(3):
                ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                        color="white" if matrix[i, j] > max_count * 0.55 else "#192b3b")
        ax.set(xticks=[0, 1, 2], xticklabels=["Negative", "Positive", "Invalid"],
               yticks=[0, 1], yticklabels=["Negative", "Positive"],
               xlabel="Predicted", ylabel="True", title=SHORT_NAMES[method])
    figure.savefig(ROOT / "results" / "confusion_matrices.png", dpi=180)
    plt.close(figure)
    # 可定位的全部错例，而不是只挑选有利例子。
    errors = []
    for index, row in enumerate(splits["test"]):
        predictions = {method: result["predictions"][index]["pred"] for method, result in results.items()}
        wrong = [method for method, prediction in predictions.items()
                 if prediction != row["label"] and method != "majority"]
        if wrong:
            errors.append({**row, "wrong_methods": wrong, "predictions": predictions})
    errors.sort(key=lambda r: (-len(r["wrong_methods"]), r["id"]))
    write_json(ROOT / "results" / "errors.json", errors)
    print("\n".join(lines))
    print("All nine result files passed exact sample/label/metric validation.")


if __name__ == "__main__":
    main()
