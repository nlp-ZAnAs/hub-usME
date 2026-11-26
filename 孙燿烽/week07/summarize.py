"""逐条核对样本 ID、实体与配置，重算指标，输出结果表、配对 bootstrap、错例与图。"""

import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import (API_METHODS, METHODS, ROOT, TYPES, api_subset_ids, config, error_categories,
                    evaluate, fingerprint, load_data, read_json, sentence_counts, write_json)

SHORT = {
    "rule": "Dictionary", "bilstm_crf": "BiLSTM-CRF",
    "roberta_linear": "RoBERTa+Linear", "roberta_linear_constrained": "RoBERTa+Linear+constr.",
    "roberta_crf": "RoBERTa+CRF", "roberta_crf_constrained": "RoBERTa+CRF+constr.",
    "qwen_zero": "Qwen1.5B zero", "qwen_few": "Qwen1.5B 5-shot", "qwen_lora": "Qwen1.5B LoRA",
    "deepseek_zero": "DeepSeek zero", "deepseek_few": "DeepSeek 5-shot",
}
SECTIONS = {
    "rule": ["rule"], "bilstm_crf": ["bilstm"],
    "roberta_linear": ["encoder_model", "encoder"], "roberta_crf": ["encoder_model", "encoder"],
    "qwen_zero": ["llm_model", "llm"], "qwen_few": ["llm_model", "llm"],
    "qwen_lora": ["llm_model", "llm"], "deepseek_zero": ["api", "llm"],
    "deepseek_few": ["api", "llm"],
}
FULL_PAIRS = [
    ("bilstm_crf", "rule"), ("roberta_linear", "bilstm_crf"),
    ("roberta_crf", "roberta_linear"),
    ("roberta_linear_constrained", "roberta_linear"),
    ("roberta_crf_constrained", "roberta_crf"),
    ("qwen_few", "qwen_zero"), ("qwen_lora", "qwen_zero"),
    ("qwen_lora", "roberta_crf_constrained"),
]
SUBSET_PAIRS = [("deepseek_few", "deepseek_zero"), ("deepseek_zero", "qwen_zero"),
                ("deepseek_few", "qwen_few"), ("qwen_lora", "deepseek_few"),
                ("roberta_crf_constrained", "deepseek_few")]


def base_method(method):
    return method.removesuffix("_constrained")


def validate(result, splits, cfg):
    method = result["method"]
    if result["split_sha256"] != {k: fingerprint(v) for k, v in splits.items()}:
        raise ValueError(f"{method}: 数据与当前切分不一致，需要重跑")
    for key in ["seed", "val_size", "api_subset_size"] + SECTIONS[base_method(method)]:
        if result["config"][key] != cfg[key]:
            raise ValueError(f"{method}: 配置项 {key} 已变化，需要重跑")
    expected = api_subset_ids() if method in API_METHODS else [r["id"] for r in splits["test"]]
    if result["eval_ids"] != expected:
        raise ValueError(f"{method}: 评测样本 ID/顺序不一致")
    index = {r["id"]: r for r in splits["test"]}
    gold = [index[i] for i in expected]
    recomputed = evaluate(gold, result["predictions"])
    stored = {k: v for k, v in result["metrics"].items() if k != "illegal_transitions"}
    if recomputed != stored:
        raise ValueError(f"{method}: 已保存指标与逐条预测不一致")
    return gold


def count_matrix(gold, preds):
    """[N, 3]：每句 strict 的 TP/FP/FN，供 bootstrap 重采样。"""
    return np.array([sentence_counts(g["entities"], p["entities"])["strict"]
                     for g, p in zip(gold, preds, strict=True)])


def micro_f1(counts):
    tp, fp, fn = counts[..., 0], counts[..., 1], counts[..., 2]
    denominator = 2 * tp + fp + fn
    return np.divide(2 * tp, denominator, out=np.zeros(tp.shape), where=denominator > 0)


def paired_bootstrap(a, b, repetitions=2000, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(repetitions, len(a)))
    diff = micro_f1(a[idx].sum(1)) - micro_f1(b[idx].sum(1))
    return {"difference": float(micro_f1(a.sum(0)) - micro_f1(b.sum(0))),
            "ci95": np.quantile(diff, [0.025, 0.975]).tolist()}


def subset_rows(result, ids):
    keep = set(ids)
    return [p for p in result["predictions"] if p["id"] in keep]


def row_for(method, result, metric, n_errors, preds):
    """所有逐句统计（非法转移、空预测、JSON 合法率、原文找不到）都按传入的 preds 重算，
    子集行因此只统计子集句子，不会混入全测试集的数字。"""
    cost = result["cost"]
    parsed = [p["parse"] for p in preds if "parse" in p]
    valid = {"ok", "repaired"}
    return {
        "method": method, "name": METHODS[method], "n": metric["n"],
        "strict_precision": metric["strict"]["precision"],
        "strict_recall": metric["strict"]["recall"], "strict_f1": metric["strict"]["f1"],
        "macro_f1": metric["macro_f1"], "boundary_f1": metric["boundary"]["f1"],
        "text_f1": metric["text"]["f1"],
        "illegal_transitions": sum(p.get("illegal_transitions", 0) for p in preds),
        "empty_predictions": sum(not p["entities"] for p in preds),
        "json_valid_rate": (sum(s["status"] in valid for s in parsed) / len(parsed)
                            if parsed else None),
        "not_in_text": sum(s["not_in_text"] for s in parsed) if parsed else None,
        "train_seconds": cost["train_seconds"], "ms_per_item": cost["milliseconds_per_item"],
        "peak_gpu_gib": cost["peak_gpu_allocated_gib"],
        "trainable_params": cost["trainable_params"], "total_params": cost["total_params"],
        **{k: len(v) for k, v in n_errors.items()},
    }


def fmt(value, digits=4):
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return f"{value:,}" if isinstance(value, int) else str(value)


def error_summary(gold, preds):
    totals = {"type_error": [], "boundary_error": [], "missed": [], "spurious": []}
    for g, p in zip(gold, preds, strict=True):
        for k, v in error_categories(g["entities"], p["entities"]).items():
            totals[k].extend({"id": g["id"], **({"detail": x})} for x in v)
    return totals


def plot(rows_full, rows_subset, per_type, errors):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [1.15, 1]})
    for ax, rows, title in [(axes[0], rows_full, "Full test (official dev, n=1343)"),
                            (axes[1], rows_subset, "Fixed API subset (n=300)")]:
        names = [SHORT[r["method"]] for r in rows]
        values = [r["strict_f1"] for r in rows]
        colors = ["#8c8c8c" if r["method"] in {"rule", "bilstm_crf"} else
                  "#4c78a8" if r["method"].startswith("roberta") else
                  "#f58518" if r["method"].startswith("qwen") else "#54a24b" for r in rows]
        y = np.arange(len(rows))[::-1]
        ax.barh(y, values, color=colors)
        ax.set_yticks(y, names)
        ax.set_xlim(0, 1)
        ax.set_xlabel("strict micro-F1 (type + exact span)")
        ax.set_title(title)
        for yi, v in zip(y, values):
            ax.text(v + 0.01, yi, f"{v:.3f}", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(ROOT / "results" / "comparison.png", dpi=150)
    plt.close(fig)

    methods = list(per_type)
    matrix = np.array([[per_type[m][t]["f1"] for t in TYPES] for m in methods])
    fig, ax = plt.subplots(figsize=(11, 0.45 * len(methods) + 1.6))
    im = ax.imshow(matrix, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(TYPES)), TYPES, rotation=30, ha="right")
    ax.set_yticks(range(len(methods)), [SHORT[m] for m in methods])
    for i in range(len(methods)):
        for j in range(len(TYPES)):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if matrix[i, j] < 0.6 else "black")
    fig.colorbar(im, ax=ax, label="strict F1")
    ax.set_title("Per-type strict F1 (full test; API rows use n=300 subset)")
    fig.tight_layout()
    fig.savefig(ROOT / "results" / "per_type_f1.png", dpi=150)
    plt.close(fig)

    kinds = ["boundary_error", "type_error", "missed", "spurious"]
    methods = list(errors)
    fig, ax = plt.subplots(figsize=(10, 0.45 * len(methods) + 1.6))
    left = np.zeros(len(methods))
    y = np.arange(len(methods))[::-1]
    for kind, color in zip(kinds, ["#4c78a8", "#f58518", "#e45756", "#72b7b2"]):
        values = np.array([errors[m][kind] for m in methods], dtype=float)
        ax.barh(y, values, left=left, label=kind, color=color)
        left += values
    ax.set_yticks(y, [SHORT[m] for m in methods])
    ax.set_xlabel("error count on fixed subset (n=300, same sentences for every method)")
    ax.legend(loc="lower right")
    ax.set_title("Error breakdown (strict view)")
    fig.tight_layout()
    fig.savefig(ROOT / "results" / "error_breakdown.png", dpi=150)
    plt.close(fig)


def main():
    cfg, splits = config(), load_data()
    subset = api_subset_ids()
    index = {r["id"]: r for r in splits["test"]}
    subset_gold = [index[i] for i in subset]
    results, missing = {}, []
    for method in METHODS:
        path = ROOT / "results" / f"{method}.json"
        if path.exists():
            results[method] = read_json(path)
        else:
            missing.append(method)
    full_rows, subset_rows_, per_type, counts_full, counts_subset = [], [], {}, {}, {}
    subset_errors, error_examples = {}, {}
    for method, result in results.items():
        gold = validate(result, splits, cfg)
        if method not in API_METHODS:
            metric = evaluate(gold, result["predictions"])
            errs = error_summary(gold, result["predictions"])
            full_rows.append(row_for(method, result, metric, errs, result["predictions"]))
            per_type[method] = metric["per_type"]
            counts_full[method] = count_matrix(gold, result["predictions"])
            preds_sub = subset_rows(result, subset)
        else:
            preds_sub = result["predictions"]
            per_type[method] = result["metrics"]["per_type"]
        metric_sub = evaluate(subset_gold, preds_sub)
        errs_sub = error_summary(subset_gold, preds_sub)
        subset_rows_.append(row_for(method, result, metric_sub, errs_sub, preds_sub))
        counts_subset[method] = count_matrix(subset_gold, preds_sub)
        subset_errors[method] = {k: len(v) for k, v in errs_sub.items()}
        error_examples[method] = errs_sub

    pairs = []
    for a, b in FULL_PAIRS:
        if a in counts_full and b in counts_full:
            pairs.append({"scope": "full", "a": a, "b": b,
                          **paired_bootstrap(counts_full[a], counts_full[b])})
    for a, b in SUBSET_PAIRS:
        if a in counts_subset and b in counts_subset:
            pairs.append({"scope": "subset", "a": a, "b": b,
                          **paired_bootstrap(counts_subset[a], counts_subset[b])})

    write_json(ROOT / "results" / "summary.json",
               {"full_test": full_rows, "api_subset": subset_rows_, "paired_bootstrap": pairs,
                "missing_methods": missing})
    write_json(ROOT / "results" / "paired_bootstrap.json", pairs)
    write_json(ROOT / "results" / "errors.json", error_examples)
    with (ROOT / "results" / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["scope"] + list(full_rows[0]))
        writer.writeheader()
        for scope, rows in [("full", full_rows), ("subset", subset_rows_)]:
            for row in rows:
                writer.writerow({"scope": scope, **row})

    lines = [
        "## 完整测试集（官方 dev，1,343 句，3,072 个实体）", "",
        "| 方法 | P | R | **strict F1** | Macro-F1 | 边界 F1 | 文本 F1 | 非法转移 | JSON 合法率 | "
        "训练/选模秒 | 毫秒/句 | 峰值显存 GiB | 可训参数 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in full_rows:
        lines.append(
            f"| {r['name']} | {fmt(r['strict_precision'])} | {fmt(r['strict_recall'])} | "
            f"**{fmt(r['strict_f1'])}** | {fmt(r['macro_f1'])} | {fmt(r['boundary_f1'])} | "
            f"{fmt(r['text_f1'])} | {fmt(r['illegal_transitions'])} | "
            f"{fmt(r['json_valid_rate'], 3)} | {fmt(r['train_seconds'], 1)} | "
            f"{fmt(r['ms_per_item'], 2)} | {fmt(r['peak_gpu_gib'], 2)} | {fmt(r['trainable_params'])} |")
    lines += ["", "## 固定子集（测试集中 seed=42 抽 300 句，所有方法同一批句子）", "",
              "| 方法 | **strict F1** | Macro-F1 | 边界 F1 | 文本 F1 | 边界错 | 类型错 | 漏抽 | 多抽 | "
              "空预测句 | JSON 合法率 | 原文找不到 | 毫秒/句 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in subset_rows_:
        lines.append(
            f"| {r['name']} | **{fmt(r['strict_f1'])}** | {fmt(r['macro_f1'])} | "
            f"{fmt(r['boundary_f1'])} | {fmt(r['text_f1'])} | {r['boundary_error']} | "
            f"{r['type_error']} | {r['missed']} | {r['spurious']} | {r['empty_predictions']} | "
            f"{fmt(r['json_valid_rate'], 3)} | {fmt(r['not_in_text'])} | {fmt(r['ms_per_item'], 1)} |")
    lines += ["", "## 配对 bootstrap（strict micro-F1，前者减后者，2,000 次，seed=42）", "",
              "| 范围 | 比较 | F1 差 | 95% 区间 |", "|---|---|---:|---:|"]
    for p in pairs:
        lo, hi = p["ci95"]
        lines.append(f"| {'全测试集' if p['scope'] == 'full' else '300 子集'} | "
                     f"{METHODS[p['a']]} − {METHODS[p['b']]} | {p['difference']:+.4f} | "
                     f"[{lo:+.4f}, {hi:+.4f}] |")
    lines += ["", "## 各类型 strict F1（API 行为 300 子集，其余为全测试集）", "",
              "| 方法 | " + " | ".join(TYPES) + " |", "|---|" + "---:|" * len(TYPES)]
    for method, pt in per_type.items():
        lines.append(f"| {METHODS[method]} | " + " | ".join(f"{pt[t]['f1']:.3f}" for t in TYPES)
                     + " |")
    if missing:
        lines += ["", "未运行：" + "、".join(METHODS[m] for m in missing)]
    (ROOT / "results" / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    plot(full_rows, subset_rows_, per_type, subset_errors)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
