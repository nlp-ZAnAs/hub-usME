# -*- coding: utf-8 -*-
"""
Week07 作业汇总：把「换编码器 / 换数据规模 / 换生成式模型」的结果拉成一张表 + 图

口径：
  - 所有判别式方法都在 dev_extract 930 句上评测，指标来自 common/evaluate_events.py；
  - 三个编码器（BERT / RoBERTa / MacBERT）使用**完全相同的训练超参**（linear 解码器、
    3 epoch、max_len 256、batch 32、lr 2e-5、seed 42），只换预训练权重；
  - 课程基线（BiLSTM-CRF / BERT+CRF / 两段式 / API / LoRA）直接读 baseline/ 下的原始
    metrics JSON，和本次实验同数据同口径，报告里会显式标注「课程基线，未重跑」。

用法：python make_summary.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parent
OUR_DISC = ROOT / "code" / "outputs" / "metrics" / "disc"
OUR_LLM = ROOT / "code" / "outputs" / "metrics" / "llm"
BASE_DISC = ROOT / "baseline" / "disc"
BASE_LLM = ROOT / "baseline" / "llm"
RESULTS = ROOT / "results"


def load(path):
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def get(metric):
    return None if metric is None else {
        "trigger": metric["trigger"]["f1"],
        "argument": metric["argument"]["f1"],
        "event": metric["event_exact"]["f1"],
        "type_macro": metric.get("type_macro_f1", 0.0),
        "n": metric.get("num_samples"),
        "meta": metric.get("meta") or {},
    }


def meta_val(m, key, default="-"):
    return (m.get("meta") or {}).get(key, default)


def fmt(v, nd=4):
    return "-" if v is None else f"{v:.{nd}f}"


def table(rows, cols):
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(c, "-")) for c in cols) + " |")
    return "\n".join(lines)


def disc_rows():
    specs = [
        ("rule", "L0 规则（词典+正则）", OUR_DISC / "rule.json", False),
        ("bilstm", "L1 BiLSTM-CRF（无预训练）", BASE_DISC / "bilstm_crf.json", True),
        ("bert_linear", "L2 BERT-base+Linear", OUR_DISC / "bert_base_linear_256.json", False),
        ("roberta", "L2 RoBERTa-wwm-ext+Linear", OUR_DISC / "roberta_linear_256.json", False),
        ("macbert", "L2 MacBERT-base+Linear", OUR_DISC / "macbert_linear_256.json", False),
        ("bert_crf", "L2 BERT-base+CRF", BASE_DISC / "bert_crf_256.json", True),
        ("pipeline", "L2 两段式流水线", BASE_DISC / "pipeline.json", True),
    ]
    out = []
    for key, label, path, baseline in specs:
        metric = get(load(path))
        if metric is None:
            continue
        illegal = (metric["meta"].get("illegal_dev") or {}).get("illegal_seq_rate")
        out.append({
            "key": key, "方法": label + ("（课程基线）" if baseline else ""),
            "触发词F1": fmt(metric["trigger"]), "论元F1": fmt(metric["argument"]),
            "事件F1": fmt(metric["event"]), "类型macroF1": fmt(metric["type_macro"]),
            "参数M": meta_val(metric, "params_m"),
            "训练s": meta_val(metric, "train_s"),
            "延迟ms/句": meta_val(metric, "ms_per_sentence"),
            "非法BIO": "-" if illegal is None else f"{illegal:.1%}",
            "_raw": metric,
        })
    return out


def encoder_scale_rows():
    specs = [
        ("BERT-base", "bert_base_linear_256"),
        ("RoBERTa-wwm-ext", "roberta_linear_256"),
        ("MacBERT-base", "macbert_linear_256"),
    ]
    rows = []
    for label, name in specs:
        full = get(load(OUR_DISC / f"{name}.json"))
        small = get(load(OUR_DISC / f"{name}_sample.json"))
        if not full:
            continue
        rows.append({
            "编码器": label,
            "全量 论元F1": fmt(full["argument"]),
            "全量 触发词F1": fmt(full["trigger"]),
            "小样本 论元F1": fmt(small["argument"]) if small else "-",
            "小样本 触发词F1": fmt(small["trigger"]) if small else "-",
            "全量训练s": meta_val(full, "train_s"),
        })
    return rows


def llm_rows():
    specs = [
        ("Qwen2-0.5B-Instruct zero-shot（n60）", OUR_LLM / "qwen2_zero_n60.json", False),
        ("Qwen3-0.6B zero-shot（n60）", OUR_LLM / "qwen3_zero_n60.json", False),
        ("Qwen2.5-1.5B-Instruct zero-shot（n60）", OUR_LLM / "qwen25_15b_zero_n60.json", False),
        ("API deepseek-v4-flash zero-shot（n200）", BASE_LLM / "api_zero_n200.json", True),
        ("API deepseek-v4-flash few-shot（n200）", BASE_LLM / "api_few_n200.json", True),
        ("Qwen3-0.6B + LoRA SFT（n200）", BASE_LLM / "sft_lora_n200.json", True),
    ]
    rows = []
    for label, path, baseline in specs:
        metric = get(load(path))
        if metric is None:
            continue
        rows.append({
            "方法": label + ("（课程基线）" if baseline else ""),
            "样本数": metric["n"],
            "触发词F1": fmt(metric["trigger"]),
            "论元F1": fmt(metric["argument"]),
            "事件F1": fmt(metric["event"]),
            "_raw": metric,
        })
    return rows


def bar_disc(rows, path):
    labels = [r["方法"].replace("（课程基线）", "") for r in rows]
    trg = [r["_raw"]["trigger"] for r in rows]
    arg = [r["_raw"]["argument"] for r in rows]
    ev = [r["_raw"]["event"] for r in rows]
    x = np.arange(len(rows))
    w = 0.27
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - w, trg, w, label="Trigger F1", color="#90caf9")
    ax.bar(x, arg, w, label="Argument F1", color="#1565c0")
    ax.bar(x + w, ev, w, label="Event F1", color="#f57c00")
    for i in range(len(rows)):
        ax.text(i, arg[i] + 0.012, f"{arg[i]:.3f}", ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=22, ha="right", fontsize=8)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("F1")
    ax.set_title("事件抽取：判别式方法对比（dev 930 句，统一口径）")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def bar_scale(rows, path):
    labels = [r["编码器"] for r in rows]
    full = [r["_full_arg"] for r in rows]
    small = [r["_small_arg"] for r in rows]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.bar(x - 0.2, full, 0.4, label="全量 5,747 句", color="#1565c0")
    ax.bar(x + 0.2, small, 0.4, label="小样本 800 句", color="#ff9800")
    for i in range(len(rows)):
        ax.text(i - 0.2, full[i] + 0.01, f"{full[i]:.3f}", ha="center", fontsize=8)
        ax.text(i + 0.2, small[i] + 0.01, f"{small[i]:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("论元 F1")
    ax.set_title("换编码器 × 换数据规模：论元 F1")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def bar_llm(rows, path):
    labels = [r["方法"].replace("（课程基线）", "").replace("（n60）", "").replace("（n200）", "")
              for r in rows]
    trg = [r["_raw"]["trigger"] for r in rows]
    arg = [r["_raw"]["argument"] for r in rows]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.bar(x - 0.18, trg, 0.36, label="Trigger F1", color="#90caf9")
    ax.bar(x + 0.18, arg, 0.36, label="Argument F1", color="#6a1b9a")
    for i in range(len(rows)):
        ax.text(i - 0.18, trg[i] + 0.01, f"{trg[i]:.3f}", ha="center", fontsize=7)
        ax.text(i + 0.18, arg[i] + 0.01, f"{arg[i]:.3f}", ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right", fontsize=7.5)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("F1")
    ax.set_title("生成式链路：本地小模型 / API / LoRA 对比（本地 n60，课程基线 n200）")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def per_type_fig(bert, mac, path):
    if not bert or not mac:
        return
    types = sorted(set(bert["per_type"]) & set(mac["per_type"]),
                   key=lambda t: bert["per_type"][t]["f1"])
    x = np.arange(len(types))
    b = [bert["per_type"][t]["f1"] for t in types]
    m = [mac["per_type"][t]["f1"] for t in types]
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.bar(x - 0.2, b, 0.4, label="BERT-base", color="#1565c0")
    ax.bar(x + 0.2, m, 0.4, label="MacBERT-base", color="#ff9800")
    ax.set_xticks(x)
    ax.set_xticklabels(types, rotation=35, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("事件级 F1")
    ax.set_title("按事件类型看：换编码器后哪些类型变好/变差")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    disc = disc_rows()
    llm = llm_rows()

    # 编码器 × 数据规模
    scale = []
    for label, name in [("BERT-base", "bert_base_linear_256"),
                        ("RoBERTa-wwm-ext", "roberta_linear_256"),
                        ("MacBERT-base", "macbert_linear_256")]:
        full = get(load(OUR_DISC / f"{name}.json"))
        # 小样本模型在小 dev 上训练，最终统一在**全量 dev 930** 上复评，保证可比
        small = get(load(OUR_DISC / f"{name}_sample_full.json")) or \
            get(load(OUR_DISC / f"{name}_sample.json"))
        if not full:
            continue
        scale.append({
            "编码器": label,
            "全量 论元F1": fmt(full["argument"]),
            "全量 触发词F1": fmt(full["trigger"]),
            "小样本 论元F1": fmt(small["argument"]) if small else "-",
            "小样本 触发词F1": fmt(small["trigger"]) if small else "-",
            "全量训练s": meta_val(full, "train_s"),
            "_full_arg": full["argument"],
            "_small_arg": small["argument"] if small else 0.0,
        })

    sections = ["# Week07 事件抽取：换模型 / 换数据 / 换生成式模型 对比", "",
                "> 数据：DuEE-Fin 派生，dev_extract 930 句（一句一事件，13 类型 / 92 槽位 / 61 角色）。",
                "> 指标：`common/evaluate_events.py` 统一口径（trigger / argument / event 三级 F1）。", ""]

    sections += ["## 1. 判别式方法主表（dev 930 句）", "",
                 table(disc, ["方法", "触发词F1", "论元F1", "事件F1", "类型macroF1",
                              "参数M", "训练s", "延迟ms/句", "非法BIO"]), ""]

    if disc:
        by = {r["key"]: r["_raw"] for r in disc}
        if "bert_linear" in by:
            base = by["bert_linear"]["argument"]
            lines = ["**换编码器（同一 decoder / 同一超参）**："]
            for k, label in [("roberta", "RoBERTa-wwm-ext"), ("macbert", "MacBERT-base")]:
                if k in by:
                    v = by[k]["argument"]
                    lines.append(f"- {label} − BERT = {v - base:+.4f}（论元 F1），"
                                 f"事件 F1 {by[k]['event'] - by['bert_linear']['event']:+.4f}")
            sections += ["", *lines, ""]

    sections += ["![判别式对比](fig_disc_compare.png)", ""]

    sections += ["## 2. 换编码器 × 换数据规模", "",
                 table(scale, ["编码器", "全量 论元F1", "全量 触发词F1", "小样本 论元F1",
                               "小样本 触发词F1", "全量训练s"]), "",
                 "![编码器与数据规模](fig_encoder_scale.png)", ""]

    sections += ["## 3. 生成式链路（本地 / API / LoRA）", "",
                 table(llm, ["方法", "样本数", "触发词F1", "论元F1", "事件F1"]), "",
                 "![生成式对比](fig_llm_compare.png)", ""]

    (RESULTS / "summary.md").write_text("\n".join(sections) + "\n", encoding="utf-8")

    bar_disc(disc, RESULTS / "fig_disc_compare.png")
    if scale:
        bar_scale(scale, RESULTS / "fig_encoder_scale.png")
    bar_llm(llm, RESULTS / "fig_llm_compare.png")
    bert_raw = load(OUR_DISC / "bert_base_linear_256.json")
    mac_raw = load(OUR_DISC / "macbert_linear_256.json")
    if bert_raw and mac_raw:
        per_type_fig(bert_raw, mac_raw, RESULTS / "fig_per_type.png")
    print("\n".join(sections))
    print(f"\n→ {RESULTS / 'summary.md'} + figures")


if __name__ == "__main__":
    main()
