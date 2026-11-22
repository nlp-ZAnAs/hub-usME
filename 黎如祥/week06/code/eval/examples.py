# -*- coding: utf-8 -*-
"""
Bad case / 典型案例抽取：把「谁对谁错」摊开看

对每个数据集：
  - BERT 对、TF-IDF 错  → 预训练知识带来的真实增益长什么样
  - TF-IDF 对、BERT 错  → 模型也会输给字面统计，为什么
  - zero-shot 的塌缩例子
输出 results/examples.md，方便直接贴进报告。

用法：
  python code/eval/examples.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.config import DATASETS, RESULTS_DIR
from common.data import label_names


def load(ds, method):
    p = RESULTS_DIR / f"pred_{ds}_{method}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def cut(s, n=80):
    s = s.replace("\n", " ").replace("|", "\\|")
    return s if len(s) <= n else s[:n] + "…"


def main():
    lines = ["# 典型案例（Bad Case 分析）", ""]
    for ds in DATASETS:
        labels = label_names(ds)
        bert = load(ds, "bert_bert-base-chinese")
        tfidf = load(ds, "tfidf")
        zero = load(ds, "qwen3_zero_shot")
        if not bert or not tfidf:
            continue
        b = {x["text"]: x for x in bert["preds"]}
        t = {x["text"]: x for x in tfidf["preds"]}
        lines += [f"## {DATASETS[ds]['display']}", ""]

        win = [k for k in b if b[k]["pred"] == b[k]["true"] and
               k in t and t[k]["pred"] != t[k]["true"]]
        lose = [k for k in b if b[k]["pred"] != b[k]["true"] and
                k in t and t[k]["pred"] == t[k]["true"]]
        lines += [f"### 1) BERT 对、TF-IDF 错（{len(win)} 条），抽 4 条", ""]
        for k in win[:4]:
            lines += [f"- 文本：{cut(k, 100)}",
                      f"  - 真值：{labels[b[k]['true']]}；BERT：{labels[b[k]['pred']]}；"
                      f"TF-IDF：{labels[t[k]['pred']]}"]
        lines += ["", f"### 2) TF-IDF 对、BERT 错（{len(lose)} 条），抽 4 条", ""]
        for k in lose[:4]:
            lines += [f"- 文本：{cut(k, 100)}",
                      f"  - 真值：{labels[b[k]['true']]}；BERT：{labels[b[k]['pred']]}；"
                      f"TF-IDF：{labels[t[k]['pred']]}"]
        if zero:
            bad = [x for x in zero["preds"] if x["pred"] != x["true"]]
            by_pred = {}
            for x in zero["preds"]:
                by_pred[labels[x["pred"]] if x["pred"] >= 0 else "解析失败"] = \
                    by_pred.get(labels[x["pred"]] if x["pred"] >= 0 else "解析失败", 0) + 1
            lines += ["", "### 3) zero-shot 的预测分布（塌缩证据）", "",
                      "| 预测标签 | 条数 |", "|---|---|"]
            for k, v in sorted(by_pred.items(), key=lambda kv: -kv[1]):
                lines.append(f"| {k} | {v} |")
            lines += ["", "失败样例（抽 3 条）："]
            for x in bad[:3]:
                lines += [f"- 文本：{cut(x['text'], 90)}",
                          f"  - 真值：{labels[x['true']]}；预测："
                          f"{labels[x['pred']] if x['pred'] >= 0 else '解析失败'}；"
                          f"原始输出：{x.get('raw','')!r}"]
        lines.append("")

    (RESULTS_DIR / "examples.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:40]))
    print(f"→ {RESULTS_DIR / 'examples.md'}")


if __name__ == "__main__":
    main()
