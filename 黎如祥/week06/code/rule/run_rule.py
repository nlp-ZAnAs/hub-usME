# -*- coding: utf-8 -*-
"""
方法一：规则 / 关键词（L0 下限）

两个数据集用两种「最朴素」的规则，目的不是刷分，而是给所有模型一个下限：
  - TNEWS：标题里出现哪个类别名就判哪类，没有命中就猜训练集最多的类（多数类基线）。
  - ChnSentiCorp：情感词典打分（正向词 + 词长，负向词 - 词长），打平猜多数类。

规则法的天花板由「命中率」决定：新闻标题很少直接写「财经」「体育」，
所以它在 15 类任务上必然接近随机；情感评论里「好/差/满意」出现得多，会好一些。
这正好说明：规则是否可用，取决于任务文本的「套路程度」，不是代码水平。

用法：
  python code/rule/run_rule.py --dataset tnews
  python code/rule/run_rule.py --dataset chnsenticorp
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.config import DATASETS, result_path
from common.data import eval_samples, label_names, load_dataset, majority_label
from common.metrics import classification_report, pred_distribution

POS_WORDS = ["很好", "不错", "满意", "喜欢", "舒适", "干净", "方便", "热情", "周到",
             "值得", "棒", "赞", "优秀", "惊喜", "舒服", "漂亮", "安静", "实惠",
             "好吃", "贴心", "细心", "礼貌", "齐全", "便利", "完美", "开心",
             "愉快", "超值", "给力", "五星", "推荐"]
NEG_WORDS = ["很差", "糟糕", "失望", "恶心", "脏", "吵", "慢", "贵", "难吃",
             "不推荐", "后悔", "破旧", "陈旧", "拥挤", "冷漠", "不耐烦", "不卫生",
             "不舒服", "不方便", "不满意", "不值", "欺骗", "恶劣", "简陋",
             "异味", "噪音", "潮湿", "发霉", "态度不好"]


def rule_tnews(text, fallback, labels):
    for i, name in enumerate(labels):
        if name in text:
            return i, "keyword"
    return fallback, "majority_fallback"


def rule_sentiment(text, fallback, labels):
    pos = sum(len(w) for w in POS_WORDS if w in text)
    neg = sum(len(w) for w in NEG_WORDS if w in text)
    if pos > neg:
        return labels.index("正面"), "lexicon"
    if neg > pos:
        return labels.index("负面"), "lexicon"
    return fallback, "majority_fallback"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DATASETS))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out_path = result_path(args.dataset, "rule")
    if out_path.exists() and not args.force:
        print(f"[skip] {out_path} 已存在")
        return

    train, _ = load_dataset(args.dataset)
    samples, meta = eval_samples(args.dataset, 500, 42)
    labels = label_names(args.dataset)
    fallback = majority_label(train)
    fn = rule_tnews if args.dataset == "tnews" else rule_sentiment

    preds, hits = [], 0
    for s in samples:
        p, how = fn(s["sentence"], fallback, labels)
        preds.append(p)
        hits += (how != "majority_fallback")

    report = classification_report([s["label"] for s in samples], preds, len(labels))
    report["method"] = "rule"
    report["pred_distribution"] = pred_distribution(preds, len(labels))
    report["rule_hit_rate"] = round(hits / len(samples), 4)

    result = {
        "method": "rule",
        "display": "规则 / 关键词",
        "family": "rule",
        "dataset": args.dataset,
        "meta": meta,
        "report": report,
        "cost": {"train_seconds": 0.0, "per_item_ms": None, "params": 0,
                 "peak_mem_gb": None},
        "preds": [{"text": s["sentence"], "true": int(s["label"]), "pred": int(p), "prob": None}
                  for s, p in zip(samples, preds)],
    }
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[{args.dataset}] rule hit_rate={report['rule_hit_rate']:.2%} "
          f"acc={report['acc_incl_unparsed']} macro_f1={report['macro_f1']}")
    print(f"→ {out_path}")


if __name__ == "__main__":
    main()
