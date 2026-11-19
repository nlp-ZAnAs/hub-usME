# -*- coding: utf-8 -*-
"""
方法二：TF-IDF + LinearSVC（不训练的强基线）

为什么在「换预训练模型」的实验里还要保留它：
  它是「没有任何预训练知识、只靠训练集字面统计」的对照组。
  如果换了个更强的预训练模型只比它高 1 个点，那这次换模型的性价比就要打问号。

实现：字符 1-2gram（中文短文本不用分词）、max_features=30 万、LinearSVC(C=1.0)。
两个数据集用完全相同的特征与分类器，只换数据。

用法：
  python code/tfidf/run_tfidf.py --dataset tnews
  python code/tfidf/run_tfidf.py --dataset chnsenticorp --C 0.5 --ngram 1,3
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.config import DATASETS, result_path
from common.data import eval_samples, label_names, load_dataset
from common.metrics import classification_report, pred_distribution


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DATASETS))
    ap.add_argument("--ngram", default="1,2")
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--leak", action="store_true", help="不去重，在原始 val 上抽样：量化泄漏虚高")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out_path = result_path(args.dataset, "tfidf") if not args.leak else \
        result_path(args.dataset, "tfidf_leak")
    if out_path.exists() and not args.force:
        print(f"[skip] {out_path} 已存在")
        return

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import LinearSVC

    train, val = load_dataset(args.dataset)
    if args.leak:
        samples = random.Random(42).sample(val, min(500, len(val)))
        meta = {"dataset": args.dataset, "n_train": len(train), "n_val": len(val),
                "n_dup": None, "n_eval": len(samples), "seed": 42, "mode": "leaky_full_val"}
    else:
        samples, meta = eval_samples(args.dataset, 500, 42)
    labels = label_names(args.dataset)
    lo, hi = (int(x) for x in args.ngram.split(","))

    vec = TfidfVectorizer(analyzer="char", ngram_range=(lo, hi), max_features=300000)
    t0 = time.time()
    Xtr = vec.fit_transform([d["sentence"] for d in train])
    Xev = vec.transform([d["sentence"] for d in samples])
    clf = LinearSVC(C=args.C).fit(Xtr, [d["label"] for d in train])
    t_inf = time.time()
    preds = clf.predict(Xev).tolist()
    infer_s = (time.time() - t_inf) / len(samples)
    fit_s = round(time.time() - t0 - infer_s * len(samples), 2)

    y_true = [d["label"] for d in samples]
    method = "tfidf_svm_leak" if args.leak else "tfidf_svm"
    report = classification_report(y_true, preds, len(labels))
    report["method"] = method
    report["pred_distribution"] = pred_distribution(preds, len(labels))

    result = {
        "method": method,
        "display": "TF-IDF + LinearSVC" + ("（不去重 val）" if args.leak else ""),
        "family": "linear",
        "dataset": args.dataset,
        "meta": meta,
        "report": report,
        "cost": {"train_seconds": fit_s, "per_item_ms": round(infer_s * 1000, 3),
                 "params": None, "peak_mem_gb": None},
        "preds": [{"text": s["sentence"], "true": int(s["label"]), "pred": int(p), "prob": None}
                  for s, p in zip(samples, preds)],
    }
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[{args.dataset}] TF-IDF ngram={args.ngram} C={args.C} fit={fit_s}s "
          f"acc={report['acc_incl_unparsed']} macro_f1={report['macro_f1']}")
    print(f"→ {out_path}")


if __name__ == "__main__":
    main()
