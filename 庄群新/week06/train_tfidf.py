"""方法二 · L1 TF-IDF + LinearSVC

四步：字 n-gram 向量化 → 稀疏矩阵 → 线性分类 → 落盘模型。
为什么用字 n-gram 而不是分词：中文标题平均 20 字，字 1-2gram 已能抓住「股/盘/楼/球/手机」，
还省掉分词依赖（jieba 在这里只是可选对照）。

它的定位：通用短文本上只是**基线**；真正的主场是「数据封闭」——
内网不许下权重、语料是自有黑话、只有几千条样本还要秒级重训。

用法：
  python train_tfidf.py                     # 训练 + 评测 500 条 + 落盘 models/tfidf_svm.joblib
  python train_tfidf.py --n 0               # 在全部 8,901 条干净样本上评测
  python train_tfidf.py --no_save           # 只跑指标，不写模型
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
import dataio
import metrics as M
import resources as R
from labels import LABELS

METHOD = "TF-IDF + LinearSVC"


def build_model():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import Pipeline
    from sklearn.svm import LinearSVC

    return Pipeline([
        ("vec", TfidfVectorizer(analyzer="char",
                                ngram_range=config.TFIDF_NGRAM,
                                max_features=config.TFIDF_MAX_FEATURES,
                                sublinear_tf=True,
                                min_df=2)),
        ("clf", LinearSVC(C=config.TFIDF_C)),   # 高维稀疏文本上的经典线性基线，秒级训练
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=config.N_EVAL)
    ap.add_argument("--out", default="tfidf_metrics.json")
    ap.add_argument("--no_save", action="store_true")
    args = ap.parse_args()

    config.ensure_dirs()
    import numpy as np
    from joblib import dump

    train, _ = dataio.load_splits()
    samples, meta = dataio.load_clean_val(args.n or None)
    texts_tr = [d["sentence"] for d in train]
    y_tr = [d["label"] for d in train]
    texts_te = [d["sentence"] for d in samples]
    y_true = [d["label"] for d in samples]

    model_path = config.MODEL_DIR / "tfidf_svm.joblib"

    with R.ResourceMonitor() as mon:
        with R.Timer() as timer:
            model = build_model()
            model.fit(texts_tr, y_tr)          # 训练：建词表 + 拟合线性分类器，一起计时间
        train_seconds = timer.seconds
        model_size_mb = R.file_size_mb(model_path) if model_path.exists() else 0.0

        # 批量推理（batch=32 口径）
        t0 = time.time()
        preds = model.predict(texts_te).tolist()
        infer_seconds = round(time.time() - t0, 3)

        # 单条推理（batch=1 口径）：前 50 条逐条 transform+predict 取中位数
        lat = []
        for t in texts_te[:50]:
            s = time.perf_counter()
            model.predict([t])
            lat.append((time.perf_counter() - s) * 1000)
        lat.sort()
        ms_b1 = round(lat[len(lat) // 2], 3)

    if not args.no_save:
        dump(model, model_path)
        model_size_mb = R.file_size_mb(model_path)

    n_features = len(model.named_steps["vec"].vocabulary_)
    clf = model.named_steps["clf"]
    # 线性模型的参数量 = 权重矩阵 15×F + 偏置 15（这就是它「小」的具体含义）
    n_params = int(clf.coef_.size + clf.intercept_.size)

    # 置信度：SVC 没有 predict_proba，用 decision_function 过 softmax（未校准，仅供 ECE 参考）
    margins = model.decision_function(texts_te)
    probs = np.exp(margins - margins.max(axis=1, keepdims=True))
    probs = probs / probs.sum(axis=1, keepdims=True)
    ece = M.expected_calibration_error(np.asarray(y_true), probs)

    report = M.classification_report(y_true, preds)
    result = {
        "method": METHOD,
        "level": "L1",
        "metrics": {**report, "group_acc": M.group_level_report(y_true, preds), "ece": ece},
        "cost": {
            "train_seconds": train_seconds,
            "infer_seconds": infer_seconds,
            "infer_ms_b1": ms_b1,
            "infer_ms_b32": round(infer_seconds * 1000 / len(samples), 3),
            "throughput": round(len(samples) / max(infer_seconds, 1e-6), 1),
            "model_size_mb": model_size_mb,
            "peak_rss_mb": round(mon.peak_rss_mb, 1),
            "peak_gpu_mb": 0.0,
            "n_params": n_params,
            "n_features": n_features,
        },
        "meta": {**meta, "env": R.env_info(),
                 "hyper": {"ngram": list(config.TFIDF_NGRAM), "C": config.TFIDF_C,
                           "max_features": config.TFIDF_MAX_FEATURES}},
    }

    (config.OUT_DIR / args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (config.OUT_DIR / "tfidf_pred.json").write_text(
        json.dumps({"y_true": y_true, "y_pred": preds,
                    "sentence": [d["sentence"] for d in samples]}, ensure_ascii=False), encoding="utf-8")

    print(f"评测 {len(samples)} 条（干净集）| 训练 + 推理计 {train_seconds}s | 词表 {n_features} 维")
    print(f"| {METHOD} | acc={report['acc']} | macro-F1={report['macro_f1']} | 覆盖 {report['coverage']} |")
    print(f"单条 {ms_b1} ms | batch32 均摊 {result['cost']['infer_ms_b32']} ms/条 | 模型 {model_size_mb} MB")
    print(f"→ 模型 {model_path if not args.no_save else '(未落盘)'}")
    print(f"→ 指标 {config.OUT_DIR / args.out}")


if __name__ == "__main__":
    main()
