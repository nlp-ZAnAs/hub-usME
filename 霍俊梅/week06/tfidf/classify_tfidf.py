"""
TF-IDF + 线性 SVM：human / ChatGPT 二分类


  1. 和规则法相比，这里不再人工定义指纹，而是让算法自己从数据里学：
     TF-IDF 把「哪些词能区分两类」交给统计（IDF 高 + 类间分布差异大的词权重自然高）。
  2. 防泄漏纪律和规则法完全一致：
     - 词典 / IDF 只在 train 上 fit（val 只 transform）；
     - 超参 C 用 train 内部交叉验证选，val 只跑最后一次。
  3. LinearSVC 输出的是 decision_function（到分隔面的带符号距离）而非概率，
     后面分流逻辑要置信度时直接用这个分数即可（分数绝对值越大越自信）。

用法：
  cd week06 && python tfidf/classify_tfidf.py [--n 2000] [--ngram 1,2] [--C 1.0] [--cv] [--out outputs/tfidf_pred.json]
"""

import argparse
import json
import sys
from pathlib import Path

import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.data import load_splits, sample_subset
from common.label import LABELS


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0, help="抽样条数，0=全部（train/val 用同一 seed 抽）")
    ap.add_argument("--ngram", default="1,2", help="词 n-gram 范围，如 1,2 或 1,3")
    ap.add_argument("--C", type=float, default=1.0, help="LinearSVC 正则强度")
    ap.add_argument("--cv", action="store_true", help="跑 train 内 5 折交叉验证对比不同 C")
    ap.add_argument("--out", default="outputs/tfidf_pred.json", help="val 预测结果落盘路径，空串不保存")
    return ap.parse_args()


def to_corpus(samples):
    """分词后用空格拼接，交给 TfidfVectorizer（空答案给空串，向量为全 0）"""
    return [" ".join(jieba.cut(s["text"] or "")) for s in samples]


def main():
    args = parse_args()
    train, val = load_splits()
    if args.n:
        train = sample_subset(train, args.n)
        val = sample_subset(val, args.n)
    print(f"train={len(train)}  val={len(val)}")

    # ---------------------------------------------------------------- 向量化
    # fit 只发生在 train：词典、IDF 都不碰 val
    lo, hi = (int(x) for x in args.ngram.split(","))
    vec = TfidfVectorizer(ngram_range=(lo, hi), sublinear_tf=True,
                          min_df=3, max_df=0.9)
    X_train = vec.fit_transform(to_corpus(train))
    X_val = vec.transform(to_corpus(val))
    feature_names = vec.get_feature_names_out()
    print(f"词典大小: {len(feature_names)}")

    # ------------------------------------------------------ 用 CV 选超参 C
    # 在 train 内部 5 折交叉验证，不碰 val
    if args.cv:
        for c in [0.1, 1.0, 10.0]:
            acc = cross_val_score(LinearSVC(C=c), X_train,
                                  [s["label"] for s in train], cv=5).mean()
            print(f"  C={c:<5} CV acc={acc:.4f}")

    clf = LinearSVC(C=args.C)
    clf.fit(X_train, [s["label"] for s in train])

    # ------------------------------------------------------------ 评估
    y_val = [s["label"] for s in val]
    pred = clf.predict(X_val)
    print("\n[val]")
    print(classification_report(y_val, pred, target_names=LABELS, digits=4))
    print("混淆矩阵 (行=实际, 列=预测):")
    print(confusion_matrix(y_val, pred))

    # -------------------------------------------- 预测结果落盘（供分流等下游使用）
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        records = [{
            "idx": s["idx"],
            "text": (s["text"] or "")[:100],
            "gold": s["label"],
            "pred": int(p),
            "score": float(d),          # decision_function，绝对值越大越自信
        } for s, p, d in zip(val, pred, clf.decision_function(X_val))]
        out_path.write_text(
            json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n预测结果已写入 {out_path}")

    # ------------------------------------------------- 模型学到了什么
    # 线性 SVM 的权重直接对应特征：看最大正/负权重词，就是它版「规则表」
    coefs = clf.coef_[0]
    names = feature_names
    order = coefs.argsort()
    print("\n最像 ChatGPT 的特征（正权重 Top 10）:")
    for i in order[::-1][:10]:
        print(f"  {names[i]:>10s}  {coefs[i]:+.3f}")
    print("最像 human 的特征（负权重 Top 10）:")
    for i in order[:10]:
        print(f"  {names[i]:>10s}  {coefs[i]:+.3f}")


if __name__ == "__main__":
    main()
