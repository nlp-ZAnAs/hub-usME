"""多数类、人工情感词、TF-IDF + NB/LR/SVM；超参数只看验证集。"""

import time
from collections import Counter

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from common import ROOT, config, load_data, metrics, save_result, seed_everything

# 在读取测试结果之前预先编写，固定使用，不根据测试错例补词。
POSITIVE = ["满意", "舒服", "舒适", "干净", "整洁", "方便", "便利", "热情",
            "周到", "友好", "推荐", "不错", "很好", "非常好", "挺好", "值得", "优美", "安静"]
NEGATIVE = ["失望", "很差", "太差", "差劲", "糟糕", "恶劣", "脏", "臭", "吵",
            "噪音", "破旧", "不满", "不值", "不推荐", "不会再", "再也不", "投诉",
            "冷漠", "不舒服", "不方便", "不干净", "不卫生", "坑人", "差评"]


def rule_predict(text, fallback=1):
    # 先删去已匹配的负面短语，避免“不舒服”同时计入“舒服”。
    negative = sum(text.count(word) for word in NEGATIVE)
    for word in sorted(NEGATIVE, key=len, reverse=True):
        text = text.replace(word, "")
    positive = sum(text.count(word) for word in POSITIVE)
    return int(positive > negative) if positive != negative else fallback


def main():
    cfg, splits = config(), load_data()
    seed_everything(cfg["seed"])
    train, val, test = (splits[k] for k in ["train", "val", "test"])
    train_x, train_y = [r["text"] for r in train], [r["label"] for r in train]
    val_x, val_y = [r["text"] for r in val], [r["label"] for r in val]
    test_x = [r["text"] for r in test]
    majority = Counter(train_y).most_common(1)[0][0]
    for method in ["majority", "rule"]:
        start = time.perf_counter()
        pred = [majority if method == "majority" else rule_predict(x, majority) for x in test_x]
        save_result(method, splits, pred, train_seconds=0,
                    predict_seconds=time.perf_counter() - start,
                    settings={"fallback": majority, "keyword_source": "fixed manual list"})
    grids = {
        "tfidf_nb": [("alpha", a, MultinomialNB(alpha=a)) for a in [0.1, 0.5, 1.0]],
        "tfidf_lr": [("C", c, LogisticRegression(C=c, max_iter=1000, random_state=cfg["seed"]))
                     for c in [0.5, 1.0, 2.0]],
        "tfidf_svm": [("C", c, LinearSVC(C=c, random_state=cfg["seed"], dual="auto"))
                      for c in [0.5, 1.0, 2.0]],
    }
    for method, grid in grids.items():
        start = time.perf_counter()
        best, best_f1, history, best_setting = None, -1, [], None
        for key, value, classifier in grid:
            model = Pipeline([
                ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(1, 2),
                                         min_df=2, max_features=50000, sublinear_tf=True)),
                ("classifier", classifier)])
            model.fit(train_x, train_y)
            score = metrics(val_y, model.predict(val_x))
            history.append({key: value, "val_macro_f1": score["macro_f1"]})
            if score["macro_f1"] > best_f1:
                best, best_f1, best_setting = model, score["macro_f1"], {key: value}
        train_seconds = time.perf_counter() - start
        start = time.perf_counter()
        predictions = best.predict(test_x).tolist()
        predict_seconds = time.perf_counter() - start
        (ROOT / "checkpoints").mkdir(exist_ok=True)
        joblib.dump(best, ROOT / "checkpoints" / (method + ".joblib"))
        save_result(method, splits, predictions, train_seconds=train_seconds,
                    predict_seconds=predict_seconds, history=history,
                    settings={"selected": best_setting, "selection": "validation macro-F1",
                              "analyzer": "char", "ngram_range": [1, 2], "min_df": 2,
                              "max_features": 50000, "sublinear_tf": True,
                              "train_time_includes": "all 3 candidates and validation"})


if __name__ == "__main__":
    main()
