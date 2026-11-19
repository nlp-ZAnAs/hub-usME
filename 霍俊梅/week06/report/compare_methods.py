"""
五路方法同口径对比（同一批 500 条 val 样本，seed=42 抽样）

教学重点：
  1. 横向比较的前提是**同一批样本**：500 条全部来自 sample_subset(val, 500)，
     规则/API 直接用各自落盘的预测结果按 idx 对齐，TF-IDF/BERT 现算。
  2. 统一看 acc 和 macro-F1：类别不均衡（56:44）时单看 acc 会高估弱方法。
  3. LoRA 结果落盘后（outputs/lora_pred.json，同为这 500 条）重跑本脚本即可自动补上。

用法：
  cd week06 && python report/compare_methods.py
"""

import json
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE_ROOT))

from common.data import load_splits, sample_subset
from common.label import LABELS
from sklearn.metrics import classification_report, f1_score

OUT_DIR = CODE_ROOT / "outputs"
N_EVAL = 500


def f1_of(y_true, y_pred):
    """acc + macro-F1（两类的 F1 平均），解析失败(-1)的样本剔除"""
    ok = [(t, p) for t, p in zip(y_true, y_pred) if p != -1]
    if not ok:
        return {"acc": 0.0, "macro_f1": 0.0, "n_fail": len(y_true)}
    y, p = [t for t, _ in ok], [q for _, q in ok]
    return {"acc": sum(a == b for a, b in ok) / len(ok),
            "macro_f1": f1_score(y, p, average="macro"),
            "n_fail": len(y_true) - len(ok)}


def eval_rule(idx2label, samples):
    """规则：读 outputs/rule_pred.json，按 idx 对齐到本批样本"""
    d = json.load(open(OUT_DIR / "rule_pred.json"))
    preds = {p["idx"]: p["pred"] for p in d["preds"]}
    return [preds[s["idx"]] for s in samples]


def eval_api(idx2label, samples):
    """GLM 零样本：读 outputs/api_zero_shot_pred.json（本来就是这 500 条）"""
    d = json.load(open(OUT_DIR / "api_zero_shot_pred.json"))
    preds = {p["idx"]: p["pred"] for p in d["preds"]}
    return [preds[s["idx"]] for s in samples]


def eval_tfidf(train, samples):
    """TF-IDF+SVM：train 全量重训（几十秒），预测本批样本"""
    import jieba
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import LinearSVC

    corpus = lambda data: [" ".join(jieba.cut(s["text"] or "")) for s in data]
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=3, max_df=0.9)
    X_train = vec.fit_transform(corpus(train))
    X = vec.transform(corpus(samples))
    clf = LinearSVC(C=1.0)
    clf.fit(X_train, [s["label"] for s in train])
    return clf.predict(X).tolist()


def eval_bert(samples):
    """BERT：读 outputs/bert_pred.json（全量 val 已推理落盘），按 idx 对齐"""
    d = json.load(open(OUT_DIR / "bert_pred.json"))
    preds = {p["idx"]: p["pred"] for p in d["preds"]}
    return [preds[s["idx"]] for s in samples]


def eval_lora(samples):
    """LoRA：读 outputs/lora_pred.json（训练完成后存在，且同为这 500 条）"""
    f = OUT_DIR / "lora_pred.json"
    if not f.exists():
        return None
    d = json.load(open(f))
    preds = {p["idx"]: p["pred"] for p in d["preds"]}
    if not all(s["idx"] in preds for s in samples):
        return None
    return [preds[s["idx"]] for s in samples]


def main():
    train, val = load_splits()
    samples = sample_subset(val, N_EVAL)          # 与 api/lora 脚本同一 seed 同一函数
    y_true = [s["label"] for s in samples]
    print(f"同一批评测样本: {len(samples)} 条（seed=42）\n")

    methods = {}

    methods["rule"] = eval_rule(None, samples)
    methods["GLM-API"] = eval_api(None, samples)
    methods["TF-IDF+SVM"] = eval_tfidf(train, samples)
    methods["BERT-FT"] = eval_bert(samples)
    lora = eval_lora(samples)
    if lora is not None:
        methods["LoRA"] = lora

    rows = {}
    for name, preds in methods.items():
        m = f1_of(y_true, preds)
        rows[name] = m
        print(f"{name:>10s}  acc={m['acc']:.4f}  macro-F1={m['macro_f1']:.4f}"
              f"{'  解析失败=' + str(m['n_fail']) if m['n_fail'] else ''}")
    if lora is None:
        print("(LoRA skip")

    (OUT_DIR / "methods_compare.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    # ------------------------------------------------------------ 直方图
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # macOS 自带中文字体，按可用性依次回退；否则中文会显示成方框
    matplotlib.rcParams["font.sans-serif"] = [
        "Arial Unicode MS", "PingFang SC", "Hiragino Sans GB", "Heiti SC"]
    matplotlib.rcParams["axes.unicode_minus"] = False   # 负号用 ASCII，避免字体缺字形

    names = list(rows)
    accs = [rows[n]["acc"] for n in names]
    f1s = [rows[n]["macro_f1"] for n in names]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(names))
    width = 0.35
    ax.bar([i - width / 2 for i in x], accs, width, label="accuracy", color="#4C72B0")
    ax.bar([i + width / 2 for i in x], f1s, width, label="macro-F1", color="#DD8452")
    for i, (a, f) in enumerate(zip(accs, f1s)):
        ax.text(i - width / 2, a + 0.01, f"{a:.3f}", ha="center", fontsize=9)
        ax.text(i + width / 2, f + 0.01, f"{f:.3f}", ha="center", fontsize=9)
    ax.set_xticks(list(x))
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("score")
    ax.set_title(f"HC3 human/ChatGPT 二分类：{len(samples)} 条同口径 val 对比")
    ax.legend()
    fig.tight_layout()
    out = OUT_DIR / "f1_compare.png"
    fig.savefig(out, dpi=150)
    print(f"\n→ 对比图 {out}｜数据 {OUT_DIR / 'methods_compare.json'}")


if __name__ == "__main__":
    main()
