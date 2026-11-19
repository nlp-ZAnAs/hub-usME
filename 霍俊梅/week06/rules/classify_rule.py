"""
规则基线：human / ChatGPT 二分类

教学重点：
  1. 规则法的核心是「打分 + 阈值」而不是 if/else 硬判：
     每条规则命中加分（ChatGPT 指纹）或减分（人类指纹），总分过阈值判 1。
  2. 阈值在 train 上调优、val 只跑一次——规则也是「模型」，
     同样要遵守训练/验证隔离，否则规则基线就会偷看答案。
  3. 这套规则写的 是 2023 年初 ChatGPT 的文风指纹（分点、套话、拒答），
     换一代模型基本失效——规则法上限低但零训练成本，适合做下界基线。

用法：
  cd week06 && python rules/classify_rule.py [--out outputs/rule_pred.json]
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.data import load_splits
from common.label import LABELS

# ---------------------------------------------------------------------------
# 规则表：(模式, 权重)。正分 = ChatGPT 指纹，负分 = 人类指纹
# ---------------------------------------------------------------------------
CHATGPT_PHRASES = {
    "首先": 1, "其次": 1, "最后": 1,
    "综上所述": 2, "总的来说": 2, "总而言之": 2,
    "需要注意的是": 2, "需要注意的是": 2,
    "希望以上": 2, "希望对您有帮助": 2, "希望这些建议": 2,
    "仅供参考": 1, "祝您": 1,
    "作为一个人工智能": 3, "作为一个 AI": 3, "作为一个AI": 3,
    "语言模型": 2, "我无法提供": 2, "我无法回答": 2, "我没有": 1,
    "建议您咨询": 2, "寻求专业": 1,
}

HUMAN_PHRASES = {
    "谢邀": -3, "本人": -1, "我当年": -2, "身边": -1,
    "哈哈": -2, "呵呵": -2, "。。。": -2, "……": -1,
    "～～": -2, "~~~": -2,
}

# 列表标记：出现在行首才算（正文里出现 "1." 多半是编号引用，信号弱）
LIST_MARK = re.compile(r"^\s*(?:[①②③④⑤⑥⑦⑧⑨⑩]|\d+[.、）)]|[（(]\d+[）)]|[-•*])\s", re.M)
BOLD_MARK = re.compile(r"\*\*")


def score(text) -> float:
    """文本 → 规则总分。>0 偏向 ChatGPT，<0 偏向 human"""
    if not text:                      # HC3 里有空答案（null/空串），算强人类信号
        return -3.0
    s = 0.0
    for pat, w in CHATGPT_PHRASES.items():
        if pat in text:
            s += w
    for pat, w in HUMAN_PHRASES.items():
        if pat in text:
            s += w

    n_list = len(LIST_MARK.findall(text))
    if n_list:
        s += min(n_list, 3)          # 分点结构，最多 +3，防止长答案刷分

    if BOLD_MARK.search(text):
        s += 2                        # markdown 加粗残留

    n_lines = text.count("\n") + 1
    if n_lines >= 3:
        s += 1                        # 多段成体系，偏 ChatGPT

    if len(text) < 15:
        s -= 3                        # 超短答案（"不会，别买"）几乎必是人类
    elif len(text) > 500:
        s += 1                        # 超长成文，偏 ChatGPT

    return s


def predict(text: str, threshold: float) -> int:
    return 1 if score(text) > threshold else 0


# ---------------------------------------------------------------------------
# 评估：准确率 + 每类 P/R/F1 + 混淆矩阵
# ---------------------------------------------------------------------------
def evaluate(samples, threshold: float) -> dict:
    tp = fp = tn = fn = 0
    for s in samples:
        pred, gold = predict(s["text"], threshold), s["label"]
        if pred == 1 and gold == 1:
            tp += 1
        elif pred == 1 and gold == 0:
            fp += 1
        elif pred == 0 and gold == 0:
            tn += 1
        else:
            fn += 1

    acc = (tp + tn) / max(tp + fp + tn + fn, 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return {"acc": acc, "precision": prec, "recall": rec, "f1": f1,
            "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn}}


def tune_threshold(train) -> float:
    """在 train 上扫阈值取准确率最高者——不碰 val"""
    scores = sorted({score(s["text"]) for s in train})
    
    candidates = [scores[0] - 1, *scores, scores[-1] + 1]
    best_t, best_acc = 0.0, -1.0
    for t in candidates:
        acc = sum(predict(s["text"], t) == s["label"] for s in train) / len(train)
        if acc > best_acc:
            best_t, best_acc = t, acc
    
    return best_t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/rule_pred.json", help="预测结果落盘路径，空串不保存")
    args = ap.parse_args()

    train, val = load_splits()
    print(f"train={len(train)}  val={len(val)}")

    threshold = tune_threshold(train)
    print(f"\n在 train 上调优的阈值: {threshold}")

    for name, data in [("train", train), ("val", val)]:
        m = evaluate(data, threshold)
        c = m.pop("confusion")
        print(f"\n[{name}] acc={m['acc']:.4f}  P={m['precision']:.4f}  "
              f"R={m['recall']:.4f}  F1={m['f1']:.4f}")
        print(f"  混淆矩阵  预测{LABELS[1]}={c['tp']+c['fp']}  预测{LABELS[0]}={c['tn']+c['fn']}")
        print(f"  实际{LABELS[1]}: 对 {c['tp']} / 错(漏) {c['fn']}")
        print(f"  实际{LABELS[0]}: 对 {c['tn']} / 错(误报) {c['fp']}")

    # 分错样本抽样：看规则漏在哪，是下一轮迭代的输入
    errors = [s for s in val if predict(s["text"], threshold) != s["label"]]
    print(f"\nval 分错 {len(errors)} 条，前 3 条示例：")
    for s in errors[:3]:
        print(f"  [idx={s['idx']}] 实际={LABELS[s['label']]}  得分={score(s['text'])}")
        print(f"  {s['text'][:60]}...")

    # -------------------------------------------- 预测结果落盘（模板与 api_zero_shot_pred.json 对齐）
    # score 相当于其它方法的置信度分数（decision_function / API 输出），
    # 分流等下游逻辑可以按同一套字段读取四种方法的结果
    if args.out:
        val_report = evaluate(list(val), threshold)   # evaluate 会 pop 掉 confusion，传副本
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({
            "meta": {
                "method": "rule",
                "n": len(val),
                "threshold": threshold,
                "report": {k: round(v, 4) for k, v in val_report.items() if k != "confusion"},
            },
            "preds": [{"idx": s["idx"],
                       "text": (s["text"] or "")[:100],
                       "true": s["label"],
                       "pred": predict(s["text"], threshold),
                       "score": score(s["text"])}
                      for s in val],
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"→ 结果写入 {out_path}")


if __name__ == "__main__":
    main()
