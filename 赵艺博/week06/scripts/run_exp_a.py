"""
实验 A：同一 TNEWS 数据，对比 bert-base-chinese vs chinese-roberta-wwm-ext，
并跑规则 / TF-IDF 作为基线。在干净验证集 500 条（seed=42）上评测。
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

CODE = Path(r"d:\BaiduNetdiskDownload\Agent开发学习资料\第六周 分类任务\week6-分类任务\week6-数据接入与分流\code")
HW = Path(__file__).resolve().parent.parent
OUT = HW / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

PY = sys.executable
N = 500


def run(cmd: list[str], log_name: str) -> int:
    log_path = OUT / log_name
    print("\n>>>", " ".join(cmd), flush=True)
    t0 = time.time()
    with log_path.open("w", encoding="utf-8") as f:
        p = subprocess.run(cmd, cwd=str(CODE), stdout=f, stderr=subprocess.STDOUT, text=True)
    elapsed = round(time.time() - t0, 1)
    print(f"exit={p.returncode} elapsed={elapsed}s log={log_path}", flush=True)
    return p.returncode


def main():
    summary = {"n_eval": N, "steps": []}

    # 1) 规则
    summary["steps"].append({"name": "rule", "rc": run(
        [PY, "rule/classify_rule.py", "--n", str(N)], "01_rule.log")})

    # 2) TF-IDF
    summary["steps"].append({"name": "tfidf", "rc": run(
        [PY, "tfidf/classify_tfidf.py", "--n", str(N)], "02_tfidf.log")})

    # 3) BERT base 训练 + 评测
    summary["steps"].append({"name": "bert_train", "rc": run(
        [PY, "bert/train_bert.py", "--model", "bert-base-chinese",
         "--ckpt_name", "bert-base-chinese_ckpt.pt", "--epochs", "3"],
        "03_bert_base_train.log")})
    summary["steps"].append({"name": "bert_eval", "rc": run(
        [PY, "bert/predict_bert.py",
         "--ckpt", str(CODE / "outputs" / "bert-base-chinese_ckpt.pt"),
         "--model", "bert-base-chinese", "--n", str(N),
         "--out", str(OUT / "bert_base_pred.json")],
        "04_bert_base_eval.log")})

    # 4) RoBERTa 训练 + 评测
    summary["steps"].append({"name": "roberta_train", "rc": run(
        [PY, "bert/train_bert.py", "--model", "chinese-roberta-wwm-ext",
         "--ckpt_name", "chinese-roberta-wwm-ext_ckpt.pt", "--epochs", "3"],
        "05_roberta_train.log")})
    summary["steps"].append({"name": "roberta_eval", "rc": run(
        [PY, "bert/predict_bert.py",
         "--ckpt", str(CODE / "outputs" / "chinese-roberta-wwm-ext_ckpt.pt"),
         "--model", "chinese-roberta-wwm-ext", "--n", str(N),
         "--out", str(OUT / "roberta_pred.json")],
        "06_roberta_eval.log")})

    (OUT / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("DONE", summary)


if __name__ == "__main__":
    main()
