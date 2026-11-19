# -*- coding: utf-8 -*-
"""
一键跑完两个数据集 × 全部方法的实验，最后调用 summarize.py 出表和图。

    python run_all.py                      # 全跑（已有结果的会跳过）
    python run_all.py --datasets tnews     # 只跑 TNEWS
    python run_all.py --methods rule tfidf
    python run_all.py --force              # 全部重跑

实验顺序按成本从低到高：先规则/TF-IDF（秒级）确认数据和评测口径，
再 LLM zero-shot，最后 BERT 微调（每个约 5 分钟）。
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable


def run(cmd, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n[{time.strftime('%H:%M:%S')}] $ {' '.join(cmd)}", flush=True)
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as f:
        p = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
    print(f"[{time.strftime('%H:%M:%S')}] exit={p.returncode} {time.time()-t0:.1f}s -> {log_path}",
          flush=True)
    return p.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["tnews", "chnsenticorp"],
                    choices=["tnews", "chnsenticorp"])
    ap.add_argument("--methods", nargs="+", default=["rule", "tfidf", "zero_shot", "bert", "lora"],
                    choices=["rule", "tfidf", "zero_shot", "bert", "lora"])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    logs = ROOT / "results" / "logs"
    force = ["--force"] if args.force else []

    for ds in args.datasets:
        if "rule" in args.methods:
            run([PY, "code/rule/run_rule.py", "--dataset", ds] + force,
                logs / f"rule_{ds}.log")
        if "tfidf" in args.methods:
            run([PY, "code/tfidf/run_tfidf.py", "--dataset", ds] + force,
                logs / f"tfidf_{ds}.log")
        if "zero_shot" in args.methods:
            run([PY, "code/llm/run_zero_shot.py", "--dataset", ds] + force,
                logs / f"zero_shot_{ds}.log")
        if "bert" in args.methods:
            for backbone in ["bert-base-chinese", "chinese-roberta-wwm-ext",
                             "chinese-macbert-base"]:
                run([PY, "code/bert/train_bert.py", "--dataset", ds,
                     "--backbone", backbone] + force,
                    logs / f"bert_{ds}_{backbone}.log")
        if "lora" in args.methods and ds == "chnsenticorp":
            # 生成式微调只在 ChnSentiCorp 上跑：TNEWS 5.3 万条对 0.6B SFT 太慢，性价比也低
            run([PY, "code/llm/train_lora.py", "--dataset", ds] + force,
                logs / f"lora_{ds}.log")

    run([PY, "code/eval/summarize.py"], logs / "summarize.log")
    run([PY, "code/eval/significance.py"], logs / "significance.log")
    run([PY, "code/eval/examples.py"], logs / "examples.log")
    print("\n全部完成：results/summary.md + significance.md + examples.md + fig_*.png")


if __name__ == "__main__":
    main()
