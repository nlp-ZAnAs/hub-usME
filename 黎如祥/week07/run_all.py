# -*- coding: utf-8 -*-
"""
Week07 一键复现：三个编码器 × 两种数据规模 + 三个生成式模型 zero-shot

    python run_all.py                 # 全跑（已有 metrics 的会跳过）
    python run_all.py --only encoders # 只跑判别式
    python run_all.py --only llm      # 只跑生成式
    python run_all.py --force         # 全部重跑

注意：GPU 任务串行执行（课程 8GB 卡同时跑两个会 OOM）；本机 16GB 也建议串行。
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CODE = ROOT / "code"
PY = sys.executable
MODELS = Path(r"D:\Env\models")
OUR_DISC = CODE / "outputs" / "metrics" / "disc"
OUR_LLM = CODE / "outputs" / "metrics" / "llm"
QWEN2 = r"E:\LRX\八斗\Videos\01_往期录播学习\week6文本分类问题\pretrain_models\Qwen2-0.5B-Instruct"


def run(cmd, log_name, skip_if, force=False):
    if skip_if.exists() and not force:
        print(f"[skip] {skip_if.name} 已存在")
        return True
    (CODE / "outputs" / "logs").mkdir(parents=True, exist_ok=True)
    log = CODE / "outputs" / "logs" / log_name
    print(f"\n[{time.strftime('%H:%M:%S')}] $ {' '.join(cmd)}")
    t0 = time.time()
    with open(log, "w", encoding="utf-8") as f:
        p = subprocess.run(cmd, cwd=CODE, stdout=f, stderr=subprocess.STDOUT)
    print(f"[{time.strftime('%H:%M:%S')}] exit={p.returncode} {time.time()-t0:.0f}s → {log}")
    return p.returncode == 0


def encoders(force):
    runs = [("bert-base-chinese", "bert_base_linear_256"),
            ("chinese-roberta-wwm-ext", "roberta_linear_256"),
            ("chinese-macbert-base", "macbert_linear_256")]
    for backbone, name in runs:
        run([PY, "bert/train.py", "--decoder", "linear", "--max_len", "256",
             "--epochs", "3", "--run_name", name, "--bert_path", str(MODELS / backbone)],
            f"run_{name}.log", OUR_DISC / f"{name}.json", force)
    for backbone, name in runs:
        # 800 句 × 21 epoch ≈ 同优化步数；训练后在全量 dev 上复评
        ok = run([PY, "bert/train.py", "--data_dir", str(CODE / "data" / "sample"),
                  "--decoder", "linear", "--max_len", "256", "--epochs", "21",
                  "--run_name", f"{name}_sample", "--bert_path", str(MODELS / backbone)],
                 f"run_{name}_sample.log", OUR_DISC / f"{name}_sample.json", force)
        if ok:
            run([PY, "bert/evaluate.py", "--ckpt", f"outputs/checkpoints/{name}_sample.pt",
                 "--run_name", f"{name}_sample_full", "--data_dir", str(CODE / "data" / "processed")],
                f"eval_{name}_sample_full.log", OUR_DISC / f"{name}_sample_full.json", force)


def llms(force):
    runs = [
        ("Qwen2-0.5B-Instruct", QWEN2, "qwen2_zero_n60"),
        ("Qwen3-0.6B", str(MODELS / "Qwen3-0.6B"), "qwen3_zero_n60"),
        ("Qwen2.5-1.5B-Instruct", str(MODELS / "Qwen2.5-1.5B-Instruct"), "qwen25_15b_zero_n60"),
    ]
    for label, path, tag in runs:
        run([PY, "llm_zero_shot/llm_extract.py", "--backend", "local", "--mode", "zero",
             "--n", "60", "--tag", tag, "--model_path", path, "--gen_batch", "8"],
            f"llm_{tag}.log", OUR_LLM / f"{tag}.json", force)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["encoders", "llm"], default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.only in (None, "encoders"):
        encoders(args.force)
    if args.only in (None, "llm"):
        llms(args.force)
    subprocess.run([PY, "make_summary.py"], cwd=ROOT, check=False)
    subprocess.run([PY, "code/eval/significance.py"], cwd=ROOT, check=False)
    print("\n完成：results/summary.md + significance.md + fig_*.png")


if __name__ == "__main__":
    main()
