"""
统计 train/val 的 token 长度分布，为 BERT 微调选 seq_len

教学重点：
  1. seq_len 不是拍脑袋定的：取太小 → 大量样本被截断，信息丢失；
     取太大 → padding 浪费算力（attention 是 O(L²)），训练慢好几倍。
  2. 常用做法：选能覆盖 ~95% 样本的长度（P95 附近），再对照
     64/128/256/512 这些标准档位取整。
  3. 用训练用的同一个 tokenizer 统计——bert-base-chinese 里
     1 个汉字 ≈ 1 个 token，但英文单词/数字/标点切法不同，别用字符数近似。

用法：
  cd week06 && python bert/token_stats.py [--model bert-base-chinese]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.data import load_splits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="bert-base-chinese")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)

    train, val = load_splits()
    for name, data in [("train", train), ("val", val)]:
        lengths = [len(tok.encode(s["text"] or "")) for s in data]
        a = np.array(lengths)

        print(f"\n===== {name}（{len(a)} 条，{args.model}）=====")
        print(f"mean={a.mean():.1f}  std={a.std():.1f}  min={a.min()}  max={a.max()}")
        for p in [50, 90, 95, 99]:
            print(f"  P{p:<3} = {np.percentile(a, p):.0f}")

        print("\n各 seq_len 档位的覆盖率（完全容纳、不截断的比例）:")
        for L in [32, 64, 128, 256, 512]:
            cover = (a <= L).mean()
            trunc = (a > L).mean()
            bar = "#" * int(cover * 40)
            print(f"  {L:>3}  覆盖 {cover:6.1%}  截断 {trunc:5.1%}  {bar}")


if __name__ == "__main__":
    main()
