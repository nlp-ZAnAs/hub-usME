# -*- coding: utf-8 -*-
"""
把下载的 ChnSentiCorp parquet 转成本项目统一的 json 格式。

源：https://hf-mirror.com/datasets/lansinuote/ChnSentiCorp
  data/train-*.parquet      9,600 条
  data/validation-*.parquet 1,200 条
  data/test-*.parquet       1,200 条（本作业不用，留作最终盲测）

输出：data/chnsenticorp/{train,val}.json，每条 {"sentence", "label"}，0=负面，1=正面。

用法：
  python code/tools/prepare_data.py --src D:\\Env\\data\\chnsenticorp
"""

import argparse
import json
from pathlib import Path

import pandas as pd

CODE_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = CODE_ROOT.parent / "data" / "chnsenticorp"


def convert(src_dir: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    mapping = {"train": "train.json", "validation": "val.json"}
    for split, fname in mapping.items():
        files = sorted(src_dir.glob(f"{split}-*.parquet"))
        if not files:
            raise FileNotFoundError(f"{src_dir} 下没有 {split}-*.parquet")
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        text_col = "text" if "text" in df.columns else df.columns[0]
        rows = [{"sentence": str(t).strip(), "label": int(y)}
                for t, y in zip(df[text_col], df["label"])]
        rows = [r for r in rows if r["sentence"]]
        (out_dir / fname).write_text(
            json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{split}: {len(rows)} -> {out_dir / fname}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=r"D:\Env\data\chnsenticorp")
    ap.add_argument("--out", default=str(DATA_DIR))
    args = ap.parse_args()
    convert(Path(args.src), Path(args.out))


if __name__ == "__main__":
    main()
