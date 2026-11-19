# -*- coding: utf-8 -*-
"""
方法四：大模型零样本（不训练，L3 对照）

同一套「给类别清单 + 只输出标签」的 prompt，两个数据集各写一版：
  - TNEWS：15 类，类别词很抽象，0.6B 模型容易塌缩到少数几类
  - ChnSentiCorp：2 类情感，标签空间小得多，zero-shot 通常明显更好

Qwen3 必须传 enable_thinking=False：否则先吐 <think>... 推理链，
「只输出类别名」的格式会被打乱，max_new_tokens=8 时还会直接截断成空回复。

用法：
  python code/llm/run_zero_shot.py --dataset tnews
  python code/llm/run_zero_shot.py --dataset chnsenticorp --batch_size 16
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.config import DATASETS, QWEN_DIR, model_path, result_path
from common.data import eval_samples, label_names
from common.metrics import classification_report, pred_distribution

IM_END = 151645

SYSTEM_TNEWS = ("你是一个新闻标题分类助手。请把给定的新闻标题归到以下类别之一，"
                "只输出类别名称，不要输出任何解释。\n可选类别：{labels}")

SYSTEM_SENTI = ("你是一个中文评论情感分类助手。判断给定评论是正面还是负面，"
                "只输出「正面」或「负面」，不要输出任何解释。")


def build_messages(dataset, text):
    if dataset == "tnews":
        return [{"role": "system", "content": SYSTEM_TNEWS.format(labels="、".join(label_names("tnews")))},
                {"role": "user", "content": text}]
    return [{"role": "system", "content": SYSTEM_SENTI},
            {"role": "user", "content": text}]


def parse_label(dataset, raw):
    text = raw.strip()
    labels = label_names(dataset)
    if dataset == "tnews":
        hits = [name for name in labels if name in text]
        if not hits:
            return -1
        return labels.index(max(hits, key=len))
    # 情感：先判负面，避免「不正面」之类被误判
    if "负面" in text or "差评" in text or "negative" in text.lower():
        return labels.index("负面")
    if "正面" in text or "好评" in text or "positive" in text.lower():
        return labels.index("正面")
    return -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DATASETS))
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_new_tokens", type=int, default=8)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out_path = result_path(args.dataset, "qwen3_zero_shot")
    if out_path.exists() and not args.force:
        print(f"[skip] {out_path} 已存在")
        return

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_dir = model_path(QWEN_DIR)
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir), dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device).eval()

    samples, meta = eval_samples(args.dataset, 500, 42)
    labels = label_names(args.dataset)

    texts = []
    for s in samples:
        texts.append(tok.apply_chat_template(
            build_messages(args.dataset, s["sentence"]), tokenize=False,
            add_generation_prompt=True, enable_thinking=False))

    preds, raws = [], []
    t0 = time.time()
    for start in range(0, len(texts), args.batch_size):
        chunk = texts[start:start + args.batch_size]
        enc = tok(chunk, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        with torch.inference_mode():
            out = model.generate(**enc, max_new_tokens=args.max_new_tokens,
                                 do_sample=False, pad_token_id=tok.pad_token_id,
                                 eos_token_id=[IM_END, tok.eos_token_id])
        new = out[:, enc["input_ids"].shape[1]:]
        for row in new:
            raw = tok.decode(row, skip_special_tokens=True)
            raws.append(raw.strip())
            preds.append(parse_label(args.dataset, raw))
        done = min(start + args.batch_size, len(texts))
        if start % (args.batch_size * 5) == 0:
            print(f"  {done}/{len(texts)} {time.time()-t0:.1f}s", flush=True)

    elapsed = time.time() - t0
    y_true = [s["label"] for s in samples]
    report = classification_report(y_true, preds, len(labels))
    report["method"] = "qwen3_zero_shot"
    report["pred_distribution"] = pred_distribution(preds, len(labels))

    result = {
        "method": "qwen3_zero_shot",
        "display": "Qwen3-0.6B zero-shot",
        "family": "llm_zero_shot",
        "dataset": args.dataset,
        "meta": meta,
        "report": report,
        "cost": {"train_seconds": 0.0, "per_item_ms": round(elapsed / len(samples) * 1000, 2),
                 "params": sum(p.numel() for p in model.parameters()), "peak_mem_gb": None},
        "preds": [{"text": s["sentence"], "true": int(s["label"]), "pred": int(p), "prob": None,
                   "raw": r} for s, p, r in zip(samples, preds, raws)],
    }
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[{args.dataset}] zero-shot acc={report['acc_incl_unparsed']} "
          f"macro_f1={report['macro_f1']} used={report['pred_distribution']['used_classes']}/{len(labels)} "
          f"unparsed={report['unparsed']} {result['cost']['per_item_ms']}ms/item")
    print(f"→ {out_path}")


if __name__ == "__main__":
    main()
