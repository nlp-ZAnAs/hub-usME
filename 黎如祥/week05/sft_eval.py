# -*- coding: utf-8 -*-
"""
Week05 作业：SFT 效果评估

在 200 条 held-out 验证集上比较：
  - response-only 的 loss / PPL / next-token 准确率
  - 生成质量：exact match、字符级 ROUGE-L、数字答案匹配、长度、是否回显问题、是否学会 <|im_end|> 停

用法：
    python sft_eval.py --model D:\\Env\\models\\Qwen2-0.5B --label base
    python sft_eval.py --model D:\\Env\\models\\Qwen2-0.5B --adapter outputs\\lora_r16\\best --label sft
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from sft_train import ROOT, PadCollator, SFTDataset, evaluate, make_loader, read_jsonl

IM_END = 151645


@torch.no_grad()
def loss_split(model, ds, pad_id, device, dtype, batch_size=16):
    """把验证 loss 拆成 prompt 段 / response 段，看 SFT 到底把哪一段学好了。"""
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        collate_fn=PadCollator(pad_id))
    tot = {"prompt": [0.0, 0], "response": [0.0, 0]}
    offset = 0
    for batch in loader:
        b = len(batch["input_ids"])
        plens = [ds.items[offset + i]["prompt_len"] for i in range(b)]
        offset += b
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast("cuda", dtype=dtype, enabled=(dtype != torch.float32)):
            out = model(**batch)
        logits = out.logits[:, :-1].float()
        tgt = batch["input_ids"][:, 1:]
        valid = batch["attention_mask"][:, 1:].bool()
        ce = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), tgt.reshape(-1), reduction="none"
        ).view(tgt.shape)
        for i, p in enumerate(plens):
            pos = torch.arange(tgt.size(1), device=device)
            pm = valid[i] & (pos < p - 1)
            rm = valid[i] & (pos >= p - 1)
            for key, mask in (("prompt", pm), ("response", rm)):
                n = int(mask.sum())
                if n:
                    tot[key][0] += float(ce[i][mask].sum())
                    tot[key][1] += n
    return (tot["prompt"][0] / max(tot["prompt"][1], 1),
            tot["response"][0] / max(tot["response"][1], 1))


def build_prompt(tok, instruction):
    messages = [{"role": "user", "content": instruction}]
    return tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def norm(s):
    return re.sub(r"\s+", "", s)


def lcs_len(a, b):
    """字符级最长公共子序列长度（两个短句，O(n*m) 足够）"""
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for ca in a:
        cur = [0]
        for j, cb in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if ca == cb else max(prev[j], cur[j - 1]))
        prev = cur
    return prev[-1]


def rouge_l(pred, ref):
    p, r = norm(pred), norm(ref)
    if not p or not r:
        return 0.0
    l = lcs_len(p, r)
    prec, rec = l / len(p), l / len(r)
    return 0.0 if l == 0 else 2 * prec * rec / (prec + rec)


NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def num_set(s):
    return {round(float(x), 4) for x in NUM_RE.findall(s)}


def numbers_match(pred, ref):
    a, b = num_set(pred), num_set(ref)
    return bool(b) and a == b


@torch.no_grad()
def generate_all(model, tok, rows, device, batch_size, max_new_tokens):
    tok.padding_side = "left"
    preds = []
    t0 = time.time()
    for start in range(0, len(rows), batch_size):
        chunk = rows[start:start + batch_size]
        texts = [build_prompt(tok, r["instruction"]) for r in chunk]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False)
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=[IM_END, tok.eos_token_id],
            pad_token_id=tok.pad_token_id,
        )
        new = out[:, enc["input_ids"].shape[1]:]
        for i in range(len(chunk)):
            ids = new[i].tolist()
            # 截到第一个 EOS
            stop = len(ids)
            for j, t in enumerate(ids):
                if t in (IM_END, tok.eos_token_id):
                    stop = j
                    break
            text = tok.decode(ids[:stop], skip_special_tokens=True).strip()
            preds.append({"text": text, "n_tokens": stop,
                          "im_end": (IM_END in ids[:stop + 1])})
        done = min(start + batch_size, len(rows))
        print(f"  generate {done}/{len(rows)}  {time.time() - t0:.1f}s", flush=True)
    tok.padding_side = "right"
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("SFT_MODEL", "Qwen/Qwen2-0.5B"))
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--val", default=str(ROOT / "data" / "sft_val.jsonl"))
    ap.add_argument("--label", default="base")
    ap.add_argument("--out-dir", default=str(ROOT / "outputs"))
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    rows = read_jsonl(args.val)
    ds = SFTDataset(rows, tok, 512, mask_prompt=True)
    loader = make_loader(ds, args.batch_size, False, tok.pad_token_id)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
    model.to(device)
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
        print(f"loaded adapter: {args.adapter}")
    model.eval()
    model.config.use_cache = True

    print(f"[{args.label}] eval loss on {len(rows)} val examples ...", flush=True)
    loss, ppl, acc = evaluate(model, loader, device, dtype)
    prompt_loss, resp_loss = loss_split(model, ds, tok.pad_token_id, device, dtype,
                                        args.batch_size)
    print(f"[{args.label}] loss={loss:.4f} ppl={ppl:.2f} acc={acc:.4f} "
          f"| prompt_ce={prompt_loss:.4f} resp_ce={resp_loss:.4f}", flush=True)

    preds = generate_all(model, tok, rows, device, args.batch_size, args.max_new_tokens)

    details = []
    for r, p in zip(rows, preds):
        out_text = p["text"]
        instr = r["instruction"]
        details.append({
            "instruction": instr,
            "reference": r["output"],
            "prediction": out_text,
            "rouge_l": round(rouge_l(out_text, r["output"]), 4),
            "exact": norm(out_text) == norm(r["output"]),
            "num_match": numbers_match(out_text, r["output"]) if num_set(r["output"]) else None,
            "len_chars": len(out_text),
            "n_tokens": p["n_tokens"],
            "im_end": p["im_end"],
            "echo_instruction": norm(instr)[:12] in norm(out_text) if len(norm(instr)) >= 12 else False,
            "empty": len(out_text) == 0,
        })

    n = len(details)
    num_rows = [d for d in details if d["num_match"] is not None]
    metrics = {
        "label": args.label,
        "model": args.model,
        "adapter": args.adapter,
        "n": n,
        "val_loss": round(loss, 4),
        "val_ppl": round(ppl, 4),
        "val_token_acc": round(acc, 4),
        "prompt_ce": round(prompt_loss, 4),
        "response_ce": round(resp_loss, 4),
        "exact_match": round(sum(d["exact"] for d in details) / n, 4),
        "rouge_l": round(sum(d["rouge_l"] for d in details) / n, 4),
        "num_match_rate": round(sum(d["num_match"] for d in num_rows) / len(num_rows), 4) if num_rows else None,
        "n_num_examples": len(num_rows),
        "avg_len_chars": round(sum(d["len_chars"] for d in details) / n, 2),
        "avg_len_tokens": round(sum(d["n_tokens"] for d in details) / n, 2),
        "im_end_rate": round(sum(d["im_end"] for d in details) / n, 4),
        "echo_instruction_rate": round(sum(d["echo_instruction"] for d in details) / n, 4),
        "empty_rate": round(sum(d["empty"] for d in details) / n, 4),
    }

    (out_dir / f"eval_{args.label}.json").write_text(
        json.dumps({"metrics": metrics, "details": details}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"saved {out_dir / f'eval_{args.label}.json'}")


if __name__ == "__main__":
    main()
