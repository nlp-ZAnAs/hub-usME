# -*- coding: utf-8 -*-
"""
eval_sft.py — 本周作业的「证据」文件：base 与 SFT 并排对照

同一批问题（取自 sft_val.jsonl，训练时没出现过），两个模型各答一次：

    base_sft.pt ：只在问句语料上预训练过   → 预期「接着列问题，不回答」
    sft.pt      ：SFT 之后                 → 预期「给出答句式」

指标说明（字符级 3M 小模型不谈「准确率」，只谈「行为对不对」）：
    有答率     ：输出非空、不是一串问句
    数字命中率 ：参考答案里的数字是否全部出现在生成里（这是「算对没算对」的代理指标）
    完全一致率 ：生成与参考答案一字不差（模板类问题才可能命中）

跑法：
    python -X utf8 eval_sft.py --n 20
"""

import argparse
import difflib
import os
import re

import torch

from model import generate, load_model
from sft_data import load_jsonl

HERE = os.path.dirname(os.path.abspath(__file__))
NUM_RE = re.compile(r"\d+(?:\.\d+)?")


def numbers(s):
    return NUM_RE.findall(s)


def norm(s):
    return re.sub(r"\s+", "", s)


def is_question_stack(gen):
    """
    第四周那个 base 模型的招牌行为：被问之后接着列问题。
    特征就是两个以上问号，或者出现了换行（它把一行答案变成了一列问题）。
    """
    return (gen.count("？") + gen.count("?")) >= 2 or gen.count("\n") >= 1


def similarity(gen, ref):
    """与参考答案的字符级相似度（difflib 的 SequenceMatcher，0~1）。
    这个指标专门用来回答「SFT 到底学到了什么」：句式对了、只有数字错，相似度会很高。"""
    return difflib.SequenceMatcher(None, norm(gen), norm(ref)).ratio()


def number_hit(gen, ref):
    """参考答案里的数字是否全部出现在生成里。参考答案没有数字时返回 None（不计分）。"""
    ref_nums = numbers(ref)
    if not ref_nums:
        return None
    gen_nums = numbers(gen)
    return all(n in gen_nums for n in ref_nums)


def exact_match(gen, ref):
    return norm(gen) == norm(ref)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=r"D:\BaiduNetdiskDownload\week5-大模型初探")
    ap.add_argument("--base", default=os.path.join(HERE, "base_sft.pt"))
    ap.add_argument("--sft", default=os.path.join(HERE, "sft.pt"))
    ap.add_argument("--n", type=int, default=20, help="取验证集前 n 条做对照")
    ap.add_argument("--max_new", type=int, default=40)
    ap.add_argument("--out", default=os.path.join(HERE, "eval_out.txt"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model, vocab, _ = load_model(args.base, device)
    sft_model, _, _ = load_model(args.sft, device)
    print(f"[device] {device}  比较 {os.path.basename(args.base)} vs {os.path.basename(args.sft)}")

    rows = load_jsonl(os.path.join(args.data_dir, "sft_val.jsonl"))[:args.n]

    lines = []
    stat = {"base": {"stack": 0, "sim": 0.0, "hit": 0, "hit_n": 0, "exact": 0},
            "sft":  {"stack": 0, "sim": 0.0, "hit": 0, "hit_n": 0, "exact": 0}}

    def emit(s):
        print(s)
        lines.append(s)

    emit("=" * 78)
    emit(f"SFT 前后对照（验证集 {len(rows)} 条，训练时没见过）")
    emit("=" * 78)

    for i, r in enumerate(rows, 1):
        prompt = "问：" + r["instruction"] + "\n答："
        g_base = generate(base_model, vocab, prompt, args.max_new, device=device)
        g_sft = generate(sft_model, vocab, prompt, args.max_new, device=device)
        ref = r["output"]

        emit(f"\n[{i}] 问：{r['instruction']}")
        emit(f"    参考答：{ref}")
        emit(f"    base → {g_base!r}")
        emit(f"    sft  → {g_sft!r}")

        for name, g in (("base", g_base), ("sft", g_sft)):
            s = stat[name]
            s["stack"] += int(is_question_stack(g))
            s["sim"] += similarity(g, ref)
            s["exact"] += int(exact_match(g, ref))
            h = number_hit(g, ref)
            if h is not None:
                s["hit_n"] += 1
                s["hit"] += int(h)

    # ── 汇总 ──
    n = len(rows)
    emit("\n" + "=" * 78)
    emit("汇总")
    emit("=" * 78)
    emit(f"{'':<16}{'问句堆叠率':>10}{'字符相似度':>12}{'数字命中':>10}{'完全一致':>8}")
    for name, label in (("base", "SFT 之前(base)"), ("sft", "SFT 之后")):
        s = stat[name]
        hit_str = f"{s['hit']}/{s['hit_n']}"
        emit(f"{label:<16}{s['stack']/max(n,1)*100:>9.1f}%{s['sim']/max(n,1):>12.2f}"
             f"{hit_str:>10}{s['exact']/max(n,1)*100:>7.1f}%")
    emit("")
    emit("读法：")
    emit("  问句堆叠率 → base 的招牌行为（被问之后接着列问题），SFT 后应明显下降；")
    emit("  字符相似度 → 句式像不像参考答案，SFT 后应上升（数字错不影响这个指标）；")
    emit("  数字命中   → 数值对不对，SFT 后仍会很低：「SFT 给形式，不给算术能力」；")
    emit("  完全一致   → 一字不差，只有模板完全记住的问题才可能命中。")

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n[eval] 明细已写入 {args.out}")


if __name__ == "__main__":
    main()
