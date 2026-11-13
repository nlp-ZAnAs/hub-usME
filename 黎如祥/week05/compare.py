# -*- coding: utf-8 -*-
"""
Week05 作业：把 base / SFT / 消融 的评估结果汇总成 markdown。

    python compare.py
    python compare.py --runs base=outputs/eval_base.json sft=outputs/eval_sft.json
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent

ROWS = [
    ("验证 loss（只在回答段）", "val_loss", "{:.4f}", True),
    ("验证 PPL", "val_ppl", "{:.2f}", True),
    ("下一 token 准确率", "val_token_acc", "{:.2%}", False),
    ("prompt 段 CE（提问）", "prompt_ce", "{:.4f}", True),
    ("response 段 CE（回答）", "response_ce", "{:.4f}", True),
    ("完全匹配率", "exact_match", "{:.2%}", False),
    ("字符级 ROUGE-L", "rouge_l", "{:.4f}", False),
    ("数字答案匹配率", "num_match_rate", "{:.2%}", False),
    ("平均回答长度（字）", "avg_len_chars", "{:.1f}", None),
    ("平均生成长度（token）", "avg_len_tokens", "{:.1f}", None),
    ("im_end 停止率", "im_end_rate", "{:.2%}", False),
    ("复述问题比例", "echo_instruction_rate", "{:.2%}", True),
]


def fmt(v, f):
    if v is None:
        return "—"
    try:
        return f.format(v)
    except Exception:
        return str(v)


def cut(s, n=70):
    s = s.replace("\n", " ").replace("|", "\\|")
    return s if len(s) <= n else s[:n] + "…"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=None,
                    help="name=path 形式，默认 base/sft/nomask")
    ap.add_argument("--out", default=str(ROOT / "outputs" / "eval_compare.md"))
    ap.add_argument("--n-examples", type=int, default=10)
    args = ap.parse_args()

    if args.runs:
        pairs = [r.split("=", 1) for r in args.runs]
    else:
        pairs = []
        for name in ["base", "sft", "nomask"]:
            p = ROOT / "outputs" / f"eval_{name}.json"
            if p.exists():
                pairs.append((name, str(p)))

    data = {}
    for name, path in pairs:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        data[name] = payload

    names = [n for n, _ in pairs]
    display = {"base": "base", "sft": "SFT (mask)", "nomask": "SFT (no mask)"}
    lines = ["# SFT 评估：base vs SFT", ""]
    header = "| 指标 | " + " | ".join(display.get(n, n) for n in names) + " |"
    lines.append(header)
    lines.append("|" + "---|" * (len(names) + 1))
    for label, key, f, lower_better in ROWS:
        cells = [fmt(data[n]["metrics"].get(key), f) for n in names]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines.append("")

    sft = data.get("sft")
    base = data.get("base")
    if base and sft:
        b = {d["instruction"]: d for d in base["details"]}
        s = {d["instruction"]: d for d in sft["details"]}
        common = [k for k in b if k in s]
        better = sum(1 for k in common if s[k]["rouge_l"] > b[k]["rouge_l"])
        worse = sum(1 for k in common if s[k]["rouge_l"] < b[k]["rouge_l"])
        same = len(common) - better - worse
        nb = sum(1 for k in common if b[k]["num_match"] is True)
        ns = sum(1 for k in common if s[k]["num_match"] is True)
        lines += [
            "## 逐条对比（200 条验证集）",
            "",
            f"- ROUGE-L：SFT 更好 {better} 条，更差 {worse} 条，持平 {same} 条",
            f"- 数字答案完全匹配：base {nb} 条 → SFT {ns} 条",
            "",
        ]

        lines += ["## 典型样本", ""]
        idx = [round(i * (len(common) - 1) / max(args.n_examples - 1, 1))
               for i in range(args.n_examples)]
        for j, k in enumerate(idx, 1):
            instr = common[k]
            d = s[instr]
            lines += [
                f"### {j}. {cut(instr, 100)}",
                "",
                f"- **参考答案**：{cut(d['reference'], 90)}",
                f"- **base**：{cut(b[instr]['prediction'], 90)}",
                f"- **SFT**：{cut(d['prediction'], 90)}",
                f"- ROUGE-L：{b[instr]['rouge_l']:.3f} → {d['rouge_l']:.3f}",
                "",
            ]

    out = Path(args.out)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
