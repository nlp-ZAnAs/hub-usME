# -*- coding: utf-8 -*-
"""
week07 作业 2：badcase 归类（边界 / 角色串味 / 触发词歧义）

用法（在 week7 课件 code 目录之外也能跑，路径全用绝对）：
    python analyze_badcase.py --pred <pred.jsonl> --gold <dev_extract.jsonl> --n 10 --out badcase.md

对齐键：id|sent_id（DuEE-Fin 的 id 是文档 id，不唯一）
判定顺序：漏抽 → 类型错 → 触发词（歧义/边界）→ 论元（角色串味/边界/多余/缺失）
"边界" = 预测文本与 gold 文本互为子串/超串（或起止相邻重叠），差的是 span 边界不是角色
"串味" = 该角色填的文本，恰好是同一句里另一个角色的 gold 文本
"""
import argparse
import json
from collections import Counter


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def gold_events(row):
    if "event" in row and isinstance(row["event"], dict):
        return [row["event"]]
    return row.get("events") or []


def classify(gold, pred, text):
    """返回 (类别, 说明)；类别为空表示该事件完全正确"""
    if not pred:
        return "漏抽", "模型没输出事件"

    g_type = gold.get("type", "")
    p_type = pred.get("type", "")
    if g_type != p_type:
        return "类型错", f"gold={g_type} pred={p_type}"

    g_trg = (gold.get("trigger") or {}).get("text", "")
    p_trg = (pred.get("trigger") or {}).get("text", "")
    if g_trg != p_trg:
        if p_trg and g_trg and (p_trg in g_trg or g_trg in p_trg):
            return "边界错(触发词)", f"gold=「{g_trg}」 pred=「{p_trg}」"
        return "触发词歧义", f"gold=「{g_trg}」 pred=「{p_trg}」"

    g_args = {a["role"]: a["text"] for a in gold.get("args") or []}
    p_args = {a["role"]: a["text"] for a in pred.get("args") or []}
    g_texts = set(g_args.values())

    for role, p_text in p_args.items():
        if role not in g_args:
            # 角色本身不该出现：可能是同类型内部的角色串味（填了别的角色）
            owner = [r for r, t in g_args.items() if t == p_text]
            if owner:
                return "角色串味", f"「{role}」填了本属于「{owner[0]}」的「{p_text}」"
            return "多余角色", f"多出「{role}」=「{p_text}」"
        g_text = g_args[role]
        if p_text == g_text:
            continue
        if p_text and g_text and (p_text in g_text or g_text in p_text):
            return "边界错(论元)", f"「{role}」gold=「{g_text}」 pred=「{p_text}」"
        owner = [r for r, t in g_args.items() if t == p_text and r != role]
        if owner:
            return "角色串味", f"「{role}」填了「{owner[0]}」的「{p_text}」"
        if p_text not in text:
            return "幻觉", f"「{role}」=「{p_text}」不在原文"
        return "论元错配", f"「{role}」gold=「{g_text}」 pred=「{p_text}」"

    missing = [r for r in g_args if r not in p_args]
    if missing:
        return "缺角色", "少抽：" + "、".join(missing)
    return "", ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--out", default="")
    ap.add_argument("--offset", type=int, default=0, help="从第几条错例开始取（保证可复现）")
    args = ap.parse_args()

    preds = {f"{r['id']}|{r['sent_id']}": r for r in load_jsonl(args.pred)}
    golds = {f"{r['id']}|{r['sent_id']}": r for r in load_jsonl(args.gold)}

    rows, stat = [], Counter()
    for key, grow in golds.items():
        text = grow.get("text", "")
        gevents = gold_events(grow)
        pevents = (preds.get(key) or {}).get("events") or []
        for i, ge in enumerate(gevents):
            pe = pevents[i] if i < len(pevents) else None
            cat, note = classify(ge, pe, text)
            if not cat:
                continue
            stat[cat] += 1
            rows.append({"key": key, "cat": cat, "note": note, "text": text,
                         "gold": ge, "pred": pe})

    total = sum(len(gold_events(r)) for r in golds.values())
    lines = [f"- 评测句数 {len(golds)}，gold 事件 {total}，错误事件 {len(rows)}"]
    for k, v in stat.most_common():
        lines.append(f"- {k}: {v}（{v/max(len(rows),1):.1%}）")

    picked = rows[args.offset: args.offset + args.n]
    if args.out:
        md = [f"# badcase 归类（{args.pred.split(chr(92))[-1]}）", ""] + lines + [""]
        md.append("| # | 类别 | 原文（截断） | gold | pred | 判定依据 |")
        md.append("|---|---|---|---|---|---|")
        for i, r in enumerate(picked, 1):
            t = r["text"].replace("|", "\\|")[:70]
            g = f"{r['gold'].get('type')}/{(r['gold'].get('trigger') or {}).get('text','')}"
            p = (f"{r['pred'].get('type')}/{(r['pred'].get('trigger') or {}).get('text','')}"
                 if r["pred"] else "（空）")
            md.append(f"| {i} | {r['cat']} | {t} | {g} | {p} | {r['note']} |")
        with open(args.out, "w", encoding="utf-8") as f:
            f.write("\n".join(md) + "\n")
        print("\n".join(lines))
        print(f"\n已写 {args.out}（取第 {args.offset}–{args.offset+len(picked)} 条错例）")
    else:
        print("\n".join(lines))
        for i, r in enumerate(picked, 1):
            print(f"{i}. [{r['cat']}] {r['note']}  | {r['text'][:60]}")


if __name__ == "__main__":
    main()
