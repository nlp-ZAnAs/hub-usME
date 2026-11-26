"""
rule_extract.py — L0 规则基线（零训练）：触发词白名单 + 正则论元（CEC）。

为什么先做规则（而不是直接上模型）：
  规则基线是最诚实的对照组 —— 不学任何参数、只靠词典和正则能拿到多少分，
  后面 BiLSTM-CRF / BERT 涨了多少点才有意义。它的失败模式也最清楚：
    - 触发词没收录        → 整句漏抽（召回天花板卡在词典覆盖率上）
    - 语义角色无固定形式  → 肇事者/受害者/袭击者/中毒者 正则根本抽不出来（只能靠模型）
    - 时间/地点/伤亡/车辆 有规律 → 正则反而很强

做法：
  1. 触发词：conf/trigger_whitelist.json（按事件类型给出触发词），**最长优先**匹配
     （否则「交通事故」会被「事故」抢走、「集体食物中毒」会被「中毒」抢走）。
  2. 论元：给定事件类型（由触发词给出），用正则抽 时间 / 地点 / 伤亡人数(中毒者) / 涉事车辆。
  3. 评测：写统一种子的预测 JSONL，交给 common/evaluate_events.py 出指标
     —— 和后面的模型同一口径（trigger / argument / event 三级 + type_macro）。

用法：
  python rule/rule_extract.py                 # 默认在 test 上评测（也可 --split dev/train）
  python rule/rule_extract.py --split dev
  python rule/rule_extract.py --show 20       # 额外打印 20 条 gold vs pred 样例
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import re
import sys
import json
import argparse
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # week07/
sys.path.insert(0, str(ROOT / "common"))

from event_data import load_schema, load_extract_split, make_pred_row, DATA_DIR  # noqa: E402
from eval_utils import evaluate_predictions, METRIC_DIR                         # noqa: E402
from log_utils import EvalLogger                                                # noqa: E402

CONF_DIR = ROOT / "data" / "conf"
WHITELIST_PATH = CONF_DIR / "trigger_whitelist.json"


# ─────────────────────────── 1. 触发词 ───────────────────────────
def load_whitelist() -> dict:
    """触发词 -> 事件类型（同一词出现在多类型时取第一个出现的类型）。"""
    with open(WHITELIST_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    word2type = {}
    for et, words in raw.items():
        if et.startswith("_"):
            continue
        for w in words:
            word2type.setdefault(w, et)
    return word2type


def find_trigger(text: str, word2type: dict):
    """最长优先扫触发词，返回 (类型, 触发词, start, end)；没命中返回 None。

    最长优先是因为触发词存在包含关系：「交通事故」里含「事故」，
    「集体食物中毒」里含「中毒」。短词先命中会把事件类型判错。
    """
    best = None  # (len(word), -pos) 越大越优先
    for word, et in word2type.items():
        pos = text.find(word)
        if pos < 0:
            continue
        key = (len(word), -pos)
        if best is None or key > best[0]:
            best = (key, et, word, pos)
    if best is None:
        return None
    _, et, word, pos = best
    return et, word, pos, pos + len(word)


# ─────────────────────────── 2. 论元正则 ───────────────────────────
_CN_NUM = r"[一二三四五六七八九十百两]{1,4}"
_PERIOD = r"(?:凌晨|上午|下午|晚上|傍晚|中午|晚)"

# 时间：长格式在前；finditer 非重叠匹配
TIME_RE = re.compile(
    r"\d{4}年\d{1,2}月\d{1,2}日(?:" + _PERIOD + r"?\d{1,2}[时点](?:\d{1,2}分)?(?:许)?)?"
    r"|" + _CN_NUM + r"月" + _CN_NUM + r"日(?:" + _CN_NUM + r"[时点](?:" + _CN_NUM + r"分)?)?"
    r"|\d{1,2}月\d{1,2}日(?:" + _PERIOD + r"?\d{1,2}[时点](?:\d{1,2}分)?(?:许)?)?"
    r"|\d{1,2}日" + _PERIOD + r"?(?:\d{1,2}[时点](?:\d{1,2}分)?(?:许)?)?"
    r"|" + _PERIOD + r"?\d{1,2}[时点](?:\d{1,2}分)?(?:许)?"
    r"|(?:昨日|昨天|今日|今天|当天|当日|近日|日前|目前|随后)" + _PERIOD + r"?(?:\d{1,2}[时点](?:\d{1,2}分)?)?"
)

# 地点：禁止以常见虚词/数字/标点开头，前缀 2~8 字，配合地名后缀
_LOC_BAD_START = "据在于至和等次该这此对了的是被由从向因当第及全明有两几众多各本前者或内外周报提位"
LOC_RE = re.compile(
    r"[^" + _LOC_BAD_START + r"\d，。、；：\s]{2,8}?"
    r"(?:省|市|县|区|镇|乡|村|州|自治区|地区|路段|高速|隧道|路口|交叉口|大桥|附近|门口|岛|交界处|境内)"
)

# 伤亡：gold 只取「数量+人/名」本身，附近需有死伤语境
CASUALTY_NUM_RE = re.compile(r"\d+[多余]?[人名]|" + _CN_NUM + r"人")
CASUALTY_CTX = ("死亡", "死", "遇难", "丧生", "身亡", "受伤", "重伤", "轻伤", "伤", "伤亡", "中毒")

VEHICLE_RE = re.compile(
    r"(?:一辆|两辆|数辆)?[^，。、；：\s与和及]{0,4}?"
    r"(?:大货车|小货车|货车|大客车|客车|面包车|轿车|摩托车|卡车|搅拌车|罐车|大巴|中巴|校车|出租车|渣土车|私家车|微型车|奔驰|斯柯达|小车)"
)


def extract_args(text: str, event_type: str, allowed_roles: set) -> list:
    """按类型抽论元；只保留该事件类型 schema 里存在的角色。"""
    args, seen = [], set()

    def add(role, s, e):
        t = text[s:e].strip()
        if t and role in allowed_roles and (role, t) not in seen:
            seen.add((role, t))
            args.append({"role": role, "text": t, "start": s, "end": e})

    # 时间
    for m in TIME_RE.finditer(text):
        if m.group(0):
            add("时间", m.start(), m.end())

    # 地点（合并相邻命中，如「四川省攀枝花市」）
    locs = [(m.start(), m.end()) for m in LOC_RE.finditer(text)]
    merged = []
    for s, e in locs:
        if merged and merged[-1][1] == s:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    for s, e in merged:
        add("地点", s, e)

    # 伤亡人数 / 中毒者（食物中毒用 中毒者）
    victim_role = "中毒者" if event_type == "食物中毒" else "伤亡人数"
    for m in CASUALTY_NUM_RE.finditer(text):
        ctx = text[max(0, m.start() - 8):min(len(text), m.end() + 8)]
        if any(k in ctx for k in CASUALTY_CTX):
            add(victim_role, m.start(), m.end())

    # 涉事车辆（仅交通事故）
    if event_type == "交通事故":
        for m in VEHICLE_RE.finditer(text):
            add("涉事车辆", m.start(), m.end())

    return args


def extract_one(text: str, schema: dict, word2type: dict):
    """一句抽一个事件；抽不到返回 None。"""
    hit = find_trigger(text, word2type)
    if hit is None:
        return None
    et, word, s, e = hit
    allowed = set(schema.get(et, []))
    return {
        "type": et,
        "trigger": {"text": word, "start": s, "end": e},
        "args": extract_args(text, et, allowed),
    }


# ─────────────────────────── 3. 失败分析（教学用）───────────────────────────
def analyze_failures(gold_rows: list, pred_rows: list) -> dict:
    """把错误拆成互斥几类：no_trigger / wrong_type / wrong_trigger / arg_missing / arg_spurious / exact。"""
    from event_data import row_event
    pred_map = {(r.get("id"), r.get("sent_id")): r for r in pred_rows}
    cats = defaultdict(int)
    n_trig_hit = 0
    for g in gold_rows:
        ge = row_event(g) or {}
        p = pred_map.get((g.get("id"), g.get("sent_id")), {"events": []})
        pe = (p.get("events") or [{}])[0] if p.get("events") else {}
        g_trig = (ge.get("trigger") or {}).get("text", "")
        p_trig = (pe.get("trigger") or {}).get("text", "")
        g_args = {(a["role"], a["text"]) for a in ge.get("args") or []}
        p_args = {(a["role"], a["text"]) for a in pe.get("args") or []}
        if not p_trig:
            cat = "no_trigger"
        else:
            n_trig_hit += 1
            if pe.get("type") != ge.get("type"):
                cat = "wrong_type"
            elif p_trig != g_trig:
                cat = "wrong_trigger"
            elif p_args == g_args:
                cat = "exact"
            elif p_args - g_args:
                cat = "arg_spurious"
            else:
                cat = "arg_missing"
        cats[cat] += 1
    total = max(len(gold_rows), 1)
    return {
        "counts": dict(cats),
        "ratios": {k: round(v / total, 4) for k, v in cats.items()},
        "trigger_hit_rate": round(n_trig_hit / total, 4),
    }


# ─────────────────────────── 4. 主流程 ───────────────────────────
def run(split: str, show: int = 0) -> dict:
    logger = EvalLogger(method="rule", root=ROOT, notes="L0 规则基线（触发词白名单 + 正则论元）")
    schema = load_schema()
    rows = load_extract_split(split)
    gold_path = DATA_DIR / f"{split}.jsonl"
    word2type = load_whitelist()

    print(f"数据：{split}（{len(rows)} 句） | 触发词 {len(word2type)} 条 | "
          f"事件类型 {len(schema)} 类")

    preds, shown = [], 0
    for r in rows:
        ev = extract_one(r["text"], schema, word2type)
        preds.append(make_pred_row(r, [ev] if ev else []))
        if shown < show:
            shown += 1
            print(f"--- {r['id']}")
            print(f"  text : {r['text']}")
            g = (r.get("event") or {})
            print(f"  gold : {(g.get('trigger') or {}).get('text')} "
                  f"{[(a['role'], a['text']) for a in g.get('args') or []]}")
            print(f"  pred : {(ev or {}).get('trigger', {}).get('text')} "
                  f"{[(a['role'], a['text']) for a in (ev or {}).get('args', [])]}")

    meta = {
        "display_name": "L0 规则（白名单+正则）",
        "train_s": 0, "params_m": 0,
        "num_triggers": len(word2type),
        "note": "零训练基线；触发词=白名单最长优先，论元=时间/地点/伤亡/车辆正则",
    }
    res = evaluate_predictions("rule", preds, meta=meta, gold_path=gold_path)

    # 每角色 F1（定位规则强/弱项）
    print("\n【per-role F1】")
    for role, m in res["per_role"].items():
        print(f"  {role:<8} P={m['precision']:.3f} R={m['recall']:.3f} "
              f"F1={m['f1']:.3f} (tp={m['tp']} fp={m['fp']} fn={m['fn']})")

    failures = analyze_failures(rows, preds)
    print("\n【失败分析】")
    for k, v in sorted(failures["counts"].items(), key=lambda x: -x[1]):
        print(f"  {k:<14} {v:5d} 条 ({failures['ratios'][k]:.1%})")
    print(f"  触发词命中率：{failures['trigger_hit_rate']:.1%}")

    res["failure_analysis"] = failures
    with open(METRIC_DIR / "rule.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)

    logger.finalize(
        metrics={k: res[k]["f1"] for k in ("trigger", "argument", "event_exact")},
        type_macro_f1=res.get("type_macro_f1"),
        num_triggers=len(word2type),
        trigger_hit_rate=failures["trigger_hit_rate"],
        failure_counts=failures["counts"],
    )
    return res


def main():
    ap = argparse.ArgumentParser(description="L0 规则基线：触发词白名单 + 正则论元（CEC）")
    ap.add_argument("--split", default="test", choices=["train", "dev", "test"])
    ap.add_argument("--show", type=int, default=0, help="打印 N 条 gold vs pred 样例")
    args = ap.parse_args()
    run(args.split, show=args.show)


if __name__ == "__main__":
    main()
