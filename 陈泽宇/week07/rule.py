"""Training-derived trigger dictionary with amount/date/percentage rules."""
import re
from collections import Counter

from data import source_span

NUMBER = r"\d[\d,，]*(?:\.\d+)?"
DATE = re.compile(r"(?:\d{4}年)?\d{1,2}月\d{1,2}日|\d{4}年")
MONEY = re.compile(NUMBER + r"\s*(?:亿|万|千)?\s*(?:美元|港元|港币|元)")
PERCENT = re.compile(NUMBER + r"\s*[%％]")
SHARES = re.compile(NUMBER + r"\s*(?:亿|万|千)?\s*股")
COMPANY = re.compile(r"[\u4e00-\u9fffA-Za-z0-9]{2,12}(?:股份有限公司|有限公司|集团|公司)")


def build_rules(rows):
    pairs = Counter((r["event"]["type"], r["event"]["trigger"]["text"]) for r in rows)
    return [(kind, word, count) for (kind, word), count in pairs.items() if word and count >= 2]


def role_pattern(role):
    if any(s in role for s in ("时间", "日期", "周期")):
        return DATE
    if any(s in role for s in ("比例", "占比", "占持", "占总")):
        return PERCENT
    if any(s in role for s in ("金额", "价格", "市值", "规模")):
        return MONEY
    if "数量" in role and any(s in role for s in ("股", "股票")):
        return SHARES
    if "公司" in role:
        return COMPANY
    return None


def extract_rule(text, schema, rules):
    matches = []
    for kind, word, count in rules:
        start = text.find(word)
        if start >= 0:
            matches.append((len(word), count, -start, kind))
    if not matches:
        return None
    length, _, negative_start, kind = max(matches)
    start = -negative_start
    trigger = source_span(text, start, start + length)
    args = []
    used = set()
    for role in schema[kind]:
        pattern = role_pattern(role)
        if pattern is None:
            continue
        choices = [m for m in pattern.finditer(text) if m.span() not in used]
        if choices:
            match = min(choices, key=lambda m: abs(m.start() - start))
            used.add(match.span())
            args.append({**source_span(text, *match.span()), "role": role})
    return {"type": kind, "trigger": trigger, "args": args}
