"""生成式链路（本地 Qwen 与 DeepSeek API）共用的 prompt、输出解析与 offset 回填。

模型只输出「类型 → 原文片段」，不输出下标；位置由程序在原文中查找。
同一份解析器与回填规则同时用于本地模型和 API，保证两条链路同口径。
"""

import json
import random
import re

from common import ROOT, TYPE_ZH, TYPES, read_json

TYPE_DEFINITIONS = {
    "address": "省市区县、街道门牌等地址或地名，如“台湾”“北京市海淀区”",
    "book": "小说、杂志、教材、工具书等书籍名称",
    "company": "企业、银行等公司名称，如“浙商银行”",
    "game": "电子游戏、网络游戏、主机游戏名称",
    "government": "各级政府、部委、法院、检察院、公安等国家机关",
    "movie": "电影、电视剧、纪录片等影视作品名称",
    "name": "真实或虚构人物的姓名、昵称",
    "organization": "学校、球队、协会、乐队、非政府组织等机构",
    "position": "职务、职称、头衔，如“总经理”“教授”“记者”",
    "scene": "旅游景点、公园、名胜古迹、风景区",
}

SYSTEM_PROMPT = "\n".join(
    ["你是中文命名实体识别助手。从用户给出的句子中抽取以下 10 类实体："]
    + [f"- {t}（{TYPE_ZH[t]}）：{TYPE_DEFINITIONS[t]}" for t in TYPES]
    + ["要求：",
       "1. 实体必须是句子中连续出现的原文片段，逐字复制，不改写、不补全；"
       "书名、影视作品、游戏名带书名号时连同《》一起抽取，如“《三国演义》”；",
       "2. 只输出一个 JSON 对象：键为上面的英文类型名，值为该类型实体文本的列表，按在句中出现的顺序；"
       "没有实体的类型不要输出；整句没有实体时输出 {}；",
       "3. 不要输出解释或任何其他文字。"])

ZH2TYPE = {v: k for k, v in TYPE_ZH.items()}


def user_prompt(text):
    return "句子：" + text


def gold_mentions(row):
    """金标 → {类型: [去重后的实体文本，按首次出现排序]}，类型按固定顺序。"""
    out = {}
    for t in TYPES:
        texts = []
        for e in row["entities"]:
            if e["type"] == t and e["text"] not in texts:
                texts.append(e["text"])
        if texts:
            out[t] = texts
    return out


def target_json(row):
    return json.dumps(gold_mentions(row), ensure_ascii=False, separators=(",", ":"))


def build_messages(text, examples=()):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for ex in examples:
        messages.append({"role": "user", "content": user_prompt(ex["text"])})
        messages.append({"role": "assistant", "content": target_json(ex)})
    messages.append({"role": "user", "content": user_prompt(text)})
    return messages


def pick_few_shot(train_rows, k, seed):
    """固定 seed 打乱训练集后做贪心集合覆盖：每次选能新覆盖最多类型的一条，
    同分取打乱顺序中靠前者；10 类全覆盖后再按顺序补足 k 条。示例只来自训练集。
    书名号不配对的实体（句子截断造成，如“生化危机4》”）是标注噪声，不作示范。"""
    rows = [r for r in train_rows
            if all(e["text"].count("《") == e["text"].count("》") for e in r["entities"])]
    random.Random(seed).shuffle(rows)
    chosen, covered = [], set()
    while covered != set(TYPES):
        if len(chosen) == k:
            raise ValueError("k 条示例无法覆盖全部 10 类；请增大 few_shot_k")
        best = max(rows, key=lambda r: len({e["type"] for e in r["entities"]} - covered))
        chosen.append(best)
        covered |= {e["type"] for e in best["entities"]}
    for row in rows:
        if len(chosen) >= k:
            break
        if row not in chosen:
            chosen.append(row)
    return chosen


# ───────────────────────────── 解析 ─────────────────────────────

def _strip_fence(text):
    match = re.search(r"```(?:json)?\s*(.*?)(?:```|$)", text, flags=re.S)
    return match.group(1) if match else text


def _balanced_object(text):
    """从第一个 { 开始做括号配平扫描；返回 (片段, 是否完整)。"""
    start = text.find("{")
    if start < 0:
        return None, False
    stack, in_string, escaped = [], False, False
    for i in range(start, len(text)):
        c = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif c == "\\":
                escaped = True
            elif c == '"':
                in_string = False
            continue
        if c == '"':
            in_string = True
        elif c in "{[":
            stack.append(c)
        elif c in "}]":
            if not stack:
                break
            stack.pop()
            if not stack:
                return text[start:i + 1], True
    # 被 max_tokens 截断：补引号并按栈逆序补右括号，宁可丢最后一个不完整元素。
    tail = text[start:]
    if in_string:
        tail += '"'
    tail = re.sub(r"[,:]\s*$", "", tail.rstrip())
    for opener in reversed(stack):
        tail += "}" if opener == "{" else "]"
    return tail, False


def parse_mentions(raw):
    """返回 (mentions{type:[text]}, stats)。stats 记录解析状态与被丢弃的字段。"""
    stats = {"status": "ok", "type_alias": 0, "invalid_type": 0, "invalid_value": 0}
    text = _strip_fence(raw.strip())
    if not text.strip():
        stats["status"] = "empty"
        return {}, stats
    fragment, complete = _balanced_object(text)
    if fragment is None:
        stats["status"] = "invalid"
        return {}, stats
    try:
        value = json.loads(fragment)
    except json.JSONDecodeError:
        stats["status"] = "invalid"
        return {}, stats
    if not isinstance(value, dict):
        stats["status"] = "invalid"
        return {}, stats
    if not complete:
        stats["status"] = "repaired"
    mentions = {}
    for key, items in value.items():
        key = str(key).strip()
        if key in ZH2TYPE:
            key = ZH2TYPE[key]
            stats["type_alias"] += 1
        if key not in TYPES:
            stats["invalid_type"] += 1
            continue
        if isinstance(items, str):
            items = [items]
        if not isinstance(items, list):
            stats["invalid_value"] += 1
            continue
        for item in items:
            if not isinstance(item, str) or not item.strip():
                stats["invalid_value"] += 1
                continue
            mentions.setdefault(key, [])
            if item.strip() not in mentions[key]:
                mentions[key].append(item.strip())
    return mentions, stats


# ───────────────────────────── offset 回填 ─────────────────────────────

def span_policy():
    """prepare_data.py 按验证集上限选定的回填策略（first / all）。"""
    return read_json(ROOT / "data" / "manifest.json")["llm_span_policy"]["selected"]


def mentions_to_spans(text, mentions, policy):
    """把 (类型, 文本) 回填为不重叠的字符 span。

    长片段优先占位（避免“北京”抢走“北京大学”的位置）；
    policy="first" 只取第一个空闲出现位置，policy="all" 取全部空闲出现位置。
    返回 (spans, n_not_in_text)。
    """
    if policy not in {"first", "all"}:
        raise ValueError("policy must be first/all")
    pairs = [(t, m) for t in TYPES for m in mentions.get(t, [])]
    pairs.sort(key=lambda x: -len(x[1]))
    taken = [False] * len(text)
    spans, missing = [], 0
    for kind, mention in pairs:
        starts, i = [], text.find(mention)
        while i != -1:
            starts.append(i)
            i = text.find(mention, i + 1)
        if not starts:
            missing += 1
            continue
        for start in starts:
            end = start + len(mention)
            if any(taken[start:end]):
                continue
            spans.append({"type": kind, "start": start, "end": end, "text": mention})
            for j in range(start, end):
                taken[j] = True
            if policy == "first":
                break
    spans.sort(key=lambda e: (e["start"], e["end"]))
    return spans, missing


def parse_to_entities(text, raw, policy):
    mentions, stats = parse_mentions(raw)
    spans, missing = mentions_to_spans(text, mentions, policy)
    stats["not_in_text"] = missing
    stats["n_mentions"] = sum(len(v) for v in mentions.values())
    return spans, stats


def summarize_parse_stats(stats_list):
    n = len(stats_list)
    status = {s: sum(x["status"] == s for x in stats_list)
              for s in ["ok", "repaired", "invalid", "empty"]}
    return {
        "n": n, "status": status,
        "json_valid_rate": (status["ok"] + status["repaired"]) / n if n else 0.0,
        "type_alias": sum(x["type_alias"] for x in stats_list),
        "invalid_type": sum(x["invalid_type"] for x in stats_list),
        "invalid_value": sum(x["invalid_value"] for x in stats_list),
        "not_in_text": sum(x["not_in_text"] for x in stats_list),
        "n_mentions": sum(x["n_mentions"] for x in stats_list),
    }
