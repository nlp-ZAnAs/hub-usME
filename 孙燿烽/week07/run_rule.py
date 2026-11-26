"""零模型基线：训练集实体词典（正向最长匹配）+ 书名号《》规则。

词典过滤的两个阈值（实体出现次数、被标注比例）只在验证集上网格选择；
《》规则的线索词在看测试结果前写定。
"""

import re
import time
from collections import Counter, defaultdict

from common import config, evaluate, load_data, save_result

CUES = {
    "movie": ["电影", "影片", "票房", "导演", "主演", "上映", "电视剧", "剧集", "影视", "观众"],
    "game": ["游戏", "玩家", "网游", "手游", "版本", "副本", "服务器", "公测", "职业"],
    "book": ["书", "小说", "作者", "出版", "读者", "杂志", "著作", "教材", "连载"],
}
BRACKET = re.compile(r"《[^《》]{1,40}》")


def build_dictionary(train_rows):
    labeled = defaultdict(Counter)
    for row in train_rows:
        for e in row["entities"]:
            labeled[e["text"]][e["type"]] += 1
    max_len = max(map(len, labeled))
    occurrences = Counter()
    for row in train_rows:
        text = row["text"]
        for i in range(len(text)):
            for j in range(i + 1, min(len(text), i + max_len) + 1):
                if text[i:j] in labeled:
                    occurrences[text[i:j]] += 1
    entries = {}
    for mention, types in labeled.items():
        count = sum(types.values())
        entries[mention] = {"type": types.most_common(1)[0][0], "count": count,
                            "label_ratio": count / max(occurrences[mention], count)}
    return entries, max_len


def bracket_type(text, start, end, fallback):
    window = text[max(0, start - 8):start] + text[end:end + 8]
    hits = {kind: sum(window.count(c) for c in cues) for kind, cues in CUES.items()}
    best = max(hits, key=hits.get)
    return best if hits[best] else fallback


def extract(text, dictionary, max_len, bracket_fallback):
    entities, i, taken = [], 0, [False] * len(text)
    while i < len(text):
        for j in range(min(len(text), i + max_len), i, -1):
            if text[i:j] in dictionary:
                entities.append({"type": dictionary[text[i:j]], "start": i, "end": j,
                                 "text": text[i:j]})
                taken[i:j] = [True] * (j - i)
                i = j
                break
        else:
            i += 1
    for m in BRACKET.finditer(text):
        if not any(taken[m.start():m.end()]):
            entities.append({"type": bracket_type(text, m.start(), m.end(), bracket_fallback),
                             "start": m.start(), "end": m.end(), "text": m.group()})
    return sorted(entities, key=lambda e: e["start"])


def main():
    cfg, splits = config(), load_data()
    start = time.perf_counter()
    entries, max_len = build_dictionary(splits["train"])
    bracketed = Counter(e["type"] for r in splits["train"] for e in r["entities"]
                        if e["text"].startswith("《"))
    fallback = bracketed.most_common(1)[0][0]
    search = []
    for min_count in cfg["rule"]["min_count_candidates"]:
        for ratio in cfg["rule"]["min_label_ratio_candidates"]:
            dictionary = {m: v["type"] for m, v in entries.items()
                          if v["count"] >= min_count and v["label_ratio"] >= ratio}
            preds = [{"id": r["id"], "entities": extract(r["text"], dictionary, max_len, fallback)}
                     for r in splits["val"]]
            f1 = evaluate(splits["val"], preds)["strict"]["f1"]
            search.append({"min_count": min_count, "min_label_ratio": ratio,
                           "dictionary_size": len(dictionary), "val_strict_f1": f1})
            print(search[-1], flush=True)
    best = max(search, key=lambda s: s["val_strict_f1"])
    dictionary = {m: v["type"] for m, v in entries.items()
                  if v["count"] >= best["min_count"] and v["label_ratio"] >= best["min_label_ratio"]}
    train_seconds = time.perf_counter() - start
    start = time.perf_counter()
    preds = [{"id": r["id"], "entities": extract(r["text"], dictionary, max_len, fallback)}
             for r in splits["test"]]
    predict_seconds = time.perf_counter() - start
    save_result("rule", splits, preds, train_seconds=train_seconds, predict_seconds=predict_seconds,
                settings={"selection": "validation strict micro-F1 over min_count × min_label_ratio",
                          "selected": best, "bracket_fallback_type": fallback,
                          "bracket_cues": CUES, "max_match_length": max_len,
                          "train_time_includes": "构建词典、统计出现次数与验证集网格搜索"},
                history=search)


if __name__ == "__main__":
    main()
