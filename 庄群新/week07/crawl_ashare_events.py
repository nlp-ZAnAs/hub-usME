# -*- coding: utf-8 -*-
"""
抓取 A 股真实事件类快讯（供事件抽取模型实测用）

数据源（均为公开可访问的资讯接口，无需登录）：
  1) 东方财富 7×24 快讯   https://np-weblist.eastmoney.com/comm/web/getFastNewsList
  2) 新浪财经 7×24 直播   https://zhibo.sina.com.cn/api/zhibo/feed

做法：两源各翻若干页 → 按 13 类事件的业务关键词打分 → 只留「像事件」的条目
      → 按正文去重 → 落盘成 articles.jsonl（{"id","title","text","url","source","time","type_hint"}）

用法：
  python crawl_ashare_events.py --target 50 --out articles.jsonl
"""

import argparse
import hashlib
import json
import re
import time
from pathlib import Path

import requests

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")
HEADERS = {"User-Agent": UA, "Referer": "https://finance.eastmoney.com/"}

# 13 类事件 → 关键词。用「业务词」而不是事件名本身，因为快讯里很少直接写「股东减持」四个字
TYPE_KEYWORDS = {
    "股份回购": ["回购"],
    "股东减持": ["减持"],
    "股东增持": ["增持"],
    "企业收购": ["收购", "并购", "受让", "拟购买"],
    "企业破产": ["破产", "重整", "清算"],
    "企业融资": ["融资", "定增", "增发", "募资", "配股", "发行股份"],
    "公司上市": ["上市", "IPO", "挂牌", "首发"],
    "中标": ["中标", "中标通知", "成交", "签约", "合同金额"],
    "亏损": ["亏损", "净利下滑", "预亏", "业绩下滑", "同比下降"],
    "被约谈": ["约谈", "问询", "监管函", "处罚", "立案"],
    "质押": ["质押"],
    "解除质押": ["解除质押", "解押", "解除股份质押"],
    "高管变动": ["辞职", "离职", "聘任", "任命", "董事长", "总经理", "财务总监", "换届"],
}

# 明显不是「企业事件」的宏观/行情类词，命中则降权
NOISE_WORDS = ["沪指", "深成指", "创业板指", "涨逾", "跌逾", "涨停板", "北向资金",
               "央行", "美联储", "汇率", "原油", "黄金", "国债", "CPI", "PMI"]


def score_item(text: str) -> tuple[int, str]:
    """返回（命中关键词数, 最匹配的事件类型）。"""
    best_type, best_hits = "", 0
    total_hits = 0
    for etype, kws in TYPE_KEYWORDS.items():
        n = sum(1 for k in kws if k in text)
        total_hits += n
        if n > best_hits:
            best_type, best_hits = etype, n
    if any(w in text for w in NOISE_WORDS):
        total_hits -= 2
    return total_hits, best_type


def clean_text(s: str) -> str:
    s = re.sub(r"\s+", " ", s or "").strip()
    s = re.sub(r"<[^>]+>", "", s)             # 去 HTML 标签
    return s


def split_title_body(raw: str) -> tuple[str, str]:
    """「【标题】正文」→ (标题, 正文全文)。正文保留原文，标题单独存。"""
    m = re.match(r"【(.+?)】(.+)", raw, re.S)
    if m:
        return m.group(1).strip(), raw.strip()
    return "", raw.strip()


# ── 数据源 1：东方财富 7×24 快讯 ──
def fetch_eastmoney(pages: int = 4, page_size: int = 50) -> list[dict]:
    out, sort_end = [], ""
    for _ in range(pages):
        url = ("https://np-weblist.eastmoney.com/comm/web/getFastNewsList"
               f"?client=web&biz=web_724&fastColumn=102&sortEnd={sort_end}"
               f"&pageSize={page_size}&req_trace=1")
        try:
            r = requests.get(url, headers=HEADERS, timeout=15)
            data = (r.json() or {}).get("data") or {}
        except Exception as e:
            print(f"[eastmoney] 第 {len(out)//page_size+1} 页失败：{type(e).__name__} {e}")
            break
        items = data.get("fastNewsList") or []
        for it in items:
            raw = clean_text(it.get("summary") or "")
            if not raw:
                continue
            code = str(it.get("code") or "")
            title, text = split_title_body(raw)
            out.append({
                "source": "东方财富7×24",
                "title": title,
                "text": text,
                "url": f"https://finance.eastmoney.com/a/{code}.html" if code else "",
                "time": str(it.get("showTime") or ""),
            })
        sort_end = str(data.get("sortEnd") or "")
        if not sort_end or not items:
            break
        time.sleep(0.6)
    return out


# ── 数据源 2：新浪财经 7×24 ──
def fetch_sina(pages: int = 4, page_size: int = 50) -> list[dict]:
    out = []
    for p in range(1, pages + 1):
        url = ("https://zhibo.sina.com.cn/api/zhibo/feed"
               f"?page={p}&page_size={page_size}&zhibo_id=152&tag_id=0&dire=f&dpc=1")
        try:
            r = requests.get(url, headers=HEADERS, timeout=15)
            feed = (((r.json() or {}).get("result") or {}).get("data") or {}).get("feed") or {}
            items = feed.get("list") or []
        except Exception as e:
            print(f"[sina] 第 {p} 页失败：{type(e).__name__} {e}")
            break
        for it in items:
            raw = clean_text(it.get("rich_text") or "")
            if not raw:
                continue
            title, text = split_title_body(raw)
            out.append({
                "source": "新浪财经7×24",
                "title": title,
                "text": text,
                "url": str(it.get("docurl") or ""),
                "time": str(it.get("create_time") or ""),
            })
        time.sleep(0.6)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=50, help="目标保留条数")
    ap.add_argument("--pages", type=int, default=6, help="每个源翻几页")
    ap.add_argument("--min_hits", type=int, default=1, help="命中关键词数下限")
    ap.add_argument("--out", default="articles.jsonl")
    args = ap.parse_args()

    pool = fetch_eastmoney(args.pages) + fetch_sina(args.pages)
    print(f"抓到原始条目 {len(pool)} 条（东财+新浪）")

    kept, seen = [], set()
    for it in pool:
        text = it["text"]
        if len(text) < 20:            # 太短，模型训的是平均 164 字的公告
            continue
        hits, etype = score_item(text)
        if hits < args.min_hits:
            continue
        key = hashlib.md5(text.encode("utf-8")).hexdigest()[:16]
        if key in seen:
            continue
        seen.add(key)
        it["type_hint"] = etype
        it["hits"] = hits
        it["id"] = f"news_{len(kept):03d}"
        kept.append(it)

    kept.sort(key=lambda x: (-x["hits"], -len(x["text"])))
    kept = kept[:args.target]
    for i, it in enumerate(kept):   # 重新按时间顺序编号，便于阅读
        it["id"] = f"news_{i:03d}"

    out_path = Path(args.out)
    with open(out_path, "w", encoding="utf-8") as f:
        for it in kept:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    from collections import Counter
    print(f"保留 {len(kept)} 条 → {out_path.resolve()}")
    print("类型分布：", dict(Counter(it["type_hint"] for it in kept)))
    print("来源分布：", dict(Counter(it["source"] for it in kept)))
    for it in kept[:3]:
        print(f"\n[{it['id']}] {it['type_hint']} | {it['text'][:120]}")


if __name__ == "__main__":
    main()
