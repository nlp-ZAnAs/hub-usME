# -*- coding: utf-8 -*-
"""
把 results.json 渲染成可视化报告（含人工核验结论）

输出：
  report_correct.html → result_correct.png   抽对的真实公司事件（原文高亮 + 论元）
  report_wrong.html   → result_wrong.png     误抽对照 + 真漏抽
  stats.png                                  统计总览（含精确率/召回/拒识）
"""

import argparse
import html
import json
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

TYPE_COLORS = {
    "股份回购": "#2f86d6", "股东减持": "#d9534f", "股东增持": "#5cb85c",
    "企业收购": "#8e6cc9", "企业破产": "#8b5a2b", "企业融资": "#e8873a",
    "公司上市": "#1f9d9d", "中标": "#c9a227", "亏损": "#b03060",
    "被约谈": "#7b7b7b", "质押": "#4b8f8c", "解除质押": "#79b8b3", "高管变动": "#3b6ea5",
}

for f in ("msyh.ttc", "msyhbd.ttc", "simhei.ttf"):
    p = Path("C:/Windows/Fonts") / f
    if p.exists():
        font_manager.fontManager.addfont(str(p))
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ── 人工核验结论（逐条看过 50 篇后标注）──
CORRECT = ["news_000", "news_005", "news_008", "news_009", "news_010", "news_011",
           "news_013", "news_017", "news_020", "news_030", "news_031", "news_038"]
FALSE_POS = ["news_001", "news_002", "news_003", "news_004", "news_006", "news_007",
             "news_012", "news_015", "news_018", "news_019", "news_022", "news_025",
             "news_027", "news_028", "news_029", "news_034", "news_042", "news_045",
             "news_046", "news_047", "news_049"]
MISSED = ["news_016", "news_026"]          # 真事件，模型输出「无事件」
SILENT_OK = ["news_014", "news_021", "news_023", "news_024", "news_032", "news_033",
             "news_035", "news_036", "news_037", "news_039", "news_040", "news_041",
             "news_043", "news_044", "news_048"]   # 确实无 13 类事件，模型沉默是对的

SHOW_CORRECT = ["news_008", "news_009", "news_010", "news_005", "news_000", "news_013"]
SHOW_WRONG = ["news_028", "news_002", "news_049", "news_046"]
SHOW_MISSED = ["news_016", "news_026"]


def build_spans(events):
    spans = []
    for ev in events:
        trg = ev.get("trigger") or {}
        if trg.get("start") is not None:
            spans.append((trg["start"], trg["end"], 0, "T", ev["type"], "触发词"))
        for a in ev.get("args") or []:
            if a.get("start") is None:
                continue
            spans.append((a["start"], a["end"], 1, "A", ev["type"], a["role"]))
    spans.sort(key=lambda s: (s[2], s[0]))
    kept, occ = [], set()
    for s in spans:
        seg = set(range(s[0], s[1]))
        if seg & occ:
            continue
        kept.append(s)
        occ |= seg
    return sorted(kept, key=lambda s: s[0])


def highlight_html(text, events):
    out, cur = [], 0
    for s, e, _p, kind, etype, label in build_spans(events):
        if s > cur:
            out.append(html.escape(text[cur:s]))
        c = "#ffb020" if kind == "T" else TYPE_COLORS.get(etype, "#4ea1ff")
        if kind == "T":
            out.append(f'<span class="tg" title="{html.escape(label)}">{html.escape(text[s:e])}</span>')
        else:
            out.append(f'<span class="ar" style="border-color:{c};background:{c}1f" '
                       f'title="{html.escape(label)}">{html.escape(text[s:e])}</span>')
        cur = e
    out.append(html.escape(text[cur:]))
    return "".join(out)


CSS = """
<style>
 body{margin:0;background:#0f1216;color:#e6e9ef;font:14px/1.7 "Microsoft YaHei",system-ui,sans-serif}
 .wrap{max-width:1360px;margin:0 auto;padding:26px 26px 40px}
 h1{font-size:22px;margin:0 0 6px}
 .sub{color:#8b95a5;font-size:12.5px;margin-bottom:20px;line-height:1.9}
 .sub b{color:#4ea1ff}
 .sec{font-size:15px;font-weight:700;margin:24px 0 12px;padding-left:10px;border-left:4px solid #4ea1ff}
 .card{background:#171b21;border:1px solid #262c35;border-radius:10px;padding:14px 16px;margin-bottom:13px}
 .head{display:flex;gap:9px;align-items:center;flex-wrap:wrap;margin-bottom:8px}
 .idx{font-size:12px;color:#8b95a5}
 .src{font-size:11.5px;color:#8b95a5}
 .tag{font-size:12px;font-weight:700;padding:2px 10px;border-radius:20px;color:#fff}
 .conf{font-size:12px;color:#8b95a5}
 .bad{font-size:11.5px;color:#ff6b6b;border:1px solid #ff6b6b55;border-radius:20px;padding:1px 8px}
 .warn{font-size:11.5px;color:#ffb020;border:1px solid #ffb02055;border-radius:20px;padding:1px 8px}
 .tg{background:#ffb02033;border-bottom:2px solid #ffb020;border-radius:3px;padding:0 2px;font-weight:700}
 .ar{border-bottom:2px solid;border-radius:3px;padding:0 2px}
 .doc{background:#0b0e12;border:1px solid #232830;border-radius:8px;padding:12px 13px;
      font-size:13.5px;line-height:2.0;margin-bottom:10px}
 .doc.bad{border-color:#ff6b6b33}
 .args{font-size:12.5px;color:#c3cbd6;line-height:1.9}
 .args b{color:#e6e9ef}
 .mdl{font-size:10.5px;color:#4ea1ff;background:#4ea1ff1f;border-radius:3px;padding:0 4px}
 .rgx{font-size:10.5px;color:#5cb85c;background:#5cb85c1f;border-radius:3px;padding:0 4px}
 .kpi{display:flex;gap:11px;flex-wrap:wrap}
 .kbox{flex:1;min-width:140px;background:#171b21;border:1px solid #262c35;border-radius:10px;padding:12px 15px}
 .kbox .v{font-size:26px;font-weight:700;color:#4ea1ff;line-height:1.25}
 .kbox .k{font-size:12px;color:#8b95a5;margin-top:2px}
 .note{background:#171b21;border:1px solid #262c35;border-left:4px solid #ffb020;
   border-radius:8px;padding:11px 14px;font-size:12.8px;color:#c3cbd6;line-height:1.85;margin-bottom:14px}
</style>"""


def page(title, sub, body):
    return (f"<!DOCTYPE html><html lang='zh-CN'><head><meta charset='utf-8'><title>{title}</title>"
            f"{CSS}</head><body><div class='wrap'><h1>{title}</h1><div class='sub'>{sub}</div>"
            f"{body}</div></body></html>")


def card(r, idx, note=""):
    evs = r["events"]
    tags = "".join(f'<span class="tag" style="background:{TYPE_COLORS.get(e["type"],"#4ea1ff")}">'
                   f'{html.escape(e["type"])}</span>' for e in evs)
    rows = []
    for e in evs:
        a = "".join(f'{html.escape(x["role"])}=<b>{html.escape(x["text"])}</b>'
                    f'<span class="{ "rgx" if x.get("source")=="regex" else "mdl"}">'
                    f'{"正则" if x.get("source")=="regex" else "模型"}</span> '
                    for x in e.get("args") or [])
        rows.append(f'<div class="args">· <b style="color:{TYPE_COLORS.get(e["type"],"#4ea1ff")}">'
                    f'{html.escape(e["type"])}</b> 触发词「{html.escape((e["trigger"] or {}).get("text",""))}」'
                    f' ｜ {a or "（未抽到论元）"}</div>')
    badge = note or (f'<span class="conf">conf {evs[0]["confidence"]:.2f}</span>' if evs else "")
    return (f'<div class="card"><div class="head"><span class="idx">#{idx:02d}</span>{tags}{badge}'
            f'<span class="src">{html.escape(r.get("source",""))} · {html.escape(r.get("time",""))[:16]}</span>'
            f'</div><div class="doc">{highlight_html(r["text"], evs)}</div>{"".join(rows)}</div>')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results.json")
    ap.add_argument("--outdir", default=".")
    args = ap.parse_args()

    rows = json.loads(Path(args.results).read_text(encoding="utf-8"))
    by_id = {r["id"]: r for r in rows}
    outdir = Path(args.outdir)
    n_doc = len(rows)
    hits = [r for r in rows if r["events"]]

    n_correct, n_fp = len(CORRECT), len(FALSE_POS)
    prec = n_correct / (n_correct + n_fp) * 100
    rec = n_correct / (n_correct + len(MISSED)) * 100
    silence = len(SILENT_OK) / (len(SILENT_OK) + n_fp) * 100
    n_ev = sum(len(r["events"]) for r in rows)
    lat = [r["latency_ms"] for r in rows]

    # ── 页 1：抽对的 ──
    kpi = (f'<div class="kpi">'
           f'<div class="kbox"><div class="v">{n_doc}</div><div class="k">真实 A 股快讯</div></div>'
           f'<div class="kbox"><div class="v">{n_correct}</div><div class="k">抽对的公司事件</div></div>'
           f'<div class="kbox"><div class="v">{prec:.0f}%</div><div class="k">篇级精确率</div></div>'
           f'<div class="kbox"><div class="v">{rec:.0f}%</div><div class="k">篇级召回率</div></div>'
           f'<div class="kbox"><div class="v">{sum(lat)/len(lat):.0f}<span style="font-size:14px"> ms</span></div>'
           f'<div class="k">平均延迟/条</div></div></div>')
    body = kpi + ('<div class="sec">抽对的真实公司事件（高亮 = 触发词与论元，颜色对应类型）</div>'
                  '<div class="note" style="border-left-color:#5cb85c"><b>读图提示：</b>'
                  '同一篇里，核心事件（回购/增持/收购）的触发词与论元都抓得准；'
                  '但个别条目会顺带冒出一个「空论元的噪声事件」（例如把「解禁上市」里的「上市」判成公司上市），'
                  '这类问题的成因与处置见下一页。</div>')
    body += "".join(card(by_id[i], k + 1) for k, i in enumerate(SHOW_CORRECT))

    # ── 页 2：误抽 + 漏抽 ──
    note = ('<div class="note"><b>这一页是模型的诚实边界。</b> 训练语料全是「含事件的公告」，'
            '模型因此没有学会「什么时候该沉默」：政策文件、人物稿、ETF 点评里出现「回购 / 并购 / 任命」等词，'
            '它仍会以 <b>0.93~1.00</b> 的置信度抽出事件，且论元往往是空的。<br>'
            '→ 上线必须补两道闸：<b>①</b> 公司主体校验（论元里没有上市公司/证券代码就丢弃）；'
            '<b>②</b> 域过滤（政策发布、研报、行情点评类文本先拦掉）。</div>')
    wbody = note + '<div class="sec">误抽：本不该有事件，却被高置信度判成事件</div>'
    for k, i in enumerate(SHOW_WRONG, 1):
        wbody += card(by_id[i], k, note='<span class="bad">误抽 · 无企业主体</span>')
    wbody += '<div class="sec">真漏抽：确实是公司事件，模型输出「无事件」</div>'
    for k, i in enumerate(SHOW_MISSED, len(SHOW_WRONG) + 1):
        wbody += card(by_id[i], k, note='<span class="warn">漏抽 · 应为 高管变动 / 中标</span>')
    wrong_html = page("模型的边界：误抽与漏抽",
                      f"同一批 {n_doc} 条语料 ｜ 误抽 {n_fp} 条（政策/人物/ETF 点评）｜ 真漏抽 {len(MISSED)} 条",
                      wbody)

    correct_html = page("A股事件抽取 · 抽对的公司事件",
                        f"模型 <b>bert_crf_full.pt</b>（BERT+CRF，149 标签，解码期硬约束，数值论元正则兜底）"
                        f" ｜ 语料：东方财富 7×24 + 新浪财经 7×24 共 <b>{n_doc}</b> 条 ｜ "
                        f"共抽出 {len(hits)} 篇 / {n_ev} 个事件", body)

    (outdir / "report_correct.html").write_text(correct_html, encoding="utf-8")
    (outdir / "report_wrong.html").write_text(wrong_html, encoding="utf-8")

    # ── 统计图 ──
    fig = plt.figure(figsize=(16, 9), facecolor="#0f1216")
    gs = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.26, left=0.07, right=0.97, top=0.85, bottom=0.08)
    fig.suptitle("A股真实快讯 · BERT+CRF 事件抽取结果总览", color="#e6e9ef", fontsize=21, x=0.07, ha="left", y=0.955)
    fig.text(0.07, 0.9, f"{n_doc} 条真实文本（东财7×24 + 新浪7×24）｜抽到 {len(hits)} 篇 / {n_ev} 个事件"
                        f"｜平均 {sum(lat)/len(lat):.0f} ms/条｜人工逐条核验", color="#8b95a5", fontsize=12)

    def style(ax, title):
        ax.set_facecolor("#171b21")
        ax.set_title(title, color="#e6e9ef", fontsize=14, loc="left")
        for sp in ax.spines.values():
            sp.set_color("#262c35")
        ax.tick_params(colors="#8b95a5")

    # 1) 四分类
    ax = fig.add_subplot(gs[0, 0])
    style(ax, "50 条语料的人工核验结论")
    cats = ["抽对", "误抽", "该沉默\n且沉默", "漏抽"]
    vals = [n_correct, n_fp, len(SILENT_OK), len(MISSED)]
    cols = ["#5cb85c", "#d9534f", "#39414d", "#ffb020"]
    b = ax.bar(cats, vals, color=cols, width=0.6)
    for r, v in zip(b, vals):
        ax.text(r.get_x() + r.get_width() / 2, v + 0.3, str(v), ha="center",
                color="#e6e9ef", fontsize=12, fontweight="bold")
    ax.set_ylim(0, max(vals) * 1.18)

    # 2) 三个比率
    ax2 = fig.add_subplot(gs[0, 1])
    style(ax2, "精确率 / 召回 / 拒识率")
    names = ["精确率\n(抽到的里有多少对)", "召回率\n(真事件抽到多少)", "拒识率\n(无事件里沉默多少)"]
    vs = [prec, rec, silence]
    bars = ax2.barh(names, vs, color=["#d9534f", "#5cb85c", "#4ea1ff"], height=0.5)
    ax2.set_xlim(0, 100)
    ax2.invert_yaxis()
    for r, v in zip(bars, vs):
        ax2.text(min(v + 2, 92), r.get_y() + r.get_height() / 2, f"{v:.0f}%",
                 va="center", color="#e6e9ef", fontsize=12, fontweight="bold")

    # 3) 类型分布（仅正确事件）
    ax3 = fig.add_subplot(gs[0, 2])
    style(ax3, "抽对事件的事件类型")
    tc = Counter(e["type"] for i in CORRECT for e in by_id[i]["events"])
    ts, cs = zip(*tc.most_common()) if tc else ([], [])
    ax3.barh(range(len(ts)), cs, color=[TYPE_COLORS.get(t, "#4ea1ff") for t in ts], height=0.58)
    ax3.set_yticks(range(len(ts)))
    ax3.set_yticklabels(ts, color="#c3cbd6", fontsize=11)
    ax3.invert_yaxis()
    for k, v in enumerate(cs):
        ax3.text(v + 0.05, k, str(v), color="#e6e9ef", va="center", fontsize=10)

    # 4) 置信度分布：抽对 vs 误抽
    ax4 = fig.add_subplot(gs[1, 0])
    style(ax4, "置信度分布（说明「高置信度 ≠ 正确」）")
    conf_ok = [e["confidence"] for i in CORRECT for e in by_id[i]["events"]]
    conf_bad = [e["confidence"] for i in FALSE_POS for e in by_id[i]["events"]]
    bins = [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.001]
    ax4.hist([conf_ok, conf_bad], bins=bins, label=["抽对", "误抽"],
             color=["#5cb85c", "#d9534f"], rwidth=0.9)
    ax4.legend(facecolor="#171b21", edgecolor="#262c35", labelcolor="#c3cbd6", fontsize=10)
    ax4.set_xlabel("confidence", color="#8b95a5")

    # 5) 论元个数
    ax5 = fig.add_subplot(gs[1, 1])
    style(ax5, "每个事件的论元个数")
    n_ok = [len(e.get("args") or []) for i in CORRECT for e in by_id[i]["events"]]
    n_bad = [len(e.get("args") or []) for i in FALSE_POS for e in by_id[i]["events"]]
    ax5.hist([n_ok, n_bad], bins=range(0, 12), label=["抽对", "误抽"],
             color=["#5cb85c", "#d9534f"], rwidth=0.9)
    ax5.legend(facecolor="#171b21", edgecolor="#262c35", labelcolor="#c3cbd6", fontsize=10)
    ax5.set_xlabel("论元数（误抽多为 0 个）", color="#8b95a5")

    # 6) 延迟
    ax6 = fig.add_subplot(gs[1, 2])
    style(ax6, "逐篇推理延迟（ms）")
    ax6.plot(range(1, len(lat) + 1), lat, color="#8e6cc9", lw=1.2, marker="o", ms=2.8)
    ax6.set_xlabel("文档序号（第 1 篇含模型预热）", color="#8b95a5")

    fig.savefig(outdir / "stats.png", dpi=150, facecolor="#0f1216")
    print(f"已生成 report_correct.html / report_wrong.html / stats.png"
          f"（精确率 {prec:.1f}% 召回 {rec:.1f}% 拒识 {silence:.1f}%）")

    edge = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")
    if not edge.exists():
        edge = Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe")
    if not edge.exists():
        print("未找到 Edge/Chrome，跳过截图")
        return
    tmp = Path(tempfile.mkdtemp(prefix="w7rpt_"))
    import shutil
    for src, dst, w, h in [("report_correct.html", "result_correct.png", 1440, 1950),
                           ("report_wrong.html", "result_wrong.png", 1440, 1720)]:
        p_src = tmp / src                                  # 纯 ASCII 路径（中文路径在 --screenshot 下会失败）
        p_tmp_out = tmp / dst
        p_src.write_text((outdir / src).read_text(encoding="utf-8"), encoding="utf-8")
        subprocess.run([str(edge), "--headless=new", "--disable-gpu", "--no-sandbox",
                        "--force-device-scale-factor=1",
                        f"--window-size={w},{h}", f"--screenshot={p_tmp_out}", str(p_src)],
                       capture_output=True, text=True, timeout=120)
        if p_tmp_out.exists():
            shutil.copyfile(p_tmp_out, outdir / dst)
            print(f"截图 {dst}: OK {p_tmp_out.stat().st_size//1024} KB")
        else:
            print(f"截图 {dst}: FAIL")


if __name__ == "__main__":
    main()
