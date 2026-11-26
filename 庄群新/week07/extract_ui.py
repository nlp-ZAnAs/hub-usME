# -*- coding: utf-8 -*-
"""
A 股事件抽取 · 可视化工具（调用第七周已训练好的 BERT+CRF 模型，硬约束解码）

两种用法：

  1) 交互式网页（推荐）
       python extract_ui.py --serve
       浏览器打开 http://127.0.0.1:8765  → 粘贴任意财经文本 → 点「抽取」→ 看高亮 + 事件卡片

  2) 批量跑语料
       python extract_ui.py --batch articles.jsonl --out results.json

模型：code/outputs/checkpoints/bert_crf_full.pt（BERT+CRF，149 标签，解码期硬约束开）
依赖：torch + transformers + 本机 bert-base-chinese（无网络请求）
"""

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# ── 定位课件代码目录（模型、schema、服务层都在里面）──
CODE_DIR = Path(os.environ.get("WEEK7_CODE", r"D:\BaiduNetdiskDownload\week7-信息抽取\code"))
sys.path.insert(0, str(CODE_DIR / "common"))
sys.path.insert(0, str(CODE_DIR / "service"))

CKPT = CODE_DIR / "outputs" / "checkpoints" / "bert_crf_full.pt"
BERT = Path(os.environ.get("WEEK7_BERT", r"D:\BaiduNetdiskDownload\week4-语言模型\bert-base-chinese"))

# 13 类事件配色（网页高亮与图例共用）
TYPE_COLORS = {
    "股份回购": "#2f86d6", "股东减持": "#d9534f", "股东增持": "#5cb85c",
    "企业收购": "#8e6cc9", "企业破产": "#8b5a2b", "企业融资": "#e8873a",
    "公司上市": "#1f9d9d", "中标": "#c9a227", "亏损": "#b03060",
    "被约谈": "#7b7b7b", "质押": "#4b8f8c", "解除质押": "#79b8b3", "高管变动": "#3b6ea5",
}


class Extractor:
    """把课件里的 ExtractService 包一层：懒加载 + 线程锁（torch 前向不并发跑）。"""

    def __init__(self, device: str = "cuda", review_threshold: float = 0.55):
        self.lock = threading.Lock()
        self.svc = None
        self.device = device
        self.review_threshold = review_threshold

    def load(self):
        if self.svc is not None:
            return
        from extract_service import ExtractService
        if not CKPT.exists():
            raise FileNotFoundError(f"找不到权重：{CKPT}")
        svc = ExtractService(
            backend="joint",
            joint_ckpt=str(CKPT),
            bert_path=str(BERT),
            device=self.device,
            review_threshold=self.review_threshold,
            use_fallback=True,       # 数值类论元用正则兜底
            hard_constraint=True,    # 解码期硬约束：禁止 I- 开头 / B-x→I-y
        )
        svc.load()
        self.svc = svc

    def extract(self, doc: dict) -> dict:
        self.load()
        with self.lock:
            t0 = time.time()
            res = self.svc.extract_document(doc)
            res["latency_ms"] = round((time.time() - t0) * 1000, 1)
        return res


EXTRACTOR = Extractor()


def build_highlight(text: str, events: list[dict]) -> list[dict]:
    """
    把 trigger / 论元 span 转成可渲染的片段列表： [{start,end,kind,label,event_idx}]
    重叠时 trigger 优先（论元把与 trigger 重叠的部分让出来）。
    """
    spans = []
    for i, ev in enumerate(events):
        trg = ev.get("trigger") or {}
        if trg.get("start") is not None:
            spans.append({"start": trg["start"], "end": trg["end"], "kind": "trigger",
                          "label": f"{ev['type']}·触发词", "event_idx": i, "priority": 0})
        for a in ev.get("args") or []:
            if a.get("start") is None:
                continue
            spans.append({"start": a["start"], "end": a["end"], "kind": "arg",
                          "label": f"{ev['type']}·{a['role']}", "event_idx": i, "priority": 1})
    spans.sort(key=lambda s: (s["priority"], s["start"]))
    # 去重叠：同一区间只留优先级高的；部分重叠则跳过后者
    kept, occupied = [], set()
    for s in spans:
        seg = set(range(s["start"], s["end"]))
        if seg & occupied:
            continue
        kept.append(s)
        occupied |= seg
    kept.sort(key=lambda s: s["start"])
    return kept


# ══════════════════════════ 网页 ══════════════════════════
PAGE = r"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>A股事件抽取 · BERT+CRF 可视化</title>
<style>
 :root{--bg:#0f1216;--panel:#171b21;--line:#262c35;--fg:#e6e9ef;--dim:#8b95a5;--acc:#4ea1ff}
 *{box-sizing:border-box}
 body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.65 "Microsoft YaHei",system-ui,sans-serif}
 .wrap{max-width:1180px;margin:0 auto;padding:22px 20px 60px}
 h1{font-size:20px;margin:0 0 4px}
 .sub{color:var(--dim);font-size:12.5px;margin-bottom:18px}
 .sub b{color:var(--acc)}
 .grid{display:grid;grid-template-columns:1fr;gap:14px}
 .card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px 16px}
 textarea{width:100%;height:130px;background:#0b0e12;color:var(--fg);border:1px solid var(--line);
   border-radius:8px;padding:10px;font:13.5px/1.7 "Microsoft YaHei",monospace;resize:vertical}
 .row{display:flex;gap:10px;align-items:center;margin-top:10px;flex-wrap:wrap}
 button{background:var(--acc);color:#08121f;border:0;border-radius:7px;padding:9px 18px;
   font-weight:700;cursor:pointer}
 button:hover{filter:brightness(1.1)}
 button.ghost{background:#20262f;color:var(--dim);font-weight:400}
 .ms{color:var(--dim);font-size:12.5px}
 .hit{font-size:16px;line-height:1.95;background:#0b0e12;border-radius:8px;padding:14px;border:1px solid var(--line)}
 .trigger{background:#ffb02022;border-bottom:2px solid #ffb020;border-radius:3px;padding:1px 2px;font-weight:700}
 .arg{border-bottom:2px solid;border-radius:3px;padding:1px 2px;background:#4ea1ff14}
 .evhead{display:flex;gap:8px;align-items:center;margin-bottom:8px;flex-wrap:wrap}
 .tag{font-size:12px;font-weight:700;padding:2px 9px;border-radius:20px;color:#fff}
 .conf{font-size:12px;color:var(--dim)}
 .warn{font-size:12px;color:#ffb020;border:1px solid #ffb02055;border-radius:20px;padding:2px 9px}
 .ok{font-size:12px;color:#5cb85c;border:1px solid #5cb85c55;border-radius:20px;padding:2px 9px}
 table{border-collapse:collapse;width:100%;font-size:13px}
 th,td{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left;vertical-align:top}
 th{color:var(--dim);font-weight:400}
 .src{font-size:11px;padding:1px 6px;border-radius:4px}
 .src.model{background:#4ea1ff22;color:#4ea1ff}
 .src.regex{background:#5cb85c22;color:#5cb85c}
 pre{background:#0b0e12;border:1px solid var(--line);border-radius:8px;padding:12px;
   overflow:auto;font-size:12.5px;max-height:340px}
 .legend{display:flex;gap:14px;flex-wrap:wrap;font-size:12px;color:var(--dim);margin-top:10px}
 .legend i{display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:5px}
 .empty{color:var(--dim);padding:26px;text-align:center;border:1px dashed var(--line);border-radius:8px}
</style></head><body><div class="wrap">
<h1>A股事件抽取 · BERT+CRF（硬约束）</h1>
<div class="sub">模型 <b>bert_crf_full.pt</b> · 149 标签 / 13 类事件 · 解码期硬约束开 · 数值论元正则兜底 ·
 阈值 <b>0.55</b> 以下标 needs_review</div>

<div class="card">
  <div class="row" style="margin:0 0 8px">
    <span class="ms">输入财经文本（公告 / 快讯 / 研报摘要均可）：</span>
    <button class="ghost" onclick="fillDemo()">填入示例</button>
  </div>
  <textarea id="ta" placeholder="粘贴一段 A 股公告或快讯，例如：华友钴业公告称，公司拟以自有资金通过集中竞价交易方式回购股份，回购金额不低于10亿元……"></textarea>
  <div class="row">
    <button id="btn" onclick="run()">开始抽取</button>
    <span class="ms" id="ms"></span>
  </div>
</div>

<div class="grid" id="out"></div>
</div>
<script>
const COLORS = __COLORS__;
function esc(s){return (s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;')}
function colorOf(t){return COLORS[t]||'#4ea1ff'}
function renderHighlight(text, spans){
  let html='', cur=0;
  for(const s of spans){
    if(s.start>cur) html+=esc(text.slice(cur,s.start));
    const c = s.kind==='trigger' ? '#ffb020' : colorOf(s.label.split('·')[0]);
    const cls = s.kind==='trigger'?'trigger':'arg';
    const st = s.kind==='arg'?`border-color:${c};background:${c}18`:'';
    html += `<span class="${cls}" style="${st}" title="${esc(s.label)}">${esc(text.slice(s.start,s.end))}</span>`;
    cur=s.end;
  }
  html+=esc(text.slice(cur));
  return html;
}
function render(res){
  const out=document.getElementById('out');
  const evs=res.events||[];
  let head=`<div class="card">
    <div class="evhead">
      <span class="tag" style="background:#2a323d">${evs.length} 个事件</span>
      <span class="conf">切句 ${res.n_sentences} 段 · 耗时 ${res.latency_ms} ms</span>
      ${res.needs_review?'<span class="warn">⚠ 建议人工复核</span>':'<span class="ok">✓ 置信度全部达标</span>'}
    </div>
    <div class="legend">${Object.entries(COLORS).map(([k,v])=>`<span><i style="background:${v}"></i>${k}</span>`).join('')}</div>
  </div>`;
  const spans = res.highlight||[];
  let docCard = `<div class="card"><div class="legend" style="margin:0 0 8px">
      <span><i style="background:#ffb020"></i>触发词</span><span>彩色下划线 = 论元（颜色对应事件类型）</span></div>
      <div class="hit">${renderHighlight(res.text||'', spans)}</div></div>`;
  let evCards = evs.map((ev,i)=>{
    const c=colorOf(ev.type);
    const rows=(ev.args||[]).map(a=>`<tr><td>${esc(a.role)}</td>
        <td><b>${esc(a.text)}</b></td>
        <td><span class="src ${a.source==='regex'?'regex':'model'}">${a.source==='regex'?'正则补':'模型'}</span></td></tr>`).join('');
    return `<div class="card"><div class="evhead">
        <span class="tag" style="background:${c}">${esc(ev.type)}</span>
        <span class="conf">触发词：<b style="color:#ffb020">${esc((ev.trigger||{}).text||'-')}</b></span>
        <span class="conf">conf ${ev.confidence} · 来自第 ${ev.sent_idx+1} 段</span>
      </div>
      <table><tr><th style="width:30%">论元角色</th><th>取值</th><th style="width:80px">来源</th></tr>${rows||'<tr><td colspan=3 style="color:#8b95a5">未抽到论元</td></tr>'}</table>
    </div>`;
  }).join('');
  if(!evs.length) evCards=`<div class="empty">这段文本没有抽到事件 —— 若它确实不含 13 类事件，说明模型判对了；若明显该有，就是漏抽（可看下方逐句置信度）。</div>`;
  let sentRows=(res.sentences||[]).map(s=>{
    const brief = s.events.length? s.events.map(e=>`${e.type}(触发词=${(e.trigger||{}).text})`).join('；') : '（无事件）';
    return `<tr><td style="color:#8b95a5">${s.sent_idx+1}</td><td>${s.confidence.toFixed(2)}</td>
      <td>${esc(brief)}${s.needs_review?` <span class="warn">${esc(s.review_reasons.join(','))}</span>`:''}</td></tr>`;
  }).join('');
  let json=`<div class="card"><div class="evhead"><span class="tag" style="background:#2a323d">原始 JSON</span>
      <button class="ghost" onclick="copyJson(this)">复制</button></div>
      <pre id="js">${esc(JSON.stringify(res.events,null,2))}</pre></div>`;
  out.innerHTML = head + docCard + evCards +
    `<div class="card"><div class="evhead"><span class="tag" style="background:#2a323d">逐段过程</span></div>
      <table><tr><th style="width:40px">#</th><th style="width:70px">conf</th><th>结果 / 复核原因</th></tr>${sentRows}</table></div>` + json;
}
function copyJson(btn){
  navigator.clipboard.writeText(document.getElementById('js').textContent);
  btn.textContent='已复制'; setTimeout(()=>btn.textContent='复制',1200);
}
async function run(){
  const text=document.getElementById('ta').value.trim();
  if(!text){document.getElementById('ms').textContent='请先输入文本';return}
  const btn=document.getElementById('btn');
  btn.disabled=true; document.getElementById('ms').textContent='模型推理中…';
  try{
    const r=await fetch('/api/extract',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({title:'web',text})});
    const res=await r.json();
    if(res.error){document.getElementById('ms').textContent='错误：'+res.error;return}
    res.text=text; render(res);
    document.getElementById('ms').textContent='完成 · '+res.latency_ms+' ms';
  }catch(e){document.getElementById('ms').textContent='请求失败：'+e}
  finally{btn.disabled=false}
}
function fillDemo(){
  document.getElementById('ta').value=
   '华友钴业公告称，公司拟以自有资金及自筹资金（含股票回购贷款），通过集中竞价交易方式回购公司股份，回购资金总额不低于人民币10亿元、不超过15亿元，回购股份用于员工持股计划。';
}
// 自动演示：url 带 ?demo=1 时自动跑一次
if(location.search.includes('demo=1')){ fillDemo(); setTimeout(run,300); }
</script></body></html>"""


def serve(host: str, port: int, open_browser: bool):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    colors_json = json.dumps(TYPE_COLORS, ensure_ascii=False)
    page = PAGE.replace("__COLORS__", colors_json).encode("utf-8")

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):  # 静音访问日志
            pass

        def _send(self, code, body: bytes, ctype="application/json; charset=utf-8"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.startswith("/api/status"):
                self._send(200, json.dumps({"ok": True, "ckpt": str(CKPT),
                                            "loaded": EXTRACTOR.svc is not None},
                                           ensure_ascii=False).encode())
                return
            self._send(200, page, "text/html; charset=utf-8")

        def do_POST(self):
            if not self.path.startswith("/api/extract"):
                self._send(404, b'{"error":"not found"}')
                return
            try:
                n = int(self.headers.get("Content-Length") or 0)
                payload = json.loads(self.rfile.read(n).decode("utf-8") or "{}")
                text = (payload.get("text") or "").strip()
                if not text:
                    self._send(400, json.dumps({"error": "文本为空"}, ensure_ascii=False).encode())
                    return
                res = EXTRACTOR.extract({"id": "web", "title": payload.get("title", ""), "text": text})
                res["highlight"] = build_highlight(text, res["events"])
                self._send(200, json.dumps(res, ensure_ascii=False).encode())
            except Exception as e:
                self._send(500, json.dumps({"error": f"{type(e).__name__}: {e}"},
                                           ensure_ascii=False).encode())

    print(f"[extract_ui] 权重 {CKPT}")
    print(f"[extract_ui] BERT {BERT}")
    print(f"[extract_ui] 服务地址 http://{host}:{port}  （Ctrl+C 退出）")
    if open_browser:
        import webbrowser
        threading.Timer(2.0, lambda: webbrowser.open(f"http://{host}:{port}/?demo=1")).start()
    ThreadingHTTPServer((host, port), H).serve_forever()


def batch(args):
    rows = [json.loads(l) for l in open(args.batch, encoding="utf-8")]
    print(f"加载模型…（{len(rows)} 篇待抽）")
    EXTRACTOR.load()
    results, t_all = [], time.time()
    for i, doc in enumerate(rows, 1):
        res = EXTRACTOR.extract({"id": doc["id"], "title": doc.get("title", ""),
                                 "text": doc["text"]})
        res["text"] = doc["text"]
        res["url"] = doc.get("url", "")
        res["source"] = doc.get("source", "")
        res["time"] = doc.get("time", "")
        res["type_hint"] = doc.get("type_hint", "")
        res["highlight"] = build_highlight(doc["text"], res["events"])
        results.append(res)
        flag = "✓" if res["events"] else "·"
        print(f"  [{i}/{len(rows)}] {doc['id']} {flag} {len(res['events'])}事件 "
              f"{res['latency_ms']:>6}ms  {(doc.get('title') or doc['text'])[:34]}")
    total = time.time() - t_all
    out = Path(args.out)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    n_ev = sum(len(r["events"]) for r in results)
    n_hit = sum(1 for r in results if r["events"])
    print(f"\n汇总：{n_hit}/{len(results)} 篇抽到事件，共 {n_ev} 个事件，"
          f"平均 {total/len(results)*1000:.0f} ms/篇 → {out.resolve()}")


def main():
    ap = argparse.ArgumentParser(description="A股事件抽取可视化")
    ap.add_argument("--serve", action="store_true", help="启动网页界面")
    ap.add_argument("--batch", default="", help="批量抽取的 jsonl")
    ap.add_argument("--out", default="results.json")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--open", action="store_true", help="启动后自动打开浏览器")
    args = ap.parse_args()
    global EXTRACTOR
    EXTRACTOR = Extractor(device=args.device)
    if args.batch:
        batch(args)
    elif args.serve:
        serve(args.host, args.port, args.open)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
