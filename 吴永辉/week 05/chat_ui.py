# -*- coding: utf-8 -*-
"""
字符级 RNN + SFT 模型的本地问答界面（零第三方依赖）。

用法：
    python chat_ui.py                # 自动挑端口、自动打开浏览器
    python chat_ui.py --port 8000    # 指定端口
    python chat_ui.py --no-browser   # 不自动开浏览器
    python chat_ui.py --device cpu   # 强制用 CPU

依赖：只用到 torch（与训练脚本一致），其余全部是标准库。
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch
import torch.nn as nn

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = SCRIPT_DIR / "sft_chat_model.pt"
DEFAULT_EXAMPLES = SCRIPT_DIR / "sft_val.jsonl"
UNK_TOKEN = "<unk>"
TURN_MARKER = "用户："  # 模型若开始编造下一轮对话，就此截断
MAX_BODY_BYTES = 64 * 1024


# --------------------------------------------------------------------------- #
# 模型
# --------------------------------------------------------------------------- #
class CharLM(nn.Module):
    """与作业脚本里的 LM 完全一致，保证 state_dict 能直接加载。"""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_layers: int,
        model_type: str,
        dropout: float,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        rnn_cls = nn.LSTM if model_type == "lstm" else nn.RNN
        self.rnn = rnn_cls(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.drop(self.embed(token_ids))
        outputs, _ = self.rnn(embedded)
        return self.fc(self.drop(outputs))


def make_prompt(instruction: str) -> str:
    """必须和训练时的模板一字不差。"""
    return f"用户：{instruction}\n助手："


def loop_cut(text: str, max_period: int = 10) -> int | None:
    """检测结尾是否出现「同一片段连续重复 3 次」，返回应截断到的位置。"""
    n = len(text)
    for period in range(1, max_period + 1):
        if n >= period * 3:
            last = text[-period:]
            if last == text[-2 * period : -period] == text[-3 * period : -2 * period]:
                return n - 3 * period
    return None


class ChatEngine:
    """把 checkpoint 装进内存，提供逐字符的流式生成。"""

    def __init__(self, model_path: Path, device_name: str = "auto") -> None:
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(f"找不到模型文件：{self.model_path}")

        try:
            checkpoint = torch.load(self.model_path, map_location="cpu", weights_only=True)
        except TypeError:  # 旧版 torch 没有 weights_only 参数
            checkpoint = torch.load(self.model_path, map_location="cpu")

        self.args = dict(checkpoint.get("args") or {})
        self.sft_config = dict(checkpoint.get("sft_config") or {})
        self.char2idx = {str(k): int(v) for k, v in checkpoint["char2idx"].items()}
        self.idx2char = {int(k): str(v) for k, v in checkpoint["idx2char"].items()}
        self.unk_id = self.char2idx.get(UNK_TOKEN, 0)
        self.vocab_size = len(self.char2idx)

        model_type = str(self.args.get("model", "rnn")).lower()
        if model_type not in {"rnn", "lstm"}:
            raise ValueError(f"不支持的模型类型：{model_type}")

        self.model = CharLM(
            vocab_size=self.vocab_size,
            embed_dim=int(self.args.get("embed_dim", 128)),
            hidden_dim=int(self.args.get("hidden_dim", 256)),
            num_layers=int(self.args.get("num_layers", 2)),
            model_type=model_type,
            dropout=float(self.args.get("dropout", 0.0)),
        )
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()

        self.device = self._pick_device(device_name)
        self.model.to(self.device)
        self.lock = threading.Lock()  # 单模型，串行生成

    @staticmethod
    def _pick_device(name: str) -> torch.device:
        name = (name or "auto").lower()
        if name == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if name.startswith("cuda") and not torch.cuda.is_available():
            print("[提示] 没有可用的 CUDA，已自动切回 CPU。", flush=True)
            return torch.device("cpu")
        return torch.device(name)

    def info(self) -> dict:
        params = sum(p.numel() for p in self.model.parameters())
        val_loss = self.sft_config.get("val_loss")
        return {
            "model_file": self.model_path.name,
            "model_type": "LSTM" if str(self.args.get("model")) == "lstm" else "RNN",
            "vocab_size": self.vocab_size,
            "embed_dim": self.args.get("embed_dim"),
            "hidden_dim": self.args.get("hidden_dim"),
            "num_layers": self.args.get("num_layers"),
            "parameters": params,
            "device": str(self.device),
            "torch": torch.__version__,
            "sft_epoch": self.sft_config.get("epoch"),
            "sft_val_loss": round(float(val_loss), 4) if isinstance(val_loss, (int, float)) else None,
            "prompt_template": "用户：{问题}\\n助手：{回答}",
        }

    # ---------------- 采样 ---------------- #
    @staticmethod
    def _pick(
        logits: torch.Tensor,
        recent_ids: list[int],
        temperature: float,
        top_k: int,
        top_p: float,
        repetition_penalty: float,
    ) -> int:
        scores = logits.detach().float().clone()

        if repetition_penalty and abs(repetition_penalty - 1.0) > 1e-3 and recent_ids:
            ids = torch.tensor(sorted(set(recent_ids)), dtype=torch.long, device=scores.device)
            values = scores.index_select(0, ids)
            adjusted = torch.where(
                values > 0, values / repetition_penalty, values * repetition_penalty
            )
            scores.index_copy_(0, ids, adjusted)

        if temperature <= 0.01:  # 温度 0 = 贪心
            return int(torch.argmax(scores).item())

        scores = scores / max(temperature, 1e-3)

        if 0 < top_k < scores.numel():
            kth = torch.topk(scores, int(top_k)).values[-1]
            scores = torch.where(scores < kth, torch.full_like(scores, float("-inf")), scores)

        probs = torch.softmax(scores, dim=-1)

        if top_p < 1.0:
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            keep = int((cumulative <= top_p).sum().item()) + 1
            keep = max(1, min(keep, sorted_probs.numel()))
            keep_idx = sorted_idx[:keep]
            sub = probs.index_select(0, keep_idx)
            sub = sub / sub.sum()
            choice = int(torch.multinomial(sub, 1).item())
            return int(keep_idx[choice].item())

        return int(torch.multinomial(probs, 1).item())

    # ---------------- 生成 ---------------- #
    def generate(
        self,
        question: str,
        *,
        max_new_chars: int = 160,
        temperature: float = 0.8,
        top_k: int = 40,
        top_p: float = 0.95,
        repetition_penalty: float = 1.15,
        should_stop=None,
    ):
        """逐字符 yield 回答（增量解码：每步只喂 1 个 token，比整段重算快得多）。"""
        prompt = make_prompt(question.strip())
        ids = [self.char2idx.get(ch, self.unk_id) for ch in prompt]
        if not ids:
            return

        hold = len(TURN_MARKER)  # 末尾留几位不急着发送，便于整段截断
        emitted = 0
        text = ""
        recent: list[int] = []

        with self.lock, torch.no_grad():
            x = torch.tensor([ids], dtype=torch.long, device=self.device)
            outputs, hidden = self.model.rnn(self.model.embed(x))
            logits = self.model.fc(outputs)[0, -1]

            for _ in range(max(1, int(max_new_chars))):
                if should_stop is not None and should_stop():
                    break

                token = self._pick(
                    logits, recent[-64:], temperature, top_k, top_p, repetition_penalty
                )
                char = self.idx2char.get(token, "")

                if char == "\n" or char == "":
                    break
                if char != UNK_TOKEN:
                    text += char

                recent.append(token)

                x = torch.tensor([[token]], dtype=torch.long, device=self.device)
                outputs, hidden = self.model.rnn(self.model.embed(x), hidden)
                logits = self.model.fc(outputs)[0, -1]

                # 截断判断：模型开始编造下一轮 / 陷入复读
                cut = None
                marker_at = text.find(TURN_MARKER)
                if marker_at >= 0:
                    cut = marker_at
                else:
                    loop_at = loop_cut(text)
                    if loop_at is not None and loop_at > 3:
                        cut = loop_at

                limit = len(text) if cut is None else cut
                safe_end = min(limit, max(0, len(text) - hold))

                while emitted < safe_end:
                    yield text[emitted]
                    emitted += 1

                if cut is not None:
                    break

            # 收尾：把剩下的部分发完（不越过截断点）
            final = text if TURN_MARKER not in text else text[: text.find(TURN_MARKER)]
            while emitted < len(final):
                yield final[emitted]
                emitted += 1

    def answer(self, question: str, **kwargs) -> str:
        return "".join(self.generate(question, **kwargs))


# --------------------------------------------------------------------------- #
# HTTP 服务
# --------------------------------------------------------------------------- #
def _clamp(value, default, low, high) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = float(default)
    return max(low, min(high, number))


class Handler(BaseHTTPRequestHandler):
    server_version = "SFTCharChat/1.0"
    protocol_version = "HTTP/1.1"
    engine: ChatEngine | None = None
    examples: list[str] = []

    # 安静一点，不刷访问日志
    def log_message(self, fmt: str, *args) -> None:  # noqa: A003
        return

    # ---------------- 工具 ---------------- #
    def _send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_text(self, text: str, content_type: str, status: int = 200) -> None:
        body = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _read_payload(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        if length > MAX_BODY_BYTES:
            raise ValueError("请求体过大")
        raw = self.rfile.read(length)
        data = json.loads(raw.decode("utf-8"))
        if not isinstance(data, dict):
            raise ValueError("请求体必须是 JSON 对象")
        return data

    @staticmethod
    def _params(payload: dict) -> dict:
        return {
            "max_new_chars": int(_clamp(payload.get("max_new_chars"), 160, 8, 800)),
            "temperature": round(_clamp(payload.get("temperature"), 0.8, 0.0, 2.0), 2),
            "top_k": int(_clamp(payload.get("top_k"), 40, 0, 500)),
            "top_p": round(_clamp(payload.get("top_p"), 0.95, 0.05, 1.0), 2),
            "repetition_penalty": round(
                _clamp(payload.get("repetition_penalty"), 1.15, 1.0, 1.6), 2
            ),
        }

    # ---------------- GET ---------------- #
    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?")[0]
        if path in {"/", "/index.html"}:
            self._send_text(HTML_PAGE, "text/html; charset=utf-8")
        elif path == "/api/info":
            self._send_json({"info": self.engine.info()})
        elif path == "/api/examples":
            self._send_json({"examples": self.examples})
        elif path == "/favicon.ico":
            self.send_response(204)
            self.send_header("Content-Length", "0")
            self.end_headers()
        else:
            self._send_json({"error": "未找到该路径"}, status=404)

    # ---------------- POST ---------------- #
    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?")[0]
        try:
            payload = self._read_payload()
        except Exception as exc:  # noqa: BLE001
            self._send_json({"error": f"请求解析失败：{exc}"}, status=400)
            return

        question = payload.get("q") or payload.get("question") or ""
        if not isinstance(question, str) or not question.strip():
            self._send_json({"error": "问题不能为空"}, status=400)
            return
        if len(question) > 3000:
            self._send_json({"error": "问题太长了（最多 3000 字）"}, status=400)
            return

        params = self._params(payload)

        if path == "/api/chat_stream":
            self._stream(question, params)
        elif path == "/api/chat":
            try:
                text = self.engine.answer(question, **params)
            except Exception as exc:  # noqa: BLE001
                self._send_json({"error": f"生成失败：{exc}"}, status=500)
                return
            self._send_json({"text": text})
        else:
            self._send_json({"error": "未找到该路径"}, status=404)

    def _stream(self, question: str, params: dict) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

        stopped = {"flag": False}
        buffer: list[str] = []

        def flush() -> None:
            if not buffer:
                return
            data = "".join(buffer).encode("utf-8")
            buffer.clear()
            self.wfile.write(data)
            self.wfile.flush()

        started = time.perf_counter()
        try:
            buffer.append(json.dumps({"meta": params}, ensure_ascii=False) + "\n")
            flush()

            for piece in self.engine.generate(question, should_stop=lambda: stopped["flag"], **params):
                buffer.append(json.dumps({"t": piece}, ensure_ascii=False) + "\n")
                flush()
            buffer.append(
                json.dumps(
                    {"done": True, "elapsed": round(time.perf_counter() - started, 2)},
                    ensure_ascii=False,
                )
                + "\n"
            )
            flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            stopped["flag"] = True  # 浏览器端点了「停止」
        except Exception as exc:  # noqa: BLE001
            try:
                buffer.append(json.dumps({"error": f"生成失败：{exc}"}, ensure_ascii=False) + "\n")
                flush()
            except Exception:  # noqa: BLE001
                pass


# --------------------------------------------------------------------------- #
# 前端页面
# --------------------------------------------------------------------------- #
HTML_PAGE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>本地问答界面 · SFT 字符级 RNN</title>
<style>
  :root{
    --bg:#0a0e18; --panel:#111a2b; --panel-2:#0d1424; --line:#1e2a44;
    --text:#e9eefc; --muted:#8a9cbd; --accent:#5b8cff; --accent-2:#8b5cff;
    --user:#22396b; --bot:#151f36;
  }
  *{box-sizing:border-box}
  html,body{height:100%}
  body{
    margin:0; background:
      radial-gradient(1200px 600px at 15% -10%, rgba(91,140,255,.18), transparent 60%),
      radial-gradient(900px 500px at 110% 10%, rgba(139,92,255,.16), transparent 55%),
      var(--bg);
    color:var(--text);
    font:15px/1.7 "Microsoft YaHei","PingFang SC",system-ui,-apple-system,"Segoe UI",sans-serif;
    -webkit-font-smoothing:antialiased;
  }
  .app{height:100dvh; max-width:960px; margin:0 auto; display:flex; flex-direction:column}
  header{
    display:flex; align-items:center; justify-content:space-between; gap:12px;
    padding:14px 18px; border-bottom:1px solid var(--line);
    background:rgba(10,14,24,.72); backdrop-filter:blur(10px);
  }
  .brand{display:flex; align-items:center; gap:12px; min-width:0}
  .logo{
    width:40px; height:40px; border-radius:12px; flex:0 0 auto;
    display:grid; place-items:center; font-weight:700; font-size:19px; color:#fff;
    background:linear-gradient(135deg,var(--accent),var(--accent-2));
    box-shadow:0 6px 18px rgba(91,140,255,.35);
  }
  h1{margin:0; font-size:16px; font-weight:700; letter-spacing:.3px}
  .sub{margin:2px 0 0; font-size:12px; color:var(--muted); white-space:nowrap; overflow:hidden; text-overflow:ellipsis}
  .tools{display:flex; gap:8px; flex:0 0 auto}
  button{
    font:inherit; color:var(--text); cursor:pointer; border-radius:10px;
    border:1px solid var(--line); background:var(--panel); padding:7px 12px;
    transition:.15s transform,.15s background,.15s border-color;
  }
  button:hover{background:#16223a; border-color:#2b3d61}
  button:active{transform:translateY(1px)}
  button:disabled{opacity:.45; cursor:not-allowed}
  button.primary{
    border:0; color:#fff; font-weight:600;
    background:linear-gradient(135deg,var(--accent),var(--accent-2));
    box-shadow:0 6px 16px rgba(91,140,255,.3); padding:9px 20px;
  }
  main{flex:1; overflow-y:auto; padding:22px 18px 8px; scroll-behavior:smooth}
  main::-webkit-scrollbar{width:8px}
  main::-webkit-scrollbar-thumb{background:#22304d; border-radius:8px}

  .empty{max-width:620px; margin:6vh auto; text-align:center; color:var(--muted)}
  .empty h2{margin:0 0 8px; font-size:20px; color:var(--text)}
  .empty p{margin:0 0 6px; font-size:14px}
  .empty code{color:#a9c1f5}
  .chips{display:flex; flex-wrap:wrap; gap:8px; justify-content:center; margin-top:18px}
  .chip{
    font-size:13px; color:#c7d6f5; background:var(--panel); border:1px solid var(--line);
    padding:7px 12px; border-radius:999px; cursor:pointer; transition:.15s;
  }
  .chip:hover{border-color:var(--accent); color:#fff; background:#16223a}

  .msg{display:flex; gap:10px; margin:0 0 18px; animation:in .22s ease-out}
  @keyframes in{from{opacity:0; transform:translateY(6px)} to{opacity:1; transform:none}}
  .msg.user{flex-direction:row-reverse}
  .avatar{
    width:32px; height:32px; border-radius:10px; flex:0 0 auto; display:grid; place-items:center;
    font-size:13px; font-weight:700; color:#dfe8ff; background:#1b2745; border:1px solid #24345a;
  }
  .msg.user .avatar{background:linear-gradient(135deg,#2b4a86,#3a63ad); border-color:#3a63ad; color:#fff}
  .col{max-width:min(78%,680px)}
  .bubble{
    padding:11px 15px; border-radius:14px; white-space:pre-wrap; word-break:break-word;
    background:var(--bot); border:1px solid #1d2947;
  }
  .msg.user .bubble{
    background:linear-gradient(135deg,var(--user),#2b4a86); border-color:#34508f; border-top-right-radius:5px;
  }
  .msg.bot .bubble{border-top-left-radius:5px}
  .meta{margin-top:6px; font-size:11.5px; color:#6f82a6; padding:0 4px}
  .msg.user .meta{text-align:right}
  .cursor::after{content:"▍"; color:var(--accent); animation:blink 1s steps(2) infinite}
  @keyframes blink{50%{opacity:0}}
  .dots{display:inline-flex; gap:4px; padding:3px 2px}
  .dots i{width:6px; height:6px; border-radius:50%; background:#5c74a5; animation:bounce 1.2s infinite}
  .dots i:nth-child(2){animation-delay:.15s}
  .dots i:nth-child(3){animation-delay:.3s}
  @keyframes bounce{0%,60%,100%{transform:translateY(0); opacity:.5} 30%{transform:translateY(-4px); opacity:1}}

  .settings{
    display:none; gap:18px; flex-wrap:wrap; padding:12px 18px;
    border-top:1px solid var(--line); background:var(--panel-2);
  }
  .settings.open{display:flex}
  .field{display:flex; flex-direction:column; gap:4px; font-size:12px; color:var(--muted); min-width:150px; flex:1}
  .field .row{display:flex; justify-content:space-between; align-items:center; gap:8px}
  .field b{color:#c9d8f7; font-variant-numeric:tabular-nums}
  input[type=range]{width:100%; accent-color:var(--accent); height:18px}

  footer{padding:12px 18px 16px; border-top:1px solid var(--line); background:rgba(10,14,24,.72); backdrop-filter:blur(10px)}
  .composer{display:flex; gap:10px; align-items:flex-end}
  textarea{
    flex:1; resize:none; min-height:46px; max-height:150px; padding:12px 14px;
    font:inherit; color:var(--text); background:var(--panel); border:1px solid var(--line);
    border-radius:12px; outline:none; transition:.15s border-color,.15s box-shadow;
  }
  textarea:focus{border-color:#3a5c9c; box-shadow:0 0 0 3px rgba(91,140,255,.14)}
  .foot{display:flex; justify-content:space-between; gap:10px; margin-top:8px; font-size:11.5px; color:#61739a}
  .foot .left{white-space:nowrap; overflow:hidden; text-overflow:ellipsis}
  @media (max-width:600px){
    header{padding:10px 12px} main{padding:16px 12px 4px} footer{padding:10px 12px 12px}
    .col{max-width:86%} .sub{display:none}
  }
</style>
</head>
<body>
<div class="app">
  <header>
    <div class="brand">
      <div class="logo">答</div>
      <div style="min-width:0">
        <h1>本地问答助手</h1>
        <p class="sub" id="sub">正在加载模型…</p>
      </div>
    </div>
    <div class="tools">
      <button id="btnSettings" type="button">⚙ 参数</button>
      <button id="btnClear" type="button">清空</button>
    </div>
  </header>

  <main id="chat">
    <div class="empty" id="empty">
      <h2>你好，我是本地训练的小助手</h2>
      <p>模型是字符级 RNN，用 <code>sft_chat_model.pt</code> 在本机推理，不联网。</p>
      <p>直接输入问题，或点下面一个示例开始：</p>
      <div class="chips" id="chips"></div>
    </div>
  </main>

  <div class="settings" id="settings">
    <label class="field">
      <span class="row">温度 <b id="vTemp">0.80</b></span>
      <input id="sTemp" type="range" min="0" max="1.5" step="0.05" value="0.8" />
    </label>
    <label class="field">
      <span class="row">Top-K <b id="vTopk">40</b></span>
      <input id="sTopk" type="range" min="0" max="200" step="5" value="40" />
    </label>
    <label class="field">
      <span class="row">重复惩罚 <b id="vRep">1.15</b></span>
      <input id="sRep" type="range" min="1" max="1.5" step="0.02" value="1.15" />
    </label>
    <label class="field">
      <span class="row">最多字数 <b id="vLen">160</b></span>
      <input id="sLen" type="range" min="32" max="512" step="16" value="160" />
    </label>
  </div>

  <footer>
    <div class="composer">
      <textarea id="input" rows="1" placeholder="输入问题，Enter 发送，Shift+Enter 换行"></textarea>
      <button id="btnStop" type="button" style="display:none">停止</button>
      <button id="btnSend" class="primary" type="button">发送</button>
    </div>
    <div class="foot">
      <span class="left" id="footInfo">字符级 RNN · SFT 微调权重</span>
      <span>Enter 发送 · Shift+Enter 换行</span>
    </div>
  </footer>
</div>

<script>
const chatEl = document.getElementById('chat');
const emptyEl = document.getElementById('empty');
const inputEl = document.getElementById('input');
const btnSend = document.getElementById('btnSend');
const btnStop = document.getElementById('btnStop');
const btnClear = document.getElementById('btnClear');
const btnSettings = document.getElementById('btnSettings');
const settingsEl = document.getElementById('settings');
const subEl = document.getElementById('sub');
const footInfo = document.getElementById('footInfo');

const state = {
  busy: false,
  ctrl: null,
  settings: { temperature: 0.8, top_k: 40, top_p: 0.95, repetition_penalty: 1.15, max_new_chars: 160 }
};

function scrollBottom(){ chatEl.scrollTop = chatEl.scrollHeight; }

function addMessage(role, text){
  emptyEl.style.display = 'none';
  const msg = document.createElement('div');
  msg.className = 'msg ' + role;
  const avatar = document.createElement('div');
  avatar.className = 'avatar';
  avatar.textContent = role === 'user' ? '你' : 'AI';
  const col = document.createElement('div');
  col.className = 'col';
  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.textContent = text;
  const meta = document.createElement('div');
  meta.className = 'meta';
  col.appendChild(bubble);
  col.appendChild(meta);
  msg.appendChild(avatar);
  msg.appendChild(col);
  chatEl.appendChild(msg);
  scrollBottom();
  return bubble;
}

async function streamChat(question, onChunk, signal){
  const payload = Object.assign({ q: question }, state.settings);
  let response;
  try {
    response = await fetch('/api/chat_stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
      signal: signal
    });
    if (!response.ok) throw new Error('HTTP ' + response.status);
  } catch (err) {
    if (err.name === 'AbortError') throw err;
    // 浏览端不支持流式时退回一次性接口
    const res2 = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
      signal: signal
    });
    const data = await res2.json();
    if (data.error) throw new Error(data.error);
    onChunk(data.text || '');
    return;
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder('utf-8');
  let buffered = '';
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffered += decoder.decode(value, { stream: true });
    let index;
    while ((index = buffered.indexOf('\n')) >= 0) {
      const line = buffered.slice(0, index).trim();
      buffered = buffered.slice(index + 1);
      if (!line) continue;
      const message = JSON.parse(line);
      if (message.error) throw new Error(message.error);
      if (message.t) onChunk(message.t);
      if (message.done) return;
    }
  }
}

function setBusy(busy){
  state.busy = busy;
  btnSend.disabled = busy;
  btnStop.style.display = busy ? 'inline-block' : 'none';
  inputEl.disabled = busy;
}

async function send(text){
  text = (text || '').trim();
  if (!text || state.busy) return;

  addMessage('user', text);
  inputEl.value = '';
  autoGrow();

  const bubble = addMessage('bot', '');
  const meta = bubble.parentElement.querySelector('.meta');
  bubble.innerHTML = '<span class="dots"><i></i><i></i><i></i></span>';

  const ctrl = new AbortController();
  state.ctrl = ctrl;
  setBusy(true);
  const startedAt = performance.now();
  let acc = '';

  try {
    await streamChat(text, (piece) => {
      if (acc === '') bubble.textContent = '';
      acc += piece;
      bubble.textContent = acc;
      bubble.classList.add('cursor');
      scrollBottom();
    }, ctrl.signal);
    bubble.classList.remove('cursor');
    const seconds = (performance.now() - startedAt) / 1000;
    meta.textContent = acc.length + ' 字 · 用时 ' + seconds.toFixed(1) + ' 秒';
    if (!acc) {
      bubble.textContent = '（模型没有给出回答，可以调低温度或再试一次）';
      bubble.style.color = '#8a9cbd';
    }
  } catch (err) {
    bubble.classList.remove('cursor');
    if (err.name === 'AbortError') {
      bubble.textContent = acc || '（已停止）';
      meta.textContent = '已手动停止';
    } else {
      bubble.textContent = (acc ? acc + '\n' : '') + '出错了：' + err.message;
      bubble.style.color = '#ff8f8f';
      meta.textContent = '请求失败';
    }
  } finally {
    state.ctrl = null;
    setBusy(false);
    scrollBottom();
    inputEl.focus();
  }
}

function autoGrow(){
  inputEl.style.height = 'auto';
  inputEl.style.height = Math.min(inputEl.scrollHeight, 150) + 'px';
}

btnSend.addEventListener('click', () => send(inputEl.value));
btnStop.addEventListener('click', () => { if (state.ctrl) state.ctrl.abort(); });
btnClear.addEventListener('click', () => {
  chatEl.querySelectorAll('.msg').forEach((node) => node.remove());
  emptyEl.style.display = 'block';
  inputEl.focus();
});
btnSettings.addEventListener('click', () => settingsEl.classList.toggle('open'));

inputEl.addEventListener('input', autoGrow);
inputEl.addEventListener('keydown', (event) => {
  if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    send(inputEl.value);
  }
});

const bindings = [
  ['sTemp', 'vTemp', 'temperature', (v) => v.toFixed(2)],
  ['sTopk', 'vTopk', 'top_k', (v) => String(v)],
  ['sRep', 'vRep', 'repetition_penalty', (v) => v.toFixed(2)],
  ['sLen', 'vLen', 'max_new_chars', (v) => String(v)]
];
bindings.forEach(([inputId, labelId, key, fmt]) => {
  const slider = document.getElementById(inputId);
  const label = document.getElementById(labelId);
  const apply = () => {
    const value = parseFloat(slider.value);
    state.settings[key] = value;
    label.textContent = fmt(value);
  };
  slider.addEventListener('input', apply);
  apply();
});

(async function bootstrap(){
  try {
    const response = await fetch('/api/info');
    const data = await response.json();
    const info = data.info || {};
    const params = info.parameters ? (info.parameters / 1e6).toFixed(2) + 'M' : '?';
    subEl.textContent = info.model_type + ' · ' + params + ' 参数 · ' + info.device +
      (info.sft_val_loss ? ' · val_loss ' + info.sft_val_loss : '');
    footInfo.textContent = '模型 ' + info.model_file + ' · 词表 ' + info.vocab_size +
      ' · torch ' + info.torch;
  } catch (err) {
    subEl.textContent = '模型信息读取失败';
  }

  try {
    const response = await fetch('/api/examples');
    const data = await response.json();
    const chips = document.getElementById('chips');
    (data.examples || []).forEach((example) => {
      const chip = document.createElement('div');
      chip.className = 'chip';
      chip.textContent = example.length > 34 ? example.slice(0, 34) + '…' : example;
      chip.title = example;
      chip.addEventListener('click', () => send(example));
      chips.appendChild(chip);
    });
  } catch (err) { /* 示例加载失败不影响使用 */ }

  // 支持 http://127.0.0.1:8000/?q=你的问题 自动提问
  const preset = new URLSearchParams(location.search).get('q');
  if (preset) { send(preset); } else { inputEl.focus(); }
})();
</script>
</body>
</html>
"""


# --------------------------------------------------------------------------- #
# 启动
# --------------------------------------------------------------------------- #
def load_examples(path: Path, limit: int = 5) -> list[str]:
    if not path.is_file():
        return [
            "给定两个数字，请选择其中的较大数并将其打印出来。 数字1: 238 数字2: 564",
            "计算一组数字的平均值和标准差。 数字：23, 45, 67, 89, 12",
            "将以下句子转换为被动语态。 句子： Tom 发现了一份工作。",
        ]
    items: list[str] = []
    try:
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                instruction = str(row.get("instruction", "")).strip()
                if instruction:
                    items.append(instruction)
                if len(items) >= limit:
                    break
    except Exception:  # noqa: BLE001
        pass
    return items


def find_free_port(preferred: int, host: str = "127.0.0.1") -> int:
    for port in range(preferred, preferred + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((host, port))
            except OSError:
                continue
            return port
    raise SystemExit("附近端口都被占用了，请用 --port 指定一个空闲端口。")


def main() -> None:
    parser = argparse.ArgumentParser(description="启动本地问答界面（字符级 RNN 模型）")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="模型权重路径")
    parser.add_argument("--port", type=int, default=8000, help="监听端口，默认 8000 起自动找空闲端口")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址，默认仅本机可访问")
    parser.add_argument("--device", default="auto", help="auto / cpu / cuda")
    parser.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    args = parser.parse_args()

    # 控制台编码自适应：能显示中文就直接用；不能（例如纯 ASCII）才切 UTF-8
    if hasattr(sys.stdout, "reconfigure"):
        try:
            encoding = (sys.stdout.encoding or "").lower()
            if encoding in {"", "ascii", "ansi_x3.4-1968", "us-ascii"}:
                sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            else:
                sys.stdout.reconfigure(errors="replace")
        except Exception:  # noqa: BLE001
            pass

    print("正在加载模型…", flush=True)
    engine = ChatEngine(args.model, device_name=args.device)
    info = engine.info()

    Handler.engine = engine
    Handler.examples = load_examples(DEFAULT_EXAMPLES)

    port = find_free_port(args.port, args.host)
    httpd = ThreadingHTTPServer((args.host, port), Handler)
    url = f"http://{args.host}:{port}/"

    print(f"模型：{info['model_file']}（{info['model_type']}，{info['vocab_size']} 字符，"
          f"{info['parameters'] / 1e6:.2f}M 参数）", flush=True)
    print(f"设备：{info['device']}", flush=True)
    print(f"问答界面已启动：{url}", flush=True)
    print("按 Ctrl+C 关闭服务。", flush=True)

    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n已关闭。", flush=True)
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
