"""
L4 LLM API zero-shot / few-shot 事件抽取（CEC）

和规则 / BiLSTM-CRF / BERT / 两段式是同任务、同数据的对照：
  前四个都在「训练」一个判别式模型；这里不训练，只靠提示词让大模型直接读句抽事件。
  它是「大模型到底能免费帮你做到什么程度」的基线，也是 few-shot 值多少点的对照组。

参考 week7-信息抽取/code/llm_zero_shot 的实现，按 CEC 做了三处适配：
  1. 数据/schema 走本项目 common/event_data.py（CEC 是 {类型:[角色]} 的 dict）；
  2. 评测走 common/eval_utils.evaluate_predictions —— 和另外四种方法同一口径、同一张表；
  3. 后端只保留 api（OpenAI 兼容）+ mock（--dry-run 离线演练），去掉本地 Qwen 分支
     （本作业要对比的是「LLM API」，本地小模型是另一条线）。

教学重点（为什么这么写）：
  1. 只让模型输出**文本 span**，不让它输出 offset：生成式偏移不可信，程序用 text.find()
     回原文对齐；找不到 = 幻觉，记数但不强行纠正。
  2. 结构化输出的健壮性：剥 ```json 围栏、括号配平、截断补全、去尾逗号……
     并统计「JSON 合法率 / 修复率 / 空回复 / 幻觉论元」——这些工程指标比 F1 更能说明成本。
  3. zero-shot vs few-shot：示例是「免费」的领域适配，但吃 prompt 长度（成本/延迟）。
     few-shot 示例只在 train 里挑，绝不从 dev 挑（否则是评测集泄漏）。

后端与环境变量（代码里不写死任何 key / 内网域名）：
  LLM_API_KEY    （或 GLM_API_KEY / DEEPSEEK_API_KEY）
  LLM_BASE_URL   留空时：有 GLM key → 智谱 endpoint；否则 DeepSeek
  LLM_MODEL      留空时：GLM → glm-4-flash；否则 deepseek-chat
  会自动读取 week07/.env（以及 week06/.env 作为便利回退，真实环境变量优先）。

用法：
  python llm_zero_shot/llm_extract.py --dry-run --mode zero          # 离线演练，不联网
  python llm_zero_shot/llm_extract.py --mode zero                    # API zero-shot，全 dev(125)
  python llm_zero_shot/llm_extract.py --mode few                     # API few-shot
  python llm_zero_shot/llm_extract.py --print-prompt                 # 只打印 system prompt
  python llm_zero_shot/llm_extract.py --mode zero --n 40             # 只跑前 40 条（省成本）

产物：
  outputs/disc/pred_<name>.jsonl         统一格式预测（evaluate_predictions 写）
  outputs/metrics/disc/<name>.json       三级指标（trigger/argument/event_exact + type_macro）
  outputs/llm/raw_<name>.jsonl           模型原始输出（bad case 分析用）
  outputs/llm/report_<name>.json         工程统计（合法率/延迟/token/幻觉数）
  outputs/llm/gold_n<N>.jsonl            子集评测时的金标子集（全量时用 dev.jsonl）
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]   # week07/
sys.path.insert(0, str(ROOT))                 # 让 `import common.*` 生效
sys.path.insert(0, str(ROOT / "common"))     # 让 common 内部 `from crf/event_data import ...` 生效

from common.eval_utils import evaluate_predictions  # noqa: E402
from common.event_data import (  # noqa: E402
    DATA_DIR, load_extract_split, load_schema, make_pred_row, row_event,
)

SEED = 42
OUTPUT_LLM_DIR = ROOT / "outputs" / "llm"


# ─────────────────────────── 环境变量 / .env ───────────────────────────
def load_env_files() -> None:
    """读 .env（不覆盖已存在的真实环境变量）。没有 python-dotenv 也能用。"""
    for cand in (ROOT / ".env", ROOT.parent / ".env"):
        if not cand.exists():
            continue
        for line in cand.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def resolve_api_config(args) -> tuple[str, str, str]:
    """返回 (api_key, base_url, model)。key 缺失时给清晰报错。"""
    key = (os.getenv("LLM_API_KEY") or os.getenv("GLM_API_KEY")
           or os.getenv("DEEPSEEK_API_KEY"))
    if not key:
        raise EnvironmentError(
            "未找到 API Key。请在 week07/.env 写入一行 `GLM_API_KEY=...`，"
            "或设置环境变量 LLM_API_KEY / GLM_API_KEY / DEEPSEEK_API_KEY；\n"
            "只想先跑通流程可以加 --dry-run（离线假模型，不联网）。")

    is_glm = bool(os.getenv("GLM_API_KEY")) and not (os.getenv("LLM_API_KEY")
                                                     or os.getenv("DEEPSEEK_API_KEY"))
    base_url = os.getenv("LLM_BASE_URL") or (
        "https://open.bigmodel.cn/api/paas/v4" if is_glm else "https://api.deepseek.com")
    model = args.model or os.getenv("LLM_MODEL") or (
        "glm-4-flash" if is_glm else "deepseek-chat")
    return key, base_url, model


# ─────────────────────────── 提示词 ───────────────────────────
SYSTEM_PROMPT_TEMPLATE = """你是中文突发事件新闻的事件抽取专家。输入是**一句**新闻，请抽取其中发生的**一个**事件。

## 事件类型与论元角色（共 {n_types} 类）
{type_lines}

## 抽取要求
1. 先判断事件类型（每句只对应一个事件）。**只能使用上面列出的 {n_types} 个类型名，逐字照抄；禁止自造类型名**（例如“爆炸”类袭击请归入“恐怖袭击”，不要输出“爆炸”）；都不符合时也要选最接近的一类。
2. 触发词：原文中最能标志该事件发生的**核心词**，通常只有 1~4 个字（如“地震”“火灾”“交通事故”“起火”“撞”“中毒”）。必须逐字来自原文；**不要**包含“发生/一起/导致/造成/引发/突然”等修饰或整个短语、整句照抄，只要那个核心词本身。
3. 论元：抽取该类型下出现的角色及其原文片段，同样必须逐字来自原文：
   - 原文里没有出现的角色，直接省略，不要猜、不要补全、不要用常识编；
   - 同一角色可以有多个论元时，用多条记录分别表示。
4. **不要输出字符下标（start/end）**：生成的偏移不可靠，程序会用 text.find() 在原文里重新对齐。
5. 数字、日期、人数要与原文写法一致（如“3人死亡”“2023年5月12日”）。

## 输出格式（严格，只输出一个 JSON 对象，不要解释、不要加 markdown 代码块）
{{"events": [{{"type": "事件类型", "trigger": {{"text": "触发词"}}, "arguments": [{{"role": "论元角色", "argument": "论元文本"}}]}}]}}
如果一个事件都抽不到，输出：{{"events": []}}"""


def schema_type_lines(schema: dict) -> str:
    """把 {类型:[角色]} 渲染成一行一类型。分隔符用「、」，避免与字段值冲突。"""
    return "\n".join(f"- {et}：" + "、".join(roles) for et, roles in schema.items())


def build_system_prompt(schema: dict) -> str:
    return SYSTEM_PROMPT_TEMPLATE.format(n_types=len(schema),
                                         type_lines=schema_type_lines(schema))


def event_to_target(event: dict) -> str:
    """金标事件 → few-shot 的标准答案串（字段名用 arguments/argument，不含 offset）。"""
    trig = (event.get("trigger") or {})
    trig_text = trig.get("text", "") if isinstance(trig, dict) else str(trig)
    args = [{"role": a.get("role", ""), "argument": a.get("text", "")}
            for a in event.get("args") or [] if a.get("role") and a.get("text")]
    obj = {"events": [{"type": event.get("type", ""),
                       "trigger": {"text": trig_text}, "arguments": args}]}
    return json.dumps(obj, ensure_ascii=False)


def build_few_shot(train_rows: list[dict], schema: dict, seed: int = SEED) -> list[dict]:
    """每类从 train 里挑 1 条作示例（确定性；绝不从 dev 挑，避免泄漏）。"""
    by_type = {}
    for r in train_rows:
        ev = row_event(r)
        if ev and ev.get("type") in schema and ev["type"] not in by_type:
            by_type[ev["type"]] = r
    examples = []
    for et in schema:
        r = by_type.get(et)
        if r:
            examples.append({"text": r["text"], "target": event_to_target(row_event(r))})
    return examples[:6]


def build_messages(text: str, schema: dict, few_shot_examples: list[dict]) -> list[dict]:
    messages = [{"role": "system", "content": build_system_prompt(schema)}]
    for ex in few_shot_examples:
        messages.append({"role": "user", "content": f"句子：{ex['text']}"})
        messages.append({"role": "assistant", "content": ex["target"]})
    messages.append({"role": "user", "content": f"句子：{text}"})
    return messages


# ─────────────────────────── 解析（生成文本 → 结构化事件）───────────────────────────
def _strip_code_fence(raw: str) -> str:
    s = raw.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9_+-]*\s*", "", s)
        s = re.sub(r"\s*```\s*$", "", s)
    return s.strip()


def _first_json_object(s: str) -> str | None:
    """从第一个 '{' 做括号配平扫描，返回完整对象；截断时返回 None。"""
    start = s.find("{")
    if start < 0:
        return None
    depth, in_str, esc = 0, False, False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[start:i + 1]
    return None


def _light_repair(s: str) -> str:
    """只去掉对象/数组末尾多余逗号（JSON 不许尾逗号）。"""
    return re.sub(r",\s*([}\]])", r"\1", s)


def _repair_brackets(s: str) -> str | None:
    """栈式括号修复：处理「少一个右括号」这种最常见的畸形 JSON。"""
    start = s.find("{")
    if start < 0:
        return None
    out, stack, in_str, esc = [], [], False, False
    for ch in s[start:]:
        if in_str:
            out.append(ch)
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            out.append(ch)
        elif ch in "{[":
            stack.append(ch)
            out.append(ch)
        elif ch in "}]":
            want = "{" if ch == "}" else "["
            if stack and stack[-1] == want:
                stack.pop()
                out.append(ch)
            elif stack:
                while stack and stack[-1] != want:
                    out.append("}" if stack.pop() == "{" else "]")
                if stack:
                    stack.pop()
                out.append(ch)
        else:
            out.append(ch)
    if in_str:
        out.append('"')
    fixed = re.sub(r",\s*$", "", "".join(out))
    while stack:
        fixed += "}" if stack.pop() == "{" else "]"
    return fixed


def _close_truncated(s: str) -> str | None:
    """输出被截断时，补引号 + 补右括号（尽力而为，最坏情况丢最后一个论元）。"""
    start = s.find("{")
    if start < 0:
        return None
    depth, in_str, esc, stack = 0, False, False, []
    for ch in s[start:]:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]" and stack:
            stack.pop()
    if not stack and not in_str:
        return None
    fixed = s[start:]
    if in_str:
        fixed += '"'
    fixed = re.sub(r',\s*"[^"]*"\s*:\s*$', "", fixed)
    fixed = re.sub(r",\s*$", "", fixed)
    for ch in reversed(stack):
        fixed += "}" if ch == "{" else "]"
    return fixed


def _try_loads(candidate: str):
    for text, repaired in ((candidate, False), (_light_repair(candidate), True)):
        try:
            return json.loads(text), repaired
        except json.JSONDecodeError:
            continue
    for fixer in (_repair_brackets, _close_truncated):
        fixed = fixer(candidate)
        if fixed:
            try:
                return json.loads(_light_repair(fixed)), True
            except json.JSONDecodeError:
                continue
    return None, False


def _align(text: str, span: str) -> tuple[int, int]:
    idx = text.find(span)
    return (idx, idx + len(span)) if idx >= 0 else (-1, -1)


def _to_event_list(obj, schema: dict, source_text: str, info: dict) -> list[dict]:
    """把解析出的对象规整成事件列表，兼容 events/event/裸对象、arguments/args 等写法。"""
    if isinstance(obj, dict) and isinstance(obj.get("events"), list):
        raw_events = obj["events"]
    elif isinstance(obj, dict) and isinstance(obj.get("event"), dict):
        raw_events = [obj["event"]]
    elif isinstance(obj, dict) and ("type" in obj or "事件类型" in obj):
        raw_events = [obj]
    elif isinstance(obj, list):
        raw_events = obj
    else:
        raw_events = []

    events = []
    for ev in raw_events:
        if not isinstance(ev, dict):
            continue
        etype = str(ev.get("type", ev.get("event_type", ev.get("事件类型", "")))).strip()
        if not etype:
            info["type_invalid"] += 1
            continue
        if etype not in schema:
            info["type_invalid"] += 1   # 不替模型纠正，原样留下（评测算 FP，才诚实）

        trig = ev.get("trigger", ev.get("触发词", ""))
        trig_text = (str(trig.get("text", trig.get("word", "")) or "").strip()
                     if isinstance(trig, dict) else str(trig or "").strip())
        if not trig_text:
            continue   # 没有触发词就没有评测意义
        t_start, t_end = _align(source_text, trig_text)
        if t_start < 0:
            info["trigger_not_in_source"] += 1

        raw_args = ev.get("arguments")
        if raw_args is None:
            raw_args = ev.get("args")
        if raw_args is None:
            raw_args = ev.get("论元") or []
        if isinstance(raw_args, dict):
            raw_args = [{"role": k, "argument": v} for k, v in raw_args.items()]

        args, seen = [], set()
        for a in raw_args:
            if isinstance(a, str):
                info["role_invalid"] += 1
                continue
            if not isinstance(a, dict):
                continue
            role = str(a.get("role", a.get("type", "")) or "").strip()
            text = str(a.get("argument", a.get("text", a.get("论元", ""))) or "").strip()
            if not role or not text:
                info["role_invalid"] += 1
                continue
            if etype in schema and role not in schema[etype]:
                info["role_invalid"] += 1
            if (role, text) in seen:
                continue
            seen.add((role, text))
            if source_text.find(text) < 0:
                info["arg_not_in_source"] += 1
            a_start, a_end = _align(source_text, text)
            args.append({"role": role, "text": text, "start": a_start, "end": a_end})

        events.append({"type": etype,
                       "trigger": {"text": trig_text, "start": t_start, "end": t_end},
                       "args": args})
    return events


def parse_model_output(raw: str, source_text: str, schema: dict) -> tuple[list[dict], dict]:
    info = {"empty_reply": False, "json_ok": False, "repaired": False, "noise_stripped": False,
            "type_invalid": 0, "role_invalid": 0,
            "trigger_not_in_source": 0, "arg_not_in_source": 0}
    if not raw or not raw.strip():
        info["empty_reply"] = True
        return [], info

    cleaned = _strip_code_fence(raw)
    candidate = _first_json_object(cleaned)
    info["noise_stripped"] = bool(candidate and candidate != raw.strip())
    obj, repaired = (None, False)
    if candidate:
        obj, repaired = _try_loads(candidate)
    if obj is None:
        obj, repaired = _try_loads(cleaned)   # 兜底：整体截断补全
    if obj is None:
        return [], info

    info["json_ok"] = True
    info["repaired"] = bool(repaired)
    return _to_event_list(obj, schema, source_text, info), info


# ─────────────────────────── 后端 ───────────────────────────
def call_api(client, messages, model, max_tokens, retries) -> dict:
    """调一次补全；限流/超时/网络抖动指数退避，失败返回空文本（不删行）。"""
    extra_body = None
    if os.getenv("LLM_DISABLE_THINKING", "0") == "1":   # 仅推理模型需要显式关思考
        extra_body = {"thinking": {"type": "disabled"}}
    last_err, t0 = "", time.time()
    for attempt in range(retries):
        try:
            kwargs = dict(model=model, messages=messages, temperature=0.0,
                          max_tokens=max_tokens)
            if extra_body:
                kwargs["extra_body"] = extra_body
            resp = client.chat.completions.create(**kwargs)
            usage = getattr(resp, "usage", None)
            return {"text": (resp.choices[0].message.content or "").strip(),
                    "prompt_tokens": getattr(usage, "prompt_tokens", 0) if usage else 0,
                    "completion_tokens": getattr(usage, "completion_tokens", 0) if usage else 0,
                    "latency_s": time.time() - t0, "error": ""}
        except Exception as exc:  # noqa: BLE001
            last_err = f"{type(exc).__name__}: {exc}"
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    return {"text": "", "prompt_tokens": 0, "completion_tokens": 0,
            "latency_s": time.time() - t0, "error": last_err}


def mock_generate(row: dict, idx: int, seed: int, schema: dict) -> str:
    """离线假模型：按金标造一个"像样"的输出，并故意注入常见毛病（仅验证流水线）。"""
    rng = random.Random(seed * 100000 + idx)
    ev = row_event(row) or {}
    keep = max(0, len(ev.get("args") or []) // 2)
    args = [{"role": a["role"], "argument": a["text"]} for a in (ev.get("args") or [])[:keep]]
    body = json.dumps({"events": [{"type": ev.get("type", ""),
                                   "trigger": {"text": (ev.get("trigger") or {}).get("text", "")},
                                   "arguments": args}]}, ensure_ascii=False)
    roll = rng.random()
    if roll < 0.05:
        return ""
    if roll < 0.10:
        return body[: int(len(body) * 0.8)]
    if roll < 0.20:
        return f"```json\n{body}\n```"
    if roll < 0.25:
        return f"好的，抽取结果如下：\n{body}"
    if roll < 0.30:
        other = rng.choice([t for t in schema if t != ev.get("type")])
        return json.dumps({"events": [{"type": other, "trigger": {"text": ""}}]},
                          ensure_ascii=False)
    return body


# ─────────────────────────── 采样 / 路径 ───────────────────────────
def canonical_sample(rows: list[dict], n: int, seed: int = SEED) -> list[dict]:
    """先按 (id,sent_id) 排序再 shuffle，取前 n —— 与文件行序无关、可复现的嵌套子集。"""
    ordered = sorted(rows, key=lambda r: (str(r.get("id", "")), str(r.get("sent_id", ""))))
    random.Random(seed).shuffle(ordered)
    return ordered if n <= 0 else ordered[:min(n, len(ordered))]


def gold_path_for(data_dir: Path, split: str) -> Path:
    for name in (f"{split}_extract.jsonl", f"{split}.jsonl"):
        p = data_dir / name
        if p.exists():
            return p
    return data_dir / f"{split}.jsonl"


# ─────────────────────────── 主流程 ───────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="LLM API zero-shot / few-shot 事件抽取（CEC）")
    p.add_argument("--backend", default="api", choices=["api", "mock"],
                   help="api=OpenAI 兼容 API；mock=离线假模型（--dry-run 等价）")
    p.add_argument("--mode", default="zero", choices=["zero", "few"])
    p.add_argument("--n", default=0, type=int,
                   help="评测样本数；0=全 dev（125 条，与其它方法同口径）")
    p.add_argument("--split", default="dev", choices=["dev", "test", "train"])
    p.add_argument("--seed", default=SEED, type=int)
    p.add_argument("--model", default="", help="模型名；留空按环境变量/厂商自动选")
    p.add_argument("--max_tokens", default=512, type=int)
    p.add_argument("--retries", default=3, type=int)
    p.add_argument("--workers", default=4, type=int, help="并发请求数（1=顺序）")
    p.add_argument("--name", default="", help="评测名（默认 llm_zero_shot / llm_few_shot）")
    p.add_argument("--data_dir", default="", help="数据目录；留空=data/processed")
    p.add_argument("--dry_run", "--dry-run", dest="dry_run", action="store_true",
                   help="离线演练：不联网，用假模型打通流程（数字无意义）")
    p.add_argument("--print_prompt", "--print-prompt", dest="print_prompt",
                   action="store_true", help="只打印 system prompt 后退出")
    p.add_argument("--no_eval", action="store_true", help="只生成预测，不跑统一评测")
    p.add_argument("--from_raw", default="", help="复用已有 raw_*.jsonl，只重跑解析+评测（不调模型）")
    return p.parse_args()


def main():
    load_env_files()
    args = parse_args()
    if args.dry_run or args.backend == "mock":
        args.backend = "mock"

    data_dir = Path(args.data_dir) if args.data_dir else DATA_DIR
    schema = load_schema()
    if args.print_prompt:
        print(build_system_prompt(schema))
        return

    all_rows = load_extract_split(args.split, data_dir)
    rows = canonical_sample(all_rows, args.n, args.seed)
    few = build_few_shot(load_extract_split("train", data_dir), schema, args.seed) \
        if args.mode == "few" else []

    OUTPUT_LLM_DIR.mkdir(parents=True, exist_ok=True)
    name = args.name or ("llm_zero_shot" if args.mode == "zero" else "llm_few_shot")
    if args.backend == "mock":
        name += "_mock"
    raw_path = OUTPUT_LLM_DIR / f"raw_{name}.jsonl"
    report_path = OUTPUT_LLM_DIR / f"report_{name}.json"

    # 子集评测要单独写金标子集，否则未采样到的句子会被算成全漏、F1 被压低
    if 0 < len(rows) < len(all_rows):
        gold_path = OUTPUT_LLM_DIR / f"gold_n{len(rows)}.jsonl"
        with open(gold_path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    else:
        gold_path = gold_path_for(data_dir, args.split)

    print(f"数据：{args.split} 采样 {len(rows)}/{len(all_rows)} 条 | 类型 {len(schema)} 类 | "
          f"backend={args.backend} mode={args.mode}")
    print(f"评测名：{name} | gold={gold_path.name if hasattr(gold_path, 'name') else gold_path}")

    # ── 后端准备 ──
    client = None
    if args.from_raw:
        raw_map = {}
        for line in open(args.from_raw, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                raw_map[f'{r.get("id")}|{r.get("sent_id")}'] = r
        print(f"[from_raw] 复用 {Path(args.from_raw).name} 的 {len(raw_map)} 条输出，只重解析")
    elif args.backend == "api":
        from openai import OpenAI
        key, base_url, model = resolve_api_config(args)
        args.model = model
        client = OpenAI(api_key=key, base_url=base_url, timeout=60.0)
        print(f"[api] base_url={base_url} model={model} max_tokens={args.max_tokens} "
              f"few_shot={bool(few)} workers={args.workers}")
    else:
        print("[mock] 离线假模型：不联网 —— 数字仅用于验证流水线，不可写进实验结论！")

    # ── 推理 ──
    stat_keys = ("empty_reply", "json_ok", "repaired", "noise_stripped", "type_invalid",
                 "role_invalid", "trigger_not_in_source", "arg_not_in_source")
    stats = {k: 0 for k in stat_keys}
    latencies, prompt_toks, completion_toks, api_errors = [], 0, 0, []

    def one(i_row):
        i, row = i_row
        if args.from_raw:
            prev = raw_map.get(f'{row.get("id")}|{row.get("sent_id")}', {})
            return {"text": prev.get("raw", ""), "prompt_tokens": 0, "completion_tokens": 0,
                    "latency_s": float(prev.get("latency_s", 0.0) or 0.0), "error": ""}
        if args.backend == "mock":
            return {"text": mock_generate(row, i + 1, args.seed, schema),
                    "prompt_tokens": 0, "completion_tokens": 0, "latency_s": 0.0, "error": ""}
        msgs = build_messages(row["text"], schema, few)
        return call_api(client, msgs, args.model, args.max_tokens, args.retries)

    t_start = time.time()
    indexed = list(enumerate(rows))
    if args.workers > 1 and not args.from_raw and args.backend == "api":
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            outs = list(pool.map(one, indexed))
    else:
        outs = [one(x) for x in indexed]

    preds = []
    with open(raw_path, "w", encoding="utf-8") as f_raw:
        for (i, row), out in zip(indexed, outs):
            if out.get("error"):
                api_errors.append({"i": i + 1, "error": out["error"]})
            events, info = parse_model_output(out["text"], row["text"], schema)
            for k in stat_keys:
                stats[k] += int(info.get(k, 0) or 0)
            latencies.append(out["latency_s"])
            prompt_toks += out["prompt_tokens"]
            completion_toks += out["completion_tokens"]
            preds.append(make_pred_row(row, events))
            f_raw.write(json.dumps({"id": row.get("id"), "sent_id": row.get("sent_id"),
                                    "text": row.get("text"), "raw": out["text"],
                                    "events": events,
                                    "latency_s": round(out["latency_s"], 3)},
                                   ensure_ascii=False) + "\n")
            if (i + 1) % 10 == 0 or i + 1 == len(rows):
                print(f"  [{i + 1:3d}/{len(rows)}] 合法 {stats['json_ok']} 条，"
                      f"平均 {(time.time() - t_start) / (i + 1):.2f}s/条", flush=True)

    n = max(len(rows), 1)
    lat_sorted = sorted(latencies)
    report = {
        "name": name, "backend": args.backend, "mode": args.mode, "n": len(rows),
        "model": "mock" if args.backend == "mock" else args.model,
        "seed": args.seed, "few_shot_examples": len(few),
        "json_valid_rate": round(100.0 * stats["json_ok"] / n, 2),
        "parse_fail": len(rows) - stats["json_ok"],
        "empty_reply": stats["empty_reply"], "repaired": stats["repaired"],
        "noise_stripped": stats["noise_stripped"], "type_invalid": stats["type_invalid"],
        "role_invalid": stats["role_invalid"],
        "trigger_not_in_source": stats["trigger_not_in_source"],
        "arg_not_in_source": stats["arg_not_in_source"],
        "latency_mean_s": round(sum(latencies) / n, 3),
        "latency_p50_s": round(lat_sorted[len(lat_sorted) // 2], 3),
        "total_s": round(time.time() - t_start, 1),
        "prompt_tokens": prompt_toks, "completion_tokens": completion_toks,
        "total_tokens": prompt_toks + completion_toks, "api_errors": len(api_errors),
        "is_mock": args.backend == "mock",
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n统计 → {report_path}（总耗时 {report['total_s']:.0f}s）")
    if api_errors[:3]:
        print("API 错误示例：", json.dumps(api_errors[:3], ensure_ascii=False)[:400])

    # ── 统一评测（和另外四种方法同一张表）──
    if args.no_eval:
        return
    meta = {
        "display_name": f"LLM API {'零样本' if args.mode == 'zero' else '少样本'}（{report['model']}）",
        "train_s": 0, "params_m": 0,
        "backend": args.backend, "mode": args.mode, "pretrained": True,
        "n": len(rows), "few_shot_examples": len(few),
        "json_valid_rate": report["json_valid_rate"], "parse_fail": report["parse_fail"],
        "empty_reply": report["empty_reply"], "repaired": report["repaired"],
        "arg_not_in_source": report["arg_not_in_source"],
        "trigger_not_in_source": report["trigger_not_in_source"],
        "ms_per_sentence": round(1000 * report["latency_mean_s"], 1),
        "total_tokens": report["total_tokens"], "is_mock": report["is_mock"],
    }
    res = evaluate_predictions(name, preds, meta=meta, gold_path=gold_path)
    print("\n" + "=" * 60)
    print(f"{name}  trigger F1={res['trigger']['f1']:.4f}  "
          f"argument F1={res['argument']['f1']:.4f}  event F1={res['event_exact']['f1']:.4f}  "
          f"type_macro={res.get('type_macro_f1', 0):.4f}")
    if report["is_mock"]:
        print("⚠ 本次是 mock/--dry-run：指标只证明流水线通了，不能当实验结果。")


if __name__ == "__main__":
    main()
