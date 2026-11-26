"""DeepSeek API（OpenAI 兼容接口）零样本 / 少样本抽取，固定 300 条测试子集。

API key 读取顺序：环境变量 DEEPSEEK_API_KEY → week07/.env（已在 .gitignore 中）。
成功响应逐条缓存到 results/api_raw/<method>.jsonl；重跑只请求缺失或 prompt 已变化的样本，
失败样本不进缓存、以空预测留在分母里，下次运行会重试。
"""

import argparse
import hashlib
import json
import os
import statistics
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

from common import ROOT, api_subset_ids, config, load_data, read_jsonl, save_result
from llm_common import (SYSTEM_PROMPT, build_messages, parse_to_entities, pick_few_shot,
                        span_policy, summarize_parse_stats)


class FatalAPIError(RuntimeError):
    """鉴权失败、模型名错误等重试无意义的错误：立即中止，避免批量写入空结果。"""


def find_api_key():
    """环境变量优先，其次 week07/.env；未配置或值为空时返回 None。"""
    key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    env_file = ROOT / ".env"
    if not key and env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            name, _, value = line.strip().partition("=")
            if name.strip() == "DEEPSEEK_API_KEY":
                key = value.strip().strip("\"'")
    return key or None


def api_key():
    key = find_api_key()
    if not key:
        raise SystemExit("未找到 DEEPSEEK_API_KEY：请在 week07/.env 的等号后填写，或设置同名环境变量")
    return key


def request(method, url, key, payload=None, timeout=60):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={
        "Authorization": f"Bearer {key}", "Content-Type": "application/json",
        "User-Agent": "week07-homework"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def chat(messages, key, api):
    payload = {"model": api["model"], "messages": messages, "temperature": 0.0,
               "max_tokens": api["max_tokens"], "stream": False} | api.get("extra_body", {})
    last_error = ""
    for attempt in range(api["retries"]):
        start = time.perf_counter()
        try:
            body = request("POST", api["base_url"].rstrip("/") + "/chat/completions", key, payload,
                           api["timeout_seconds"])
            choice = body["choices"][0]
            return {"text": choice["message"].get("content") or "",
                    "finish_reason": choice.get("finish_reason"),
                    "usage": body.get("usage", {}), "model": body.get("model"),
                    "latency_s": time.perf_counter() - start}
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:300]
            if exc.code in (400, 401, 402, 403, 404, 422):
                raise FatalAPIError(f"HTTP {exc.code}: {detail}") from exc
            last_error = f"HTTP {exc.code}: {detail}"
        except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        time.sleep(2 ** attempt)
    return {"error": last_error}


def prompt_hash(messages, api):
    raw = json.dumps({"model": api["model"], "max_tokens": api["max_tokens"],
                      "extra_body": api.get("extra_body", {}), "messages": messages},
                     ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()


def check(key, api):
    body = request("GET", api["base_url"].rstrip("/") + "/models", key, timeout=30)
    print("可用模型：", [m["id"] for m in body.get("data", [])])
    print("当前配置：", api["model"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["zero", "few"])
    parser.add_argument("--check", action="store_true", help="只列出可用模型，不产生费用")
    args = parser.parse_args()
    cfg = config()
    api, key = cfg["api"], api_key()
    if args.check:
        check(key, api)
        return
    if not args.mode:
        parser.error("需要 --mode zero|few 或 --check")
    method = f"deepseek_{args.mode}"
    splits = load_data()
    index = {r["id"]: r for r in splits["test"]}
    rows = [index[i] for i in api_subset_ids()]
    examples = pick_few_shot(splits["train"], cfg["llm"]["few_shot_k"], cfg["seed"]) \
        if args.mode == "few" else []
    messages = {r["id"]: build_messages(r["text"], examples) for r in rows}
    hashes = {i: prompt_hash(m, api) for i, m in messages.items()}

    cache_path = ROOT / "results" / "api_raw" / f"{method}.jsonl"
    cache = {}
    if cache_path.exists():
        for rec in read_jsonl(cache_path):
            if hashes.get(rec["id"]) == rec["prompt_sha256"]:
                cache[rec["id"]] = rec
    todo = [r["id"] for r in rows if r["id"] not in cache]
    print(f"{method}: 缓存命中 {len(cache)}，需请求 {len(todo)}", flush=True)
    lock, errors = threading.Lock(), {}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    wall_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=api["workers"]) as pool, \
            cache_path.open("a", encoding="utf-8") as sink:
        futures = {pool.submit(chat, messages[i], key, api): i for i in todo}
        for done, future in enumerate(as_completed(futures), 1):
            sample_id = futures[future]
            try:
                out = future.result()
            except FatalAPIError:
                for f in futures:
                    f.cancel()
                raise
            if "error" in out:
                errors[sample_id] = out["error"]
            else:
                rec = {"id": sample_id, "prompt_sha256": hashes[sample_id], **out}
                with lock:
                    sink.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    sink.flush()
                cache[sample_id] = rec
            if done % 25 == 0:
                print(f"{method}: {done}/{len(todo)}", flush=True)
    wall_seconds = time.perf_counter() - wall_start

    policy = span_policy()
    preds, stats = [], []
    for row in rows:
        raw = cache.get(row["id"], {}).get("text", "")
        entities, s = parse_to_entities(row["text"], raw, policy)
        if row["id"] in errors:
            s["status"] = "request_failed"
        preds.append({"id": row["id"], "entities": entities, "raw": raw, "parse": s})
        stats.append(s)
    records = [cache[r["id"]] for r in rows if r["id"] in cache]
    latencies = sorted(r["latency_s"] for r in records)
    usage_keys = ["prompt_tokens", "completion_tokens", "prompt_cache_hit_tokens",
                  "prompt_cache_miss_tokens"]
    usage = {k: sum(r["usage"].get(k, 0) or 0 for r in records) for k in usage_keys}
    parse = summarize_parse_stats(stats)
    parse["request_failed"] = len(errors)
    save_result(method, splits, preds, eval_ids=[r["id"] for r in rows],
                predict_seconds=sum(latencies),
                settings={"provider": "DeepSeek", "base_url": api["base_url"],
                          "model": api["model"], "served_models": sorted({r.get("model") for r in records} - {None}),
                          "temperature": 0.0, "max_tokens": api["max_tokens"],
                          "concurrency": api["workers"], "system_prompt": SYSTEM_PROMPT,
                          "span_policy": policy, "few_shot_ids": [r["id"] for r in examples],
                          "predict_time_includes": "逐条请求延迟之和（含网络）；墙钟时间见 extra"},
                extra={"parse": parse, "usage": usage,
                       "finish_reasons": {k: sum(r.get("finish_reason") == k for r in records)
                                          for k in {r.get("finish_reason") for r in records}},
                       "latency_seconds": {"mean": statistics.mean(latencies) if latencies else 0,
                                           "p50": latencies[len(latencies) // 2] if latencies else 0,
                                           "p95": latencies[int(len(latencies) * 0.95)] if latencies else 0},
                       "wall_seconds_this_run": wall_seconds, "errors": errors})
    print("parse:", parse, "usage:", usage, flush=True)


if __name__ == "__main__":
    main()
