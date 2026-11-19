"""
方法四：GLM API 零样本二分类（human / ChatGPT，不训练，走网络）

教学重点：
  1. 这是**最省事的一条路**：不加载本地权重、不占显存，只要一个 key。
     冷启动验证、跨领域临时任务，它都是第一选择。
  2. 但成本和延迟都是「按次计费」的：500 条跑一遍要几十秒到几分钟，
     高 QPS 场景必须换成本地小模型或微调——这就是前三种方法存在的理由。
  3. 写法就是 OpenAI 兼容接口的事实标准：
          client = OpenAI(base_url=..., api_key=...)
          client.chat.completions.create(model=..., messages=[...])
     智谱 GLM 官方就是 OpenAI 兼容协议，换 DeepSeek/本地 vLLM 只改环境变量，代码不动。

环境变量：
  GLM_API_KEY / LLM_API_KEY   必填，https://open.bigmodel.cn 的 key
  LLM_BASE_URL                默认 https://open.bigmodel.cn/api/paas/v4
  LLM_MODEL                   默认 glm-4-flash（官方免费档，不限量）

用法：
  python llm_api_zero_shot/classify_api_zero_shot.py --n 500
  python llm_api_zero_shot/classify_api_zero_shot.py --n 500 --workers 8
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE_ROOT))

from common.data import load_splits, sample_subset
from common.label import LABELS
from sklearn.metrics import classification_report, confusion_matrix

# 长 answer 截断：文风信号集中在前段（套话/分点开头），截断省 token 效果几乎不损失
MAX_TEXT_CHARS = 400

SYSTEM = (
    "你要判断一段中文答案是「人类用户手写的回答」还是「ChatGPT 生成的回答」。\n"
    "判断线索：ChatGPT 倾向书面语、结构化分点（首先/其次/最后）、套话和免责声明、"
    "语气礼貌中立；人类回答更口语化、有个人经历、情绪化表达或很短很随意。\n"
    "只输出 human 或 chatGPT 中的一个，不要输出任何其他内容。"
)


def make_client():
    """
    构造 OpenAI 兼容客户端。base_url / model 从环境变量读：
    线上换模型只改配置不改代码。
    """
    from openai import OpenAI

    key = os.getenv("GLM_API_KEY") or os.getenv("LLM_API_KEY")
    if not key:
        raise SystemExit("缺少 GLM_API_KEY（或 LLM_API_KEY）环境变量")
    return OpenAI(
        api_key=key,
        base_url=os.getenv("LLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4"),
        timeout=60,
        max_retries=2,          # 偶发 429/网络抖动自动重试，不要把整批跑挂
    )


def parse_label(text: str) -> int:
    """从返回文本里抽类别名（大小写不敏感）；抽不到返回 -1（统计解析失败率）"""
    t = text.lower()
    hits = [n for n in LABELS if n.lower() in t]
    return LABELS.index(max(hits, key=len)) if hits else -1


def classify_one(client, model, text, max_tokens=8):
    """单条请求：返回 (预测 label, 原始输出, 用量)。异常吞掉记为失败，保证整批跑完"""
    try:
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": SYSTEM},
                      {"role": "user", "content": text[:MAX_TEXT_CHARS]}],
            temperature=0,          # 分类要稳定，不要随机性
            max_tokens=max_tokens,  # 只需一个词，给 8 个 token 足够
        )
        raw = (r.choices[0].message.content or "").strip()
        usage = {
            "prompt_tokens": getattr(r.usage, "prompt_tokens", 0),
            "completion_tokens": getattr(r.usage, "completion_tokens", 0),
        }
        return parse_label(raw), raw, usage
    except Exception as e:          # 网络错误 / 限流 / 超时 → 记为失败，不中断整批
        return -1, f"ERROR: {e}", {"prompt_tokens": 0, "completion_tokens": 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--workers", type=int, default=8, help="并发数；太高会被限流")
    ap.add_argument("--out", default="outputs/api_zero_shot_pred.json")
    args = ap.parse_args()

    model = os.getenv("LLM_MODEL", "glm-4-flash")
    client = make_client()

    _, val = load_splits()
    samples = sample_subset(val, args.n)
    print(f"model={model}  n={len(samples)}  workers={args.workers}")

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(lambda d: classify_one(client, model, d["text"] or ""), samples))
    elapsed = round(time.time() - t0, 1)

    preds = [r[0] for r in results]
    raws = [r[1] for r in results]
    tin = sum(r[2]["prompt_tokens"] for r in results)
    tout = sum(r[2]["completion_tokens"] for r in results)

    n_fail = sum(p == -1 for p in preds)          # 解析/请求失败率，零样本流程的健康指标
    y_true = [d["label"] for d in samples]
    print(f"\n{elapsed}s（{elapsed/len(samples):.2f}s/条）  tokens in/out = {tin}/{tout}"
          f"  解析失败 {n_fail}/{len(samples)}")

    ok = [(t, p) for t, p in zip(y_true, preds) if p != -1]
    if ok:
        y, p = [t for t, _ in ok], [q for _, q in ok]
        print(classification_report(y, p, target_names=LABELS, digits=4))
        print("混淆矩阵 (行=实际, 列=预测):")
        print(confusion_matrix(y, p))

    out = CODE_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"meta": {"model": model, "n": len(samples), "elapsed_s": elapsed,
                  "tokens_in": tin, "tokens_out": tout, "parse_fail": n_fail},
         "preds": [{"idx": d["idx"], "text": (d["text"] or "")[:100],
                    "true": d["label"], "pred": p, "raw": r}
                   for d, p, r in zip(samples, preds, raws)]},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"→ 结果写入 {out}")


if __name__ == "__main__":
    main()
