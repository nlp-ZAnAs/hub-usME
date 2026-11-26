"""单句实体抽取：用已训练/已配置的方法对任意句子输出 {类型, 文本, start, end}。

示例：
    python extract.py --method roberta_crf --constrained --text "浙商银行行长叶老桂在杭州出席了发布会。"
    python extract.py --method qwen_lora --text "《王者荣耀》的玩家在成都举办了线下赛。"
    python extract.py --method deepseek_few --text "故宫博物院院长王旭东接受了采访。"
"""

import argparse
import json

from common import ROOT, config, load_data, read_json
from llm_common import build_messages, parse_to_entities, pick_few_shot, span_policy

SEQUENCE = ["bilstm_crf", "roberta_linear", "roberta_crf"]
LOCAL_LLM = ["qwen_zero", "qwen_few", "qwen_lora"]
API = ["deepseek_zero", "deepseek_few"]


def run_rule(text, splits):
    from collections import Counter
    from run_rule import build_dictionary, extract
    selected = read_json(ROOT / "results" / "rule.json")["settings"]["selected"]
    entries, max_len = build_dictionary(splits["train"])
    dictionary = {m: v["type"] for m, v in entries.items()
                  if v["count"] >= selected["min_count"]
                  and v["label_ratio"] >= selected["min_label_ratio"]}
    fallback = Counter(e["type"] for r in splits["train"] for e in r["entities"]
                       if e["text"].startswith("《")).most_common(1)[0][0]
    return {"entities": extract(text, dictionary, max_len, fallback)}


def run_sequence(method, text, constrained):
    import torch
    from run_sequence import load_tagger, tag_text
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, char_encoder = load_tagger(method, config(), device)
    return tag_text(model, char_encoder, text, "constrained" if constrained else "raw", device)


def run_local_llm(method, text, splits):
    from run_llm_local import generate, load_base
    from common import checkpoint_path
    cfg = config()
    tokenizer, model = load_base(cfg, "cuda")
    if method == "qwen_lora":
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, checkpoint_path("qwen_lora"))
    examples = pick_few_shot(splits["train"], cfg["llm"]["few_shot_k"], cfg["seed"]) \
        if method == "qwen_few" else []
    raws, _, _ = generate(model, tokenizer, [{"text": text}], examples, cfg, "cuda")
    entities, stats = parse_to_entities(text, raws[0], span_policy())
    return {"entities": entities, "raw": raws[0], "parse": stats}


def run_api(method, text, splits):
    from run_api import api_key, chat
    cfg = config()
    examples = pick_few_shot(splits["train"], cfg["llm"]["few_shot_k"], cfg["seed"]) \
        if method == "deepseek_few" else []
    out = chat(build_messages(text, examples), api_key(), cfg["api"])
    if "error" in out:
        raise RuntimeError(out["error"])
    entities, stats = parse_to_entities(text, out["text"], span_policy())
    return {"entities": entities, "raw": out["text"], "parse": stats, "usage": out["usage"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=["rule"] + SEQUENCE + LOCAL_LLM + API)
    parser.add_argument("--text", required=True)
    parser.add_argument("--constrained", action="store_true",
                        help="序列标注方法使用 BIO 约束解码（不重训）")
    args = parser.parse_args()
    text = args.text.strip()
    if not text:
        parser.error("文本不能为空")
    splits = load_data()
    if args.method == "rule":
        out = run_rule(text, splits)
    elif args.method in SEQUENCE:
        out = run_sequence(args.method, text, args.constrained)
    elif args.method in LOCAL_LLM:
        out = run_local_llm(args.method, text, splits)
    else:
        out = run_api(args.method, text, splits)
    print(json.dumps({"method": args.method, "text": text, **out}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
