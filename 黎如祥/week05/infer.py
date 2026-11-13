# -*- coding: utf-8 -*-
"""
Week05 作业：用训好的 LoRA 权重做推理，同一输入对比「微调前 / 微调后」。

    python infer.py --adapter outputs/lora_r16/best --prompt "给定两个数字，请选择其中的较大数并将其打印出来。 数字1: 238 数字2: 564"
    python infer.py --adapter outputs/lora_r16/best          # 进入交互模式

同一个模型、同一套权重，只是 disable/enable LoRA adapter，最能说明 SFT 改了什么。
"""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent
IM_END = 151645


def build_inputs(tok, prompt):
    messages = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return tok(text, return_tensors="pt", add_special_tokens=False).to("cuda")


def generate(model, tok, inputs, max_new_tokens, sample, temperature):
    kwargs = dict(
        max_new_tokens=max_new_tokens,
        pad_token_id=tok.pad_token_id,
        eos_token_id=[IM_END, tok.eos_token_id],
    )
    if sample:
        kwargs.update(do_sample=True, temperature=temperature, top_p=0.9)
    else:
        kwargs.update(do_sample=False)
    with torch.inference_mode():
        out = model.generate(**inputs, **kwargs)
    new = out[0][inputs["input_ids"].shape[1]:]
    return tok.decode(new, skip_special_tokens=True).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None,
                    help="基座路径；不填则读 adapter_config.json 或 SFT_MODEL 环境变量")
    ap.add_argument("--adapter", default=str(ROOT / "outputs" / "lora_r16" / "best"))
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.8)
    args = ap.parse_args()

    from peft import PeftModel

    model_id = args.model or os.environ.get("SFT_MODEL")
    if model_id is None:
        cfg_path = Path(args.adapter) / "adapter_config.json"
        if cfg_path.exists():
            model_id = json.loads(cfg_path.read_text(encoding="utf-8")).get(
                "base_model_name_or_path")
    model_id = model_id or "Qwen/Qwen2-0.5B"

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).cuda()
    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()

    def run(prompt):
        inputs = build_inputs(tok, prompt)
        with model.disable_adapter():
            before = generate(model, tok, inputs, args.max_new_tokens, args.sample, args.temperature)
        after = generate(model, tok, inputs, args.max_new_tokens, args.sample, args.temperature)
        print("\n" + "-" * 78)
        print("指令：", prompt)
        print("  微调前：", before)
        print("  微调后：", after)

    if args.prompt:
        run(args.prompt)
        return
    print("输入指令后回车（空行退出）：")
    while True:
        try:
            prompt = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not prompt:
            break
        run(prompt)


if __name__ == "__main__":
    main()
