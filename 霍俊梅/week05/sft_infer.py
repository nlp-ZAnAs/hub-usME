"""SFT 推理演示：微调前后对比 + 终端交互问答。"""
import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

DEMO_QUESTIONS = [
    "给出两个数字，求它们的平均值。 25, 41",
    "给出一些适合夏天穿的衣服款式。",
    "请解释以下缩写的含义：NLP。",
    "为以下句子生成一个标题。 2021年东京奥运会宣布将无观众举行",
]


def generate(model, tokenizer, question, max_new_tokens=128, disable_adapter=False,
             temperature=0.2):
    """对单个问题生成回答（chat 模板 + 采样解码）。

    disable_adapter=True 时临时关闭 LoRA，走纯 base 权重 —— 用于真实的
    "微调前"对比。注意 PeftModel.from_pretrained(base) 会原地改造 base，
    直接用 base 生成其实带着 adapter，对比无效。

    eos 传 [endoftext, im_end] 双停止：SFT 教模型用 eos_token 收尾，
    im_end 兜底兼容，防止答完不停出现乱码尾巴。

    temperature 默认 0.2：低温度采样更稳定（0.7 时偶发把 33 采成 34）。
    """
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    from contextlib import nullcontext
    ctx = model.disable_adapter() if disable_adapter else nullcontext()
    with ctx:
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
            eos_token_id=[tokenizer.eos_token_id, im_end_id],
            pad_token_id=tokenizer.pad_token_id,
        )
    # 只取新生成的部分（去掉 prompt）
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description="SFT 推理演示")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B",
                        help="基座模型名或本地目录（如 ./models）")
    parser.add_argument("--adapter", required=True, help="训练产出的 output/adapter 目录")
    parser.add_argument("--temperature", type=float, default=0.2,
                        help="采样温度，越低越稳定（0=贪婪）")
    args = parser.parse_args()

    if torch.cuda.is_available():
        dtype = torch.bfloat16
    elif torch.backends.mps.is_available():
        dtype = torch.float16
    else:
        dtype = torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # 加载一次模型，挂上 LoRA 适配器（微调后）
    # 注意：wrap 会原地改造 base，"微调前"必须用 disable_adapter 关掉 LoRA 才真实
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype
    ).to("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    # 前后对比
    print("=" * 60)
    print("微调前 vs 微调后对比")
    print("=" * 60)
    for q in DEMO_QUESTIONS:
        print(f"\n【问】{q}")
        with torch.no_grad():
            print(f"  [微调前] {generate(model, tokenizer, q, disable_adapter=True, temperature=args.temperature)}")
            print(f"  [微调后] {generate(model, tokenizer, q)}")

    # 4) 交互问答
    print("\n" + "=" * 60)
    print("交互问答（输入 quit 退出）")
    print("=" * 60)
    while True:
        try:
            q = input("\n你: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q or q.lower() == "quit":
            break
        with torch.no_grad():
            print(f"模型: {generate(model, tokenizer, q, temperature=args.temperature)}")


if __name__ == "__main__":
    main()
