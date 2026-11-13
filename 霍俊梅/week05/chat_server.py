"""SFT Chatbot：Gradio 网页界面 + TextIteratorStreamer 流式输出 + 多轮对话。"""
import argparse
import os
import threading

import gradio as gr
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
)


def _as_text(content):
    """把 Gradio 6 的消息 content 归一化成 str。

    Gradio 6 ChatInterface 传回的历史 content 是多模态 list 格式
    （如 [{"type":"text","text":"..."}]），而 Qwen chat 模板只接受 str，
    直接透传会报 jinja 的 str+list TypeError。
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return str(content)


def build_prompt(tokenizer, history, message):
    """把多轮历史 + 新消息拼成 Qwen chat 模板（末尾带生成提示）。"""
    messages = [{"role": m.get("role", "user"), "content": _as_text(m["content"])}
                for m in history]
    messages.append({"role": "user", "content": _as_text(message)})
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def load_model(model_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if torch.cuda.is_available():
        dtype = torch.bfloat16
    elif torch.backends.mps.is_available():
        dtype = torch.float16
    else:
        dtype = torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=dtype
    ).to("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model.eval()
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="SFT 模型 Chatbot（流式 + 多轮）")
    parser.add_argument("--model", default="./output/merged",
                        help="模型目录：./output/merged（微调后）或 ./models-qwen0.5b（底座）")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    if not os.path.isdir(args.model):
        raise FileNotFoundError(f"模型目录不存在: {args.model}")
    model, tokenizer = load_model(args.model)
    print(f"模型已加载: {args.model}")

    def respond(message, history):
        """Gradio 流式回调：history 为 type='messages' 格式，逐 token yield。"""
        inputs = tokenizer(
            build_prompt(tokenizer, history, message), return_tensors="pt"
        ).to(model.device)

        streamer = TextIteratorStreamer(
            tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        thread = threading.Thread(
            target=model.generate,
            kwargs=dict(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=0.9,
                eos_token_id=[
                    tokenizer.eos_token_id,
                    tokenizer.convert_tokens_to_ids("<|im_end|>"),
                ],
                streamer=streamer,
            ),
        )
        thread.start()

        partial = ""
        try:
            for token in streamer:
                partial += token
                yield partial
        except Exception as e:
            yield partial + f"\n\n[生成出错: {e}]"
        thread.join()

    demo = gr.ChatInterface(
        fn=respond,
        title="week05 SFT Chatbot",
        description="微调模型流式问答，支持多轮对话（清空按钮重开话题）",
    )
    demo.queue().launch(server_port=args.port)


if __name__ == "__main__":
    main()
