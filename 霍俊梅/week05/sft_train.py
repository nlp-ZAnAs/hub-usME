"""Qwen2.5-0.5B SFT：用 instruction/output 数据做 LoRA 微调（课程作业 week05）。"""
import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from torch.utils.data import Dataset


def build_example(tokenizer, instruction, output, max_len=128):
    """拼 Qwen chat 模板并做 loss mask。

    prompt（instruction）部分的 labels 置 -100 —— 只在回答部分计算交叉熵，
    让模型学"怎么答"而不是"怎么问"。这是 SFT 与普通 LM 训练的核心区别。
    """
    messages = [{"role": "user", "content": instruction}]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    full_text = prompt_text + output + tokenizer.eos_token
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
    full_ids = tokenizer(full_text, add_special_tokens=False).input_ids

    input_ids = full_ids[:max_len]
    labels = list(input_ids)
    for i in range(min(len(prompt_ids), len(input_ids))):
        labels[i] = -100
    return {"input_ids": input_ids, "labels": labels}


class SFTDataset(Dataset):
    """读取 jsonl（每行 {"instruction": ..., "output": ...}），预先编码全部样本。"""

    def __init__(self, path, tokenizer, max_len=512):
        self.samples = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                self.samples.append(
                    build_example(tokenizer, d["instruction"], d["output"], max_len)
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]


class Collator:
    """同一个 batch 内动态 padding 到最长序列，padding 位置 label=-100、attention=0。"""

    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, batch):
        maxlen = max(len(b["input_ids"]) for b in batch)
        input_ids, labels, attn = [], [], []
        for b in batch:
            pad = maxlen - len(b["input_ids"])
            input_ids.append(b["input_ids"] + [self.pad_id] * pad)
            labels.append(b["labels"] + [-100] * pad)
            attn.append([1] * len(b["input_ids"]) + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(attn),
        }


def plot_loss_curve(log_history, save_path):
    """从 Trainer 的 log_history 画 train/val loss 曲线。"""
    steps = [h["step"] for h in log_history if "loss" in h]
    train = [h["loss"] for h in log_history if "loss" in h]
    epochs = [h["epoch"] for h in log_history if "eval_loss" in h]
    val = [h["eval_loss"] for h in log_history if "eval_loss" in h]

    plt.figure(figsize=(8, 5))
    plt.plot(steps, train, label="train loss")
    if val:
        # val 是按 epoch 记录的，按比例映射到 step 轴
        plt.plot([steps[-1] * e / epochs[-1] for e in epochs], val, "o-", label="val loss")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.title("SFT loss curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"loss 曲线已保存: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Qwen2.5-0.5B LoRA SFT")
    parser.add_argument("--train", required=True, help="sft_train.jsonl 路径")
    parser.add_argument("--val", required=True, help="sft_val.jsonl 路径")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B",
                        help="模型名或本地目录（如 ./models）")
    parser.add_argument("--output", default="./output")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--logging_steps", type=int, default=20)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}")

    # 1. tokenizer + 数据
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    train_ds = SFTDataset(args.train, tokenizer, args.max_len)
    val_ds = SFTDataset(args.val, tokenizer, args.max_len)
    print(f"train: {len(train_ds)} 条, val: {len(val_ds)} 条")

    # 2. 模型 + LoRA（CUDA 用 bf16+sdpa；MPS/CPU 用 fp32，fp16 实测拖慢收敛）
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    kwargs = {"torch_dtype": dtype}
    if device == "cuda":
        kwargs["attn_implementation"] = "sdpa"
    model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs)
    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    # 3. 训练
    targs = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        bf16=(device == "cuda"),
        # MPS 上 fp16 混合精度会拖慢收敛（实测），Mac 用 fp32
        logging_steps=args.logging_steps,
        eval_strategy="epoch",
        save_strategy="no",
        report_to="none",
        seed=42,
    )
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=Collator(tokenizer.pad_token_id),
    )
    trainer.train()

    # 4. 保存：log、曲线、adapter、合并模型
    os.makedirs(args.output, exist_ok=True)
    with open(f"{args.output}/log_history.json", "w", encoding="utf-8") as f:
        json.dump(trainer.state.log_history, f, indent=2)
    plot_loss_curve(trainer.state.log_history, f"{args.output}/loss_curve.png")

    adapter_dir = f"{args.output}/adapter"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"LoRA adapter 已保存: {adapter_dir}")

    merged_dir = f"{args.output}/merged"
    merged = model.merge_and_unload()
    merged.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)
    print(f"合并后完整模型已保存: {merged_dir}")


if __name__ == "__main__":
    main()
