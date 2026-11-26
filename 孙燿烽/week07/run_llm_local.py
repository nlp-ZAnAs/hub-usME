"""本地 Qwen2.5-1.5B-Instruct：零样本、少样本（5 条训练集示例）、LoRA SFT。

三者共用 llm_common 里的 system prompt、解析器和 offset 回填规则；
SFT 的输入就是零样本 prompt，只在「类型 → 实体」JSON 与结束符上计算 loss。
"""

import argparse
import gc
import math
import os
import time
from itertools import islice

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

from common import (ROOT, checkpoint_path, config, evaluate, load_data, model_path, save_result,
                    seed_everything, write_json)
from llm_common import (SYSTEM_PROMPT, build_messages, parse_to_entities, pick_few_shot,
                        span_policy, summarize_parse_stats, target_json)

END_TOKEN = "<|im_end|>"


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def render(tokenizer, text, examples):
    return tokenizer.apply_chat_template(build_messages(text, examples), tokenize=False,
                                         add_generation_prompt=True)


def generate(model, tokenizer, rows, examples, cfg, device):
    """贪心解码；按 prompt 长度分桶以减少左侧 padding，输出按原顺序返回。"""
    model.eval()
    tokenizer.padding_side = "left"
    prompts = [render(tokenizer, r["text"], examples) for r in rows]
    lengths = [len(tokenizer(p, add_special_tokens=False)["input_ids"]) for p in prompts]
    order = sorted(range(len(rows)), key=lambda i: -lengths[i])
    end_id = tokenizer.convert_tokens_to_ids(END_TOKEN)
    raws, truncated = [None] * len(rows), 0
    batch_size = cfg["llm"]["eval_batch_size"]
    sync()
    start = time.perf_counter()
    with torch.inference_mode():
        for offset in range(0, len(order), batch_size):
            index = order[offset:offset + batch_size]
            encoded = tokenizer([prompts[i] for i in index], add_special_tokens=False,
                                padding=True, return_tensors="pt").to(device)
            output = model.generate(**encoded, max_new_tokens=cfg["llm"]["max_new_tokens"],
                                    do_sample=False, temperature=None, top_p=None, top_k=None,
                                    pad_token_id=tokenizer.pad_token_id, eos_token_id=end_id)
            new_tokens = output[:, encoded["input_ids"].shape[1]:]
            for i, tokens in zip(index, new_tokens):
                truncated += int(end_id not in tokens.tolist())
                raws[i] = tokenizer.decode(tokens, skip_special_tokens=True)
    sync()
    tokenizer.padding_side = "right"
    return raws, time.perf_counter() - start, {"max_prompt_tokens": max(lengths),
                                               "mean_prompt_tokens": sum(lengths) / len(lengths),
                                               "generation_truncated": truncated}


def to_predictions(rows, raws, policy):
    preds, stats = [], []
    for row, raw in zip(rows, raws, strict=True):
        entities, s = parse_to_entities(row["text"], raw, policy)
        preds.append({"id": row["id"], "entities": entities, "raw": raw, "parse": s})
        stats.append(s)
    return preds, summarize_parse_stats(stats)


# ───────────────────────────── SFT ─────────────────────────────

def sft_item(tokenizer, row):
    prompt = tokenizer(render(tokenizer, row["text"], ()), add_special_tokens=False)["input_ids"]
    answer = tokenizer(target_json(row), add_special_tokens=False)["input_ids"]
    answer = answer + [tokenizer.convert_tokens_to_ids(END_TOKEN)]
    return {"input_ids": prompt + answer, "labels": [-100] * len(prompt) + answer}


def collate(rows, pad_id):
    length = max(len(r["input_ids"]) for r in rows)
    inputs = torch.full((len(rows), length), pad_id, dtype=torch.long)
    labels = torch.full_like(inputs, -100)
    mask = torch.zeros_like(inputs)
    for i, row in enumerate(rows):
        n = len(row["input_ids"])
        inputs[i, :n] = torch.tensor(row["input_ids"])
        labels[i, :n] = torch.tensor(row["labels"])
        mask[i, :n] = 1
    return {"input_ids": inputs, "attention_mask": mask, "labels": labels}


def answer_loss_sum(model, batch):
    """只为答案前一位置展开词表 logits；与标准 masked causal-LM loss 等价（见 test_core）。"""
    targets = batch["labels"][:, 1:]
    positions = torch.where((targets != -100).any(dim=0))[0]
    if positions.numel() == 0:
        raise ValueError("所有答案都被 mask，拒绝无监督信号的批次")
    output = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                   logits_to_keep=positions, use_cache=False)
    chosen = targets[:, positions]
    loss = F.cross_entropy(output.logits.float().reshape(-1, output.logits.shape[-1]),
                           chosen.reshape(-1), ignore_index=-100, reduction="sum")
    return loss, int((chosen != -100).sum())


def groups(loader, size):
    batches = iter(loader)
    while group := list(islice(batches, size)):
        yield group


def train_lora(model, tokenizer, splits, cfg, device, policy):
    from peft import LoraConfig, TaskType, get_peft_model
    llm = cfg["llm"]
    targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=llm["lora_r"], lora_alpha=llm["lora_alpha"],
        lora_dropout=llm["lora_dropout"], target_modules=targets))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"qwen_lora: trainable={trainable:,} / total={total:,}", flush=True)
    items = [sft_item(tokenizer, r) for r in splits["train"]]
    loader = DataLoader(items, batch_size=llm["batch_size"], shuffle=True,
                        collate_fn=lambda b: collate(b, tokenizer.pad_token_id))
    steps_per_epoch = math.ceil(len(loader) / llm["grad_accum"])
    total_steps = steps_per_epoch * llm["epochs"]
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=llm["lora_lr"], weight_decay=llm["weight_decay"])
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * llm["warmup_ratio"]),
                                                total_steps)
    history, best_f1, selected_epoch, steps = [], -1.0, None, 0
    sync()
    start = time.perf_counter()
    for epoch in range(1, llm["epochs"] + 1):
        model.train()
        running, tokens = 0.0, 0
        for group in groups(loader, llm["grad_accum"]):
            # 用整个累积窗口的答案 token 数归一化；最后不足一个窗口的批次也照常更新。
            group_tokens = sum(int((b["labels"][:, 1:] != -100).sum()) for b in group)
            optimizer.zero_grad(set_to_none=True)
            for batch in group:
                loss_sum, count = answer_loss_sum(model, {k: v.to(device) for k, v in batch.items()})
                if not torch.isfinite(loss_sum):
                    raise RuntimeError("SFT loss 非有限数")
                (loss_sum / group_tokens).backward()
                running += loss_sum.item()
                tokens += count
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            scheduler.step()
            steps += 1
            if steps % 50 == 0:
                print(f"qwen_lora epoch={epoch} step={steps}/{total_steps} "
                      f"loss={running / tokens:.4f} elapsed={time.perf_counter() - start:.0f}s",
                      flush=True)
        raws, _, _ = generate(model, tokenizer, splits["val"], (), cfg, device)
        preds, parse = to_predictions(splits["val"], raws, policy)
        val_f1 = evaluate(splits["val"], preds)["strict"]["f1"]
        row = {"epoch": epoch, "train_loss": running / tokens, "val_strict_f1": val_f1,
               "val_json_valid_rate": parse["json_valid_rate"], "optimizer_steps": steps,
               "elapsed_seconds": time.perf_counter() - start}
        history.append(row)
        print("qwen_lora", row, flush=True)
        if val_f1 > best_f1:
            best_f1, selected_epoch = val_f1, epoch
            model.save_pretrained(checkpoint_path("qwen_lora"))
        write_json(ROOT / "results" / "qwen_lora_training.json", history)
    sync()
    info = {"trainable": trainable, "total": total, "history": history,
            "selected_epoch": selected_epoch, "train_seconds": time.perf_counter() - start,
            "lora_target_modules": targets,
            "max_train_tokens": max(len(i["input_ids"]) for i in items),
            "max_answer_tokens": max(sum(x != -100 for x in i["labels"]) for i in items)}
    del model, optimizer, scheduler
    gc.collect()
    torch.cuda.empty_cache()
    return info


def load_base(cfg, device):
    source = model_path(cfg["llm_model"])
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(source, local_files_only=True, dtype=torch.bfloat16,
                                                 attn_implementation="sdpa").to(device)
    return tokenizer, model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=["qwen_zero", "qwen_few", "qwen_lora"])
    args = parser.parse_args()
    cfg, splits = config(), load_data()
    seed_everything(cfg["seed"])
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("本脚本按 BF16 GPU 配置；CPU 上 1.5B 生成过慢，请显式调整后再运行")
    device = "cuda"
    torch.cuda.reset_peak_memory_stats()
    policy = span_policy()
    tokenizer, model = load_base(cfg, device)
    examples = pick_few_shot(splits["train"], cfg["llm"]["few_shot_k"], cfg["seed"]) \
        if args.method == "qwen_few" else []
    settings = {"model": cfg["llm_model"], "precision": "BF16", "decode": "greedy",
                "system_prompt": SYSTEM_PROMPT, "span_policy": policy,
                "few_shot_ids": [r["id"] for r in examples],
                "predict_time_includes": "chat template、tokenize、生成；不含解析与权重加载"}
    train = {"trainable": 0, "total": sum(p.numel() for p in model.parameters()),
             "history": [], "selected_epoch": None, "train_seconds": 0.0}
    if args.method == "qwen_lora":
        train = train_lora(model, tokenizer, splits, cfg, device, policy)
        from peft import PeftModel
        _, model = load_base(cfg, device)
        model = PeftModel.from_pretrained(model, checkpoint_path("qwen_lora"))
        settings.update({k: cfg["llm"][k] for k in ["epochs", "batch_size", "grad_accum", "lora_lr",
                                                    "lora_r", "lora_alpha", "lora_dropout"]})
        settings.update({"selection": "validation strict micro-F1（全部 1,000 条验证集生成）",
                         "lora_target_modules": train["lora_target_modules"],
                         "gradient_checkpointing": True,
                         "max_train_tokens": train["max_train_tokens"],
                         "max_answer_tokens": train["max_answer_tokens"],
                         "train_time_includes": "SFT、每轮验证集生成与 adapter 保存"})
    raws, seconds, gen_info = generate(model, tokenizer, splits["test"], examples, cfg, device)
    preds, parse = to_predictions(splits["test"], raws, policy)
    save_result(args.method, splits, preds, train_seconds=train["train_seconds"],
                predict_seconds=seconds, settings=settings, history=train["history"],
                selected_epoch=train["selected_epoch"], trainable_params=train["trainable"],
                total_params=train["total"],
                peak_gpu_gib=torch.cuda.max_memory_allocated() / 1024**3,
                extra={"parse": parse, **gen_info})
    print("parse:", parse, gen_info, flush=True)


if __name__ == "__main__":
    main()
