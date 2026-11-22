"""RBT3 分类头微调，以及 Qwen 的 zero-shot / LoRA / 全量 SFT 对比。"""

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
from transformers import (AutoModelForCausalLM, AutoModelForSequenceClassification,
                          AutoTokenizer, get_linear_schedule_with_warmup)

from common import (LABELS, ROOT, config, load_data, metrics, save_result,
                    seed_everything, write_json)

SYSTEM = "你是酒店评论情感分类助手。根据评论的整体态度判断情感，只输出“正面”或“负面”，不要解释。"


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def move(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def checkpoint_path(method):
    return ROOT / "checkpoints" / method


def model_path(model_id):
    return ROOT / ".cache" / model_id.split("/")[-1]


def parse_label(text):
    # 容忍包裹标点，但拒绝“正面或负面”及长解释，避免宽松包含匹配误计成功。
    cleaned = text.strip().strip("\"'“”‘’。，.!！：: \n")
    return LABELS.index(cleaned) if cleaned in LABELS else -1


def make_prompt(tokenizer, text):
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": "评论：" + text + "\n情感："}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def make_sft_item(tokenizer, row, max_length):
    prompt = tokenizer(make_prompt(tokenizer, row["text"]), add_special_tokens=False)["input_ids"]
    answer = tokenizer(LABELS[row["label"]], add_special_tokens=False)["input_ids"]
    answer = answer + [tokenizer.eos_token_id]
    if len(prompt) + len(answer) > max_length:
        raise ValueError("输入超过 llm_max_length；增大配置，不能截掉答案或静默改变输入")
    return {"input_ids": prompt + answer, "labels": [-100] * len(prompt) + answer}


def sft_collate(rows, pad_id):
    length = max(len(r["input_ids"]) for r in rows)
    inputs = torch.full((len(rows), length), pad_id, dtype=torch.long)
    labels = torch.full_like(inputs, -100)
    mask = torch.zeros_like(inputs)
    for index, row in enumerate(rows):
        n = len(row["input_ids"])
        inputs[index, :n] = torch.tensor(row["input_ids"])
        labels[index, :n] = torch.tensor(row["labels"])
        mask[index, :n] = 1
    return {"input_ids": inputs, "attention_mask": mask, "labels": labels}


def answer_loss_sum(model, batch):
    """只展开答案前位置的词表 logits，与标准 causal LM 的 masked CE 等价。

    右侧补齐的不同样本可能在不同位置开始答题，取所有有效位置的并集。
    这样避免为几百个 prompt token 分配 151936 维 logits，节省 8GB 卡显存。
    """
    targets = batch["labels"][:, 1:]
    positions = torch.where((targets != -100).any(dim=0))[0]
    if positions.numel() == 0:
        raise ValueError("所有答案被 mask；拒绝无监督信号的训练批次")
    output = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                   logits_to_keep=positions, use_cache=False)
    chosen = targets[:, positions]
    loss = F.cross_entropy(output.logits.float().reshape(-1, output.logits.shape[-1]),
                           chosen.reshape(-1), ignore_index=-100, reduction="sum")
    return loss, int((chosen != -100).sum())


def batch_groups(loader, accumulation):
    if accumulation < 1:
        raise ValueError("gradient accumulation must be positive")
    batches = iter(loader)
    while group := list(islice(batches, accumulation)):
        yield group


def llm_predict(model, tokenizer, rows, cfg, device):
    model.eval()
    previous_side = tokenizer.padding_side
    tokenizer.padding_side = "left"  # decoder-only 批量生成必须左补齐
    predictions, raw = [], []
    sync()
    start = time.perf_counter()
    with torch.inference_mode():
        for offset in range(0, len(rows), cfg["llm_eval_batch_size"]):
            texts = [make_prompt(tokenizer, r["text"])
                     for r in rows[offset:offset + cfg["llm_eval_batch_size"]]]
            encoded = tokenizer(texts, add_special_tokens=False, padding=True, return_tensors="pt")
            if encoded["input_ids"].shape[1] + cfg["max_new_tokens"] > cfg["llm_max_length"]:
                raise ValueError("推理输入超过配置长度")
            encoded = move(encoded, device)
            generated = model.generate(
                **encoded, max_new_tokens=cfg["max_new_tokens"], do_sample=False,
                temperature=None, top_p=None, top_k=None,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                use_cache=True)
            answers = tokenizer.batch_decode(
                generated[:, encoded["input_ids"].shape[1]:], skip_special_tokens=True)
            raw.extend(answers)
            predictions.extend(parse_label(answer) for answer in answers)
    sync()
    tokenizer.padding_side = previous_side
    return predictions, raw, time.perf_counter() - start


def encoder_loader(tokenizer, rows, batch_size, max_length, shuffle=False):
    encoded = tokenizer([r["text"] for r in rows], add_special_tokens=True,
                        truncation=False, padding=False)
    if max(map(len, encoded["input_ids"])) > max_length:
        raise ValueError("Encoder 输入超过配置；禁止隐式二次截断")
    items = [{k: v[i] for k, v in encoded.items()} | {"labels": rows[i]["label"]}
             for i in range(len(rows))]

    def collate(batch):
        labels = torch.tensor([x["labels"] for x in batch])
        inputs = [{k: v for k, v in x.items() if k != "labels"} for x in batch]
        return dict(tokenizer.pad(inputs, padding=True, return_tensors="pt"), labels=labels)

    return DataLoader(items, batch_size=batch_size, shuffle=shuffle, collate_fn=collate)


def encoder_predict(model, loader, device):
    model.eval()
    predictions = []
    sync()
    start = time.perf_counter()
    with torch.inference_mode(), torch.autocast(device_type=device, dtype=torch.bfloat16,
                                                enabled=device == "cuda"):
        for batch in loader:
            batch = move(batch, device)
            batch.pop("labels")
            predictions.extend(model(**batch).logits.argmax(-1).cpu().tolist())
    sync()
    return predictions, time.perf_counter() - start


def train_encoder(cfg, splits, device):
    source = model_path(cfg["encoder_model"])
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        source, num_labels=2, local_files_only=True, attn_implementation="sdpa").to(device)
    loaders = {k: encoder_loader(tokenizer, rows, cfg["encoder_batch_size"],
                               cfg["encoder_max_length"], shuffle=k == "train")
               for k, rows in splits.items()}
    params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["encoder_lr"], weight_decay=0.01)
    steps = len(loaders["train"]) * cfg["encoder_epochs"]
    scheduler = get_linear_schedule_with_warmup(optimizer, int(steps * 0.1), steps)
    best_f1, selected_epoch, history = -1, None, []
    sync()
    start = time.perf_counter()
    for epoch in range(1, cfg["encoder_epochs"] + 1):
        model.train()
        loss_sum, n = 0., 0
        for batch in loaders["train"]:
            batch = move(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
                loss = model(**batch).loss
            if not torch.isfinite(loss):
                raise RuntimeError("RBT3 loss 非有限数")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            scheduler.step()
            size = len(batch["labels"])
            loss_sum += loss.item() * size
            n += size
        val_pred, _ = encoder_predict(model, loaders["val"], device)
        score = metrics([r["label"] for r in splits["val"]], val_pred)
        row = {"epoch": epoch, "train_loss": loss_sum / n, "val_macro_f1": score["macro_f1"],
               "val_accuracy": score["accuracy"], "elapsed_seconds": time.perf_counter() - start}
        history.append(row)
        print("rbt3", row, flush=True)
        if score["macro_f1"] > best_f1:
            best_f1, selected_epoch = score["macro_f1"], epoch
            model.save_pretrained(checkpoint_path("rbt3"))
            tokenizer.save_pretrained(checkpoint_path("rbt3"))
        write_json(ROOT / "results" / "rbt3_training.json", history)
    sync()
    train_seconds = time.perf_counter() - start
    del model, optimizer, scheduler
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint_path("rbt3"), local_files_only=True, attn_implementation="sdpa").to(device)
    pred, predict_seconds = encoder_predict(model, loaders["test"], device)
    return save_result(
        "rbt3", splits, pred, train_seconds=train_seconds, predict_seconds=predict_seconds,
        settings={"model": cfg["encoder_model"], "precision": "FP32 weights + BF16 autocast",
                  "selection": "validation macro-F1", "weight_decay": 0.01,
                  "warmup_ratio": 0.1, "gradient_clip": 1.0,
                  "train_time_includes": "training, validation and best checkpoint saving",
                  "predict_time_includes": "padding, model forward; excludes cached tokenization"},
        history=history, selected_epoch=selected_epoch, trainable_params=params, total_params=params,
        peak_gpu_gib=torch.cuda.max_memory_allocated() / 1024**3 if device == "cuda" else 0)


def run_llm(method, cfg, splits, device):
    source = model_path(cfg["llm_model"])
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        source, local_files_only=True, dtype=dtype, attn_implementation="sdpa").to(device)
    settings = {"model": cfg["llm_model"], "system_prompt": SYSTEM, "precision": str(dtype),
                "selection": "validation macro-F1", "decode": "greedy; exact label parser",
                "predict_time_includes": "tokenization, generation and parsing",
                "train_time_includes": "SFT, validation generation and best checkpoint saving"}
    history, train_seconds, selected_epoch = [], 0., None
    trainable = 0
    total = sum(p.numel() for p in model.parameters())
    if method != "qwen_zero":
        if method == "qwen_lora":
            from peft import LoraConfig, TaskType, get_peft_model
            targets = ["q_proj", "k_proj", "v_proj", "o_proj"]
            model = get_peft_model(model, LoraConfig(
                task_type=TaskType.CAUSAL_LM, r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"],
                lora_dropout=0.05, target_modules=targets))
            settings["lora_target_modules"] = targets
            lr = cfg["lora_lr"]
        else:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            lr = cfg["full_lr"]
        settings["lr"] = lr
        settings["gradient_checkpointing"] = method == "qwen_full"
        settings["weight_decay"] = 0.01
        settings["warmup_ratio"] = 0.1
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"{method}: trainable={trainable:,} / total={total:,}", flush=True)
        items = [make_sft_item(tokenizer, r, cfg["llm_max_length"]) for r in splits["train"]]
        settings["max_observed_train_tokens"] = max(len(r["input_ids"]) for r in items)
        loader = DataLoader(items, batch_size=cfg["llm_batch_size"], shuffle=True,
                            collate_fn=lambda batch: sft_collate(batch, tokenizer.pad_token_id))
        steps_per_epoch = math.ceil(len(loader) / cfg["llm_grad_accum"])
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                      lr=lr, weight_decay=0.01)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, int(steps_per_epoch * cfg["llm_epochs"] * 0.1),
            steps_per_epoch * cfg["llm_epochs"])
        best_f1, steps = -1, 0
        sync()
        start = time.perf_counter()
        for epoch in range(1, cfg["llm_epochs"] + 1):
            model.train()
            running, tokens = 0., 0
            for group in batch_groups(loader, cfg["llm_grad_accum"]):
                # 用整个累积窗口的答案 token 数归一化；最后不足一个窗口也要更新。
                group_tokens = sum(int((b["labels"][:, 1:] != -100).sum()) for b in group)
                optimizer.zero_grad(set_to_none=True)
                for batch in group:
                    loss_sum, count = answer_loss_sum(model, move(batch, device))
                    if not torch.isfinite(loss_sum):
                        raise RuntimeError("SFT loss 非有限数")
                    (loss_sum / group_tokens).backward()
                    running += loss_sum.item()
                    tokens += count
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
                optimizer.step()
                scheduler.step()
                steps += 1
                if steps % 25 == 0:
                    print(f"{method} epoch={epoch} steps={steps} loss={running / tokens:.4f} "
                          f"elapsed={time.perf_counter() - start:.0f}s", flush=True)
            val_pred, _, _ = llm_predict(model, tokenizer, splits["val"], cfg, device)
            score = metrics([r["label"] for r in splits["val"]], val_pred)
            row = {"epoch": epoch, "train_loss": running / tokens,
                   "val_macro_f1": score["macro_f1"], "val_accuracy": score["accuracy"],
                   "val_invalid": score["invalid"], "optimizer_steps": steps,
                   "elapsed_seconds": time.perf_counter() - start}
            history.append(row)
            print(method, row, flush=True)
            if score["macro_f1"] > best_f1:
                best_f1, selected_epoch = score["macro_f1"], epoch
                model.save_pretrained(checkpoint_path(method))
                tokenizer.save_pretrained(checkpoint_path(method))
            write_json(ROOT / "results" / (method + "_training.json"), history)
        sync()
        train_seconds = time.perf_counter() - start
        del model, optimizer, scheduler
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        if method == "qwen_lora":
            from peft import PeftModel
            model = AutoModelForCausalLM.from_pretrained(
                source, local_files_only=True, dtype=dtype, attn_implementation="sdpa").to(device)
            model = PeftModel.from_pretrained(model, checkpoint_path(method))
        else:
            model = AutoModelForCausalLM.from_pretrained(
                checkpoint_path(method), local_files_only=True, dtype=dtype,
                attn_implementation="sdpa").to(device)
    pred, raw, predict_seconds = llm_predict(model, tokenizer, splits["test"], cfg, device)
    return save_result(method, splits, pred, train_seconds=train_seconds,
                       predict_seconds=predict_seconds, settings=settings, history=history,
                       raw=raw, trainable_params=trainable, total_params=total,
                       selected_epoch=selected_epoch,
                       peak_gpu_gib=torch.cuda.max_memory_allocated() / 1024**3 if device == "cuda" else 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True,
                        choices=["rbt3", "qwen_zero", "qwen_lora", "qwen_full"])
    args = parser.parse_args()
    cfg, splits = config(), load_data()
    seed_everything(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("当前配置要求支持 BF16 的 GPU；请明确调整精度后再训练")
        torch.cuda.reset_peak_memory_stats()
    if args.method == "rbt3":
        train_encoder(cfg, splits, device)
    else:
        run_llm(args.method, cfg, splits, device)


if __name__ == "__main__":
    main()
