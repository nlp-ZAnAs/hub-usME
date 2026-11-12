"""Run answer-only LoRA SFT and save the adapter with the lowest validation NLL."""
import argparse
import math
import random
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from torch.nn import functional as F

from sft import ROOT, collate, completion_logits, evaluate, load_base, load_config, read_split
from sft import select_device, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--base", type=Path, help="Optional local snapshot of the configured base")
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--out", type=Path, default=ROOT / "runs" / "sft")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--max-updates", type=int, help="Stop after this many updates for a smoke test")
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = select_device(args.device)
    torch.manual_seed(cfg["seed"])
    rng = random.Random(cfg["seed"])
    args.out.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load_base(cfg, args.base, device)
    train, train_stats = read_split(args.data / "sft_train.jsonl", tokenizer, cfg["max_length"])
    val, val_stats = read_split(args.data / "sft_val.jsonl", tokenizer, cfg["max_length"])
    overlap = set(x.instruction for x in train) & set(x.instruction for x in val)
    report = {
        "config": cfg, "train": train_stats, "val": val_stats, "overlap": len(overlap),
        "torch": str(torch.__version__), "device": (torch.cuda.get_device_name(device)
                                                    if device.type == "cuda" else str(device)),
        "base_dtype": str(model.dtype), "history": [],
    }
    write_json(args.out / "history.json", report)
    report["base_val"] = evaluate(model, val, tokenizer, device, cfg["micro_batch"])
    print("Base validation", report["base_val"], flush=True)
    model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", **cfg["lora"]))
    model.print_trainable_parameters()
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    report["trainable_parameters"] = sum(p.numel() for p in params)
    report["total_parameters"] = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    batch_size = cfg["micro_batch"]
    update_size = batch_size * cfg["accumulate"]
    updates = math.ceil(len(train) / update_size) * cfg["epochs"]
    if args.max_updates:
        updates = min(updates, args.max_updates)
    warmup = min(cfg["warmup_steps"], updates // 2)
    best = math.inf
    step = 0
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        rng.shuffle(train)
        # Sort within shuffled groups to reduce padding while varying batches each epoch.
        train_nll = torch.zeros((), device=device, dtype=torch.float64)
        train_tokens = 0
        for start in range(0, len(train), update_size):
            group = sorted(train[start:start + update_size], key=lambda row: len(row.ids))
            count = sum(row.supervised for row in group)
            step += 1
            fraction = (step - warmup) / max(1, updates - warmup)
            scale = step / warmup if step <= warmup else 0.1 + 0.9 * (1 + math.cos(math.pi * fraction)) / 2
            optimizer.param_groups[0]["lr"] = cfg["lr"] * scale
            optimizer.zero_grad(set_to_none=True)
            for offset in range(0, len(group), batch_size):
                x, mask, y = collate(group[offset:offset + batch_size], tokenizer.pad_token_id, device)
                logits, targets = completion_logits(model, x, mask, y)
                loss_sum = F.cross_entropy(logits.float(), targets, reduction="sum")
                (loss_sum / count).backward()
                train_nll += loss_sum.detach()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0, error_if_nonfinite=True)
            optimizer.step()
            train_tokens += count
            if step % 20 == 0:
                print(f"epoch {epoch} | update {step}/{updates} | "
                      f"train NLL {train_nll.item() / train_tokens:.3f}", flush=True)
            if step >= updates:
                break
        scores = evaluate(model, val, tokenizer, device, batch_size)
        row = {"epoch": epoch, "updates": step, "train_nll": train_nll.item() / train_tokens,
               "val": scores, "elapsed_seconds": time.perf_counter() - started}
        if scores["nll"] < best:
            best = scores["nll"]
            report["best_epoch"] = epoch
            model.save_pretrained(args.out / "adapter", safe_serialization=True)
            # Record the hub ID and immutable revision even when training uses a local cache.
            adapter_config = args.out / "adapter" / "adapter_config.json"
            ac = load_config(adapter_config)
            ac.update(base_model_name_or_path=cfg["base_model"], revision=cfg["revision"])
            write_json(adapter_config, ac)
        report["history"].append(row)
        if device.type == "cuda":
            report["peak_allocated_mib"] = torch.cuda.max_memory_allocated(device) / 1024**2
        write_json(args.out / "history.json", report)
        print(row, flush=True)
        if step >= updates:
            break


if __name__ == "__main__":
    main()
