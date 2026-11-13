"""Shared data, model and generation utilities for Week 05."""
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch
from peft import PeftModel
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

ROOT = Path(__file__).resolve().parent
IGNORE = -100
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"


def load_config(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                          encoding="utf-8")


def select_device(value):
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    if value == "cpu":
        torch.set_num_threads(min(8, torch.get_num_threads()))
    return torch.device(value)


def load_base(cfg, base, device):
    source = str(base or cfg["base_model"])
    kwargs = {"revision": cfg["revision"]} if base is None else {}
    tokenizer = AutoTokenizer.from_pretrained(source, **kwargs)
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(source, torch_dtype=dtype, attn_implementation="sdpa",
                                                 **kwargs).to(device)
    for marker in (IM_START, IM_END):
        if tokenizer.encode(marker, add_special_tokens=False) != [tokenizer.convert_tokens_to_ids(marker)]:
            raise ValueError(f"Expected one token for {marker}")
    return model, tokenizer


def prompt_text(instruction):
    return f"{IM_START}user\n{instruction.strip()}{IM_END}\n{IM_START}assistant\n"


def prompt_ids(tokenizer, instruction):
    return tokenizer.encode(prompt_text(instruction), add_special_tokens=False)


@dataclass
class Example:
    instruction: str
    answer: str
    ids: list
    labels: list
    supervised: int


def read_split(path, tokenizer, max_length):
    raw = Path(path).read_bytes()
    rows = []
    for line_no, line in enumerate(raw.decode("utf-8-sig").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        instruction = row["instruction"].strip()
        answer = row["output"].strip()
        if not instruction or not answer:
            raise ValueError(f"{path}:{line_no} empty instruction or answer")
        if any(token in instruction or token in answer for token in (IM_START, IM_END, tokenizer.eos_token)):
            raise ValueError(f"{path}:{line_no} contains a reserved control token")
        prefix = prompt_ids(tokenizer, instruction)
        completion = tokenizer.encode(answer, add_special_tokens=False)
        completion.append(tokenizer.convert_tokens_to_ids(IM_END))
        ids = prefix + completion
        if len(ids) > max_length:
            raise ValueError(f"{path}:{line_no} has {len(ids)} tokens. Increase max_length.")
        rows.append(Example(instruction, answer, ids, [IGNORE] * len(prefix) + completion,
                            len(completion)))
    if not rows:
        raise ValueError(f"{path} contains no usable samples")
    return rows, {
        "sha256": hashlib.sha256(raw).hexdigest(), "examples": len(rows),
        "max_sequence_tokens": max(len(row.ids) for row in rows),
        "input_tokens": sum(len(row.ids) for row in rows),
        "supervised_tokens": sum(row.supervised for row in rows),
    }


def collate(rows, pad_id, device):
    width = max(len(row.ids) for row in rows)
    inputs, masks, labels = [], [], []
    for row in rows:
        pad = width - len(row.ids)
        inputs.append(row.ids + [pad_id] * pad)
        masks.append([1] * len(row.ids) + [0] * pad)
        labels.append(row.labels + [IGNORE] * pad)
    return tuple(torch.tensor(values, dtype=torch.long, device=device)
                 for values in (inputs, masks, labels))


def completion_logits(model, inputs, mask, labels):
    # PeftModel injects its trainable layers into the base decoder.
    base = model.get_base_model() if isinstance(model, PeftModel) else model
    hidden = base.model(input_ids=inputs, attention_mask=mask, use_cache=False).last_hidden_state
    shifted = labels[:, 1:]
    supervised = shifted != IGNORE
    return base.lm_head(hidden[:, :-1][supervised]), shifted[supervised]


@torch.inference_mode()
def evaluate(model, rows, tokenizer, device, batch_size):
    model.eval()
    nll_sum = torch.zeros((), device=device, dtype=torch.float64)
    correct = torch.zeros((), device=device, dtype=torch.long)
    tokens = sum(row.supervised for row in rows)
    for start in range(0, len(rows), batch_size):
        x, mask, y = collate(rows[start:start + batch_size], tokenizer.pad_token_id, device)
        logits, target = completion_logits(model, x, mask, y)
        nll_sum += F.cross_entropy(logits.float(), target, reduction="sum")
        correct += (logits.argmax(dim=-1) == target).sum()
    nll = nll_sum.item() / tokens
    return {"nll": nll, "ppl": math.exp(nll), "token_accuracy": correct.item() / tokens,
            "answer_tokens": tokens, "examples": len(rows)}


@torch.inference_mode()
def generate(model, tokenizer, instructions, device, max_new_tokens=96):
    model.eval()
    prompts = [prompt_ids(tokenizer, instruction) for instruction in instructions]
    width = max(map(len, prompts))
    pad_id = tokenizer.pad_token_id
    end_ids = sorted({tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids(IM_END)})
    inputs = torch.tensor([[pad_id] * (width - len(p)) + p for p in prompts], device=device)
    mask = torch.tensor([[0] * (width - len(p)) + [1] * len(p) for p in prompts], device=device)
    config = GenerationConfig(do_sample=False, max_new_tokens=max_new_tokens,
                              pad_token_id=pad_id, eos_token_id=end_ids, use_cache=True)
    generated = model.generate(input_ids=inputs, attention_mask=mask, generation_config=config)
    results = []
    for row in generated[:, width:].tolist():
        stop = next((i for i, token in enumerate(row) if token in end_ids), len(row))
        results.append({"text": tokenizer.decode(row[:stop], skip_special_tokens=True).strip(),
                        "stopped": stop < len(row), "new_tokens": stop})
    return results
