"""序列标注方法：字符 BiLSTM-CRF（无预训练）、RoBERTa-wwm + Linear、RoBERTa-wwm + CRF。

RoBERTa 两种解码头各训练一次；测试时同一个 checkpoint 再做一次「约束解码」，
单独写成 *_constrained 结果，用来回答“不重训、只改解码”值多少。
"""

import argparse
import copy
import gc
import os
import time
from collections import Counter

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from common import (LABEL2ID, LABELS, ROOT, bio_to_spans, checkpoint_path, config, count_illegal,
                    evaluate, load_data, model_path, save_result, seed_everything, spans_to_bio,
                    transition_constraints, write_json)
from crf import CRF, constrained_argmax, constraint_scores

PAD, UNK = 0, 1


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# ───────────────────────────── 模型 ─────────────────────────────

class BiLSTMCRF(nn.Module):
    def __init__(self, vocab_size, cfg):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, cfg["embed_dim"], padding_idx=PAD)
        self.lstm = nn.LSTM(cfg["embed_dim"], cfg["hidden_dim"], batch_first=True,
                            bidirectional=True)
        self.dropout = nn.Dropout(cfg["dropout"])
        self.head = nn.Linear(2 * cfg["hidden_dim"], len(LABELS))
        self.crf = CRF(len(LABELS))

    def emissions(self, input_ids, mask):
        x = self.dropout(self.embed(input_ids))
        packed = nn.utils.rnn.pack_padded_sequence(x, mask.sum(1).cpu(), batch_first=True,
                                                   enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True,
                                                  total_length=input_ids.shape[1])
        return self.head(self.dropout(out)).float()


class EncoderTagger(nn.Module):
    """[CLS] + 逐字 token + [SEP]；只取中间与字符一一对应的位置作为发射分数。"""

    def __init__(self, source, decoder, dropout):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(source, local_files_only=True,
                                                 attn_implementation="sdpa")
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(self.encoder.config.hidden_size, len(LABELS))
        self.crf = CRF(len(LABELS)) if decoder == "crf" else None

    def emissions(self, input_ids, mask):
        # input_ids 每行为 [CLS] 字符… [SEP] [PAD]…，长度 = 本批最长句 + 2。
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        attention = (positions.unsqueeze(0) < mask.sum(1, keepdim=True) + 2).long()
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention).last_hidden_state
        return self.head(self.dropout(hidden[:, 1:-1])).float()


# ───────────────────────────── 数据 ─────────────────────────────

class CharEncoder:
    """把每个字符映射到一个 id，保证「字符 ↔ token」严格一一对应，不依赖子词切分。"""

    def __init__(self, kind, rows=None, tokenizer=None, vocab=None):
        self.kind, self.tokenizer = kind, tokenizer
        if kind == "bilstm":
            if vocab is None:
                counts = Counter(c for r in rows for c in r["text"])
                vocab = {c: i + 2 for i, c in enumerate(sorted(counts))}
            self.vocab = vocab
        else:
            self.cache = {}

    def char_id(self, char):
        if self.kind == "bilstm":
            return self.vocab.get(char, UNK)
        if char not in self.cache:
            pieces = self.tokenizer.tokenize(char)
            self.cache[char] = (self.tokenizer.convert_tokens_to_ids(pieces[0]) if len(pieces) == 1
                                else self.tokenizer.unk_token_id)
        return self.cache[char]

    def encode(self, text):
        ids = [self.char_id(c) for c in text]
        if self.kind == "encoder":
            ids = [self.tokenizer.cls_token_id] + ids + [self.tokenizer.sep_token_id]
        return ids

    def unk_rate(self, rows):
        unk = UNK if self.kind == "bilstm" else self.tokenizer.unk_token_id
        chars = [c for r in rows for c in r["text"]]
        return sum(self.char_id(c) == unk for c in chars) / len(chars)


def make_loader(rows, char_encoder, batch_size, shuffle, pad_id):
    items = [(char_encoder.encode(r["text"]),
              [LABEL2ID[t] for t in spans_to_bio(r["text"], r["entities"])]) for r in rows]
    extra = 2 if char_encoder.kind == "encoder" else 0

    def collate(batch):
        length = max(len(tags) for _, tags in batch)
        ids = torch.full((len(batch), length + extra), pad_id, dtype=torch.long)
        tags = torch.zeros((len(batch), length), dtype=torch.long)
        mask = torch.zeros((len(batch), length), dtype=torch.bool)
        for i, (x, y) in enumerate(batch):
            n = len(y)
            ids[i, :len(x)] = torch.tensor(x)   # encoder: [CLS] 字符… [SEP]，SEP 紧跟句末
            tags[i, :n] = torch.tensor(y)
            mask[i, :n] = True
        return ids, tags, mask

    return DataLoader(items, batch_size=batch_size, shuffle=shuffle, collate_fn=collate)


# ───────────────────────────── 训练 / 预测 ─────────────────────────────

def loss_fn(model, ids, tags, mask, device):
    with torch.autocast(device_type=device, dtype=torch.bfloat16,
                        enabled=device == "cuda" and isinstance(model, EncoderTagger)):
        emissions = model.emissions(ids, mask)
    if model.crf is not None:
        return model.crf.nll(emissions, tags, mask)
    return nn.functional.cross_entropy(emissions[mask], tags[mask])


def forward_all(model, loader, device):
    """返回每句的发射分数（CPU, float32）与 mask，以及前向耗时。"""
    model.eval()
    outputs = []
    sync()
    start = time.perf_counter()
    with torch.inference_mode():
        for ids, _, mask in loader:
            ids, mask = ids.to(device), mask.to(device)
            with torch.autocast(device_type=device, dtype=torch.bfloat16,
                                enabled=device == "cuda" and isinstance(model, EncoderTagger)):
                emissions = model.emissions(ids, mask)
            outputs.append((emissions.float(), mask))
    sync()
    return outputs, time.perf_counter() - start


def decode_all(model, outputs, mode, device):
    """mode: raw（Linear 取 argmax / CRF 维特比）或 constrained（加 BIO 硬约束）。"""
    constraints = constraint_scores(*transition_constraints(), device=device)
    paths = []
    sync()
    start = time.perf_counter()
    with torch.inference_mode():
        for emissions, mask in outputs:
            if model.crf is not None:
                paths.extend(model.crf.decode(emissions, mask,
                                              constraints if mode == "constrained" else None))
            elif mode == "constrained":
                paths.extend(constrained_argmax(emissions, mask, constraints))
            else:
                lengths = mask.sum(1).tolist()
                best = emissions.argmax(-1).cpu().tolist()
                paths.extend(row[:n] for row, n in zip(best, lengths))
    sync()
    return paths, time.perf_counter() - start


def to_predictions(rows, paths):
    preds = []
    for row, path in zip(rows, paths, strict=True):
        tags = [LABELS[i] for i in path]
        preds.append({"id": row["id"], "entities": bio_to_spans(row["text"], tags),
                      "illegal_transitions": count_illegal(tags)})
    return preds


def strict_f1(model, loader, rows, device):
    outputs, _ = forward_all(model, loader, device)
    paths, _ = decode_all(model, outputs, "raw", device)
    return evaluate(rows, to_predictions(rows, paths))["strict"]["f1"]


def build(method, cfg, splits, device):
    if method == "bilstm_crf":
        char_encoder = CharEncoder("bilstm", rows=splits["train"])
        model = BiLSTMCRF(len(char_encoder.vocab) + 2, cfg["bilstm"])
        pad_id = PAD
        settings = {"vocab_size": len(char_encoder.vocab) + 2, **cfg["bilstm"],
                    "embedding": "随机初始化，训练集字符表"}
    else:
        source = model_path(cfg["encoder_model"])
        tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
        char_encoder = CharEncoder("encoder", tokenizer=tokenizer)
        decoder = method.split("_")[1]
        model = EncoderTagger(source, decoder, cfg["encoder"]["dropout"])
        pad_id = tokenizer.pad_token_id
        settings = {"model": cfg["encoder_model"], "decoder": decoder, **cfg["encoder"],
                    "precision": "FP32 权重 + BF16 autocast；CRF 用 FP32",
                    "tokenization": "逐字映射到词表 id，字符与 token 一一对应"}
    settings["unk_rate_test"] = char_encoder.unk_rate(splits["test"])
    return model.to(device), char_encoder, pad_id, settings


def optimizer_for(method, model, cfg, total_steps):
    if method == "bilstm_crf":
        return torch.optim.Adam(model.parameters(), lr=cfg["bilstm"]["lr"]), None
    enc = cfg["encoder"]
    no_decay = ("bias", "LayerNorm.weight", "LayerNorm.bias")
    encoder_params = list(model.encoder.named_parameters())
    groups = [
        {"params": [p for n, p in encoder_params if not n.endswith(no_decay)],
         "lr": enc["lr"], "weight_decay": enc["weight_decay"]},
        {"params": [p for n, p in encoder_params if n.endswith(no_decay)],
         "lr": enc["lr"], "weight_decay": 0.0},
        # 随机初始化的分类头和 CRF 转移矩阵需要更大的学习率，否则几轮内学不出转移结构。
        {"params": [p for n, p in model.named_parameters() if not n.startswith("encoder.")],
         "lr": enc["head_lr"], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(groups)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * enc["warmup_ratio"]), total_steps)
    return optimizer, scheduler


def run(method, cfg, splits, device):
    model, char_encoder, pad_id, settings = build(method, cfg, splits, device)
    hp = cfg["bilstm"] if method == "bilstm_crf" else cfg["encoder"]
    loaders = {k: make_loader(rows, char_encoder, hp["batch_size"], k == "train", pad_id)
               for k, rows in splits.items()}
    total_steps = len(loaders["train"]) * hp["epochs"]
    optimizer, scheduler = optimizer_for(method, model, cfg, total_steps)
    params = sum(p.numel() for p in model.parameters())
    best_f1, best_state, selected_epoch, history = -1.0, None, None, []
    sync()
    start = time.perf_counter()
    for epoch in range(1, hp["epochs"] + 1):
        model.train()
        loss_sum, n = 0.0, 0
        for ids, tags, mask in loaders["train"]:
            ids, tags, mask = ids.to(device), tags.to(device), mask.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model, ids, tags, mask, device)
            if not torch.isfinite(loss):
                raise RuntimeError(f"{method}: loss 非有限数")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0 if scheduler else 5.0)
            optimizer.step()
            if scheduler:
                scheduler.step()
            loss_sum += loss.item() * len(ids)
            n += len(ids)
        val_f1 = strict_f1(model, loaders["val"], splits["val"], device)
        row = {"epoch": epoch, "train_loss": loss_sum / n, "val_strict_f1": val_f1,
               "elapsed_seconds": time.perf_counter() - start}
        history.append(row)
        print(method, row, flush=True)
        if val_f1 > best_f1:
            best_f1, selected_epoch = val_f1, epoch
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        write_json(ROOT / "results" / f"{method}_training.json", history)
    sync()
    train_seconds = time.perf_counter() - start
    model.load_state_dict(best_state)
    checkpoint_path(method).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "method": method,
                "vocab": getattr(char_encoder, "vocab", None), "settings": settings},
               checkpoint_path(method).with_suffix(".pt"))
    del optimizer, scheduler
    gc.collect()

    outputs, forward_seconds = forward_all(model, loaders["test"], device)
    peak = torch.cuda.max_memory_allocated() / 1024**3 if device == "cuda" else 0.0
    settings.update({"selection": "validation strict micro-F1（不约束解码）",
                     "train_time_includes": "训练、每轮验证集解码与最佳权重拷贝",
                     "predict_time_includes": "测试集前向 + 解码；不含权重加载"})
    modes = ["raw"] if method == "bilstm_crf" else ["raw", "constrained"]
    for mode in modes:
        paths, decode_seconds = decode_all(model, outputs, mode, device)
        name = method if mode == "raw" else method + "_constrained"
        save_result(name, splits, to_predictions(splits["test"], paths),
                    train_seconds=train_seconds if mode == "raw" else 0.0,
                    predict_seconds=forward_seconds + decode_seconds,
                    settings=settings | {"decode": mode,
                                         "reuses_checkpoint_of": method if mode != "raw" else None},
                    history=history if mode == "raw" else [], selected_epoch=selected_epoch,
                    trainable_params=params, total_params=params, peak_gpu_gib=peak,
                    extra={"forward_seconds": forward_seconds, "decode_seconds": decode_seconds})


def load_tagger(method, cfg, device):
    """从 checkpoints/<method>.pt 恢复模型与字符编码器（供 extract.py 单句推理）。"""
    saved = torch.load(checkpoint_path(method).with_suffix(".pt"), map_location="cpu")
    if method == "bilstm_crf":
        char_encoder = CharEncoder("bilstm", vocab=saved["vocab"])
        model = BiLSTMCRF(len(saved["vocab"]) + 2, cfg["bilstm"])
    else:
        source = model_path(cfg["encoder_model"])
        tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
        char_encoder = CharEncoder("encoder", tokenizer=tokenizer)
        model = EncoderTagger(source, method.split("_")[1], cfg["encoder"]["dropout"])
    model.load_state_dict(saved["state_dict"])
    return model.to(device).eval(), char_encoder


def tag_text(model, char_encoder, text, mode, device):
    if char_encoder.kind == "encoder" and len(text) > 510:
        raise ValueError("RoBERTa 最多 512 个位置；请先切句再抽取")
    row = {"id": "input", "text": text, "entities": []}
    pad = PAD if char_encoder.kind == "bilstm" else char_encoder.tokenizer.pad_token_id
    loader = make_loader([row], char_encoder, 1, False, pad)
    outputs, _ = forward_all(model, loader, device)
    paths, _ = decode_all(model, outputs, mode, device)
    return to_predictions([row], paths)[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True,
                        choices=["bilstm_crf", "roberta_linear", "roberta_crf"])
    args = parser.parse_args()
    cfg, splits = config(), load_data()
    seed_everything(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    run(args.method, cfg, splits, device)


if __name__ == "__main__":
    main()
