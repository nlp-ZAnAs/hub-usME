r"""在 week04 字符级语言模型上进行监督微调（SFT）。

复用 ../week04/language_model_transformer.py 中的 TransformerLM；只依赖 PyTorch。
训练输入为「指令 + 回答」，仅回答和 EOS 参与 loss，指令和右侧 padding 忽略。
旧字表编号、已有权重保持对应关系；新增字符和位置的参数在微调中学习。

在仓库根目录运行（也可以把下述解释器换成已安装 PyTorch 的 python）：
    .venv-transformer-gpu/bin/python 孙燿烽/week05/sft_transformer.py train
    .venv-transformer-gpu/bin/python 孙燿烽/week05/sft_transformer.py evaluate
    .venv-transformer-gpu/bin/python 孙燿烽/week05/sft_transformer.py generate \
        --prompt "给定两个数字，请选择其中的较大数。数字1: 238 数字2: 564"

所有默认路径相对于本文件，不受运行目录影响。train 的常用参数：
    --epochs 10 --batch-size 32 --lr 1e-4 --max-length 128 --patience 3
    --base-checkpoint 孙燿烽/week04/best_transformer.pt
    --output-dir 孙燿烽/week05/sft_output
显式传入的相对路径仍相对于运行目录。

训练产物默认写入 week05/sft_output：
    best_sft_transformer.pt：最佳验证 loss 对应的模型、字表、结构和提示模板。
    metrics.json：适配字表/长度后的微调前基线、各轮答案 loss/PPL、最佳轮数。
    examples.json：固定验证问题、参考回答、微调前/后的贪心生成结果。

注意：答案 PPL 不能直接与 week04 的普通文本 PPL 比较。小模型及语料中的
错误答案会限制实际回答质量。--train-limit/--val-limit 仅用于快速检查流程，
默认 0 表示使用完整数据；本脚本保存推理权重，不保存优化器断点。
"""

import argparse
from functools import partial
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
# 直接复用上一周的模型定义，避免 week05 维护另一份相同代码。
sys.path.insert(0, str(HERE.parent / "week04"))
from language_model_transformer import TransformerLM


DEFAULT_OUTPUT = HERE / "sft_output"
DEFAULT_CHECKPOINT = DEFAULT_OUTPUT / "best_sft_transformer.pt"
PROMPT_TEMPLATE = "指令：\n{instruction}\n回答：\n"
PAD, EOS, UNK = "<PAD>", "<EOS>", "<UNK>"
IGNORE_INDEX = -100


# ─────────────────────────── 数据与标签 ───────────────────────────

def load_records(path, limit=0):
    """逐行读取并检查格式；保留原文中的空格、换行，不悄悄过滤坏数据。"""
    records = []
    with Path(path).open(encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} 不是合法 JSON：{exc.msg}") from exc
            if not isinstance(row, dict) or any(
                not isinstance(row.get(key), str) or not row[key].strip()
                for key in ("instruction", "output")
            ):
                raise ValueError(f"{path}:{line_number} 需要非空字符串 instruction 和 output。")
            records.append({"instruction": row["instruction"], "output": row["output"]})
    if not records:
        raise ValueError(f"{path} 中没有有效问答。")
    return records[:limit] if limit else records


def format_prompt(instruction):
    return PROMPT_TEMPLATE.format(instruction=instruction)


def extend_vocab(base_vocab, train_records):
    """在旧编号末尾追加特殊 token 和训练集新字符；不读取验证集来建表。"""
    vocab = dict(base_vocab)
    if set(vocab.values()) != set(range(len(vocab))):
        raise ValueError("基础字表的编号必须唯一且从 0 连续递增。")
    for token in (PAD, EOS, UNK):
        if token not in vocab:
            vocab[token] = len(vocab)
    characters = set()
    for row in train_records:
        characters.update(format_prompt(row["instruction"]) + row["output"])
    for char in sorted(characters):
        if char not in vocab:
            vocab[char] = len(vocab)
    return vocab


def encode(text, vocab):
    return [vocab.get(char, vocab[UNK]) for char in text]


class SFTDataset(Dataset):
    def __init__(self, records, vocab, max_length):
        self.samples = []
        unknown_prompt = unknown_answer = 0
        for number, row in enumerate(records, 1):
            prompt_ids = encode(format_prompt(row["instruction"]), vocab)
            answer_ids = encode(row["output"], vocab) + [vocab[EOS]]
            ids = prompt_ids + answer_ids
            # 标签先按完整序列标记，再整体左移，保留第一个答案字符的监督。
            labels = [IGNORE_INDEX] * len(prompt_ids) + answer_ids
            if len(ids) - 1 > max_length:
                raise ValueError(
                    f"第 {number} 条样本需要 {len(ids) - 1} 个输入位置，"
                    f"超过 max_length={max_length}；请增大 --max-length。"
                    "本脚本不截断问题或答案。"
                )
            self.samples.append((torch.tensor(ids[:-1]), torch.tensor(labels[1:])))
            unknown_prompt += prompt_ids.count(vocab[UNK])
            unknown_answer += answer_ids.count(vocab[UNK])
        if not self.samples:
            raise ValueError("数据集不能为空。")
        self.stats = {
            "samples": len(self.samples),
            "max_input_tokens": max(x.numel() for x, _ in self.samples),
            "answer_tokens_including_eos": sum(
                int((y != IGNORE_INDEX).sum()) for _, y in self.samples
            ),
            "unknown_prompt_tokens": unknown_prompt,
            "unknown_answer_tokens": unknown_answer,
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def collate_batch(batch, pad_id):
    xs, ys = zip(*batch)
    # 必须右侧补齐：原模型的因果掩码会阻止真实 token 看到后面的 PAD。
    # PAD 位置的预测也通过 -100 排除，因此无需修改原模型的 forward。
    return (
        pad_sequence(xs, batch_first=True, padding_value=pad_id),
        pad_sequence(ys, batch_first=True, padding_value=IGNORE_INDEX),
    )


def make_loader(dataset, vocab, batch_size, shuffle=False):
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        collate_fn=partial(collate_batch, pad_id=vocab[PAD]),
    )


# ─────────────────────────── 权重适配 ───────────────────────────

def build_finetune_model(checkpoint, vocab, max_length):
    """扩大字表和位置嵌入；逐行复制旧权重，其余 Transformer 权重严格加载。"""
    if checkpoint.get("model_type") != "transformer":
        raise ValueError("--base-checkpoint 应指向 week04 的 Transformer 预训练权重。")
    base_args = checkpoint["args"]
    config = {key: base_args[key] for key in (
        "embed_dim", "num_layers", "nhead", "ffn_dim", "dropout"
    )}
    config.update(vocab_size=len(vocab), max_seq_len=max_length)
    model = TransformerLM(**config)
    target_state = model.state_dict()
    adapted_state = {}
    expandable = {"embed.weight", "fc.weight", "fc.bias", "position_embed.weight"}
    for key, old_tensor in checkpoint["model_state"].items():
        if key not in target_state:
            raise ValueError(f"预训练权重包含不兼容的参数：{key}")
        new_tensor = target_state[key]
        if key in expandable:
            if old_tensor.shape[1:] != new_tensor.shape[1:] or old_tensor.size(0) > new_tensor.size(0):
                raise ValueError(f"不能缩小或改变 {key} 的结构：{old_tensor.shape} -> {new_tensor.shape}")
            new_tensor[:old_tensor.size(0)].copy_(old_tensor)
            adapted_state[key] = new_tensor
        else:
            adapted_state[key] = old_tensor
    model.load_state_dict(adapted_state, strict=True)
    return model, config


def load_sft_model(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("model_type") != "sft_transformer":
        raise ValueError("请使用本脚本训练得到的 best_sft_transformer.pt。")
    if checkpoint["prompt_template"] != PROMPT_TEMPLATE:
        raise ValueError("权重中的提示模板与当前脚本不一致。")
    model = TransformerLM(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return model.to(device).eval(), checkpoint["char2idx"]


# ─────────────────────────── 训练、评估、生成 ───────────────────────────

def run_epoch(model, loader, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_tokens = 0
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            count = int((y != IGNORE_INDEX).sum())
            if count == 0:
                raise ValueError("当前 batch 没有可用于监督的答案 token。")
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss_sum = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), y.reshape(-1),
                ignore_index=IGNORE_INDEX, reduction="sum",
            )
            loss = loss_sum / count
            if not torch.isfinite(loss):
                raise FloatingPointError("loss 出现 NaN/Inf，训练已停止。")
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
            # 各批次按答案 token 数加权，指令和 padding 不计入分母。
            total_loss += loss_sum.item()
            total_tokens += count
    if total_tokens == 0:
        raise ValueError("数据集中没有答案 token。")
    average = total_loss / total_tokens
    return {"loss": average, "ppl": math.exp(average), "answer_tokens": total_tokens}


@torch.inference_mode()
def generate_answer(model, vocab, instruction, device, max_new_tokens=80, temperature=0.0, top_k=30):
    model.eval()
    ids = encode(format_prompt(instruction), vocab)
    if len(ids) > model.max_seq_len:
        raise ValueError(f"指令长度 {len(ids)} 超过模型上下文 {model.max_seq_len}，请缩短指令。")
    unknown_count = ids.count(vocab[UNK])
    inverse_vocab = {index: token for token, index in vocab.items()}
    generated = []
    reason = "max_new_tokens"
    for _ in range(max_new_tokens):
        if len(ids) > model.max_seq_len:
            reason = "context_limit"
            break
        logits = model(torch.tensor([ids], device=device))[0, -1].clone()
        logits[[vocab[PAD], vocab[UNK]]] = -torch.inf
        if temperature == 0:
            token = int(logits.argmax())
        else:
            logits /= temperature
            if top_k > 0:
                values, indices = torch.topk(logits, min(top_k, logits.numel()))
                token = int(indices[torch.multinomial(values.softmax(-1), 1)].item())
            else:
                token = int(torch.multinomial(logits.softmax(-1), 1).item())
        if token == vocab[EOS]:
            reason = "eos"
            break
        ids.append(token)
        generated.append(inverse_vocab[token])
    return {"text": "".join(generated), "stop_reason": reason, "unknown_prompt_tokens": unknown_count}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def train(args, device):
    train_rows = load_records(args.train_file, args.train_limit)
    val_rows = load_records(args.val_file, args.val_limit)
    base = torch.load(args.base_checkpoint, map_location="cpu", weights_only=True)
    vocab = extend_vocab(base["char2idx"], train_rows)
    train_ds = SFTDataset(train_rows, vocab, args.max_length)
    val_ds = SFTDataset(val_rows, vocab, args.max_length)
    model, config = build_finetune_model(base, vocab, args.max_length)
    model.to(device)
    train_loader = make_loader(train_ds, vocab, args.batch_size, shuffle=True)
    val_loader = make_loader(val_ds, vocab, args.batch_size)
    save_path = args.output_dir / "best_sft_transformer.pt"
    if save_path.resolve() == args.base_checkpoint.resolve():
        raise ValueError("输出路径不能覆盖基础模型权重。")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"device: {device} | 参数量: {sum(p.numel() for p in model.parameters()):,}", flush=True)
    print(f"字表: {len(base['char2idx'])} -> {len(vocab)} | 最大输入长度: {args.max_length}", flush=True)
    print(f"训练数据: {train_ds.stats}\n验证数据: {val_ds.stats}", flush=True)

    baseline = run_epoch(model, val_loader, device)
    print(f"微调前（扩表/扩展位置后）Val Loss: {baseline['loss']:.4f} | PPL: {baseline['ppl']:.2f}", flush=True)
    examples = []
    for row in val_rows[:args.examples]:
        examples.append({
            **row,
            "before": generate_answer(model, vocab, row["instruction"], device),
        })
    report = {
        "base_checkpoint": str(args.base_checkpoint.resolve()),
        "settings": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "model_config": config, "train_data": train_ds.stats, "val_data": val_ds.stats,
        "baseline_note": "与 SFT 使用同一扩展字表、位置长度和答案 loss；新增参数尚未训练。",
        "baseline": baseline, "epochs": [],
    }
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    best_loss = math.inf
    stale_epochs = 0
    print("Epoch  Train Loss  Train PPL  Val Loss  Val PPL  Seconds", flush=True)
    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        train_metrics = run_epoch(model, train_loader, device, optimizer)
        val_metrics = run_epoch(model, val_loader, device)
        improved = val_metrics["loss"] < best_loss
        report["epochs"].append({
            "epoch": epoch, "train": train_metrics, "validation": val_metrics,
            "seconds": round(time.perf_counter() - started, 3), "improved": improved,
        })
        if improved:
            best_loss = val_metrics["loss"]
            stale_epochs = 0
            report.update(best_epoch=epoch, best_validation=val_metrics)
            torch.save({
                "model_type": "sft_transformer", "model_config": config,
                "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                "char2idx": vocab, "prompt_template": PROMPT_TEMPLATE,
                "epoch": epoch, "validation": val_metrics,
                "base_checkpoint": str(args.base_checkpoint.resolve()),
            }, save_path)
        else:
            stale_epochs += 1
        write_json(args.output_dir / "metrics.json", report)
        print(
            f"{epoch:5d}  {train_metrics['loss']:10.4f}  {train_metrics['ppl']:9.2f}  "
            f"{val_metrics['loss']:8.4f}  {val_metrics['ppl']:7.2f}  "
            f"{report['epochs'][-1]['seconds']:7.2f}{' *' if improved else ''}", flush=True,
        )
        if stale_epochs >= args.patience:
            print(f"连续 {args.patience} 轮验证 loss 未下降，提前结束。", flush=True)
            break

    # 对比使用验证集最优权重，避免拿最后一轮替代最优轮。
    best = torch.load(save_path, map_location="cpu", weights_only=True)
    model.load_state_dict(best["model_state"])
    for example in examples:
        example["after"] = generate_answer(model, vocab, example["instruction"], device)
    write_json(args.output_dir / "examples.json", examples)
    print(f"最佳轮数: {report['best_epoch']} | 验证 PPL: {report['best_validation']['ppl']:.2f}", flush=True)
    print(f"权重、指标和回答对比已保存到: {args.output_dir.resolve()}", flush=True)


# ─────────────────────────── 命令行 ───────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    common.add_argument("--seed", type=int, default=42)
    commands = parser.add_subparsers(dest="command", required=True)

    training = commands.add_parser("train", parents=[common], help="加载 week04 权重并进行 SFT")
    training.add_argument("--base-checkpoint", type=Path, default=HERE.parent / "week04" / "best_transformer.pt")
    training.add_argument("--train-file", type=Path, default=HERE / "sft_train" / "sft_train.jsonl")
    training.add_argument("--val-file", type=Path, default=HERE / "sft_train" / "sft_val.jsonl")
    training.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    training.add_argument("--max-length", type=int, default=128)
    training.add_argument("--batch-size", type=int, default=32)
    training.add_argument("--lr", type=float, default=1e-4)
    training.add_argument("--epochs", type=int, default=10)
    training.add_argument("--patience", type=int, default=3)
    training.add_argument("--train-limit", type=int, default=0, help="快速检查用；0 表示全部")
    training.add_argument("--val-limit", type=int, default=0, help="快速检查用；0 表示全部")
    training.add_argument("--examples", type=int, default=5, help="保存前后对比的验证样本数；0 表示跳过")

    evaluation = commands.add_parser("evaluate", parents=[common], help="评估 SFT 权重的答案 loss/PPL")
    evaluation.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    evaluation.add_argument("--val-file", type=Path, default=HERE / "sft_train" / "sft_val.jsonl")
    evaluation.add_argument("--val-limit", type=int, default=0)
    evaluation.add_argument("--batch-size", type=int, default=32)

    generation = commands.add_parser("generate", parents=[common], help="输入指令并生成回答")
    generation.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    generation.add_argument("--prompt", required=True)
    generation.add_argument("--max-new-tokens", type=int, default=80)
    generation.add_argument("--temperature", type=float, default=0.0, help="0 为贪心；大于 0 为随机采样")
    generation.add_argument("--top-k", type=int, default=30, help="采样时使用；0 表示不限制")
    args = parser.parse_args()
    for key in ("max_length", "batch_size", "epochs", "patience", "max_new_tokens"):
        if hasattr(args, key) and getattr(args, key) < 1:
            parser.error(f"--{key.replace('_', '-')} 必须是正整数。")
    for key in ("train_limit", "val_limit", "examples", "top_k"):
        if hasattr(args, key) and getattr(args, key) < 0:
            parser.error(f"--{key.replace('_', '-')} 不能为负数。")
    if args.command == "train" and (not math.isfinite(args.lr) or args.lr <= 0):
        parser.error("--lr 必须为有限正数。")
    if args.command == "generate":
        if not args.prompt.strip():
            parser.error("--prompt 不能为空。")
        if not math.isfinite(args.temperature) or args.temperature < 0:
            parser.error("--temperature 必须为有限非负数。")
    return args


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    # 小模型限制 CPU 线程数，避免线程调度开销超过计算本身。
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("当前 PyTorch 无法使用 CUDA，请指定 --device cpu 或使用 GPU 环境。")
    if args.command == "train":
        train(args, device)
    elif args.command == "evaluate":
        model, vocab = load_sft_model(args.checkpoint, device)
        rows = load_records(args.val_file, args.val_limit)
        dataset = SFTDataset(rows, vocab, model.max_seq_len)
        metrics = run_epoch(model, make_loader(dataset, vocab, args.batch_size), device)
        print(json.dumps({"data": dataset.stats, **metrics}, ensure_ascii=False, indent=2))
    else:
        model, vocab = load_sft_model(args.checkpoint, device)
        result = generate_answer(
            model, vocab, args.prompt, device, args.max_new_tokens, args.temperature, args.top_k,
        )
        print(result["text"])
        print(f"\n停止原因: {result['stop_reason']} | 指令中未知字符数: {result['unknown_prompt_tokens']}")


if __name__ == "__main__":
    main()
