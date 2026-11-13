"""Compare the base and the saved SFT adapter on the provided validation split."""
import argparse
import unicodedata
from pathlib import Path

from peft import PeftModel

from sft import ROOT, evaluate, generate, load_base, load_config, read_split, select_device, write_json


def normalize(text):
    return "".join(unicodedata.normalize("NFKC", text).casefold().split())


def rouge_l(prediction, reference):
    a, b = normalize(prediction), normalize(reference)
    previous = [0] * (len(b) + 1)
    for char in a:
        current = [0]
        for index, other in enumerate(b, start=1):
            current.append(previous[index - 1] + 1 if char == other
                           else max(previous[index], current[-1]))
        previous = current
    return 2 * previous[-1] / max(1, len(a) + len(b))


def run_generation(model, tokenizer, rows, device, batch_size, max_new):
    outputs = []
    for start in range(0, len(rows), batch_size):
        outputs.extend(generate(model, tokenizer, [row.instruction for row in rows[start:start + batch_size]],
                                device, max_new))
    return outputs


def summarize(rows, outputs):
    total = len(rows)
    return {
        "examples": total,
        "rouge_l_f1": sum(rouge_l(output["text"], row.answer)
                         for row, output in zip(rows, outputs, strict=True)) / total,
        "exact_match": sum(normalize(output["text"]) == normalize(row.answer)
                           for row, output in zip(rows, outputs, strict=True)) / total,
        "end_token_rate": sum(output["stopped"] for output in outputs) / total,
        "avg_new_tokens": sum(output["new_tokens"] for output in outputs) / total,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "sft_val.jsonl")
    parser.add_argument("--base", type=Path)
    parser.add_argument("--adapter", type=Path, default=ROOT / "adapter")
    parser.add_argument("--out", type=Path, default=ROOT / "runs" / "comparison")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--generation-batch", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    args = parser.parse_args()
    cfg = load_config(args.config)
    args.out.mkdir(parents=True, exist_ok=True)
    device = select_device(args.device)
    model, tokenizer = load_base(cfg, args.base, device)
    rows, stats = read_split(args.data, tokenizer, cfg["max_length"])
    base_val = evaluate(model, rows, tokenizer, device, cfg["micro_batch"])
    base_outputs = run_generation(model, tokenizer, rows, device, args.generation_batch, args.max_new_tokens)
    print("Base generation complete", flush=True)
    model = PeftModel.from_pretrained(model, args.adapter)
    sft_val = evaluate(model, rows, tokenizer, device, cfg["micro_batch"])
    sft_outputs = run_generation(model, tokenizer, rows, device, args.generation_batch, args.max_new_tokens)
    result = {
        "split": "validation", "data_sha256": stats["sha256"],
        "decode": "greedy", "max_new_tokens": args.max_new_tokens,
        "base": {"teacher_forced": base_val, "generated": summarize(rows, base_outputs)},
        "sft": {"teacher_forced": sft_val, "generated": summarize(rows, sft_outputs)},
    }
    write_json(args.out / "comparison.json", result)
    write_json(args.out / "generations.json", [
        {"row": i, "instruction": row.instruction, "reference": row.answer,
         "base": base, "sft": sft}
        for i, (row, base, sft) in enumerate(zip(rows, base_outputs, sft_outputs, strict=True), start=1)
    ])
    print(result, flush=True)


if __name__ == "__main__":
    main()
