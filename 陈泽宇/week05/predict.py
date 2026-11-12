"""Load the base model or its Week 05 adapter and answer a single instruction."""
import argparse
import json
from pathlib import Path

from peft import PeftModel

from sft import ROOT, generate, load_base, load_config, select_device


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--base", type=Path)
    parser.add_argument("--adapter", type=Path, default=ROOT / "adapter")
    parser.add_argument("--base-only", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    args = parser.parse_args()
    device = select_device(args.device)
    model, tokenizer = load_base(load_config(args.config), args.base, device)
    if not args.base_only:
        model = PeftModel.from_pretrained(model, args.adapter)
    result = generate(model, tokenizer, [args.instruction], device, args.max_new_tokens)[0]
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
