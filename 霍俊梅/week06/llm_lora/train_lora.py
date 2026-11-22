"""
方法五：Qwen3-0.6B 指令微调（LoRA 与全量微调两种模式）

教学重点：
  1. SFT 的关键不是模型，而是三件事：**chat 模板、Loss Masking、数据格式**。
  2. Loss Masking：prompt 部分 labels 全设 -100（不算 loss），只在答案 token 上算；
     不 mask 的话模型学的是「复述模板」，不是「答标签」。
  3. 两种微调模式的分工：
       LoRA（默认）       只训低秩旁路，2.29M / 596M = 0.38%，显存约 3 GB，adapter 9 MB
       全量微调 --full_ft 596M 参数全更新，显存 6~8 GB（8G 卡要开梯度检查点），权重 1.2 GB
     但成本差一个量级 —— 这就是「什么时候不值得全量微调」的答案。
  4. Qwen3 必须 enable_thinking=False，且训练与推理的模板必须完全一致。

用法：
  python llm_lora/train_lora.py                                   # LoRA，5000 条 × 2 epoch
  python llm_lora/train_lora.py --num_train 1000 --epochs 1        # 课堂快速版
  python llm_lora/train_lora.py --full_ft --num_train 3000 --epochs 2   # 全量微调
  python llm_lora/train_lora.py --max_steps 20                     # 冒烟测流程
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.data import load_splits, sample_subset
from common.label import LABELS
from common.model_paths import qwen3_path

CODE_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = CODE_ROOT / "outputs"

SYSTEM = (
    "你要判断一段中文答案是「人类用户手写的回答」还是「ChatGPT 生成的回答」。\n"
    "判断线索：ChatGPT 倾向书面语、结构化分点（首先/其次/最后）、套话和免责声明、"
    "语气礼貌中立；人类回答更口语化、有个人经历、情绪化表达或很短很随意。\n"
    "只输出 human 或 chatGPT 中的一个，不要输出任何其他内容。"
)

MAX_TEXT_CHARS = 400      # 与 api_zero_shot 保持一致：文风信号在前段


def build_messages(text: str):
    """训练/推理共用同一套消息结构——模板不一致是 SFT 排查不到位的头号原因"""
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": text[:MAX_TEXT_CHARS]},
    ]


def answer_of(label: int) -> str:
    return LABELS[label]          # "human" / "chatGPT"


def to_ids(x):
    """transformers 5.x 的 apply_chat_template(tokenize=True) 可能返回
    tokenizers.Encoding 而不是 list，统一转成 python list"""
    return list(x.ids) if hasattr(x, "ids") else list(x)


class SFTDataset(Dataset):
    """
    每条样本 = prompt 部分（不算 loss） + 答案部分（算 loss）

    实现：分别对 prompt / 完整对话套 chat 模板，
    prompt 的 token 数就是 loss mask 的分界线（labels 前 n 个设 -100）。
    """

    def __init__(self, data, tok, max_length=256):
        self.data = data
        self.tok = tok
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        d = self.data[i]
        msgs = build_messages(d["text"] or "")
        # prompt：到 assistant 开头为止（enable_thinking=False 跳过 Qwen3 的思考段）
        prompt_text = self.tok.apply_chat_template(
            msgs, add_generation_prompt=True, enable_thinking=False,
            tokenize=False)
        # 完整对话：多带一条 assistant 答案
        full_text = self.tok.apply_chat_template(
            msgs + [{"role": "assistant", "content": answer_of(d["label"])}],
            enable_thinking=False, tokenize=False)

        prompt_ids = self.tok(prompt_text, add_special_tokens=False)["input_ids"]
        full_ids = self.tok(full_text, add_special_tokens=False)["input_ids"]

        # 截断到 max_length：从左侧截 prompt（保住答案段），
        # 且至少留 1 个答案 token，避免整条 labels 全 -100（loss 会变 NaN）
        if len(full_ids) > self.max_length:
            n_cut = len(full_ids) - self.max_length
            full_ids = full_ids[n_cut:]
            prompt_ids = prompt_ids[n_cut:]
        n_prompt = min(len(prompt_ids), max(len(full_ids) - 1, 0))
        labels = [-100] * n_prompt + full_ids[n_prompt:]   # ← Loss Masking 在这一行
        return {"input_ids": full_ids, "labels": labels}


def collate(batch, pad_id):
    maxlen = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, attn = [], [], []
    for b in batch:
        n = maxlen - len(b["input_ids"])
        input_ids.append(b["input_ids"] + [pad_id] * n)
        labels.append(b["labels"] + [-100] * n)            # padding 也不算 loss
        attn.append([1] * len(b["input_ids"]) + [0] * n)
    return (torch.tensor(input_ids), torch.tensor(labels), torch.tensor(attn))


def parse_label(text: str) -> int:
    t = text.lower()
    hits = [n for n in LABELS if n.lower() in t]
    return LABELS.index(max(hits, key=len)) if hits else -1


@torch.no_grad()
def evaluate(model, tok, samples, device, batch_size=8):
    """推理与训练用同一模板；贪心解码，逐条解析 human/chatGPT"""
    model.eval()
    preds = []
    for i in range(0, len(samples), batch_size):
        chunk = samples[i:i + batch_size]
        prompts = [tok.apply_chat_template(
            build_messages(d["text"] or ""), add_generation_prompt=True,
            enable_thinking=False, tokenize=False) for d in chunk]
        enc = tok(prompts, return_tensors="pt", padding=True,
                  padding_side="left", add_special_tokens=False).to(device)
        out = model.generate(**enc, max_new_tokens=8, do_sample=False,
                             pad_token_id=tok.pad_token_id)
        for ids in out[:, enc["input_ids"].shape[1]:]:     # 只看新生成的部分
            preds.append(parse_label(tok.decode(ids, skip_special_tokens=True)))
    model.train()
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=qwen3_path)
    ap.add_argument("--num_train", type=int, default=5000)
    ap.add_argument("--num_val", type=int, default=500, help="评估条数（生成式评估慢，抽部分）")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=4, help="MPS 显存有限，batch 8 会 OOM")
    ap.add_argument("--lr", type=float, default=None, help="默认 LoRA 1e-4 / 全量 1e-5")
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--max_length", type=int, default=256, help="截断长度；seq×batch×vocab 的 logits 是显存大头")
    ap.add_argument("--full_ft", action="store_true", help="全量微调（默认 LoRA）")
    ap.add_argument("--max_steps", type=int, default=0, help=">0 时冒烟测试")
    ap.add_argument("--out", default="outputs/lora_pred.json")
    args = ap.parse_args()

    from peft import LoraConfig, get_peft_model, TaskType
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available()
              else "cpu")
    lr = args.lr if args.lr is not None else (1e-5 if args.full_ft else 1e-4)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16)
    model.to(device)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    if args.full_ft:
        model.train()
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        lcfg = LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_r * 2, lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lcfg)
        model.train()
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        model.print_trainable_parameters()
    print(f"device={device}  mode={'full_ft' if args.full_ft else 'LoRA'}  "
          f"trainable={n_trainable/1e6:.2f}M  lr={lr}")

    train, val = load_splits()
    train = sample_subset(train, args.num_train)
    val = sample_subset(val, args.num_val)
    dl = DataLoader(SFTDataset(train, tok, args.max_length), batch_size=args.batch_size,
                    shuffle=True, collate_fn=lambda b: collate(b, tok.pad_token_id))

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=0.01)
    total_steps = args.max_steps or len(dl) * args.epochs

    step, t0, running = 0, time.time(), 0.0
    done = False
    for epoch in range(1, args.epochs + 1):
        if done:
            break
        for input_ids, labels, attn in dl:
            out = model(input_ids=input_ids.to(device), attention_mask=attn.to(device),
                        labels=labels.to(device))
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            optimizer.zero_grad()
            step += 1
            running += out.loss.item()
            if step % 50 == 0:
                print(f"  step {step}/{total_steps}  loss={running/50:.4f}  "
                      f"({time.time()-t0:.0f}s)", flush=True)
                running = 0.0
                if device == "mps":
                    torch.mps.empty_cache()   # 变长 batch 易碎片化，定期归还系统
            if step >= total_steps:
                done = True
                break

    # ------------------------------------------------------------ 保存
    tag = "fullft" if args.full_ft else f"lora_r{args.lora_r}"
    save_dir = OUT_DIR / f"qwen3_{tag}"
    if args.full_ft:
        model.save_pretrained(save_dir)
    else:
        model.save_pretrained(save_dir)          # peft 模型：只存 adapter（几 MB）
    print(f"模型已保存 → {save_dir}")

    # ------------------------------------------------------------ 评估
    print(f"\n评估 {len(val)} 条（贪心解码）...")
    preds = evaluate(model, tok, val, device)
    y_true = [d["label"] for d in val]
    n_fail = sum(p == -1 for p in preds)
    print(f"解析失败 {n_fail}/{len(val)}")
    ok = [(t, p) for t, p in zip(y_true, preds) if p != -1]
    if ok:
        print(classification_report([t for t, _ in ok], [p for _, p in ok],
                                    target_names=LABELS, digits=4))
        print("混淆矩阵 (行=实际, 列=预测):")
        print(confusion_matrix([t for t, _ in ok], [p for _, p in ok]))

    out_path = CODE_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "meta": {"method": f"sft_{'fullft' if args.full_ft else 'lora'}",
                 "model": str(args.model), "n_train": len(train), "n_val": len(val),
                 "epochs": args.epochs, "lr": lr, "parse_fail": n_fail},
        "preds": [{"idx": d["idx"], "text": (d["text"] or "")[:100],
                   "true": t, "pred": p}
                  for d, t, p in zip(val, y_true, preds)],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"→ 结果写入 {out_path}")


if __name__ == "__main__":
    main()
