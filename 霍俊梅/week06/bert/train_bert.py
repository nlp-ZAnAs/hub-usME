"""
BERT 微调：human / ChatGPT 二分类（方法三，L2 主力）

教学重点：
  1. 和 TF-IDF 的本质区别：TF-IDF 把文本当「词袋」，BERT 通过 12 层
     attention 看到词序和上下文——「哈哈」出现在开头还是结尾、句式是否
     成体系，这些信号词袋模型全部丢失，这就是上限差距的来源。
  2. 微调的纪律和前两路一致：tokenizer/模型在 train 上更新参数，
     val 只用来评估和选 best checkpoint，绝不参与梯度更新。
  3. 动态 padding：按 batch 内最长序列补齐，而不是全部 pad 到 max_length。
     统计过 P50 才 100 出头，静态 pad 到 256 会浪费一半以上的计算。

用法：
  cd week06 && python bert/train_bert.py [--n 2000] [--max_length 256] [--epochs 2]
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.data import load_splits, sample_subset
from common.label import LABELS

CODE_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = CODE_ROOT / "outputs"
BERT_DIR = "bert-base-chinese"


class AnswerDataset(Dataset):
    """只存原始文本；tokenize 延迟到 collate_fn，配合动态 padding"""

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        d = self.data[i]
        return {"text": d["text"] or "", "label": d["label"]}


def make_collate(tok, max_length):
    """一个 batch 一起 tokenize：truncation 按 max_length，padding 按批内最长"""
    def collate(batch):
        enc = tok([b["text"] for b in batch],
                  truncation=True, max_length=max_length, padding=True,
                  return_tensors="pt")
        enc["labels"] = torch.tensor([b["label"] for b in batch])
        return enc
    return collate


class BertClassifier(nn.Module):
    """BERT + dropout + 线性分类头。pool 可选 cls / mean，消融实验用"""

    def __init__(self, model_dir, num_labels=2, pool="cls", dropout=0.1):
        super().__init__()
        from transformers import AutoModel
        self.bert = AutoModel.from_pretrained(str(model_dir))
        self.pool = pool
        hidden = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_labels)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                        return_dict=True)
        hs = out.last_hidden_state                       # (B, L, H)
        if self.pool == "cls":
            vec = hs[:, 0]                               # [CLS] 向量
        else:                                            # mean：对有效 token 求平均
            m = attention_mask.unsqueeze(-1).float()     # (B, L, 1) padding 位为 0
            vec = (hs * m).sum(1) / m.sum(1).clamp(min=1e-9)
        return self.classifier(self.dropout(vec))        # (B, 2) 未经 softmax 的 logits


@torch.no_grad()
def evaluate(model, val_dl, device):
    model.eval()
    golds, preds, scores = [], [], []
    for batch in val_dl:
        logits = model(batch["input_ids"].to(device),
                       batch["attention_mask"].to(device))
        probs = logits.softmax(-1)                    # 概率作为置信度，供下游分流用
        preds += logits.argmax(-1).cpu().tolist()
        scores += probs.max(-1).values.cpu().tolist()  # GPU 算完搬回 CPU 再转 list
        golds += batch["labels"].tolist()
    model.train()
    report = classification_report(golds, preds, target_names=LABELS,
                                   digits=4, output_dict=True)
    matrix = confusion_matrix(golds, preds)
    return report, matrix, preds, scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bert-dir", default=BERT_DIR, help="bert 模型目录或名称")
    ap.add_argument("--n", type=int, default=0, help="抽样条数，0=全部（调试用）")
    ap.add_argument("--max_length", type=int, default=256, help="截断长度，见 token_stats.py")
    ap.add_argument("--batch_size", type=int, default=16, help="MPS 显存有限，不够就调小")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-5, help="BERT 微调标准学习率 2e-5~5e-5")
    ap.add_argument("--pool", default="cls", choices=["cls", "mean"])
    ap.add_argument("--out", default="outputs/bert_pred.json", help="预测结果落盘路径")
    args = ap.parse_args()

    from transformers import AutoTokenizer, get_linear_schedule_with_warmup

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available()
              else "cpu")
    print(f"device={device}")

    tokenizer = AutoTokenizer.from_pretrained(args.bert_dir)
    model = BertClassifier(args.bert_dir, pool=args.pool).to(device)

    train, val = load_splits()
    if args.n:
        train = sample_subset(train, args.n)
        val = sample_subset(val, args.n)
    collate = make_collate(tokenizer, args.max_length)
    train_dl = DataLoader(AnswerDataset(train), batch_size=args.batch_size,
                          shuffle=True, collate_fn=collate)
    val_dl = DataLoader(AnswerDataset(val), batch_size=args.batch_size * 2,
                        shuffle=False, collate_fn=collate)
    print(f"train={len(train)}  val={len(val)}  "
          f"batches/epoch={len(train_dl)}  max_length={args.max_length}")

    # AdamW + 线性 warmup 衰减：BERT 微调的标配，lr 峰值小、先热身再衰减
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=0.01)
    total_steps = len(train_dl) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps)
    loss_fn = nn.CrossEntropyLoss()

    best_acc, best_path = 0.0, OUT_DIR / "bert_best.pt"
    OUT_DIR.mkdir(exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        t0, running = time.time(), 0.0
        for step, batch in enumerate(train_dl, 1):
            logits = model(batch["input_ids"].to(device),
                           batch["attention_mask"].to(device))
            loss = loss_fn(logits, batch["labels"].to(device))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 防梯度爆炸
            optimizer.step()
            scheduler.step()
            running += loss.item()
            if step % 100 == 0:
                print(f"  epoch {epoch} step {step}/{len(train_dl)} "
                      f"loss={running/step:.4f} ({time.time()-t0:.0f}s)")

        report, matrix, _, _ = evaluate(model, val_dl, device)
        acc = report["accuracy"]
        print(f"[epoch {epoch}] loss={running/len(train_dl):.4f}  val acc={acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), best_path)
            print(f"  已保存 best checkpoint → {best_path}")

    # 加载 best，出最终报告
    model.load_state_dict(torch.load(best_path, map_location=device))
    report, matrix, preds, scores = evaluate(model, val_dl, device)
    print("\n[best checkpoint · val]")
    for cls in LABELS:
        m = report[cls]
        print(f"  {cls:>8s}  P={m['precision']:.4f}  R={m['recall']:.4f}  F1={m['f1-score']:.4f}")
    print(f"  accuracy={report['accuracy']:.4f}")
    print("混淆矩阵 (行=实际, 列=预测):")
    print(matrix)

    # -------------------------------------------- 预测结果落盘（模板与其余方法对齐）
    # score = softmax 最大概率，即模型对该条预测的置信度，下游分流按它路由
    import json
    out_path = CODE_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "meta": {
            "method": "bert_ft",
            "model": str(args.bert_dir),
            "n": len(val),
            "epochs": args.epochs,
            "lr": args.lr,
            "pool": args.pool,
            "report": {"acc": round(report["accuracy"], 4),
                       "macro_f1": round(report["macro avg"]["f1-score"], 4)},
        },
        "preds": [{"idx": s["idx"], "text": (s["text"] or "")[:100],
                   "true": s["label"], "pred": p, "score": round(sc, 4)}
                  for s, p, sc in zip(val, preds, scores)],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"→ 结果写入 {out_path}")


if __name__ == "__main__":
    main()
