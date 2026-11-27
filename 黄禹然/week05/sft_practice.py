"""
sft_practice.py — 自己动手重写的 SFT 训练脚本（功能与 sft.py 一致，写法不同）

【和答案版的关系】
    依赖完全相同：复用 transformer_lm.py 的 TransformerLM / encode / SPECIAL_TOKENS，
    加载它训出的 base_lm.pt 基座，用同一份字表继续训练。
    差别只在实现方式，方便对照学习：
        1) 标签 y 的构造：答案版用 for 循环逐位赋值，这里用一次切片 y[start:end] = x[start+1:end+1]，
           效果完全等价，但更贴近向量化思维；
        2) 凑 batch：答案版手工建 (B,T) 大张量再逐行填，这里用 torch.nn.utils.rnn.pad_sequence；
        3) padding 掩码：这里从"每条样本的真实长度"推导，而不是靠 x == pad_id 判断
           （因为 <|end|> 本身是真实 token，不能拿它当 padding 的判定依据）；
        4) 日志取数用 loss.item()，不会触发 float(loss) 的 requires_grad 警告。

【SFT 的三件事（和答案版口径一致）】
    1) 把 (instruction, output) 拼成 <|user|>问题<|assistant|>回答<|end|>；
    2) loss mask：模板、问题、padding 的标签填 -100，模型只在回答段被监督；
    3) 训练 + 用"回答段 PPL"评测，存验证集最优的 checkpoint。

用法:
    python sft_practice.py                          # 从 base_lm.pt 接着训，存 sft_practice.pt
    python sft_practice.py --epochs 20 --lr 1e-3    # 调参
    python sft_practice.py --eval-only              # 只评测
    python sft_practice.py --generate "列出三种健康的早餐食品。"
    python sft_practice.py --no-loss-mask           # 消融对照：模板和问题也参与 loss
    python sft_practice.py --save-best last         # 存最后一轮（复现过拟合用）
"""

import argparse
import json
import math
import os
import random

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

# 结构、字表、编解码全部复用基座文件：字表不一致 = embedding 形状对不上 = 无法接着训
from transformer_lm import SPECIAL_TOKENS, TransformerLM, encode

HERE = os.path.dirname(os.path.abspath(__file__))
# 三个特殊标记的字符串（id 早已在 base_lm.pt 的 char2idx 里固化为 0/1/2）
USER, ASSISTANT, END = SPECIAL_TOKENS

IGNORE_LABEL = -100  # PyTorch cross_entropy 的约定：这个标签的位置不算 loss、不回传梯度


# ─────────────────────── 数据 ───────────────────────

def load_jsonl(path):
    """每行一个 JSON 对象，取 instruction / output 组成 (问题, 回答) 对。"""
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                pairs.append((rec["instruction"], rec["output"]))
    return pairs


def make_prompt(question):
    """对话模板前半段：<|user|>问题<|assistant|>。这段是"输入侧"，永远不算 loss。"""
    return f"{USER}{question}{ASSISTANT}"


class ConversationDataset(Dataset):
    """
    一条样本 = 完整对话的 id 序列 + 前缀长度（模板+问题占多少个 token）。

    存"前缀长度"而不是提前算好 y，是因为 loss mask 开关（--no-loss-mask）要到
    __getitem__ 时才生效，这样消融实验不用重建数据集。
    """

    def __init__(self, pairs, char2idx, seq_len, use_loss_mask=True):
        self.items = []
        self.use_loss_mask = use_loss_mask
        self.num_dropped = 0

        for question, answer in pairs:
            prompt_ids = encode(make_prompt(question), char2idx)
            answer_ids = encode(answer + END, char2idx)  # 回答末尾补 <|end|>，教会"答完就停"
            if not prompt_ids or not answer_ids:
                self.num_dropped += 1
                continue

            # 超长时砍回答的尾巴：模板和问题必须完整，否则模型不知道"从哪开始答"
            if len(prompt_ids) + len(answer_ids) > seq_len:
                budget = seq_len - len(prompt_ids)
                if budget < 4:  # 砍完连 4 个字的回答都放不下，这条样本没价值
                    self.num_dropped += 1
                    continue
                answer_ids = answer_ids[:budget]

            if len(answer_ids) < 2:  # 回答太短，学不到东西
                self.num_dropped += 1
                continue

            self.items.append((prompt_ids + answer_ids, len(prompt_ids)))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        ids, prefix_len = self.items[index]
        x = torch.tensor(ids, dtype=torch.long)

        # 关键一步：y[t] 永远是"位置 t 处模型该预测的下一个字"
        #   开 loss mask：只有 [prefix_len-1, len-2] 参与 —— 第一个回答字是在
        #   <|assistant|> 那个位置被预测的，所以起点要往前挪一位（off-by-one 的精髓）；
        #   关 loss mask：整条序列都参与（消融对照用）
        y = torch.full((len(ids),), IGNORE_LABEL, dtype=torch.long)
        if self.use_loss_mask:
            y[prefix_len - 1: len(ids) - 1] = x[prefix_len: len(ids)]
        else:
            y[:-1] = x[1:]
        return x, y


def make_batch(batch, pad_id):
    """
    把变长样本 pad 成 (B, T) 的一批。返回 x / y / padding 掩码 / 有效 token 数。

    掩码必须从真实长度推导：真实序列末尾的 <|end|> 和填充用的 <|end|> 是同一个 id，
    用 x == pad_id 判断会把真实的结束符也误判成 padding。
    另外：-100 只挡 loss，挡不住注意力，所以注意力侧还需要这份掩码（两套 mask 各管各的）。
    """
    xs, ys = zip(*batch)
    x = pad_sequence(xs, batch_first=True, padding_value=pad_id)
    y = pad_sequence(ys, batch_first=True, padding_value=IGNORE_LABEL)

    lengths = torch.tensor([len(x) for x in xs])
    pad_mask = torch.arange(x.size(1))[None, :] >= lengths[:, None]  # True = padding

    num_labels = int((y != IGNORE_LABEL).sum())
    return x, y, pad_mask, num_labels


# ─────────────────────── 模型加载 ───────────────────────

def load_base_model(ckpt_path, device, seq_len):
    """按 ckpt 里记的超参重建模型、载入基座权重——"接着训"的全部含义。"""
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["args"]
    model = TransformerLM(
        vocab_size=len(ckpt["char2idx"]),
        d_model=cfg["d_model"], n_head=cfg["heads"], n_layer=cfg["layers"],
        max_len=cfg["seq_len"], dropout=cfg["dropout"],
        use_pos=not cfg.get("no_pos", False),
        causal=not cfg.get("no_mask", False),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])

    # 位置编码表只有 cfg["seq_len"] 行，SFT 想用更长的上下文必须先扩表
    if seq_len > cfg["seq_len"]:
        raise SystemExit(f"--seq_len {seq_len} 超过基座的 {cfg['seq_len']}，位置编码不够长")
    return model, ckpt


# ─────────────────────── 评测与生成 ───────────────────────

@torch.no_grad()
def evaluate(model, loader, device):
    """验证集上的"回答段"loss / PPL：ignore_index 自动跳过模板、问题、padding。"""
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for x, y, pad_mask, n in loader:
        logits = model(x.to(device), key_padding_mask=pad_mask.to(device))
        # reduction="sum" 再手动除 token 数：最后一个不满的 batch 不会被不公平地加权
        total_loss += float(F.cross_entropy(
            logits.flatten(0, 1), y.to(device).flatten(),
            ignore_index=IGNORE_LABEL, reduction="sum"))
        total_tokens += n
    avg_loss = total_loss / max(1, total_tokens)
    return avg_loss, math.exp(avg_loss)


@torch.no_grad()
def generate_answer(model, char2idx, idx2char, question, seq_len,
                    max_new_tokens=40, temperature=0.7):
    """从 <|assistant|> 之后逐字生成，采到 <|end|> 就停——"答完就停"是 SFT 教出来的。"""
    model.eval()
    device = next(model.parameters()).device
    prompt_ids = encode(make_prompt(question), char2idx)
    sequence = list(prompt_ids)
    end_id = char2idx[END]

    for _ in range(max_new_tokens):
        logits = model(torch.tensor([sequence[-seq_len:]], device=device))[0, -1]
        if temperature <= 0.01:
            next_id = int(logits.argmax())  # 温度≈0 退化为贪心解码
        else:
            next_id = int(torch.multinomial(F.softmax(logits / temperature, dim=-1), 1))
        sequence.append(next_id)
        if next_id == end_id:
            break

    # 只把回答段翻译回文字，并去掉结束符
    return "".join(idx2char[i] for i in sequence[len(prompt_ids):] if i != end_id)


# ─────────────────────── 训练主流程 ───────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=os.path.join(HERE, "base_lm.pt"))
    parser.add_argument("--train", default=os.path.join(HERE, "sft_train.jsonl"))
    parser.add_argument("--val", default=os.path.join(HERE, "sft_val.jsonl"))
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    # SFT 学习率要比预训练小一个数量级：基座已会写字，这里只调格式
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save", default=os.path.join(HERE, "sft_practice.pt"))
    parser.add_argument("--save-best", choices=["val", "last"], default="val",
                        help="val=存验证最优（防过拟合）；last=存最后一轮（复现过拟合用）")
    parser.add_argument("--no-loss-mask", dest="no_loss_mask", action="store_true",
                        help="消融对照：模板和问题也参与 loss")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--generate", default=None, help="只生成一条回答看看")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    if not os.path.exists(args.base):
        raise SystemExit(f"找不到基座 {args.base}，先跑 python transformer_lm.py")

    model, ckpt = load_base_model(args.base, device, args.seq_len)
    char2idx, idx2char = ckpt["char2idx"], ckpt["idx2char"]
    pad_id = char2idx[END]   # 用 <|end|> 当填充：它的标签永远是 -100，填充字符学不到
    vocab_size = len(char2idx)

    print(f"模板 = {USER}问题{ASSISTANT}回答{END}   "
          f"loss mask = {'开' if not args.no_loss_mask else '**关（消融）**'}")
    print(f"参数量 {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M   "
          f"字表 {vocab_size}（{len(SPECIAL_TOKENS)} 个特殊标记）   device: {device}")

    train_set = ConversationDataset(load_jsonl(args.train), char2idx, args.seq_len,
                                    use_loss_mask=not args.no_loss_mask)
    # 验证集永远开 loss mask：无论训练口径怎么变，评测口径不变，数字才能横向比较
    val_set = ConversationDataset(load_jsonl(args.val), char2idx, args.seq_len,
                                  use_loss_mask=True)
    print(f"训练样本 {len(train_set)}（丢弃 {train_set.num_dropped}）   "
          f"验证样本 {len(val_set)}（丢弃 {val_set.num_dropped}）")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              drop_last=True,
                              collate_fn=lambda b: make_batch(b, pad_id))
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            collate_fn=lambda b: make_batch(b, pad_id))

    if args.generate:
        print(f"问: {args.generate}")
        print(f"答: {generate_answer(model, char2idx, idx2char, args.generate, args.seq_len)}")
        return

    if args.eval_only:
        loss, ppl = evaluate(model, val_loader, device)
        print(f"[eval-only] 回答段 loss={loss:.4f}  PPL={ppl:.2f}")
        return

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_loss, best_epoch = float("inf"), 0
    print(f"\n{'Epoch':>6} {'TrainLoss':>10} {'ValLoss':>9} {'ValPPL(回答段)':>15}")
    print("-" * 46)

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum, token_sum = 0.0, 0
        for x, y, pad_mask, n in train_loader:
            logits = model(x.to(device), key_padding_mask=pad_mask.to(device))
            # ★ SFT 的核心就这一行的 ignore_index：-100 的位置（模板/问题/padding）
            #   不产生梯度，模型"只学怎么答，不背怎么问"
            loss = F.cross_entropy(logits.flatten(0, 1), y.to(device).flatten(),
                                   ignore_index=IGNORE_LABEL)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            # .item() 显式脱离计算图取数值，比 float(loss) 规范
            loss_sum += loss.item() * n
            token_sum += n

        val_loss, val_ppl = evaluate(model, val_loader, device)
        mark = ""
        # val 模式只在验证集变好时覆盖保存；last 模式每轮都覆盖（最后一轮留下）
        if args.save_best == "last" or val_loss < best_loss:
            if val_loss < best_loss:
                best_loss, best_epoch = val_loss, epoch
            mark = " *"
            torch.save({
                "model_state": model.state_dict(),
                "char2idx": char2idx, "idx2char": idx2char,
                "args": ckpt["args"],
                "sft": {"loss_mask": not args.no_loss_mask, "seq_len": args.seq_len,
                        "epochs": args.epochs, "lr": args.lr,
                        "train_loss": loss_sum / token_sum,
                        "val_loss": val_loss, "val_ppl": val_ppl, "epoch": epoch},
            }, args.save)
        print(f"{epoch:>6} {loss_sum / token_sum:>10.4f} {val_loss:>9.4f} {val_ppl:>15.2f}{mark}")

    if args.save_best == "last":
        print(f"\n已保存最后一轮（epoch {args.epochs}）→ {args.save}")
    else:
        print(f"\n最佳回答段 PPL: {math.exp(best_loss):.2f}（epoch {best_epoch}）已保存 {args.save}")

    print("\n生成示例：")
    for question, _ in load_jsonl(args.val)[:3]:
        print(f"  问: {question}")
        print(f"  答: {generate_answer(model, char2idx, idx2char, question, args.seq_len)}")


if __name__ == "__main__":
    main()
