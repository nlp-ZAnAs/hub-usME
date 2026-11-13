# -*- coding: utf-8 -*-
"""
sft_data.py — SFT 的数据层（本周作业最核心的一个文件）

一句话：**SFT 和普通语言模型的区别，几乎全在这里。**

    普通语言模型（第四周）              SFT（本周）
    ─────────────────────────────────  ──────────────────────────────────────
    输入   任意文本切 64 字窗口          「问：{问题}\n答：{答案}<eos>」
    labels 每个字都算 loss（错一位）      问句段全填 -100（不算），答句段才算
    学什么 「下一个字是什么」            「被问之后该怎么答」

为什么问句段要填 -100？
    -100 是 PyTorch cross_entropy 的 ignore_index 约定。填了它，这些位置既不产生
    梯度、也不计入分母。于是「模型把问题原文背下来」这件事得不到任何奖励，模型只能
    靠**答句**拿分 —— 这就是 SFT 把「续写」变成「问答」的全部秘密。

模板为什么不用 Qwen 那套 <|im_start|>？
    这是字符级词表，实测 4000 个字里 `<`、`|`、`>` 三个字符都不存在，硬塞进去只会
    变成 <unk>，白白浪费 12 个位置。所以改用纯中文标记「问：」「答：」，这 5 个字符
    都在表内，模型第四周就见过它们，迁移成本最低。
"""

from collections import Counter

import torch
from torch.utils.data import Dataset

# 三个特殊符号的名字（会追加到字表尾部）
PAD_NAME = "<pad>"
EOS_NAME = "<eos>"
UNK_NAME = "<unk>"

# 一条 SFT 样本的模板：问题段 + 答案段
PROMPT_PREFIX = "问："
ANSWER_PREFIX = "\n答："

IGNORE_INDEX = -100   # 不计 loss 的标签值（PyTorch 约定）


# ─────────────────────────────── 词表 ───────────────────────────────

class Vocab:
    """
    字符级词表 + 三个特殊符号。

    特殊符号为什么排在 4000 之后，而不是 0/1/2？
        因为第四周那 4000 个字的 id 已经跟着 embedding 一起训练过了。id 一旦整体挪动，
        旧权重就全对不上号（第 5 个字变成第 7 个字）。所以新符号一律「追加在尾部」，
        旧 id 一个都不碰 —— 这是给已有 checkpoint 扩词表的标准做法。
    """

    def __init__(self, char2idx, idx2char, pad_id, eos_id, unk_id):
        self.char2idx = char2idx
        self.idx2char = idx2char
        self.pad_id = pad_id
        self.eos_id = eos_id
        self.unk_id = unk_id

    def __len__(self):
        return len(self.char2idx)

    @classmethod
    def from_ckpt(cls, ck):
        sp = ck["special"]
        return cls(ck["char2idx"], ck["idx2char"], sp["pad"], sp["eos"], sp["unk"])

    def to_ckpt_fields(self):
        """写回 checkpoint 时要带上的三个字段。"""
        return {
            "char2idx": self.char2idx,
            "idx2char": self.idx2char,
            "special": {"pad": self.pad_id, "eos": self.eos_id, "unk": self.unk_id},
        }

    def encode(self, text):
        """逐字符编码。表外字符落到 <unk>：保住位置、只丢内容，比直接删掉干净。"""
        return [self.char2idx.get(ch, self.unk_id) for ch in text]

    def decode(self, ids):
        """id → 字符串。特殊符号不参与还原（<pad>/<eos>/<unk> 不是真字符）。"""
        specials = {self.pad_id, self.eos_id, self.unk_id}
        return "".join(self.idx2char.get(int(i), "")
                       for i in ids if int(i) not in specials)


def extend_vocab(base_char2idx, extra_text):
    """
    在原 4000 字表后面追加：[<pad>, <eos>, <unk>, 语料里出现的新字符]。

    - 新字符按出现频次从高到低排：高频字排在前面，能更早拿到足够的训练信号；
    - 返回 (char2idx, idx2char, pad_id, eos_id, unk_id)。
    """
    char2idx = dict(base_char2idx)
    idx2char = {i: c for c, i in char2idx.items()}

    # 旧 id 区间 [0, n)，新符号从这里开始
    n = len(char2idx)
    pad_id, eos_id, unk_id = n, n + 1, n + 2

    # 三个特殊符号必须登记进字表本身。
    # 不登记的话 char2idx 的长度还是 4000，后面第一个新字符（空格，出现 8700 次最多）
    # 就会拿到 id=4000 —— 正好是 pad_id，空格和 padding 撞车，训练直接废掉。
    for name, sid in ((PAD_NAME, pad_id), (EOS_NAME, eos_id), (UNK_NAME, unk_id)):
        char2idx[name] = sid
        idx2char[sid] = name

    # 统计语料里「表外」的字符
    oov_chars = [c for c, _ in Counter(extra_text).most_common() if c not in char2idx]
    for c in oov_chars:
        char2idx[c] = len(char2idx)
        idx2char[len(char2idx) - 1] = c

    return char2idx, idx2char, pad_id, eos_id, unk_id


# ─────────────────────────────── 样本构造 ───────────────────────────────

def build_prompt(instruction):
    """问句段：`问：{问题}\n答：`  —— 这段不计 loss，但模型生成时要吃进去。"""
    return PROMPT_PREFIX + instruction + ANSWER_PREFIX


def load_jsonl(path):
    """读 sft_train.jsonl / sft_val.jsonl（每行一个 {"instruction","output"}）。"""
    import json
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class SFTDataset(Dataset):
    """
    一条样本长这样：

        input_ids = encode("问：问题\\n答：")  +  encode(答案)  +  [<eos>]
        labels    = [-100 ...同长度...]        +  encode(答案)  +  [<eos>]
                     └── 问句段，不计 loss ──┘   └──── 只有答句段计 loss ────┘

    为什么 labels 要和 input_ids 等长、而不是错一位？
        第 t 位的 label 就是「input 第 t 个字」。训练时用 logits[:, :-1] 去对 labels[:, 1:]，
        等价于第四周的 y = x[1:]。保持等长，是为了让 -100 的遮盖位置一眼能看出边界。
    """

    def __init__(self, rows, vocab, max_len=128, verbose=True):
        self.rows = rows
        self.vocab = vocab
        self.max_len = max_len

        # 先全量编码一遍，统计被截断的样本（截断会丢掉结尾的 <eos>，模型学不会「停」）
        self.items = []
        n_truncated = 0
        for r in rows:
            prompt_ids = vocab.encode(build_prompt(r["instruction"]))
            answer_ids = vocab.encode(r["output"]) + [vocab.eos_id]
            ids = prompt_ids + answer_ids
            labels = [IGNORE_INDEX] * len(prompt_ids) + answer_ids

            if len(ids) > max_len:
                n_truncated += 1
                ids, labels = ids[:max_len], labels[:max_len]

            self.items.append((torch.tensor(ids, dtype=torch.long),
                               torch.tensor(labels, dtype=torch.long)))

        if verbose and n_truncated:
            print(f"[warn] {n_truncated} 条样本超过 max_len={max_len} 被截断"
                  f"（占比 {n_truncated / max(len(rows), 1) * 100:.1f}%）")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        ids, labels = self.items[i]
        return {"input_ids": ids, "labels": labels}


def make_collate_fn(pad_id):
    """
    把 batch 里的样本 pad 到等长。

    三个张量：input_ids（用 pad_id 填）、labels（用 -100 填，绝不学 padding）、
    pad_mask（bool，True = 忽略该位置）。
    注意：padding 位置在 labels 里也必须是 -100，否则模型会认真地学「怎么补 pad」。
    """
    def collate(batch):
        B = len(batch)
        T = max(len(b["input_ids"]) for b in batch)

        input_ids = torch.full((B, T), pad_id, dtype=torch.long)
        labels = torch.full((B, T), IGNORE_INDEX, dtype=torch.long)
        pad_mask = torch.ones((B, T), dtype=torch.bool)      # 默认全忽略，再逐条打开

        for i, b in enumerate(batch):
            n = len(b["input_ids"])
            input_ids[i, :n] = b["input_ids"]
            labels[i, :n] = b["labels"]
            pad_mask[i, :n] = False                          # 真实 token 不忽略

        return {"input_ids": input_ids, "labels": labels, "pad_mask": pad_mask}

    return collate


# ─────────────────────────────── 可视化自检 ───────────────────────────────

def preview_one(vocab, dataset, index=0):
    """
    把一条样本摊开打印，训练前必看：

        哪一段被 -100 抹掉了、哪一段计 loss、有几个 token 在学。
    这一眼不看清，后面 loss 掉得再漂亮也可能是「模型在背问题」。
    """
    item = dataset[index]
    ids = item["input_ids"].tolist()
    labels = item["labels"].tolist()
    n_loss = sum(1 for x in labels if x != IGNORE_INDEX)
    prompt_len = len(ids) - n_loss

    print(f"\n── 样本 #{index} ── 共 {len(ids)} token，其中计 loss 的 {n_loss} 个")
    print(f"问句段（{prompt_len} token，labels=-100，不计 loss）：")
    print("   " + vocab.decode(ids[:prompt_len]).replace("\n", "\\n"))
    print("答句段（{0} token，计 loss）：".format(n_loss))
    print("   " + vocab.decode(ids[prompt_len:]))

    print("前 40 个位置 id / label 对照（label=-100 表示不学）：")
    print("  ids   :", ids[:40])
    print("  labels:", labels[:40])
