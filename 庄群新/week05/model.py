# -*- coding: utf-8 -*-
"""
model.py — 字符级 Transformer 语言模型（沿用第四周 tf_mask.pt 的结构）

为什么结构必须和第四周一模一样？
    SFT 不是从零训，而是「接着第四周的权重往下训」。state_dict 的每个键名和维度
    都必须对得上，否则 load_state_dict 会报 missing / unexpected keys，模型直接废掉。
    所以这里原封不动搬了第四周的结构：
        Token Embedding → + 可学习位置编码 → Dropout
        → N × TransformerEncoderLayer（Pre-LN / GELU / FFN=4d）
        → 最终 LayerNorm → 输出头（与 embedding 共享权重，weight tying）

相对第四周只有两处改动：
    1) forward 多一个 pad_mask 参数。
       第四周是固定窗口 + drop_last，一个 batch 里每条长度完全一样，用不着 padding。
       本周样本是「问：…答：…」一问一答，长度天然参差，必须 pad 到同长；
       而 pad 出来的位置不能让模型看见（否则它会学「补空格」），所以要传给
       TransformerEncoder 的 src_key_padding_mask（True = 该位置忽略）。
    2) max_len 支持从外部传（第四周写死 = seq_len = 64，本周要扩到 128）。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerLM(nn.Module):
    """和第四周完全同构的字符级自回归语言模型。"""

    def __init__(self, vocab_size, d_model=192, n_head=4, n_layer=4, max_len=128,
                 dropout=0.1, use_pos=True, causal=True):
        super().__init__()
        self.causal = causal
        self.use_pos = use_pos
        self.vocab_size = vocab_size
        self.max_len = max_len

        # 输入侧：字 embedding + 位置 embedding
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model) if use_pos else None
        self.drop = nn.Dropout(dropout)

        # 序列建模：直接用 torch 自带的 Decoder-only 式 EncoderLayer（因果 mask）
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_head, dim_feedforward=4 * d_model,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=n_layer, norm=nn.LayerNorm(d_model),
            enable_nested_tensor=False,
        )

        # 输出头：不加 bias，才能和输入 embedding 共享同一份权重（weight tying）
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok.weight

        # GPT 常规初始化：小方差，避免共享权重下初始 logits 过大
        nn.init.normal_(self.tok.weight, std=0.02)
        if self.pos is not None:
            nn.init.normal_(self.pos.weight, std=0.02)

    def forward(self, idx, pad_mask=None):
        """
        idx      : (B, T) 字符 id
        pad_mask : (B, T) bool，True 表示「这是 padding，别管它」；None 表示没有 padding
        返回      : (B, T, V) 每个位置对下一个字的 logits
        """
        B, T = idx.shape
        x = self.tok(idx)
        if self.use_pos:
            x = x + self.pos(torch.arange(T, device=idx.device))
        x = self.drop(x)

        # 因果掩码：位置 t 只能看 0..t。生成时下一字还不存在，这是硬约束。
        # 用 bool 掩码（True = 不许看），和 src_key_padding_mask 的类型保持一致；
        # 一个 float(-inf) 一个 bool 混着传，PyTorch 会报「mask 类型不一致」的弃用警告。
        mask = None
        if self.causal:
            mask = torch.triu(torch.ones(T, T, device=idx.device, dtype=torch.bool),
                              diagonal=1)

        # 同时传 attn_mask（因果）和 src_key_padding_mask（pad），两者会被合并
        x = self.encoder(x, mask=mask, src_key_padding_mask=pad_mask)
        return self.head(x)


@torch.no_grad()
def generate(model, vocab, prompt, max_new_tokens=40, temperature=0.0, top_k=0,
             ban_newline=True, device="cpu"):
    """
    自回归逐字生成（推理时必须逐字来，训练时才是整段并行）。

    temperature <= 0 用贪心，结果可复现，适合做评估对照；
    temperature > 0 配合 top_k 做采样，适合看「像不像人话」。
    """
    model.eval()
    ids = vocab.encode(prompt)
    if not ids:
        return ""
    out = list(ids)

    for _ in range(max_new_tokens):
        x = torch.tensor([out[-model.max_len:]], dtype=torch.long, device=device)
        logits = model(x)[0, -1].float()

        if temperature <= 0.01:
            nxt = int(logits.argmax())
        else:
            logits = logits / temperature
            if ban_newline and "\n" in vocab.char2idx:
                logits[vocab.char2idx["\n"]] = float("-inf")
            if top_k and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.numel()))
                logits[logits < v[-1]] = float("-inf")
            nxt = int(torch.multinomial(F.softmax(logits, dim=-1), 1))

        if nxt == vocab.eos_id:      # 模型自己说「说完了」
            break
        out.append(nxt)

    return vocab.decode(out[len(ids):])


def load_model(ckpt_path, device, dropout=0.0):
    """
    加载一份 SFT 检查点（base_sft.pt 或 sft.pt），按 checkpoint 里存的 args 原样重建模型。

    dropout：推理传 0.0；继续训练传 0.1（和第四周训练时一致，保留正则化）。
    返回 (model, vocab, args)
    """
    from sft_data import Vocab   # 放函数内 import，避免模块级循环依赖

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    a = ck["args"]
    model = TransformerLM(
        vocab_size=len(ck["char2idx"]),
        d_model=a["d_model"], n_head=a["heads"], n_layer=a["layers"],
        max_len=a["max_len"], dropout=dropout,
        use_pos=a.get("use_pos", True), causal=a.get("causal", True),
    )
    model.load_state_dict(ck["model_state"])
    model.to(device).eval()
    return model, Vocab.from_ckpt(ck), a
