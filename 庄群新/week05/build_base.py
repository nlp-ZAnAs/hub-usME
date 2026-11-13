# -*- coding: utf-8 -*-
"""
build_base.py — 把第四周的 tf_mask.pt 扩成 SFT 可直接用的 base_sft.pt

为什么必须有这一步？两个硬约束，都是拿你的 checkpoint 实测出来的：

  1) 字表缺 5.28% 的字符
     SFT 语料里有 120 个字符不在第四周的 4000 字表内：空格（8700 次最多）、
     = + [ ] % ℃ ° × ÷ ……
     这些恰恰是答案里的关键符号（「= 564」「25℃」），直接丢掉等于把答案弄残。

  2) 位置编码只有 64 位，装不下一问一答
     实测样本长度：p50=65、p90=92、max=118。
     max_len=64 会截掉 50.7% 的样本（一半的答案尾巴被砍），128 才能 100% 覆盖。

做法：**不重新训练，只「扩表 + 拷旧权重」**
  · token embedding：4000 → 4123 行，前 4000 行原样拷，新增行留随机初始化
  · 位置编码：64 → 128 行，前 64 行原样拷，后 64 行随机初始化
  · 注意力 / FFN / LayerNorm：全部原样拷
  这样第四周学到的字符级语言能力一点没丢，同时又装得下新符号和更长的序列。
"""

import argparse
import os
import torch

from model import TransformerLM, generate
from sft_data import Vocab, extend_vocab, load_jsonl

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BASE_CKPT = os.path.abspath(os.path.join(HERE, "..", "week04", "tf_mask.pt"))
DEFAULT_DATA_DIR = r"D:\BaiduNetdiskDownload\week5-大模型初探"


def build_base(base_ckpt=DEFAULT_BASE_CKPT, data_dir=DEFAULT_DATA_DIR,
               max_len=128, verbose=True):
    """
    返回 (model, vocab, args)，并把检查点写到 week05/base_sft.pt。
    """
    ck = torch.load(base_ckpt, map_location="cpu", weights_only=False)
    base_args = ck.get("args", {})

    # 第四周的模型配置，原样继承（改了就跟旧权重对不上）
    d_model = base_args.get("d_model", 192)
    n_head = base_args.get("heads", 4)
    n_layer = base_args.get("layers", 4)
    base_max_len = base_args.get("seq_len", 64)
    use_pos = not base_args.get("no_pos", False)
    causal = not base_args.get("no_mask", False)

    # ── 1) 收集 SFT 语料的全部字符，据此扩字表 ──
    extra_text = []
    for fn in ("sft_train.jsonl", "sft_val.jsonl"):
        for r in load_jsonl(os.path.join(data_dir, fn)):
            extra_text.append(r["instruction"])
            extra_text.append(r["output"])

    char2idx, idx2char, pad_id, eos_id, unk_id = extend_vocab(
        ck["char2idx"], "".join(extra_text))
    vocab = Vocab(char2idx, idx2char, pad_id, eos_id, unk_id)

    # ── 2) 按新尺寸搭一个空模型，再把旧权重一块块拷进去 ──
    model = TransformerLM(
        vocab_size=len(vocab), d_model=d_model, n_head=n_head, n_layer=n_layer,
        max_len=max_len, dropout=0.0, use_pos=use_pos, causal=causal,
    )

    old = ck["model_state"]
    new = model.state_dict()          # state_dict 返回的是参数引用，改它就是改模型

    # embedding：只拷旧行数（前 n_old_vocab 行），多出来的新符号行保持 std=0.02 随机值
    n_old_vocab = old["tok.weight"].shape[0]
    new["tok.weight"][:n_old_vocab].copy_(old["tok.weight"])
    # 输出头与输入 embedding 共享权重，拷完 tok 就自动带上了，不用单独处理 head.weight

    # 位置编码：只拷前 base_max_len 行，新增位置随机初始化
    if use_pos and "pos.weight" in old:
        n_old_pos = old["pos.weight"].shape[0]
        new["pos.weight"][:n_old_pos].copy_(old["pos.weight"])

    # 其余所有层（注意力、FFN、LayerNorm）必须逐键、逐形状对上
    for k, v in old.items():
        if k in ("tok.weight", "head.weight", "pos.weight"):
            continue
        assert k in new, f"旧权重里有新模型没有的键：{k}"
        assert new[k].shape == v.shape, f"{k} 形状不匹配：{tuple(new[k].shape)} vs {tuple(v.shape)}"
        new[k].copy_(v)

    model.eval()

    out_args = {
        "d_model": d_model, "heads": n_head, "layers": n_layer,
        "max_len": max_len, "use_pos": use_pos, "causal": causal,
        "base_ckpt": os.path.basename(base_ckpt),
    }
    out_path = os.path.join(HERE, "base_sft.pt")
    ck_out = dict(vocab.to_ckpt_fields())
    ck_out["model_state"] = model.state_dict()
    ck_out["args"] = out_args
    torch.save(ck_out, out_path)

    if verbose:
        n_par = sum(p.numel() for p in model.parameters())
        print(f"[build_base] 第四周字表 {len(ck['char2idx'])} → {len(vocab)}")
        print(f"[build_base] 位置编码 {base_max_len} → {max_len}")
        print(f"[build_base] 参数量 {n_par/1e6:.2f}M   pad={pad_id} eos={eos_id} unk={unk_id}")
        print(f"[build_base] 已保存 {out_path}")

        # 体检：拿原句喂它，确认扩表没把第四周的能力弄坏
        # 这时候它还没做 SFT，所以应该「接着列问题」—— 这正是我们要的 base 行为
        for q in ("问：今天天气怎么样\n答：", "问：123 和 456 哪个大\n答："):
            print(f"[build_base] {q!r} → {generate(model, vocab, q, 25, device='cpu')!r}")

    return model, vocab, out_args


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_ckpt", default=DEFAULT_BASE_CKPT, help="第四周的 tf_mask.pt")
    ap.add_argument("--data_dir", default=DEFAULT_DATA_DIR, help="sft_*.jsonl 所在目录")
    ap.add_argument("--max_len", type=int, default=128)
    args = ap.parse_args()
    build_base(args.base_ckpt, args.data_dir, args.max_len)


if __name__ == "__main__":
    main()
