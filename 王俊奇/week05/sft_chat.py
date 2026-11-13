"""
基于 SFT 权重的交互式问答入口：输入一个问题，模型按训练模板生成回答。
用法:
    python sft_chat.py
    python sft_chat.py --model_path best_model_sft.pt --top_k 20 --gen_len 40
"""

import argparse
import torch

from language_model import TransformerLM, generate_answer, normalize


def load_model(model_path, device):
    """按 checkpoint 里的结构超参构建模型并加载权重，推理时关闭 dropout。"""
    cfg = torch.load(model_path, map_location=device, weights_only=False)
    pa = cfg["args"]
    model = TransformerLM(
        vocab_size=len(cfg["char2idx"]),
        embed_dim=pa["embed_dim"],
        num_layers=pa["num_layers"],
        nhead=pa["nhead"],
        dim_feedforward=pa["dim_feedforward"],
        dropout=0.0,
        max_len=pa["max_len"],
    ).to(device)
    model.load_state_dict(cfg["model_state"])
    model.eval()
    return model, cfg["char2idx"], cfg["idx2char"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="best_model_sft.pt")
    parser.add_argument("--top_k", type=int, default=20, help="top-k 采样范围，1=贪心")
    parser.add_argument("--gen_len", type=int, default=40, help="回答最大字符数")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, char2idx, idx2char = load_model(args.model_path, device)
    print(f"已加载 {args.model_path}  device: {device}  词表: {len(char2idx)}")
    print("输入问题回车即可提问，q = 退出；空格会自动删除，其他字表外字符会提示重输。")

    while True:
        raw = input("\n问题> ").strip()
        if raw == "q":
            break
        if not raw:
            continue

        question = normalize(raw)
        invalid = sorted({c for c in question if c not in char2idx})
        if invalid:
            # 用转义形式输出，避免 Windows 控制台无法编码生僻字
            shown = [c.encode("unicode_escape").decode("ascii") for c in invalid]
            print(f"以下字符不在模型字表内，请改用中文表述：{shown}")
            continue

        answer = generate_answer(model, char2idx, idx2char, question, args.gen_len, device, args.top_k)
        print(f"回答> {answer}")


if __name__ == "__main__":
    main()
