"""字符级因果 Transformer，复用 week04 的网络结构和训练参数。"""
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

SPECIAL_TOKENS = ("<pad>", "<unk>", "<bos>", "<sep>", "<eos>")
ROOT = Path(__file__).resolve().parent
INITIAL_CHECKPOINT = ROOT.parent / "week04/best_model_transformer_matched.pt"

class CausalBlock(nn.Module):
    """多头因果自注意力 + 前馈网络，两个子层均使用预归一化和残差连接。"""
    def __init__(self, width, heads, ff_dim, dropout):
        super().__init__()
        self.heads = heads
        self.attn_norm = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.proj = nn.Linear(width, width)
        self.dropout = nn.Dropout(dropout)
        self.ff = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, ff_dim), nn.GELU(),
            nn.Linear(ff_dim, width), nn.Dropout(dropout),
        )

    def forward(self, x):
        batch, length, width = x.shape
        qkv = self.qkv(self.attn_norm(x)).reshape(
            batch, length, 3, self.heads, width // self.heads
        )
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attended = attended.transpose(1, 2).reshape(batch, length, width)
        x = x + self.dropout(self.proj(attended))
        return x + self.ff(x)


class TransformerLM(nn.Module):
    def __init__(self, vocab_size, cfg):
        super().__init__()
        width, heads = cfg["embed_dim"], cfg["heads"]
        if heads < 1 or width % heads:
            raise ValueError("embed_dim 必须是 heads 的整数倍")
        self.max_len = cfg["seq_len"]
        self.embed = nn.Embedding(vocab_size, width)
        self.position = nn.Embedding(self.max_len, width)
        self.drop = nn.Dropout(cfg["dropout"])
        self.blocks = nn.Sequential(*[
            CausalBlock(width, heads, cfg["hidden_dim"], cfg["dropout"])
            for _ in range(cfg["num_layers"])
        ])
        self.norm = nn.LayerNorm(width)
        # 字表按频次排序。自适应 softmax 对常见字直接分类，对低频字分组，
        # 可在 CPU 上计算真实的归一化概率，而无需在训练时展开全部词表。
        cutoffs = [n for n in (256, 1024) if n < vocab_size]
        if cutoffs:
            self.output = nn.AdaptiveLogSoftmaxWithLoss(
                width, vocab_size, cutoffs, div_value=2, head_bias=True
            )
        else:
            self.output = nn.Linear(width, vocab_size)

    def features(self, x):
        if x.size(1) > self.max_len:
            raise ValueError("输入长度超过模型上下文窗口")
        pos = torch.arange(x.size(1), device=x.device)
        return self.norm(self.blocks(self.drop(self.embed(x) + self.position(pos))))

    def token_loss(self, x, y):
        h = self.features(x).flatten(0, 1)
        targets = y.flatten()
        valid = targets != -100
        h, targets = h[valid], targets[valid]
        if targets.numel() == 0:
            raise ValueError("当前批次没有有效训练目标")
        if isinstance(self.output, nn.AdaptiveLogSoftmaxWithLoss):
            return self.output(h, targets).loss
        return F.cross_entropy(self.output(h), targets)

    def forward(self, x):
        h = self.features(x)
        if isinstance(self.output, nn.AdaptiveLogSoftmaxWithLoss):
            return self.output.log_prob(h.flatten(0, 1)).reshape(*x.shape, -1)
        return self.output(h)



class Tokenizer:
    def __init__(self, vocab):
        self.vocab = vocab
        self.inverse = {index: char for char, index in vocab.items()}
        for token in SPECIAL_TOKENS:
            if token not in vocab:
                raise ValueError(f"字表缺少特殊 token {token}")
        self.pad, self.unk, self.bos, self.sep, self.eos = [
            vocab[token] for token in SPECIAL_TOKENS
        ]

    def encode(self, text):
        return [self.vocab.get(char, self.unk) for char in text]

    def decode(self, ids):
        return "".join(self.inverse[index] if index != self.unk else "�"
                       for index in ids if index not in (self.pad, self.bos, self.sep, self.eos))


def load_model(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = TransformerLM(len(checkpoint["vocab"]), checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    return model.eval(), Tokenizer(checkpoint["vocab"]), checkpoint


def initialize_from_previous(path, seq_len=256):
    """扩充特殊 token 和位置表，其余参数逐项复用旧模型。"""
    old = torch.load(path, map_location="cpu", weights_only=True)
    vocab = dict(old["char2idx"])
    if old["args"]["model"] != "transformer":
        raise ValueError("初始化权重必须为 Transformer")
    for token in SPECIAL_TOKENS:
        vocab[token] = len(vocab)
    config = {key: old["args"][key] for key in
              ("embed_dim", "hidden_dim", "heads", "num_layers", "dropout")}
    config["seq_len"] = seq_len
    config["dropout"] = 0.1
    model = TransformerLM(len(vocab), config)
    state = model.state_dict()
    for name, value in old["model_state"].items():
        target = state[name]
        if target.shape == value.shape:
            target.copy_(value)
        elif name == "position.weight":
            for start in range(0, len(target), len(value)):
                count = min(len(value), len(target) - start)
                target[start:start + count].copy_(value[:count])
        elif name in ("embed.weight", "output.tail.1.1.weight"):
            target[:len(value)].copy_(value)
        else:
            raise ValueError(f"无法迁移参数 {name}: {value.shape} -> {target.shape}")
    model.load_state_dict(state)
    return model, Tokenizer(vocab), config


def answer_example(row, tokenizer, seq_len):
    prompt = [tokenizer.bos] + tokenizer.encode(row["question"]) + [tokenizer.sep]
    answer = tokenizer.encode(row["answer"]) + [tokenizer.eos]
    sequence = prompt + answer
    if len(sequence) - 1 > seq_len:
        raise ValueError("问答样本超出上下文长度，请重新准备数据")
    # y[t] 对应 sequence[t+1]，从第一个答案字符开始参与损失。
    x = torch.tensor(sequence[:-1], dtype=torch.long)
    y = torch.tensor([-100] * (len(prompt) - 1) + answer, dtype=torch.long)
    return x, y


def collate_examples(batch):
    xs, ys = zip(*batch)
    # 右侧填充不会影响有效位置的因果注意力。填充目标统一忽略。
    return (nn.utils.rnn.pad_sequence(xs, batch_first=True, padding_value=0),
            nn.utils.rnn.pad_sequence(ys, batch_first=True, padding_value=-100))
