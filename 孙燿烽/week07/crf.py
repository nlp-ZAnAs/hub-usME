"""线性链 CRF：前向算法求 logZ、负对数似然、维特比解码；可选解码期 BIO 硬约束。

mask 必须是前缀形式（每行先全 1 后全 0），且第一列全为 1。
约束只作用于解码：训练仍是标准 CRF，比较「同一个 checkpoint 只换解码」的效果。
"""

import torch
import torch.nn as nn


def constraint_scores(allowed_start, allowed, device):
    """布尔约束矩阵 → 加性分数（合法 0，非法 -inf）。"""
    start = torch.zeros(len(allowed_start), device=device)
    start[~torch.as_tensor(allowed_start, device=device)] = float("-inf")
    trans = torch.zeros(allowed.shape, device=device)
    trans[~torch.as_tensor(allowed, device=device)] = float("-inf")
    return start, trans


def viterbi(emissions, mask, trans, start, end):
    """emissions [B,L,C]；返回每条序列（按 mask 长度）的最优标签路径。"""
    batch, length, _ = emissions.shape
    score = start + emissions[:, 0]
    history = []
    for i in range(1, length):
        best, index = (score.unsqueeze(2) + trans.unsqueeze(0)).max(dim=1)
        score = torch.where(mask[:, i, None], best + emissions[:, i], score)
        history.append(index)
    score = score + end
    lengths = mask.sum(1).tolist()
    last = score.argmax(1).tolist()
    history = [h.cpu() for h in history]
    paths = []
    for b in range(batch):
        tag = last[b]
        path = [tag]
        for i in range(lengths[b] - 1, 0, -1):
            tag = int(history[i - 1][b, tag])
            path.append(tag)
        paths.append(path[::-1])
    return paths


class CRF(nn.Module):
    def __init__(self, num_tags):
        super().__init__()
        self.start = nn.Parameter(torch.empty(num_tags).uniform_(-0.1, 0.1))
        self.end = nn.Parameter(torch.empty(num_tags).uniform_(-0.1, 0.1))
        self.trans = nn.Parameter(torch.empty(num_tags, num_tags).uniform_(-0.1, 0.1))

    @staticmethod
    def _check(mask):
        if not bool(mask[:, 0].all()):
            raise ValueError("CRF mask 的第一列必须全为 1")

    def log_partition(self, emissions, mask):
        self._check(mask)
        alpha = self.start + emissions[:, 0]
        for i in range(1, emissions.shape[1]):
            step = torch.logsumexp(alpha.unsqueeze(2) + self.trans.unsqueeze(0)
                                   + emissions[:, i].unsqueeze(1), dim=1)
            alpha = torch.where(mask[:, i, None], step, alpha)
        return torch.logsumexp(alpha + self.end, dim=1)

    def path_score(self, emissions, tags, mask):
        self._check(mask)
        batch = torch.arange(emissions.shape[0], device=emissions.device)
        score = self.start[tags[:, 0]] + emissions[batch, 0, tags[:, 0]]
        for i in range(1, emissions.shape[1]):
            step = self.trans[tags[:, i - 1], tags[:, i]] + emissions[batch, i, tags[:, i]]
            score = score + step * mask[:, i]
        last = tags[batch, mask.sum(1) - 1]
        return score + self.end[last]

    def nll(self, emissions, tags, mask):
        """按 batch 平均的负对数似然；emissions 应为 float32。"""
        return (self.log_partition(emissions, mask) - self.path_score(emissions, tags, mask)).mean()

    @torch.no_grad()
    def decode(self, emissions, mask, constraints=None):
        self._check(mask)
        start, trans = self.start, self.trans
        if constraints is not None:
            extra_start, extra_trans = constraints
            start, trans = start + extra_start, trans + extra_trans
        return viterbi(emissions, mask, trans, start, self.end)


@torch.no_grad()
def constrained_argmax(emissions, mask, constraints):
    """无 CRF 时的约束解码：在 log-softmax 分数上只走合法转移的维特比路径。"""
    log_probs = torch.log_softmax(emissions.float(), dim=-1)
    start, trans = constraints
    end = torch.zeros(emissions.shape[-1], device=emissions.device)
    return viterbi(log_probs, mask, trans, start, end)
