# -*- coding: utf-8 -*-
"""
ablation_bert_vs_roberta.py —— BERT / RoBERTa 基座消融实验 + Macro-F1 可视化

放在 code/bert/ 目录下运行（和 train_bert.py 同级）：

    # 一键全跑（14 个 config，约 1.5 小时）
    python bert/ablation_bert_vs_roberta.py

    # 冒烟测试（每 config 只跑 20 步，2 分钟验证流程通不通）
    python bert/ablation_bert_vs_roberta.py --smoke

    # 只跑 BERT 这一个基座
    python bert/ablation_bert_vs_roberta.py --backbones bert

    # 想更严谨：每 config 跑 3 个种子（耗时 ×3）
    python bert/ablation_bert_vs_roberta.py --seeds 42,43,44

    # 只重新出图（用上次存的结果，不重新训练）
    python bert/ablation_bert_vs_roberta.py --plot_only

输出：
    outputs/ablation/results.json      每个 config 的逐类 F1 与汇总
    outputs/ablation/results.csv       扁平表格，方便贴 Excel
    outputs/ablation/ablation_f1.png   三联图（池化 / 技巧 / 热力图）

设计要点（为什么这么写）：
  1. 【绕开 bug】common/model_paths.py 里 roberta_path() 误写成了 resolve(BERT_DIR)，
     这里直接 resolve(ROBERTA_DIR)，不调用那个函数。
  2. 【固定种子】每个 config、每个种子都重设 random / numpy / torch 以及 cudnn，
     否则 ±0.5% 的「提升」全是噪声。
  3. 【两阶段消融】
       阶段一（pool）：3 种池化，固定 CE + 类别加权  → 选出最佳池化
       阶段二（tech）：在【最佳池化】上对比 5 种技巧
     这样技巧的对比不受池化选择干扰。
  4. 【可续跑】每跑完一个 config 就写盘，中途崩了不丢结果；加 --plot_only 直接出图。
"""

import argparse
import json
import math
import random
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

CODE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE_ROOT))

from common.data import load_splits                       # noqa: E402
from common.labels import LABELS                          # noqa: E402
from common.model_paths import resolve, BERT_DIR, ROBERTA_DIR   # noqa: E402

OUT_DIR = CODE_ROOT / "outputs" / "ablation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# PyTorch 的 LR 调度器在「GradScaler 调 opt.step()」时会误报
# "call lr_scheduler.step() before optimizer.step()"。
# 实测确认：实际施加的 LR 序列与 schedule 完全一致（0.333 -> 0.667 -> 1.0 …），
# 属于启发式检查的假阳性，这里精确屏蔽这一条，不影响其它警告。
warnings.filterwarnings(
    "ignore",
    message=r"Detected call of `lr_scheduler\.step\(\)` before `optimizer\.step\(\)`",
)


# ─────────────────────────── 基座路径 ───────────────────────────
# 注意：不要用 model_paths.roberta_path()，那个函数有 bug（返回的是 BERT 目录）
BACKBONE_DIRS = {
    "bert":    BERT_DIR,        # bert-base-chinese
    "roberta": ROBERTA_DIR,     # iic--nlp_roberta_backbone_base_std
}


def resolve_backbone(name):
    """返回 (Path, 真实 model_type)，顺便打印出来让人确认没走错目录。"""
    from transformers import AutoConfig
    p = resolve(BACKBONE_DIRS[name])
    cfg = AutoConfig.from_pretrained(str(p))
    return p, cfg


# ─────────────────────────── 可复现性 ───────────────────────────
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # 让 cudnn 的降级更稳（deterministic 下部分算子会被禁用）
    torch.backends.cudnn.enabled = True


# ─────────────────────────── 数据 ───────────────────────────
class TnewsDataset(Dataset):
    """把 {"sentence","label"} 变成模型输入张量。"""

    def __init__(self, data, tok, max_length=64):
        self.data, self.tok, self.max_length = data, tok, max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        d = self.data[i]
        enc = self.tok(d["sentence"], truncation=True, max_length=self.max_length,
                       padding="max_length", return_tensors="pt")
        # 只取 input_ids / attention_mask —— RoBERTa 系不需要 token_type_ids
        return {"input_ids": enc["input_ids"][0],
                "attention_mask": enc["attention_mask"][0],
                "labels": torch.tensor(d["label"])}


# ─────────────────────────── 模型 ───────────────────────────
class BertClassifier(nn.Module):
    """预训练编码器 + 池化 + 线性头。pool ∈ {cls, mean, max}。"""

    def __init__(self, model_dir, num_labels, pool="cls", dropout=0.1):
        super().__init__()
        from transformers import AutoModel
        self.encoder = AutoModel.from_pretrained(str(model_dir))
        self.pool = pool
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_labels)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask,
                           return_dict=True)
        hs = out.last_hidden_state                      # [B, L, H]
        if self.pool == "cls":
            vec = hs[:, 0]
        else:
            m = attention_mask.unsqueeze(-1).float()    # [B, L, 1]
            if self.pool == "mean":
                vec = (hs * m).sum(1) / m.sum(1).clamp(min=1e-9)
            else:                                       # max：PAD 压到 -1e9
                vec = hs.masked_fill(m == 0, -1e9).max(1).values
        return self.classifier(self.dropout(vec))


class FGM:
    """Fast Gradient Method 对抗训练（只扰动 word_embeddings）。"""

    def __init__(self, model, emb_name="word_embeddings", eps=1.0):
        self.model, self.emb_name, self.eps = model, emb_name, eps
        self.backup = {}

    def attack(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name and param.grad is not None:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    param.data.add_(self.eps * param.grad / norm)

    def restore(self):
        for name in self.backup:
            self.model.get_parameter(name).data = self.backup[name]
        self.backup = {}


def focal_loss(logits, targets, weight=None, gamma=2.0):
    """L = -alpha_t * (1 - p_t)^gamma * log p_t"""
    logp = F.log_softmax(logits, dim=-1)
    logp_t = logp.gather(1, targets.unsqueeze(1)).squeeze(1)
    p_t = logp_t.exp()
    alpha = weight[targets] if weight is not None else 1.0
    return (-alpha * ((1.0 - p_t) ** gamma) * logp_t).mean()


# ─────────────────────────── 评估 ───────────────────────────
@torch.no_grad()
def evaluate(model, loader, device, num_classes):
    from sklearn.metrics import accuracy_score, f1_score
    model.eval()
    ys, ps = [], []
    for batch in loader:
        logits = model(batch["input_ids"].to(device),
                       batch["attention_mask"].to(device))
        ps.extend(logits.argmax(-1).cpu().tolist())
        ys.extend(batch["labels"].tolist())
    per_class = f1_score(ys, ps, average=None, labels=list(range(num_classes)),
                         zero_division=0)
    return {
        "acc": float(accuracy_score(ys, ps)),
        "macro_f1": float(f1_score(ys, ps, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(ys, ps, average="weighted", zero_division=0)),
        "per_class_f1": [round(float(v), 4) for v in per_class],
        "n_eval": len(ys),
    }


# ─────────────────────────── 单个 config 训练 ───────────────────────────
def run_one(cfg, backbone, seed, args, tok, train, val, device):
    """训练一个 (backbone, cfg, seed) 组合，返回该次的最佳 macro-F1 与逐类 F1。"""
    set_seed(seed)

    model_dir, _ = resolve_backbone(backbone)
    train_dl = DataLoader(TnewsDataset(train, tok, args.max_length),
                          batch_size=args.batch_size, shuffle=True)
    val_dl = DataLoader(TnewsDataset(val, tok, args.max_length),
                        batch_size=args.batch_size)

    model = BertClassifier(model_dir, len(LABELS), pool=cfg["pool"]).to(device)

    # 分层学习率：预训练层小步，随机初始化的分类头大步
    head_ids = {id(p) for p in model.classifier.parameters()}
    head_params = [p for p in model.parameters() if id(p) in head_ids]
    enc_params = [p for p in model.parameters() if id(p) not in head_ids]
    opt = torch.optim.AdamW([
        {"params": enc_params,  "lr": args.lr},
        {"params": head_params, "lr": args.lr * args.head_lr_mult},
    ], weight_decay=0.01)

    # 类别加权（23.2x 不均衡，不加权时尾部类 recall 会趋近 0）
    if cfg["class_weight"]:
        from sklearn.utils.class_weight import compute_class_weight
        cw = compute_class_weight("balanced", classes=np.arange(len(LABELS)),
                                  y=[d["label"] for d in train])
        weight = torch.tensor(cw, dtype=torch.float).to(device)
    else:
        weight = None

    # 损失函数（Focal 与 Label Smoothing 不叠加，Focal 优先）
    if cfg["focal"] > 0:
        crit = lambda lg, y: focal_loss(lg, y, weight, cfg["focal"])   # noqa: E731
        crit_desc = f"Focal(gamma={cfg['focal']})"
    else:
        crit = nn.CrossEntropyLoss(weight=weight, label_smoothing=cfg["label_smoothing"])
        crit_desc = "CE" + (f"+LS({cfg['label_smoothing']})" if cfg["label_smoothing"] > 0 else "")

    steps_per_epoch = min(len(train_dl), args.max_steps) if args.max_steps else len(train_dl)
    total_steps = steps_per_epoch * args.epochs
    warmup = max(1, int(total_steps * 0.1))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s / warmup) if s < warmup
        else max(0.0, (total_steps - s) / max(1, total_steps - warmup)))
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
    fgm = FGM(model, eps=args.fgm_eps) if cfg["fgm"] else None

    best_f1, best_detail, best_epoch = -1.0, None, 0
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running, n_step = 0.0, 0
        for i, batch in enumerate(train_dl):
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            y = batch["labels"].to(device)

            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=(device == "cuda")):
                loss = crit(model(ids, mask), y)
            scaler.scale(loss).backward()

            if fgm is not None:
                fgm.attack()
                with torch.autocast("cuda", dtype=torch.float16, enabled=(device == "cuda")):
                    loss_adv = crit(model(ids, mask), y)
                scaler.scale(loss_adv).backward()
                fgm.restore()

            scaler.step(opt)
            scaler.update()
            sched.step()
            running += loss.item()
            n_step += 1
            if args.max_steps and n_step >= args.max_steps:
                break

        res = evaluate(model, val_dl, device, len(LABELS))
        if res["macro_f1"] > best_f1:
            best_f1, best_detail, best_epoch = res["macro_f1"], res, epoch

    del model, opt, scaler
    if device == "cuda":
        torch.cuda.empty_cache()

    return {
        "backbone": backbone,
        "config": cfg["name"],
        "group": cfg["group"],
        "seed": seed,
        "pool": cfg["pool"],
        "loss": crit_desc,
        "class_weight": cfg["class_weight"],
        "fgm": cfg["fgm"],
        "best_epoch": best_epoch,
        "best_macro_f1": round(best_f1, 4),
        "acc": round(best_detail["acc"], 4),
        "weighted_f1": round(best_detail["weighted_f1"], 4),
        "per_class_f1": best_detail["per_class_f1"],
        "train_seconds": round(time.time() - t0, 1),
        "smoke": bool(args.max_steps),
    }


# ─────────────────────────── 配置矩阵 ───────────────────────────
def build_pool_configs():
    """阶段一：三种池化，固定 CE + 类别加权。"""
    return [
        dict(name="pool=cls",  group="pool", pool="cls",  class_weight=True,
             label_smoothing=0.0, focal=0.0, fgm=False),
        dict(name="pool=mean", group="pool", pool="mean", class_weight=True,
             label_smoothing=0.0, focal=0.0, fgm=False),
        dict(name="pool=max",  group="pool", pool="max",  class_weight=True,
             label_smoothing=0.0, focal=0.0, fgm=False),
    ]


def build_tech_configs(best_pool):
    """阶段二：在最佳池化上对比 5 种技巧。"""
    base = dict(pool=best_pool, label_smoothing=0.0, focal=0.0, fgm=False)
    return [
        dict(name="baseline",          group="tech", class_weight=True,  **base),
        dict(name="no_class_weight",   group="tech", class_weight=False, **base),
        dict(name="label_smoothing",   group="tech", class_weight=True,
             **{**base, "label_smoothing": 0.1}),
        dict(name="focal_gamma=2",     group="tech", class_weight=True,
             **{**base, "focal": 2.0}),
        dict(name="fgm",               group="tech", class_weight=True,
             **{**base, "fgm": True}),
    ]


# ─────────────────────────── 可视化 ───────────────────────────
def plot_results(rows, out_png, smoke=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    # 聚合成 (backbone, config) -> mean/std
    from collections import defaultdict
    agg = defaultdict(list)
    for r in rows:
        agg[(r["backbone"], r["config"])].append(r["best_macro_f1"])
    stat = {k: (float(np.mean(v)), float(np.std(v)), len(v)) for k, v in agg.items()}

    backbones = sorted({k[0] for k in stat})
    colors = {"bert": "#4C72B0", "roberta": "#DD8452"}

    # 保证 config 顺序稳定
    order = []
    for nm in ["pool=cls", "pool=mean", "pool=max",
               "baseline", "no_class_weight", "label_smoothing",
               "focal_gamma=2", "fgm"]:
        if (backbones[0], nm) in stat or (backbones[-1], nm) in stat:
            order.append(nm)
    pool_cfgs = [c for c in order if c.startswith("pool=")]
    tech_cfgs = [c for c in order if not c.startswith("pool=")]

    fig = plt.figure(figsize=(16, 5.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.4, 1.3], wspace=0.30,
                          top=0.80, bottom=0.20, left=0.05, right=0.985)

    def grouped_bar(ax, cfgs, title, baseline_cfg=None):
        x = np.arange(len(cfgs))
        w = 0.8 / max(1, len(backbones))
        for bi, bb in enumerate(backbones):
            means = [stat.get((bb, c), (float("nan"),) * 3)[0] for c in cfgs]
            stds = [stat.get((bb, c), (0, 0, 0))[1] for c in cfgs]
            pos = x + (bi - (len(backbones) - 1) / 2) * w
            ax.bar(pos, means, width=w * 0.92, yerr=stds, capsize=3,
                   label=bb.upper(), color=colors.get(bb, None), alpha=0.9,
                   error_kw=dict(lw=1))
            for p, m in zip(pos, means):
                if not math.isnan(m):
                    ax.text(p, m + 0.004, f"{m:.3f}", ha="center", va="bottom",
                            fontsize=7.5, rotation=90)
        if baseline_cfg and (backbones[0], baseline_cfg) in stat:
            b = stat[(backbones[0], baseline_cfg)][0]
            ax.axhline(b, ls=":", lw=1, color="gray",
                       label=f"baseline {b:.3f}" if not smoke else None)
        ax.set_xticks(x)
        ax.set_xticklabels(cfgs, rotation=18, ha="right", fontsize=9)
        ax.set_ylabel("best val Macro-F1")
        ax.set_title(title, fontsize=11)
        ax.grid(axis="y", alpha=0.25)
        lo = min(m for m, _, _ in stat.values() if not math.isnan(m))
        hi = max(m for m, _, _ in stat.values() if not math.isnan(m))
        pad = max(0.01, (hi - lo) * 0.35)
        ax.set_ylim(max(0, lo - pad), hi + pad * 1.6)
        ax.legend(fontsize=8, loc="upper right")

    grouped_bar(fig.add_subplot(gs[0, 0]), pool_cfgs,
                "① 池化方式消融\n(CE + 类别加权)", baseline_cfg="pool=cls")
    grouped_bar(fig.add_subplot(gs[0, 1]), tech_cfgs,
                "② 训练技巧消融\n(在最佳池化上)", baseline_cfg="baseline")

    # 面板③：相对 baseline 的 ΔF1（最直接回答「提升多少」）
    ax3 = fig.add_subplot(gs[0, 2])
    deltas = {}
    for bb in backbones:
        base = stat.get((bb, "baseline"), (None,))[0]
        if base is None:
            continue
        deltas[bb] = [(c, stat[(bb, c)][0] - base)
                      for c in tech_cfgs if (bb, c) in stat and c != "baseline"]
    x = np.arange(len(tech_cfgs) - 1) if tech_cfgs and tech_cfgs[0] == "baseline" else np.arange(len(tech_cfgs))
    w = 0.8 / max(1, len(backbones))
    for bi, bb in enumerate(backbones):
        if bb not in deltas:
            continue
        labs = [c for c, _ in deltas[bb]]
        vals = [d for _, d in deltas[bb]]
        pos = np.arange(len(labs)) + (bi - (len(backbones) - 1) / 2) * w
        ax3.bar(pos, vals, width=w * 0.92, label=bb.upper(),
                color=colors.get(bb, None), alpha=0.9)
        for p, v in zip(pos, vals):
            ax3.text(p, v + (0.0008 if v >= 0 else -0.0008), f"{v:+.4f}",
                     ha="center", va="bottom" if v >= 0 else "top", fontsize=7.5)
    ax3.axhline(0, color="black", lw=1)
    ax3.set_xticks(np.arange(len(deltas.get(backbones[0], []))))
    ax3.set_xticklabels([c for c, _ in deltas.get(backbones[0], [])],
                        rotation=18, ha="right", fontsize=9)
    ax3.set_ylabel("Δ Macro-F1（相对 baseline）")
    ax3.set_title("③ 各技巧相对 baseline 的增益\n(负值 = 变差)", fontsize=11)
    ax3.grid(axis="y", alpha=0.25)
    ax3.legend(fontsize=8)

    title = "BERT vs RoBERTa 消融实验（%d 个 config × %d 基座）" % (len(order), len(backbones))
    if smoke:
        title += "   [冒烟模式：步数不足，数值无意义]"
    fig.suptitle(title, fontsize=12)
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    print(f"[plot] 已保存 {out_png}")
    return stat


# ─────────────────────────── 主流程 ───────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", default="bert,roberta", help="逗号分隔")
    ap.add_argument("--seeds", default="42", help="逗号分隔，严谨起见建议 42,43,44")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--head_lr_mult", type=float, default=5.0)
    ap.add_argument("--max_length", type=int, default=64)
    ap.add_argument("--fgm_eps", type=float, default=1.0)
    ap.add_argument("--max_steps", type=int, default=0, help="每 epoch 最多步数（冒烟用）")
    ap.add_argument("--max_eval", type=int, default=0, help="验证集最多条数（冒烟用）")
    ap.add_argument("--smoke", default="", action="store_true", help="等价于 --max_steps 20 --max_eval 500")
    ap.add_argument("--plot_only", default="true",action="store_true", help="只用已有 results.json 重画")
    ap.add_argument("--out_tag", default="", help="结果文件名后缀，便于分开多次实验")
    args = ap.parse_args()

    if args.smoke:
        args.max_steps, args.max_eval = 20, 500

    tag = args.out_tag or ("smoke" if args.smoke else "main")
    res_json = OUT_DIR / f"results{'_' + tag if tag != 'main' else ''}.json"
    res_csv = OUT_DIR / f"results{'_' + tag if tag != 'main' else ''}.csv"
    res_png = OUT_DIR / f"ablation_f1{'_' + tag if tag != 'main' else ''}.png"

    # ── 只出图 ──
    if args.plot_only:
        if not res_json.exists():
            raise SystemExit(f"找不到 {res_json}，先跑一次完整实验")
        rows = json.loads(res_json.read_text(encoding="utf-8"))
        plot_results(rows, res_png, smoke=any(r.get("smoke") for r in rows))
        return

    from transformers import AutoTokenizer, logging as hf_logging
    hf_logging.set_verbosity_error()          # 压掉 LOAD REPORT 的噪音

    device = "cuda" if torch.cuda.is_available() else "cpu"
    backbones = [b.strip() for b in args.backbones.split(",") if b.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    print("=" * 78)
    print("BERT vs RoBERTa 消融实验")
    print("=" * 78)
    print(f"  device   : {device}")
    print(f"  backbones: {backbones}")
    print(f"  seeds    : {seeds}")
    print(f"  epochs   : {args.epochs}   batch={args.batch_size}  lr={args.lr}  max_len={args.max_length}")
    if args.max_steps:
        print(f"  [冒烟] max_steps={args.max_steps}  max_eval={args.max_eval}")
    print()

    # ── 数据（只读一次）──
    train, val = load_splits()
    if args.max_eval:
        val = val[: args.max_eval]
    print(f"  训练 {len(train)} 条 / 验证 {len(val)} 条 / 类别 {len(LABELS)} 个")

    # ── 校验基座目录，避免又走错到同一个模型 ──
    print()
    print("  基座自检：")
    dirs = {}
    for bb in backbones:
        p, cfg = resolve_backbone(bb)
        dirs[bb] = p
        print(f"    {bb:<8} -> {p.name}")
        print(f"             model_type={cfg.model_type} hidden={cfg.hidden_size} "
              f"vocab={cfg.vocab_size}")
    if len({str(v) for v in dirs.values()}) < len(dirs):
        raise SystemExit(
            "\n[中止] 有两个基座解析到了同一个目录！\n"
            "  请检查 common/model_paths.py 里的 roberta_path() ——\n"
            "  它现在写的是 resolve(BERT_DIR)，应该是 resolve(ROBERTA_DIR)。\n"
        )
    print()

    results = []
    if res_json.exists():                     # 续跑：读回已完成的结果
        results = json.loads(res_json.read_text(encoding="utf-8"))
        done = {(r["backbone"], r["config"], r["seed"]) for r in results}
        print(f"  [续跑] 已有 {len(results)} 条结果，将跳过重复项\n")
    else:
        done = set()

    def save():
        res_json.write_text(json.dumps(results, ensure_ascii=False, indent=2),
                            encoding="utf-8")
        if results:
            import csv
            keys = ["backbone", "config", "group", "seed", "pool", "loss",
                    "class_weight", "fgm", "best_epoch", "best_macro_f1",
                    "acc", "weighted_f1", "train_seconds"]
            with open(res_csv, "w", encoding="utf-8-sig", newline="") as f:
                wr = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
                wr.writeheader()
                for r in results:
                    wr.writerow(r)

    t_start = time.time()
    for bb in backbones:
        tok = AutoTokenizer.from_pretrained(str(dirs[bb]))

        # ══ 阶段一：池化 ══
        print("─" * 78)
        print(f"[{bb.upper()}] 阶段一：池化方式消融")
        print("─" * 78)
        pool_rows = []
        for cfg in build_pool_configs():
            for seed in seeds:
                if (bb, cfg["name"], seed) in done:
                    print(f"  {cfg['name']:<18} seed={seed}  (已完成，跳过)")
                    continue
                r = run_one(cfg, bb, seed, args, tok, train, val, device)
                results.append(r)
                pool_rows.append(r)
                done.add((bb, cfg["name"], seed))
                save()
                print(f"  {cfg['name']:<18} seed={seed}  "
                      f"f1={r['best_macro_f1']:.4f}  acc={r['acc']:.4f}  "
                      f"({r['train_seconds']:.0f}s)")

        # 选最佳池化（用本次 + 历史数据一起算）
        pool_cfgs = [c["name"] for c in build_pool_configs()]
        best_pool, best_score = "cls", -1.0
        for pc in pool_cfgs:
            v = [r["best_macro_f1"] for r in results
                 if r["backbone"] == bb and r["config"] == pc]
            if v and np.mean(v) > best_score:
                best_pool, best_score = pc.replace("pool=", ""), float(np.mean(v))
        print(f"\n  → 最佳池化：{best_pool}（macro-F1 {best_score:.4f}）\n")

        # ══ 阶段二：技巧 ══
        print("─" * 78)
        print(f"[{bb.upper()}] 阶段二：训练技巧消融（pool={best_pool}）")
        print("─" * 78)
        for cfg in build_tech_configs(best_pool):
            for seed in seeds:
                # baseline 若与阶段一的 pool=cls/mean/max 完全等价，直接复用
                same_as_pool = (cfg["name"] == "baseline"
                                and f"pool={best_pool}" in pool_cfgs)
                if same_as_pool and (bb, f"pool={best_pool}", seed) in done:
                    reused = next((r for r in results
                                   if r["backbone"] == bb and r["seed"] == seed
                                   and r["config"] == f"pool={best_pool}"), None)
                    if reused is not None and not any(
                            r["backbone"] == bb and r["config"] == "baseline"
                            and r["seed"] == seed for r in results):
                        r2 = dict(reused)
                        r2["config"], r2["group"] = "baseline", "tech"
                        results.append(r2)
                        done.add((bb, "baseline", seed))
                        save()
                        print(f"  baseline           seed={seed}  "
                              f"f1={r2['best_macro_f1']:.4f}  (复用 pool={best_pool} 的结果)")
                    continue
                if (bb, cfg["name"], seed) in done:
                    print(f"  {cfg['name']:<18} seed={seed}  (已完成，跳过)")
                    continue
                r = run_one(cfg, bb, seed, args, tok, train, val, device)
                results.append(r)
                done.add((bb, cfg["name"], seed))
                save()
                extra = "  [FGM ×1.5]" if cfg["fgm"] else ""
                print(f"  {cfg['name']:<18} seed={seed}  "
                      f"f1={r['best_macro_f1']:.4f}  acc={r['acc']:.4f}  "
                      f"({r['train_seconds']:.0f}s){extra}")
        print()

    save()

    # ══ 汇总表 ══
    print("=" * 78)
    print("汇总（best val Macro-F1）")
    print("=" * 78)
    from collections import defaultdict
    agg = defaultdict(list)
    for r in results:
        agg[(r["backbone"], r["config"])].append(r["best_macro_f1"])
    cfgs = [c["name"] for c in build_pool_configs()] + \
           [c["name"] for c in build_tech_configs("x")]
    seen, order = set(), []
    for c in cfgs:
        if c not in seen:
            seen.add(c); order.append(c)
    hdr = "  %-30s" % "config" + "".join("%22s" % b.upper()
                                         for b in backbones) + "%16s" % "Δ(roberta-bert)"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for c in order:
        cells, means = [], {}
        for bb in backbones:
            v = agg.get((bb, c), [])
            if v:
                means[bb] = float(np.mean(v))
                cells.append("%.4f±%.4f(n=%d)" % (np.mean(v), np.std(v), len(v)))
            else:
                cells.append("-")
        delta = ""
        if len(means) == 2:
            delta = "%+.4f" % (means["roberta"] - means["bert"])
        print("  %-30s" % c + "".join("%22s" % x for x in cells) + "%16s" % delta)
    print()
    print(f"  总耗时 {(time.time() - t_start)/60:.1f} 分钟")
    print(f"  结果文件：{res_json}")
    print(f"           {res_csv}")

    plot_results(results, res_png, smoke=bool(args.max_steps))


if __name__ == "__main__":
    main()
