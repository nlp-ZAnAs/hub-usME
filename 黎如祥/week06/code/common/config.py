# -*- coding: utf-8 -*-
"""
全局配置：数据集注册表 + 预训练模型注册表

Week06 作业：固定评测协议，换「预训练模型」和「数据集」做对比实验。
所有脚本都从这里取标签、路径、长度设置，保证横向可比。
"""

import os
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parent.parent      # week06/code
PROJECT_ROOT = CODE_ROOT.parent                         # week06
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

# 模型权重不放进作业仓库（每个 400MB~1.5GB），统一放在这里
MODELS_ROOT = Path(os.environ.get("WEEK6_MODELS", r"D:\Env\models"))
# 训练好的 checkpoint 也不进仓库（BERT 一个 400MB），可用 --save_dir 覆盖
CKPT_DIR = Path(os.environ.get("WEEK6_CKPT", r"D:\Env\week6_ckpt"))

# --------------------------------------------------------------------------- #
# 数据集
# --------------------------------------------------------------------------- #
DATASETS = {
    "tnews": {
        "display": "TNEWS 新闻标题（15 类）",
        "dir": "tnews",
        "labels": ["故事", "文化", "娱乐", "体育", "财经",
                   "房产", "汽车", "教育", "科技", "军事",
                   "旅游", "国际", "证券", "农业", "电竞"],
        "max_len": 64,
        "batch_size": 64,
        "text_key": "sentence",
        "rule": "keyword",
    },
    "chnsenticorp": {
        "display": "ChnSentiCorp 中文情感（2 类，混合领域）",
        "dir": "chnsenticorp",
        "labels": ["负面", "正面"],
        "max_len": 256,
        "batch_size": 32,
        "text_key": "sentence",
        "rule": "sentiment_lexicon",
    },
}

# --------------------------------------------------------------------------- #
# 预训练骨干（同尺寸、同架构家族，只换预训练目标/语料）
# --------------------------------------------------------------------------- #
BACKBONES = {
    "bert-base-chinese": {
        "display": "BERT-base-chinese",
        "params": "110M",
        "note": "Google 原版，MLM + NSP，中文维基",
    },
    "chinese-roberta-wwm-ext": {
        "display": "RoBERTa-wwm-ext",
        "params": "102M",
        "note": "哈工大，全词掩码 + 去掉 NSP，更多语料",
    },
    "chinese-macbert-base": {
        "display": "MacBERT-base",
        "params": "102M",
        "note": "哈工大，纠错式掩码（MLM as correction）",
    },
}

QWEN_DIR = "Qwen3-0.6B"


def model_path(name: str) -> Path:
    """返回本地模型目录；不存在就明确报错，不静默联网下载。"""
    p = MODELS_ROOT / name
    if not (p / "config.json").exists():
        raise FileNotFoundError(
            f"未找到模型 {name}：{p}\n"
            f"请把权重放到 {MODELS_ROOT}，或设置环境变量 WEEK6_MODELS")
    return p


def dataset_paths(name: str):
    cfg = DATASETS[name]
    d = DATA_DIR / cfg["dir"]
    return d / "train.json", d / "val.json"


def result_path(dataset: str, method: str) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR / f"pred_{dataset}_{method}.json"
