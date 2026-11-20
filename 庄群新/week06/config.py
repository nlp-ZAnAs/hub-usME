"""第 6 周作业 · 全局配置

口径统一说明（所有方法都从这里取，避免「各脚本各写一套」）：
  - 评测集：TNEWS 验证集**去掉与训练集重复的 1,099 条**后的干净集，再按 seed=42 抽样 N_EVAL 条
  - 主指标：Macro-F1（15 类等权）
  - 预训练底座：week4 目录里的 bert-base-chinese（本机已验证可加载）
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
OUT_DIR = ROOT / "outputs"
FIG_DIR = ROOT / "figures"
LOG_DIR = ROOT / "logs"

# 预训练模型：直接用 week4 那份（换成别的模型只需改这一行）
BERT_PRETRAINED = Path(r"D:\BaiduNetdiskDownload\week4-语言模型\bert-base-chinese")

NUM_LABELS = 15
SEED = 42

# 评测条数：500 = 与课件 summary 表可直接对齐；0 = 全部 8,901 条干净样本
N_EVAL = 500

# BERT 训练超参（默认值对齐课件：cls 池化 / 3 epoch / 分层 lr / 类别加权 / AMP）
BERT_EPOCHS = 3
BERT_BATCH = 32
BERT_LR = 2e-5
BERT_HEAD_LR_MULT = 5.0
BERT_MAX_LEN = 64
BERT_CLASS_WEIGHT = True
BERT_POOL = "cls"

# TF-IDF 超参（字 n-gram，中文短标题比词更稳）
TFIDF_NGRAM = (1, 2)
TFIDF_MAX_FEATURES = 300_000
TFIDF_C = 1.0


def ensure_dirs():
    for d in (DATA_DIR, MODEL_DIR, OUT_DIR, FIG_DIR, LOG_DIR):
        d.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    ensure_dirs()
    print(f"ROOT           = {ROOT}")
    print(f"BERT_PRETRAINED= {BERT_PRETRAINED}  (存在: {BERT_PRETRAINED.is_dir()})")
    for d in (DATA_DIR, MODEL_DIR, OUT_DIR, FIG_DIR, LOG_DIR):
        print(f"{d.name:<8} = {d}")
