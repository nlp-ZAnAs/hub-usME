"""
本地模型权重路径（全项目共用）

优先用本地已下载的权重（离线可用、加载快）；不存在时返回模型名，
transformers 会自己去 HF Hub / 缓存里找。换机器/换目录只改这一个文件。
"""

from pathlib import Path

MODELS_ROOT = Path.home() / "badouai" / "models"

qwen3_path = str(MODELS_ROOT / "Qwen3-0.6B")
bert_path = str(MODELS_ROOT / "bert-base-chinese")   # 若本地没有则各脚本用默认名
if not Path(bert_path).exists():
    bert_path = str(Path.home() / "badouai" / "week4-LM" / "bert-base-chinese")
