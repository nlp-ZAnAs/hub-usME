"""
本地预训练模型定位（离线课堂必备）

为什么不能写死路径：
  同一个项目会存在于两套目录树里（母版 `项目材料准备/` 与交付副本 `八斗课程/weekN/`），
  脚本所在层级不同，`../pretrain_models` 在其中一套里必然失效。
  做法：从脚本所在目录**向上逐级查找** `pretrain_models/<模型名>`，
  并且要求目录里真的有权重文件——否则会拿到半下载的空目录（离线课堂最常见的坑）。

用法：
  from common.model_paths import qwen3_path, bert_path
"""

from pathlib import Path

QWEN_DIR = "models--Qwen--Qwen2.5-0.5B-Instruct"
BERT_DIR = "bert-base-chinese"
PRETRAIN = "pretrain_models"


def _has_weights(p: Path) -> bool:
    if not p.is_dir():
        return False
    if any(p.glob("*.safetensors")):
        return True
    return (p / "pytorch_model.bin").exists()


def resolve(dir_name: str, extra_roots=None) -> Path:
    """
    候选来源：
      - extra_roots（如 F:/models/hub）下的 models--<dir_name>
      - 项目树里的 pretrain_models/<dir_name>
    Hub 缓存要下钻到 snapshots/<commit>/，返回那个真正含 config.json 的目录。
    """
    here = Path(__file__).resolve()
    candidates = []
    for root in (extra_roots or []):
        root = Path(root)
        candidates.extend([
            root / f"models--{dir_name}",
            root / dir_name,
        ])
    for parent in here.parents:
        candidates.append(parent / PRETRAIN / dir_name)

    for c in candidates:
        if not c.is_dir():
            continue
        # Hub 缓存：取 snapshots 下最新的一个含权重的版本
        snap_root = c / "snapshots"
        if snap_root.is_dir():
            for snap in sorted(snap_root.iterdir(), reverse=True):
                if _has_weights(snap):
                    return snap
        if _has_weights(c):
            return c

    tried = "\n  ".join(str(c) for c in candidates)
    raise FileNotFoundError(f"未找到本地模型 {dir_name}，尝试过：\n  {tried}")


def qwen3_path() -> Path:
    return resolve(QWEN_DIR, extra_roots=[r"F:\models\hub"])


def bert_path() -> Path:
    return resolve(BERT_DIR, extra_roots=[r"F:\models\hub"])


if __name__ == "__main__":
    # 自检：课堂排查「模型找不到」时先跑这个
    print(f"{QWEN_DIR:<20} -> {qwen3_path()}")
    print(f"{BERT_DIR:<20} -> {bert_path()}")
