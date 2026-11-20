"""标签口径：TNEWS 15 类 → 投研 4 组主题 + 相关性

为什么单独一个文件：规则法、TF-IDF、BERT 三路输出必须用同一张表，
否则同一个 0.5 的准确率在不同脚本里含义都不同，数字不可比。

标签口径是业务定义，不是模型能力 —— 换领域改这张映射表，不用重标数据。
"""

LABELS = [
    "故事", "文化", "娱乐", "体育", "财经",
    "房产", "汽车", "教育", "科技", "军事",
    "旅游", "国际", "证券", "农业", "电竞",
]
LABEL2ID = {name: i for i, name in enumerate(LABELS)}

GROUP_MAP = {
    "财经": "金融", "证券": "金融", "房产": "金融",
    "科技": "产业", "汽车": "产业", "农业": "产业",
    "国际": "宏观", "军事": "宏观",
    "教育": "噪音", "娱乐": "噪音", "体育": "噪音",
    "电竞": "噪音", "文化": "噪音", "故事": "噪音", "旅游": "噪音",
}
GROUPS = ["金融", "产业", "宏观", "噪音"]
RELEVANT_GROUPS = {"金融", "产业"}


def to_group(label_id: int) -> str:
    return GROUP_MAP[LABELS[label_id]]


def is_relevant(label_id: int) -> bool:
    return to_group(label_id) in RELEVANT_GROUPS


def queue_name(label_id: int, conf: float, conf_threshold: float = 0.8) -> str:
    """分类结果 → 队列动作：不相关 drop / 低置信 review / 高置信 research"""
    if not is_relevant(label_id):
        return "drop"
    return "research" if conf >= conf_threshold else "review"
