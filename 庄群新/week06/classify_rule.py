"""方法一 · L0 规则 / 关键词

思路：给每个类别配一组关键词（类别名 + 几行业黑话），标题里命中就判该类；
      一个都没命中 → 猜训练集里最多的类（兜底）。
为什么留着它：公告标题有强套路（"年度业绩预告""回购""减持"），一条正则/词典就能覆盖，
      省掉一次模型调用。但它的上限很低——命中率上不去，就到此为止。

与课件的差别：课件用的是「只拿 15 个类别名当关键词」，命中率 6.6%、F1 0.072；
             这里把每个类补了 3~8 个业务词，命中率会高一些，但仍然是三种方法里最弱的。

用法：
  python classify_rule.py            # 500 条干净验证样本
  python classify_rule.py --n 0      # 全部 8,901 条
"""

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
import dataio
import metrics as M
import resources as R
from labels import LABELS, to_group

# 关键词表：长度优先匹配（"年度业绩预告" 应该压过 "业绩"）
KEYWORDS = {
    "故事": ["故事", "小说", "传奇", "民间"],
    "文化": ["文化", "文物", "非遗", "读书", "书法", "传统"],
    "娱乐": ["明星", "电影", "综艺", "演员", "演唱会", "粉丝", "票房"],
    "体育": ["足球", "篮球", "比赛", "球员", "赛季", "夺冠", "国足", "奥运"],
    "财经": ["财报", "营收", "净利润", "业绩预告", "经济", "GDP", "央行", "降息"],
    "房产": ["房价", "楼市", "房地产", "楼盘", "物业", "限购"],
    "汽车": ["汽车", "新车", "车企", "车型", "新能源车", "销量"],
    "教育": ["高考", "学校", "学生", "教师", "招生", "考试", "考研"],
    "科技": ["手机", "芯片", "人工智能", "技术", "发布会", "系统", "算力", "大模型"],
    "军事": ["军演", "战机", "导弹", "部队", "国防", "演习"],
    "旅游": ["旅游", "景区", "游客", "航班", "酒店", "签证"],
    "国际": ["美国", "俄罗斯", "外交", "总统", "联合国", "访华", "欧盟"],
    "证券": ["股票", "股价", "涨停", "A股", "基金", "券商", "回购", "减持", "并购"],
    "农业": ["农业", "种植", "猪肉", "粮食", "养殖", "丰收"],
    "电竞": ["电竞", "战队", "LPL", "KPL", "选手", "手游"],
}

# 按关键词长度降序，长词优先命中（避免 "业绩" 抢走 "业绩预告" 的语义）
_RULES = sorted(
    ((kw, LABELS.index(name)) for name, kws in KEYWORDS.items() for kw in kws),
    key=lambda x: -len(x[0]),
)


def predict_one(text: str, fallback: int):
    for kw, label_id in _RULES:
        if kw in text:
            return label_id
    return fallback


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=config.N_EVAL, help="0 = 全部干净验证样本")
    ap.add_argument("--out", default="rule_metrics.json")
    args = ap.parse_args()

    config.ensure_dirs()
    # 监控包住「读数据 + 建表 + 推理」全过程，否则峰值内存只会测到最后那几毫秒
    with R.ResourceMonitor(interval=0.05) as mon:
        train, _ = dataio.load_splits()
        samples, meta = dataio.load_clean_val(args.n or None)

        fallback = Counter(d["label"] for d in train).most_common(1)[0][0]

        with R.Timer() as train_timer:
            # 规则法没有「训练」，只有建表这一步（这里已在模块级完成）
            pass

        t0 = time.time()
        preds, latencies = [], []
        for d in samples:
            s = time.perf_counter()
            preds.append(predict_one(d["sentence"], fallback))
            latencies.append((time.perf_counter() - s) * 1000)
        infer_seconds = round(time.time() - t0, 3)

    latencies.sort()
    y_true = [d["label"] for d in samples]
    report = M.classification_report(y_true, preds)
    hit_rate = round(sum(1 for d, p in zip(samples, preds)
                         if any(kw in d["sentence"] for kw, _ in _RULES)) / len(samples), 4)

    result = {
        "method": "规则 / 关键词",
        "level": "L0",
        "metrics": {
            **report,
            "group_acc": M.group_level_report(y_true, preds),
            "hit_rate": hit_rate,
            "fallback_class": LABELS[fallback],
        },
        "cost": {
            "train_seconds": 0.0,
            "infer_seconds": infer_seconds,
            "infer_ms_b1": round(sum(latencies) / len(latencies), 3),
            "infer_ms_b32": round(sum(latencies) / len(latencies), 3),
            "throughput": round(len(samples) / max(infer_seconds, 1e-6), 1),
            "model_size_mb": round(R.file_size_mb(Path(__file__)), 3),
            "peak_rss_mb": round(mon.peak_rss_mb, 1),
            "peak_gpu_mb": 0.0,
            "n_params": 0,
        },
        "meta": {**meta, "env": R.env_info(), "fallback_majority": LABELS[fallback]},
    }

    (config.OUT_DIR / args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (config.OUT_DIR / "rule_pred.json").write_text(
        json.dumps({"y_true": y_true, "y_pred": preds,
                    "sentence": [d["sentence"] for d in samples]}, ensure_ascii=False), encoding="utf-8")

    print(f"评测 {meta['n_eval']} 条（干净集，seed={meta['seed']}）| 关键词命中率 {hit_rate:.1%}")
    print(f"| 规则/关键词 | acc={report['acc']} | macro-F1={report['macro_f1']} | 覆盖 {report['coverage']} |")
    print(f"耗时 {infer_seconds}s（{result['cost']['infer_ms_b1']} ms/条）| 峰值内存 {result['cost']['peak_rss_mb']} MB")
    print(f"→ {config.OUT_DIR / args.out}")


if __name__ == "__main__":
    main()
