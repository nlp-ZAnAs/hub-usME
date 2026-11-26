# week07 · 中文突发事件事件抽取（CEC）

在 **CEC（Chinese Emergency Corpus）** 上做「一句一事件」的触发词 + 论元抽取，用**统一评测口径**横向对比 6 种方法：规则基线 → BiLSTM-CRF → BERT/RoBERTa+CRF → 两段式流水线 → LLM API 零/少样本。


---

## 1. 数据统计

**语料**：CEC 突发事件新闻，按句切分，一句对应一个事件；共 **770 句**（train / dev / test = **507 / 125 / 138**）。数据源统一为 `data/processed/{train,dev,test}.jsonl`。

### 1.1 事件类型与论元角色（5 类 / 8 个角色）

| 事件类型 | 论元角色 | 角色数 |
|---|---|---|
| 地震 | 时间、地点、伤亡人数 | 3 |
| 火灾 | 时间、地点、伤亡人数 | 3 |
| 交通事故 | 时间、地点、肇事者、受害者、涉事车辆、伤亡人数 | 6 |
| 恐怖袭击 | 时间、地点、袭击者、受害者、伤亡人数 | 5 |
| 食物中毒 | 时间、地点、中毒者 | 3 |

去重后共有 **8 个角色**：时间、地点、伤亡人数、肇事者、受害者、涉事车辆、袭击者、中毒者。定义见 `data/conf/event_schema.json`。

### 1.2 各 split 的规模与类型分布

| split | 句数 | 地震 | 火灾 | 交通事故 | 恐怖袭击 | 食物中毒 | 平均句长(字) | 平均论元/句 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 507 | 120 | 100 | 143 | 85 | 59 | 57.7 | 2.04 |
| dev | 125 | 28 | 23 | 38 | 25 | 11 | 57.6 | 2.05 |
| test | 138 | 32 | 34 | 33 | 18 | 21 | 56.6 | 2.17 |
| **合计** | **770** | 180 | 157 | 214 | 128 | 91 | — | — |

要点：
- 类别**不均衡**：交通事故最多（214），食物中毒最少（91）；dev 里食物中毒仅 11 句，单类别指标波动较大。
- 句子较短（平均 ~57 字，最长 167 字），`max_len=256` 足以覆盖，dev 截断数为 0。
- 每句平均 ~2 个论元，且很多句子只有「时间/地点」，论元本就稀疏 → 论元 F1 天然偏低。

---

## 2. 统一评测口径

所有方法都调用 `common/evaluate_events.py`（经 `common/eval_utils.evaluate_predictions` 落盘到 `outputs/metrics/disc/<name>.json`），保证「谁高谁低」可比：

- **对齐键**：`id|sent_id`（同一文档可能多句，避免互相覆盖）。
- **trigger**：预测触发词文本与金标**完全相等**才算命中。
- **argument**：按 `(角色, 论元文本)` 的 `Counter` 求交集；多抽 = FP，漏抽 = FN。
- **event_exact**：类型 + 触发词 + 全部论元三者全等才算对（最严格）。
- **type_macro**：只看事件类型，逐类算 P/R/F1 再**宏平均**；类别集合 = 金标类型 ∪ 预测类型，出现越界类型名（如「爆炸」）会多算一个 F1=0 的类，拉低宏平均。
- 预测文件**必须保留空预测行**（解析失败/没抽到），否则分母变小、F1 虚高。

指标含义与各方法的解析/对齐细节见 `common/README.md`、`README` 各脚本头部注释。

---

## 3. 统一口径对比（dev，125 句）

| 方法 | 类型 | trigger F1 | argument F1 | event F1 | type_macro | 备注 |
|---|---|---:|---:|---:|---:|---|
| L0 规则基线 | 零训练 | 0.5920 | 0.2068 | 0.0480 | 0.8704 | 触发词白名单 + 正则论元 |
| BiLSTM-CRF | 从零训练（1.0M） | 0.6894 | 0.4934 | 0.2809 | 0.9363 | 40 epoch；软约束，非法序列 28.8% |
| BERT-base + CRF | 预训练（102M） | 0.7184 | 0.5502 | 0.3673 | **0.9656** | 15 epoch；软约束，非法序列 36.0% |
| RoBERTa-base + CRF | 预训练（102M） | 0.6971 | **0.5988** | 0.3817 | 0.9369 | 15 epoch；软约束，非法序列 48.0% |
| RoBERTa-base + CRF（硬约束） | 预训练（102M） | 0.6840 | 0.5938 | **0.4242** | 0.8909 | 非法转移置 -1e4，非法 BIO = 0 |
| 两段式（类型→论元专家） | 预训练，5+1 模型 | **0.7220** | 0.5299 | 0.3154 | 0.9507 | 类型准确率 98.4% |
| 两段式（类型给金标的上界） | — | 0.7303 | 0.5379 | 0.3154 | 0.9577 | 隔离「类型判错」的损失 |
| LLM API 零样本（glm-4-flash） | 零训练 | 0.6400 | 0.3505 | 0.1040 | 0.8117 | JSON 合法率 100% |
| LLM API 少样本（glm-4-flash） | 零训练 | 0.6240 | 0.4704 | 0.2320 | 0.6937 | 幻觉论元 52→12 |

> 加粗为各列最优（含上界分析行）。`roberta*.json` 的 `meta.display_name` 仍是早先复制的 “BERT-12L+CRF”，以本表为准。

### 3.1 非法 BIO 统计（判别式模型，dev 原始输出）

| 模型 | 非法序列率 | 非法起点 | 非法转移 |
|---|---:|---:|---:|
| BiLSTM-CRF（软约束） | 28.8% | 2 | 43 |
| BERT-base + CRF（软约束） | 36.0% | 0 | 83 |
| RoBERTa-base + CRF（软约束） | 48.0% | 2 | 124 |
| RoBERTa-base + CRF（硬约束） | 0% | 0 | 0 |

CRF 的软约束只是「倾向」合法，硬约束把非法转移/开头分数置 -1e4，才能压到 0 —— 且对 RoBERTa 的 event F1 有正收益（0.382 → 0.424）。

---

## 4. 主要结论

1. **有监督 > 零样本 LLM > 规则**：trigger/argument/event 三级上，BERT/RoBERTa/两段式最高，零样本 LLM 全面强于规则，但仍明显落后有监督模型；少样本把 LLM 的 argument/event 拉近一步（0.35→0.47，0.10→0.23），仍低于训练模型。
2. **event_exact 的瓶颈在论元**：event 全等要求触发词 + 全部论元都对，而论元稀疏、边界细碎，是所有方法这列最低的原因。
3. **两段式不是越拆越好**：类型准确率已达 98.4%，`oracle_type` 上界与端到端几乎重合，说明损失不在「类型判错」，而在拆开后各专家只见 ~1/5 数据、论元略松 → event 反而低于联合模型。
4. **硬约束值得**：非法 BIO 归零，且 RoBERTa 的 event F1 提升约 4 个点。
5. **LLM 的 type_macro 要结合 per-type 看**：越界类型名（「爆炸」「疾病」）会被宏平均按 0 分计入；5 个 schema 内类型 F1 实际都在 0.95 以上。

---

## 5. 目录结构

```
week07/
├── common/            # 公共底座：event_data / evaluate_events / crf / models / eval_utils / log_utils
├── data/
│   ├── processed/     # train.jsonl / dev.jsonl / test.jsonl（统一数据源）
│   └── conf/          # event_schema.json、trigger_whitelist.json
├── rule/              # L0 规则基线（零训练）
├── bilstm_crf/        # 字符级 BiLSTM-CRF（从零训练）
├── bert/              # BERT/RoBERTa + (CRF | Linear) 联合抽取
├── bert_pipeline/     # 两段式：类型分类 → 类型条件论元专家
├── llm_zero_shot/     # LLM API 零/少样本（OpenAI 兼容 + 离线 mock）
└── outputs/
    ├── checkpoints/   # *.pt
    ├── metrics/disc/  # 统一指标 JSON（每方法一个）
    ├── disc/          # pred_*.jsonl 预测
    ├── logs/          # 训练/评测日志
    └── llm/           # LLM raw 输出与工程统计
```

---

## 6. 复现

环境：Python 3 + `torch` / `transformers` / `openai`；Mac 自动用 MPS。训练与评测均在 `week07/` 下执行。

```bash
# L0 规则基线
python rule/rule_extract.py --split dev

# BiLSTM-CRF（从零，默认 40 epoch）
python bilstm_crf/train.py
python bilstm_crf/train.py --decoder linear        # 消融：去掉 CRF

# BERT-base + CRF 联合抽取（默认 15 epoch, lr 3e-5）
python bert/train.py
python bert/train.py --decoder linear --hard_constraint   # 消融

# RoBERTa-base + CRF（复用 bert/train.py，只换底座与 run_name）
python bert/train.py --bert_path ../../../models/roberta-base --run_name roberta
python bert/train.py --bert_path ../../../models/roberta-base --hard_constraint --run_name roberta_hard

# 两段式流水线（类型分类 → 5 个类型条件论元专家）
python bert_pipeline/train.py
python bert_pipeline/train.py --only_experts       # 只重训论元专家
python bert_pipeline/train.py --skip_train         # 只重新组合评测

# LLM API 零/少样本（在 .env 写 GLM_API_KEY=...；或 export LLM_API_KEY/LLM_BASE_URL/LLM_MODEL）
python llm_zero_shot/llm_extract.py --mode zero
python llm_zero_shot/llm_extract.py --mode few
python llm_zero_shot/llm_extract.py --dry-run      # 离线 mock，不联网，验证流水线
```

每次评测都会写出 `outputs/metrics/disc/<name>.json`，用上表的字段即可拼出对比。

---

## 7. 关键参数速查

| 方法 | run_name | 关键超参 |
|---|---|---|
| BiLSTM-CRF | `bilstm_crf` | emb 128 / hidden 256×2 / lr 1e-3 / 40 epoch |
| BERT-base + CRF | `bert_crf_256` | 12 层 / lr 3e-5 / head×5 / 15 epoch |
| RoBERTa-base + CRF | `roberta`, `roberta_hard` | 同上，底座换 `roberta-base`；后者加 `--hard_constraint` |
| 两段式 | `pipeline` | 类型 6 epoch / 论元专家 12 层·30 epoch·lr 3e-5 |
| LLM 零/少样本 | `llm_zero_shot`, `llm_few_shot` | `glm-4-flash`，temperature 0，全 dev 125 句 |
