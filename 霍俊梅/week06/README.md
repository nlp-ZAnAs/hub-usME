# week06：数据接入与分流 —— HC3 human/ChatGPT 二分类

基于 [HC3-Chinese](https://github.com/Hello-SimpleAI/chatgpt-comparison-detection) 数据集，
对一段中文答案做二分类：`human`（人写，label=0）还是 `chatGPT`（AI 生成，label=1）。
用同一个任务横向对比五种方法，理解「成本 vs 效果」的取舍，以及分类输出如何支撑下游分流。

## 目录结构

```
week06/
├── data/                    # all.jsonl 及各领域子集（12853 行 → 39781 个答案样本）
│   ├── train.json / val.json   # 切分并展开后的训练/验证集
├── common/                  # 共用模块
│   ├── data.py              #   load_splits / sample_subset（seed=42 固定抽样）
│   ├── label.py             #   LABELS = ["human", "chatGPT"]
│   ├── split.py             #   数据切分：先按行切（防同题泄漏），再展开成样本
│   └── model_paths.py       #   本地权重路径统一管理
├── rules/                   # 方法一：规则打分
├── tfidf/                   # 方法二：TF-IDF + 线性 SVM
├── bert/                    # 方法三：BERT 微调（+ token_stats.py 定 seq_len）
├── llm_api_zero_shot/       # 方法四：GLM-4-flash API 零样本
├── llm_lora/                # 方法五：Qwen3-0.6B LoRA 指令微调
├── report/compare_methods.py  # 五路同口径对比 + 直方图
└── outputs/                 # 各方法预测结果、模型 checkpoint、对比图
```

## 数据处理

- `all.jsonl` 共 12853 行，每行含 `human_answers` / `chatgpt_answers` 两个列表；
- 每行按行先切 train/val（9:1，防止同一问题的两类答案跨集泄漏），再展开成
  `{"text": 答案, "label": 0/1}` 样本，共 39753 条（train 35728 / val 4025）；
- 标签分布约 56:44（human 略多），train 与 val 比例一致，无需重采样；
- 做二分类只关注 `answer` 文本，不用 `question`（两类共享同一问题，无判别信息）。

## 五路方法

| # | 方法 | 脚本 | 训练成本 | 说明 |
|---|------|------|---------|------|
| 1 | 规则打分 | `rules/classify_rule.py` | 0 | 手写 ChatGPT/人类指纹词表 + 分点/长度启发式，阈值在 train 上调优 |
| 2 | TF-IDF + SVM | `tfidf/classify_tfidf.py` | 秒级 | jieba 分词 → 词 n-gram TF-IDF → LinearSVC |
| 3 | BERT 微调 | `bert/train_bert.py` | ~10 min (MPS) | bert-base-chinese + cls/mean pooling 分类头，8000 条 × 1 epoch |
| 4 | GLM API 零样本 | `llm_api_zero_shot/classify_api_zero_shot.py` | 0（按次计费） | glm-4-flash 免费档，OpenAI 兼容接口 |
| 5 | Qwen3-0.6B LoRA | `llm_lora/train_lora.py` | ~20 min (MPS) | chat 模板 SFT + Loss Masking，只训 1.7% 参数 |

所有方法共同遵守的防泄漏纪律：

- 词表 / IDF / 阈值 / 超参只在 train 上定（val 只 `transform`）；
- val 只做最终评估，不参与任何拟合；
- 横向对比统一用 `sample_subset(val, 500)`（seed=42）——**同一批样本**是可比的前提。

## 对比结果（`report/compare_methods.py`）

同一批 500 条 val 样本（seed=42 抽样），acc / macro-F1：

| 方法 | acc | macro-F1 | 备注 |
|------|------|---------|------|
| rule | 0.6520 | 0.5606 | 偏向判 human，ChatGPT 召回低 |
| GLM-API | 0.6081 | 0.5269 | 同样偏 human；5/500 解析失败 |
| TF-IDF+SVM | 0.9560 | 0.9550 | 词袋特征已能抓住大部分文风差异 |
| **BERT-FT** | **0.9920** | **0.9918** | 8000 条 × 1 epoch，4025 条全量 val acc 0.9903 |
| LoRA | 0.9660 | 0.9649 | Qwen3-0.6B，仅 500 条 × 1 epoch、10M 可训参数 |

![macro-F1 对比](outputs/f1_compare.png)

运行方式：

```bash
cd week06 && python report/compare_methods.py
# 规则 / API 直接读 outputs/*.json 按 idx 对齐；
# TF-IDF 现场重训全量 train；BERT 加载 outputs/bert_best.pt 推理；
# LoRA 结果落盘（outputs/lora_pred.json，同 500 条）后自动纳入。
```

### 结果解读

1. **规则 vs 零样本 LLM（~0.6）**：两者靠的都是「显式指纹」，抓不到就保守判 human，
   犯错方向都一样。会生成文本 ≠ 会判别文本。
2. **TF-IDF（0.956）**：SVM 权重表就是它学到的「规则表」（通常/例如/包括 → ChatGPT，
   就是/其实/你好 → human），印证规则法的方向对、覆盖有限。
3. **BERT（0.992）**：比 TF-IDF 多的 3.6 个点来自词序/句式信息——词袋模型全部丢失的部分。
4. **LoRA（0.966）**：0.6B 生成式模型只训 500 条、10M 可训参数就逼近 TF-IDF；
   但代价是训练/推理成本高一个量级（SFT 要对 15 万词表算交叉熵、推理逐 token 生成），
   且 chatGPT 类召回（0.925）低于 BERT——同样的任务，「分类式」结构天然比「生成式」省。

## 产出的下游用途（分流）

各方法统一落盘 `{"idx", "text", "true", "pred", "score"}`（score 为置信度：
规则分数 / decision_function / API 输出），下游分流只需一个读取器：

```
pred + score ──> 高置信 human   ──> research 队列
             ──> 高置信 chatGPT ──> drop（噪音过滤）
             ──> 低置信          ──> review（人工 / LLM 兜底）
```

`outputs/bert_best.pt`、`outputs/qwen3_lora_r16/` 可直接加载做在线推理。

## 复现命令

```bash
python common/split.py                                        # 数据切分
python rules/classify_rule.py                                 # 方法一
python tfidf/classify_tfidf.py --cv                           # 方法二
python bert/token_stats.py && python bert/train_bert.py --n 8000 --epochs 1   # 方法三
export GLM_API_KEY=... && python llm_api_zero_shot/classify_api_zero_shot.py --n 500  # 方法四
python llm_lora/train_lora.py --num_train 500 --epochs 1      # 方法五
python report/compare_methods.py                              # 汇总对比
```
