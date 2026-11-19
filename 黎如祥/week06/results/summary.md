# Week06 分类方法对比：结果汇总

## ChnSentiCorp 中文情感（2 类，混合领域）

| 方法 | 家族 | 参数 | Acc（含失败） | Macro-F1 | 覆盖类别 | 训练耗时 | 推理 ms/条(批量) |
|---|---|---|---|---|---|---|---|
| 规则 / 关键词 | rule | — | 0.742 | 0.7397 | 2/2 | 0s | — |
| TF-IDF + LinearSVC | linear | — | 0.900 | 0.8978 | 2/2 | 2s | <0.1ms |
| BERT-base-chinese 微调 | finetune | 102.3M | 0.936 | 0.9343 | 2/2 | 3.8min | 2.0ms |
| MacBERT-base 微调 | finetune | 102.3M | 0.948 | 0.9466 | 2/2 | 3.5min | 1.8ms |
| RoBERTa-wwm-ext 微调 | finetune | 102.3M | 0.946 | 0.9445 | 2/2 | 3.9min | 1.9ms |
| Qwen3-0.6B zero-shot | llm_zero_shot | 596.0M | 0.762 | 0.7313 | 2/2 | 0s | 44.6ms |
| Qwen3-0.6B + LoRA SFT | llm_lora | 601.1M | 0.944 | 0.9424 | 2/2 | 13.5min | 37.2ms |

> 训练 9600 条 / 验证 1200 条，去重 338 条（28.2%），评测 500 条，seed=42。

**MacBERT-base 微调 最常混淆的类对**：正面→负面（17 条）；负面→正面（9 条）

## TNEWS 新闻标题（15 类）

| 方法 | 家族 | 参数 | Acc（含失败） | Macro-F1 | 覆盖类别 | 训练耗时 | 推理 ms/条(批量) |
|---|---|---|---|---|---|---|---|
| 规则 / 关键词 | rule | — | 0.126 | 0.0716 | 10/15 | 0s | — |
| TF-IDF + LinearSVC | linear | — | 0.512 | 0.4721 | 15/15 | 8s | <0.1ms |
| BERT-base-chinese 微调 | finetune | 102.3M | 0.564 | 0.5547 | 15/15 | 5.9min | 0.5ms |
| MacBERT-base 微调 | finetune | 102.3M | 0.578 | 0.571 | 15/15 | 5.3min | 0.5ms |
| RoBERTa-wwm-ext 微调 | finetune | 102.3M | 0.564 | 0.5574 | 15/15 | 5.7min | 0.5ms |
| Qwen3-0.6B zero-shot | llm_zero_shot | 596.0M | 0.220 | 0.1209 | 9/15 | 0s | 10.6ms |

> 训练 53360 条 / 验证 10000 条，去重 1099 条（11.0%），评测 500 条，seed=42。

**MacBERT-base 微调 最常混淆的类对**：科技→财经（13 条）；国际→军事（10 条）；财经→国际（6 条）；国际→旅游（6 条）；财经→旅游（4 条）

## 跨数据集对比（每族最好成绩）

| 方法族 | TNEWS Macro-F1 | ChnSentiCorp Macro-F1 |
|---|---|---|
| rule | 0.072 | 0.740 |
| linear | 0.472 | 0.898 |
| finetune | 0.571 | 0.947 |
| llm_zero_shot | 0.121 | 0.731 |

## 数据泄漏检查：TF-IDF 在「不去重」的 val 上会虚高多少

| 数据集 | 干净集 Macro-F1 | 含重复 val Macro-F1 | 虚高 |
|---|---|---|---|
| ChnSentiCorp 中文情感（2 类，混合领域） | 0.898 | 0.922 | +0.024 |
| TNEWS 新闻标题（15 类） | 0.472 | 0.497 | +0.025 |
