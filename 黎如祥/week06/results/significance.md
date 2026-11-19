# Macro-F1 配对 bootstrap 检验（2000 次重采样）

> 差值 = 模型 A − 模型 B；95% CI 跨 0 表示差异不显著。

## TNEWS 新闻标题（15 类）

| 对比（A vs B） | A Macro-F1 | B Macro-F1 | 差值 | 95% CI | 结论 |
|---|---|---|---|---|---|
| RoBERTa-wwm-ext 微调 vs BERT-base-chinese 微调 | 0.557 | 0.555 | +0.002 | [-0.033, +0.038] | 不显著 |
| MacBERT-base 微调 vs BERT-base-chinese 微调 | 0.571 | 0.555 | +0.016 | [-0.018, +0.051] | 不显著 |
| TF-IDF + LinearSVC vs BERT-base-chinese 微调 | 0.472 | 0.555 | -0.082 | [-0.129, -0.032] | 显著 |
| MacBERT-base 微调 vs TF-IDF + LinearSVC | 0.571 | 0.472 | +0.097 | [+0.048, +0.149] | 显著 |

## ChnSentiCorp 中文情感（2 类，混合领域）

| 对比（A vs B） | A Macro-F1 | B Macro-F1 | 差值 | 95% CI | 结论 |
|---|---|---|---|---|---|
| RoBERTa-wwm-ext 微调 vs BERT-base-chinese 微调 | 0.945 | 0.934 | +0.010 | [-0.008, +0.027] | 不显著 |
| MacBERT-base 微调 vs BERT-base-chinese 微调 | 0.947 | 0.934 | +0.012 | [-0.004, +0.029] | 不显著 |
| TF-IDF + LinearSVC vs BERT-base-chinese 微调 | 0.898 | 0.934 | -0.037 | [-0.064, -0.012] | 显著 |
| Qwen3-0.6B + LoRA SFT vs BERT-base-chinese 微调 | 0.942 | 0.934 | +0.008 | [-0.011, +0.028] | 不显著 |
| MacBERT-base 微调 vs TF-IDF + LinearSVC | 0.947 | 0.898 | +0.049 | [+0.026, +0.073] | 显著 |

