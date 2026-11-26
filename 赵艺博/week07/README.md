# week07 作业：金融公告事件抽取

对照课件收口页「课后作业」五件事；并按「换不同模型做抽取」加了一组 **BERT vs RoBERTa** 对照。

## 目录

```
week07/
├── README.md
├── 作业报告.md
├── scripts/run_exp_models.py
└── outputs/
    ├── badcases_10.json
    ├── loss_curve_bert_crf_256.png
    └── *.log
```

## 作业对应

| # | 课件要求 | 本包 |
|---|----------|------|
| 1 | 四路（规则/BERT/两段式/LLM）同口径对比表 | `作业报告.md` §2 |
| 2 | 10 条 badcase 归类 | `outputs/badcases_10.json` + 报告 §3 |
| 3 | extract_event 跑通公告→事件列表 | `outputs/05_demo.log` |
| 4 | 简历口径四个数 | 报告 §5 |
| 5 | steps.jsonl 画 loss 曲线 | `outputs/loss_curve_bert_crf_256.png` |
| + | 换预训练：bert-base vs chinese-roberta-wwm-ext | 报告 §2.2 |

## 复现（换模型）

```powershell
cd "…\week7-信息抽取\code"
python bert/train.py --device xpu --decoder linear --model bert-base-chinese --epochs 3 --run_name hw_bert_linear_256
python bert/train.py --device xpu --decoder linear --model chinese-roberta-wwm-ext --epochs 3 --run_name hw_roberta_linear_256
```
