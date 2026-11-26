# week07：换模型做事件抽取对比

同一设置下对比 `bert-base-chinese` 与 `chinese-roberta-wwm-ext`。

## 目录

```
week07/
├── README.md
├── 作业报告.md
├── scripts/run_exp_models.py
└── outputs/
    ├── hw_bert_linear_256.json
    ├── hw_roberta_linear_256.json
    ├── 02_bert_linear_train.log
    └── 03_roberta_linear_train.log
```

## 复现

```powershell
cd "…\week7-信息抽取\code"
python bert/train.py --device xpu --decoder linear --model bert-base-chinese --epochs 3 --run_name hw_bert_linear_256
python bert/train.py --device xpu --decoder linear --model chinese-roberta-wwm-ext --epochs 3 --run_name hw_roberta_linear_256
```
