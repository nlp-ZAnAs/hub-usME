# week06 作业提交包

## 作业对应关系（课件）

- **课后作业**：自己做实验 → 本组 = **换预训练模型**（BERT vs RoBERTa）
- **混淆分析**：科技 ↔ 财经（见 `作业报告.md` §3）
- **数字口径**：见 `作业报告.md` §2 / §4

## 目录

```
week06/
├── README.md                 # 本说明
├── 作业报告.md               # 实验结论 + 易混类 + 简历口径
├── scripts/run_exp_a.py      # 复现脚本
└── outputs/                  # 预测 JSON + 训练/评测日志
```

## 结果摘要（干净 val 500，seed=42）

| 方法 | Acc | Macro-F1 |
|------|-----|----------|
| 规则 | 0.126 | 0.0716 |
| TF-IDF | 0.512 | 0.4721 |
| bert-base-chinese | **0.554** | **0.5388** |
| chinese-roberta-wwm-ext | 0.532 | 0.5206 |

## 复现（课包代码）

课包根目录：

`D:\BaiduNetdiskDownload\Agent开发学习资料\第六周 分类任务\week6-分类任务\week6-数据接入与分流\code`

```powershell
cd "……\week6-数据接入与分流\code"
python ..\..\..\..\..\Agent开发作业\赵艺博\week06\scripts\run_exp_a.py
# 或按 scripts/run_exp_a.py 内命令逐步跑
```

预训练权重：`Agent开发学习资料\pretrain_models\{bert-base-chinese,chinese-roberta-wwm-ext}`  
训练得到的 ckpt：`code\outputs\bert-base-chinese_ckpt.pt`、`chinese-roberta-wwm-ext_ckpt.pt`（约数百 MB，未纳入本提交包）

## 说明

- 训练设备默认 CPU（本机 Arc + PyTorch XPU 反传不稳定）。
- 评测进程偶发 `0xC0000005` 退出，**预测文件已写出则不影响结果**。
