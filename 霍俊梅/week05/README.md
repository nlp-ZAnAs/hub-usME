# week05: Qwen2.5-0.5B SFT 指令微调

用 3000 条中文指令问答数据对 Qwen2.5-0.5B 做 LoRA 微调，目标是跑通 SFT 全流程。
全部在本机 Mac（16GB 内存，MPS）上完成。

## 环境

```bash
pip install torch transformers peft accelerate matplotlib gradio pytest
```

国内网络先配置 HuggingFace 镜像（重要，否则模型下载会超时）：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

## 目录结构

```
week05/
├── sft_train/          # 课程数据（已有）
│   ├── sft_train.jsonl
│   └── sft_val.jsonl
├── sft_train.py        # LoRA 训练
├── sft_infer.py        # 前后对比 + 终端交互
├── chat_server.py      # Gradio 流式 chatbot
└── README.md
```

## 下载模型

本次用 Qwen2.5-0.5B，重新下载到独立目录：

```bash
huggingface-cli download Qwen/Qwen2.5-0.5B --local-dir ./models/Qwen2.5-0.5B
```

## 冒烟测试（先跑这个，几分钟确认能跑通）

用 100 条数据、1 个 epoch 快速验证全流程（数据用 head 截取）：

```bash
head -n 100 sft_train/sft_train.jsonl > /tmp/smoke_train.jsonl
head -n 20  sft_train/sft_val.jsonl   > /tmp/smoke_val.jsonl
python sft_train.py --model ./models/Qwen2.5-0.5B --train /tmp/smoke_train.jsonl --val /tmp/smoke_val.jsonl --output ./output-smoke
```


## 全量训练（约 1-3 小时，建议别开太多应用）

```bash
python sft_train.py --model ./models/Qwen2.5-0.5B --train sft_train/sft_train.jsonl --val sft_train/sft_val.jsonl --output ./output
```

产出：

- `output/loss_curve.png` — train/val loss 曲线
- `output/log_history.json` — 原始训练日志
- `output/adapter/` — LoRA 适配器（几 MB）
- `output/merged/` — 合并后的完整模型（约 1GB）

## 推理演示（微调前后对比 + 交互问答）

```bash
python sft_infer.py --model ./models/Qwen2.5-0.5B --adapter ./output/adapter
```

## Chatbot（Gradio 网页 + 流式输出 + 多轮对话）

```bash
python chat_server.py --model ./output/merged
```

浏览器打开 http://127.0.0.1:7860 ，逐字流式回答，支持多轮追问，点清空按钮重开话题。
（没跑过训练时也可以先用底座模型体验：`python chat_server.py --model ./models/Qwen2.5-0.5B`）
