# 字符级 Transformer 预训练与问答 SFT

本实验复用 week04 已训练的 Transformer，完成继续预训练、监督微调和逐 token 问答生成。每个普通字符对应一个 token。模型使用 2 层因果自注意力、4 个注意力头、192 维字符向量、516 维前馈层和可学习的位置向量。

本次预训练从之前的权重开始，继续学习原文本的下一个 token。上下文窗口从 64 扩展到 256，原位置向量重复用于初始化扩展部分，随后全部参与训练。字符表沿用原有 4000 字，新增 PAD、UNK、BOS、SEP、EOS 五个特殊 token，总参数量为 1,808,442。

## 文件和入口

| 文件 | 用途 |
| --- | --- |
| model.py | Transformer、字符编码、权重迁移和答案损失掩码 |
| prepare_data.py | 文本去重、短问答筛选及固定数据划分 |
| train.py | 继续预训练和全参数 SFT 两个阶段 |
| generate.py | 输入问题并逐字符生成答案，也支持预训练文本续写 |
| test_model.py | 在问答测试集计算答案 PPL，保存生成示例 |
| ../week04/best_model_transformer_matched.pt | 复用上一周的 Transformer 初始化权重 |
| checkpoints/pretrained.pt | 本次继续预训练后保存的最佳权重 |
| checkpoints/sft.pt | 从 pretrained.pt 开始监督微调的最佳权重 |
| checkpoints/*.metrics.json | 两阶段的逐轮训练记录和最佳轮次 |
| ../week04/corpus_general.txt | 复用上一周的原预训练语料 |
| data/pretrain_train.txt、pretrain_val.txt | 去重后的本次预训练划分 |
| data/sft_train.jsonl、sft_val.jsonl、sft_test.jsonl | 问答训练、验证、测试数据 |
| data/vocab.json | 字符与特殊 token 编号 |
| data/manifest.json | 数据数量、筛选规则相关统计、来源和 SHA256 指纹 |
| evaluation.json | 预训练与 SFT 权重在同一问答测试集上的答案 PPL 和生成示例 |

## 运行

运行环境为 Python 3.12、PyTorch 2.6.0，已在 CPU 上验证。请在克隆的仓库中进入本目录运行，完整克隆同时包含需要复用的 week04 文件。

```powershell
cd "陶欣荣/week05"
python -m pip install torch==2.6.0
python generate.py
```

输入问题并回车，输入 /exit 结束。每次问题独立处理。也可以单次运行。

```powershell
python generate.py --question "中国的首都是哪里？"
python generate.py --question "请解释什么是计算机。" --temperature 0.7 --top_k 40
python generate.py --checkpoint checkpoints/pretrained.pt --raw --question "今天的天气"
python test_model.py
```

默认生成采用贪心选择和 1.1 的重复惩罚，最多生成 128 个字符，或在预测到 EOS 时结束。模型的总上下文限制为 256 token，超过限制的问题会提示缩短，生成也会在上下文容量耗尽时停止。输出仍可能重复、跑题或包含事实错误，可在 evaluation.json 中检查真实结果。

## 复现训练

已提交本次实际使用的划分后数据。重新筛选数据时，prepare_data.py 会按需下载公开原始问答全集并校验指纹，原文本和初始化权重从相邻的 week04 目录读取。复现全部步骤时，以下命令按顺序执行。训练会写入对应阶段的权重和指标，最佳轮次由验证损失选择。

```powershell
python prepare_data.py --samples 6000 --seq_len 256 --seed 42
python train.py pretrain --epochs 1
python train.py sft --epochs 8
python test_model.py
```

两个训练阶段默认均为 batch size 32、学习率 0.0003、AdamW 优化器、0.1 dropout。所有参数参与更新，包括词嵌入和位置编码。训练脚本支持 --save 指定新的输出位置。SFT 使用 --init 指定预训练权重，默认读取 checkpoints/pretrained.pt。

## 数据与训练目标

原语料包含许多零散问题和句子，适合继续学习文本分布。原始共 395,286 行，去重后有 246,229 行，本次分为 233,917 行训练和 12,312 行验证。旧权重曾使用原语料训练，因此本次预训练验证 PPL 用于观察继续训练过程。

SFT 使用公开的 Alpaca GPT-4 中文合成指令数据，下载文件实际含 48,818 条记录。将 instruction 与非空 input 拼成问题，将 output 作为答案，按规范化后的问题去重，保留问题最多 100 字、答案最多 180 字、总长度适合 256 token 且词表外字符比例不超过 2% 的样本。固定抽取 6000 条，按 5400 / 300 / 300 分为训练、验证和测试集。超过长度限制的样本整体过滤，答案文本按来源保留。

问答格式为 `<bos>问题<sep>答案<eos>`。训练输入和目标错开一个 token，SFT 将问题、分隔标记和填充位置的目标设为 -100，只对答案及 EOS 计算损失。这样模型学习在给定问题后生成答案，并学会结束回答。词表外字符编码为 UNK。

PPL 为有效目标 token 的平均负对数概率取指数。预训练 PPL 统计文本的所有有效目标，SFT PPL 只统计答案和 EOS。对比预训练与 SFT 时，test_model.py 统一使用问答测试集和答案掩码。测试集用于训练结束后的评估，PPL 需要结合具体生成内容判断回答质量。

## 数据来源

中文指令数据来自 [Instruction-Tuning-with-GPT-4 / GPT-4-LLM](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM)，原始文件为 data/alpaca_gpt4_data_zh.json。原仓库说明已保存为 data/SOURCE_README.md，数据采用 CC BY-NC 4.0，项目说明限定研究用途。本目录用于教学实验。来源内容是合成问答，准确性与措辞可能存在问题。

## 本次实际训练结果

继续预训练完成 1 轮，文本验证 PPL 为 32.48。SFT 完成 8 轮，最佳权重为第 8 轮，问答验证答案 PPL 为 34.93。

| 问答测试集指标 | 继续预训练权重 | SFT 权重 |
| --- | --- | --- |
| 答案 PPL | 165.58 | 36.36 |
| 计分 token 数，包含 EOS | 23,369 | 23,369 |

问答测试集共 300 条，生成示例按固定顺序取前 8 条，完整记录在 evaluation.json。PPL 下降表示参考答案的平均概率上升。该实验模型的实际回答仍有明显事实错误、重复和跑题问题。

交互式实测中，输入“中国的首都是哪里？”，模型输出“印度最大的国家是美国的国家，它是一个国家的国家，它们在巴西里（Sange Arang）。”；输入“什么是人工智能？”，输出“我是一个人工智能助手，我无法确定您想要我为您提供帮助。”。这些输出说明模型已具备回答格式，但知识正确性和指令执行能力较弱。本实验完成训练流程，以结果记录为准。

## 提交内容

提交保留代码、两阶段最佳权重、逐轮指标、整理后的训练与评估数据以及生成结果。需要重新筛选数据时，原始问答全集由准备脚本下载。依赖的 week04 文件使用仓库内相对路径。
