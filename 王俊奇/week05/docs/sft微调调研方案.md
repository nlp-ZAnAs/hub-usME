# 调研方案：基于已有 Transformer 预训练权重做 SFT

## 一、背景与现状

### 1.1 现有工程

- `language_model.py`：字符级语言模型训练脚本，`load_corpus("*.txt")` 拼接纯文本 → `build_vocab(text, 4000)` 建字表 → 按行 `shuffle` 取 5% 验证 → `CharDataset` 定长切块 → `TransformerLM` → 全 token 交叉熵 → 保存 checkpoint。
- `pinyin_ime.py`：拼音输入法推理，内部复制了一份 `LM / PositionalEncoding / TransformerLM` 定义，按 checkpoint 里的 `args` 构建模型；`beam_search` 把 `prefix + partial` 整段送入模型，取最后一位 logits 打分。
- `best_model_transformer.pt` 已训练完成，checkpoint 含四个键：`model_state / char2idx / idx2char / args`。

### 1.2 预训练 checkpoint 实际超参

从权重文件读出，实施时一律以 `args` 为准，不得按文档默认值重建：

| 参数 | 值 | 参数 | 值 |
|------|-----|------|-----|
| model | transformer | embed_dim | 128 |
| nhead | 4 | num_layers | 2 |
| dim_feedforward | 256 | dropout | 0.1 |
| seq_len | 64 | stride | 64 |
| batch_size | 128 | lr | 3e-4 |
| max_vocab | 4000 | val_ratio | 0.05 |

`dim_feedforward` 实际为 256（早期方案文档里写的是 512），若按 512 重建模型会直接形状不匹配。词表大小 4000，`embed.weight` 为 4000×128，换行符 `\n` 在字表内，半角空格不在。

### 1.3 SFT 数据现状

- `sft_train.jsonl` 3000 条、`sft_val.jsonl` 200 条，字段均为 `instruction / output`；train 与 val 的指令无重复，指令也不出现在 `corpus_general.txt` 中。
- 相对旧字表共有 118 种未登录字符、出现 10145 次（半角空格、`@`、`[`、`]`、`=`、`√`、`×`、`÷`、`℃`、生僻字等）。
- 模板 `用户：{instruction}\n助手：{output}` 的 6 个模板字符（含冒号、换行）全部在旧字表内，无需新增 token。
- 模板长度：min 25 / 中位 65 / p90 92 / max 118。删除半角空格后，完全落在字表内的样本为 train 2452 条、val 168 条；保留样本长度 min 27 / 中位 62 / p90 88 / max 116，output 平均 26.3 字符、最长 64。

### 1.4 已经排除的做法

把 SFT 数据并入 `corpus_general.txt` 再训练，无论是否加载旧权重都不采用：全 token loss 没有 prompt 边界，产出仍是 base model；按行 `splitlines + shuffle` 会把 `用户：…` 与 `助手：…` 拆成两行独立打乱，QA 配对被破坏且验证集泄漏；`seq_len=64` 装不下一条 QA。预训练语料、原权重、原字表三者保持原样。

## 二、改造目标

1. 以 `best_model_transformer.pt` 为起点做指令微调，产出 `best_model_sft.pt`，原权重文件不被覆盖。
2. checkpoint 结构保持四个键不变，字表固定 4000、不重建、不扩表。
3. 复用已有的 `TransformerLM` 定义与 `run_epoch` 训练循环，不新增第三份模型定义；`pinyin_ime.py` 保持可用。
4. SFT 作为 `language_model.py` 的独立分支（`--sft`），预训练分支的行为与结果不变。
5. 验证只用 `sft_val.jsonl`，指标只统计 output 部分 token。

## 三、技术方案

### 3.1 模板与数据规范化：消除未登录字符的触发场景

固定模板：

```
用户：{instruction}\n助手：{output}\n
```

模板末尾的换行是回答结束符：训练时它参与 loss，生成时遇到它即停止；没有它模型会一直续写。推理时输入 prefix 为 `用户：{instruction}\n助手：`，模型从此处续写回答。

定义唯一的 `normalize(text)`：删除半角空格（数据里无全角空格），不做其他替换、不做静默过滤。规范化同时作用于 `instruction` 与 `output`，训练与推理共用同一函数，保证进入模型的字符必然在字表内。

规范化后若整条样本仍含字表外字符，则整条剔除，不做近似字符替换。原因是替换会制造错误标签，例如把 `3的平方根是√3` 的 `√` 删掉就变成 `3的平方根是3`。按此规则保留 train 2452 条、val 168 条。

`SFTDataset` 内直接用 `char2idx[c]` 取 id，不写 `if c in char2idx` 之类的过滤分支：合法性在数据预处理阶段已经保证，越界时直接报错比静默丢字符更容易发现问题。

### 3.2 样本构造与 loss mask

每条样本独立成块，不跨样本拼接：

- `ids = [char2idx[c] for c in normalize 后的模板文本]`，`x = ids[:-1]`，`y = ids[1:]`。
- `y` 中 prompt 部分（直到 `助手：` 结束，即 output 首字符之前）置 `-100`；右侧 padding 位置同样置 `-100`。只对 output 的 token 计算 loss，这才是 SFT 与继续预训练的区别。
- 统一 padding 到 `seq_len=128`（含结束符后最长样本 117）。pad 的 id 复用字表内已有字符（取 `。` 的 id），与结束符 `\n` 分开，避免结束信号被大量填充字符淹没；其在 `y` 上被 mask。因果掩码保证右侧 padding 不影响前面有效位置的 logits，因此不需要额外的 attention mask。
- `criterion` 改为 `nn.CrossEntropyLoss(ignore_index=-100)`；`run_epoch` 中 `total_tokens` 的分母由 `y.numel()` 改为 `(y != -100).sum().item()`，否则被 mask 的位置仍计入分母，PPL 会虚低。这两处改动对预训练分支无影响（预训练的 `y` 中没有 `-100`）。
- train / val 直接取两个 jsonl，不复用 `text.splitlines()` 的按行随机划分。

### 3.3 加载权重与训练流程

1. `cfg = torch.load("best_model_transformer.pt", map_location=device)`，取 `model_state / char2idx / idx2char / args`。
2. 用 `args` 里的结构超参构建 `TransformerLM`（`vocab_size=len(char2idx)`），`load_state_dict(cfg["model_state"])`，不重建字表、不重新初始化。
3. 新建 Adam 优化器（checkpoint 未保存优化器状态），按 SFT 学习率训练。
4. 每个 epoch 在 sft val 上算 output-only 的 loss / PPL，取最优保存 `best_model_sft.pt`；保存结构仍为四个键，`args` 以原 `args` 为基础覆盖训练超参并保留全部结构字段，保证 `pinyin_ime.py` 仍能按 `cfg["model"]` 等字段构建模型。
5. 模板字符串作为脚本级常量，推理侧 import 同一常量，不往 checkpoint 里加新键。

### 3.4 评估与生成验证

- 定量：sft val 168 条上的 output-only loss / PPL，与原模型在同一批数据上的同口径指标对比。
- 定性：抽查若干条 val 指令，自回归生成回答并与 `output` 对照。
- 生成不能直接调用 `pinyin_ime.py` 的 `beam_search`：它的候选字被 `pingyin_map` 限制在拼音表内，且无上文时走 unigram 字频先验，都是输入法专属逻辑。生成入口只复用"整段送入 `TransformerLM`、取最后一位 logits"这一打分方式，用 top-k 采样（k=20）逐字自回归：对最近 12 个已出现字符压低 logits，连续 3 字重复时提前停止，避免字符级模型陷入重复循环；把新字拼回 prefix，遇到 `\n` 或达到长度上限（40）停止。生成时对用户输入的指令同样调用 `normalize()`。

## 四、超参建议

| 参数 | 建议值 | 理由 |
|------|--------|------|
| lr | 1e-4 | 预训练为 3e-4，微调降一档，避免破坏已学到的字符分布 |
| epochs | 30 | 4 轮时 val PPL 从 107 降到 70 且仍在下降，属明显欠拟合 |
| batch_size | 32 | 小数据集，批次过大则每轮更新次数太少 |
| seq_len | 128 | 含结束符后保留样本最长 117；`max_len=512` 的位置编码足够 |
| sft_top_k | 20 | 生成抽查用采样而非贪心，配合重复惩罚抑制退化输出 |
| dropout | 0.1 | 沿用 checkpoint 结构超参 |
| 保存路径 | best_model_sft.pt | 不覆盖 `best_model_transformer.pt` |

## 五、改动清单（实施时执行）

1. `language_model.py`
   - 新增 `--sft` 开关及 `--sft_train / --sft_val / --model_path / --sft_save` 参数。
   - 新增 `SFT_TEMPLATE` 常量、`normalize()`、`load_sft_samples()`、`SFTDataset`（含 prompt mask 与 padding）。
   - `run_epoch` 的 token 分母改为有效 token 数；`criterion` 增加 `ignore_index=-100`。
   - `main()` 增加 SFT 分支：加载 checkpoint → 构建模型 → 装载权重 → 训练 → 按 sft val PPL 保存 `best_model_sft.pt`。
   - 新增生成入口（复用 `TransformerLM` 的最后一位 logits 打分）。
2. `pinyin_ime.py`：不改动，保持输入法可用；生成入口从 `language_model` import 模型定义与模板常量，不再复制一份模型类。
3. 训练命令：
   - `python language_model.py --sft --model_path best_model_transformer.pt --sft_train sft_train.jsonl --sft_val sft_val.jsonl --seq_len 128 --batch_size 32 --lr 1e-4 --epochs 30`
4. 效果核对：查看 sft val 的 output-only PPL 是否逐轮下降，再抽查生成样例。

## 六、风险与取舍

- 词表固定 4000，模型永远无法输出被规范化或剔除的符号（`@`、`√`、`×` 等）。已通过剔除含这些字符的样本消除"想生成却生成不出"的矛盾，这是本方案主动接受的边界，不做扩表或替代表。
- 规范化后样本降到 2452 / 168 条，且是字符级小模型，能学到的是固定模板与短答对齐；复杂计算、长推理类指令基本学不动，属预期结果。
- `sft_train.jsonl` 中存在错答案（如 `(8*4)/(2+6)` 标为 `1`、`12 和 25` 标为"不互质"），会被当成正确监督信号学进去。这是数据质量问题，应在数据侧清洗，不在代码里加修正或兜底逻辑。
- prompt 与 output 之间不新增分隔 token，边界完全由 `\n` 与 `助手：` 承担；因此模板一旦确定，训练、生成、评估三处必须共用同一个常量。
- 不做通用语料回放：本阶段目标是对齐指令格式，回放会重新引入按行 shuffle 与验证集泄漏，且与已确定的"不动预训练语料"冲突。

## 七、交互式问答入口调研（sft_chat.py）

### 7.1 目标

在已有 SFT 权重上提供命令行问答入口：输入一个问题，按训练模板生成回答。不修改 `language_model.py` 与 `pinyin_ime.py`。

### 7.2 复用与新增

- 复用 `language_model.py` 的 `TransformerLM`、`generate_answer`、`normalize`、`SFT_PROMPT`，不复制模型定义、不改动现有文件。
- 新增独立脚本 `sft_chat.py`：按 checkpoint 的 `args` 构建模型（推理 `dropout=0.0`）、`load_state_dict`、`model.eval()`，提供 `--model_path / --top_k / --gen_len` 参数。
- 生成完全走 `generate_answer`（top-k 采样 + 重复惩罚 + 结束符停止），与训练后的抽查口径一致，避免两处解码逻辑分叉。

### 7.3 输入约束

用户输入不可控，而字表固定 4000。交互入口在调用生成前显式检查所有字符，列出字表外字符并要求重输，不做静默替换、不做字符兜底；半角空格由 `normalize()` 统一删除。这样 `generate_answer` 的越界异常不会被触发。

### 7.4 运行

- `python sft_chat.py`
- `python sft_chat.py --top_k 1`（确定性贪心，仍带重复惩罚）
- `python sft_chat.py --model_path best_model_transformer.pt`（对比未微调的预训练模型）

### 7.5 已知限制

当前 SFT 模型 val PPL 33.93，输出仍可能答非所问，这是数据量与模型容量的上限（见第六节），交互入口无法改善生成质量，只负责把已有能力暴露出来。
