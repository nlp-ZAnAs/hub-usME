# week07 · 信息抽取（事件抽取）作业

对应课件：`D:\BaiduNetdiskDownload\week7-信息抽取`（PPT 43 页 + `code/` 六条方法链路）
本目录只放**作业脚本与产出**，训练/评测代码仍在课件的 `code/` 里（不改逻辑）。

## 文件

| 文件 | 作业项 | 说明 |
|---|---|---|
| `analyze_badcase.py` | 作业 2 | 预测 vs 金标逐句对齐，把错误归到「漏抽/类型错/边界/角色串味/多余/缺角色」；对齐键 `id|sent_id` |
| `badcase.md` | 作业 2 | 674 条错例分布 + 10 条案例（自动判定 + 人工复核）+ 三类处方 |
| `compare_chart.py` | 作业 1 | 读 `code/outputs/metrics/disc/*.json` 出六路对比图；LLM 三行来自课件 `outputs/llm`，图上已标注 |
| `compare_methods.png` | 作业 1 | 左 dev 930（本机实跑），右 n=200（后三行为课件结果） |
| `plot_loss.py` | 作业 5 | 读 `code/outputs/logs/*_steps.jsonl`，量级差 >10 倍自动拆面板（CRF 的 loss 是 logZ−score） |
| `loss_curve.png` | 作业 5 | BERT+Linear 与 BERT+CRF 双面板 loss 曲线，虚线为 epoch 边界 |
| `demo_events.json` | 作业 3 | `service/demo_extract.py --n 3` 输出的篇章事件流（带全文 offset / confidence / needs_review） |

## 跑一遍

```powershell
$py = "C:\Users\Corey1016\.conda\envs\py312\python.exe"
$code = "D:\BaiduNetdiskDownload\week7-信息抽取\code"
cd $code

# 作业 3：篇章事件流（CPU 足够，3 篇约 30 s）
& $py service/demo_extract.py --n 3 --device cpu --joint_ckpt outputs/checkpoints/bert_crf_full.pt --out outputs/demo/week7_homework_events.json

cd D:\BaiduNetdiskDownload\AILearningSpace\Python\hub-usME\庄群新\week07

# 作业 2
& $py analyze_badcase.py --pred "$code\outputs\disc\pred_bert_linear_full.jsonl" `
    --gold "$code\data\processed\dev_extract.jsonl" --n 10 --out badcase_raw.md

# 作业 5
& $py plot_loss.py --logs "$code\outputs\logs" --names bert_linear_full bert_crf_full --out loss_curve.png

# 作业 1
& $py compare_chart.py
```

## 本机口径（写报告时必须带上）

- 4 GB 显存：`batch 8 + grad_accum 2` → 970 步；课件 `batch 32` → 486 步，所以本机绝对指标偏高，**只做本机内部横向比较**
- LLM 两路**未在本机实跑**（无 API key，HF 下载超时），表中数据来自课件 `code/outputs/llm`
- 课件代码两处环境适配：`BertTokenizer` → `BertTokenizerFast`（慢版无 `word_ids()`）、`pretrain_models/bert-base-chinese` 硬链接到 week4 权重

---

# 落地实测：真实 A 股快讯的事件抽取（可视化）

用训好的 `bert_crf_full.pt` 去抽网上抓来的真实快讯，看看它在**域外文本**上到底行不行。

| 文件 | 说明 |
|---|---|
| `extract_ui.py` | **可视化程序**：`--serve` 起网页（粘贴文本→高亮+事件卡片+原 JSON），`--batch` 批量跑 jsonl |
| `crawl_ashare_events.py` | 抓取脚本：东方财富 7×24 快讯 + 新浪财经 7×24，按 13 类事件关键词筛选去重 |
| `articles.jsonl` | 抓到的 50 条真实文本（含 url / 来源 / 时间） |
| `results.json` | 50 条逐篇抽取结果（事件流 + 全文 offset + 置信度 + 延迟） |
| `render_report.py` | 渲染报告：`report_correct.html` / `report_wrong.html` + `stats.png`，并用 Edge 无头模式截图 |
| `result_correct.png` | 抽对的公司事件（原文彩色高亮 + 论元表） |
| `result_wrong.png` | 误抽对照 + 真漏抽（模型的边界） |
| `stats.png` | 统计总览：精确率 / 召回 / 拒识 / 置信度分布 / 延迟 |
| `ui_screenshot.png` | 网页程序界面截图 |

## 跑一遍

```powershell
$py = "C:\Users\Corey1016\.conda\envs\py312\python.exe"
cd D:\BaiduNetdiskDownload\AILearningSpace\Python\hub-usME\庄群新\week07

# 1) 抓 50 条（两个源，约 10 s）
& $py crawl_ashare_events.py --target 50 --pages 6 --out articles.jsonl

# 2) 可视化界面（浏览器开 http://127.0.0.1:8765）
& $py extract_ui.py --serve --open

# 3) 批量抽 + 出报告图
& $py extract_ui.py --batch articles.jsonl --out results.json
& $py render_report.py --results results.json --outdir .
```

## 实测结论（50 条语料，人工逐条核验）

| 指标 | 值 | 说明 |
|---|---|---|
| 抽对 | 12 篇 | 回购 / 增持 / 收购 / 融资类公告，触发词与论元基本都对 |
| 误抽 | 21 篇 | 政策文件、人物稿、ETF 点评、研报 —— 出现「回购/并购/任命」等词就硬抽，且置信度 0.93~1.00、论元常为空 |
| 该沉默且沉默 | 15 篇 | 宏观数据、行情点评、会议报道 |
| 漏抽 | 2 篇 | 重庆信托「代为履行董事长职务」（真高管变动）、中国化学「新签合同额 3132.78 亿」 |
| 篇级精确率 / 召回率 / 拒识率 | 36% / 86% / 42% | **召回强、精确率弱** —— 训练语料全是「含事件的公告」，模型没学会「什么时候该沉默」 |
| 延迟 | 平均 122 ms/条（第 1 条含预热 1.9 s） | 单条 96~549 字，GPU 3050Ti |

**上线必须补的两道闸**（已在 `render_report.py` 的说明里给出）：
1. **公司主体校验**：事件论元里没有任何上市公司名/证券代码 → 丢弃；
2. **域过滤**：政策发布、研报、行情点评类文本先拦掉（可用关键词或小分类器）。

> 附带发现：`老挝正式退出联合国最不发达国家行列` 被判成「企业融资」（conf 0.97）、
> `黄仁勋重回英伟达` 这类人物稿被判成「高管变动」（conf 0.94）——
> 高置信度不等于正确，这正是课件 L5「模型会抽错但看起来对」的现场证据。

