# 第 6 周作业 —— 一次跑完三种方法并出图
# 用法：powershell -ExecutionPolicy Bypass -File .\run_all.ps1
#       或逐步执行下面四行命令，便于观察每一步的输出

$ErrorActionPreference = "Stop"
$PY = "C:\Users\Corey1016\.conda\envs\py312\python.exe"
Set-Location $PSScriptRoot

Write-Host "=== 0) 配置自检 ===" -ForegroundColor Cyan
& $PY config.py

Write-Host "`n=== 1) 方法一：规则/关键词（L0）===" -ForegroundColor Cyan
& $PY classify_rule.py --n 500

Write-Host "`n=== 2) 方法二：TF-IDF + LinearSVC（L1）===" -ForegroundColor Cyan
& $PY train_tfidf.py --n 500

Write-Host "`n=== 3) 方法三：BERT 微调（L2，约 20~40 分钟）===" -ForegroundColor Cyan
if (Test-Path ".\models\bert_ckpt.pt") {
    Write-Host "已存在 bert_ckpt.pt → 只重新评测（想重训删掉该文件或加 --epochs 3 手动跑）" -ForegroundColor Yellow
    & $PY train_bert.py --n 500 --eval_only
} else {
    & $PY train_bert.py --n 500 --epochs 3
}

Write-Host "`n=== 4) 可视化对比 ===" -ForegroundColor Cyan
& $PY plot_compare.py

Write-Host "`n=== 5) 类别加权 A/B（同一份数据只改 loss 权重，约 15 分钟）===" -ForegroundColor Cyan
if (Test-Path ".\models\bert_ckpt_nw.pt") {
    Write-Host "已存在 bert_ckpt_nw.pt → 只重新评测" -ForegroundColor Yellow
    & $PY train_bert.py --n 500 --eval_only --no_class_weight --tag _nw
} else {
    & $PY train_bert.py --n 500 --epochs 3 --no_class_weight --tag _nw
}
& $PY plot_class_weight.py

Write-Host "`n全部完成：图表在 figures\，指标表在 outputs\summary.md 与 outputs\class_weight_compare.md" -ForegroundColor Green
