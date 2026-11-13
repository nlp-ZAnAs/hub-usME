# 第五周作业一键脚本（固定用 conda py312 的解释器）
#   .\run.ps1 base     # 第一步：把第四周 tf_mask.pt 扩成 base_sft.pt
#   .\run.ps1 train    # 第二步：GPU 上跑 SFT
#   .\run.ps1 eval     # 第三步：base 与 SFT 并排对照
#   .\run.ps1 auto     # 一条龙（缺哪步补哪步）

param(
    [ValidateSet("base", "train", "eval", "auto")]
    [string]$Step = "auto",
    [int]$Epochs = 8
)

$PY = "C:\Users\Corey1016\.conda\envs\py312\python.exe"
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path

if (-not (Test-Path -LiteralPath $PY)) { throw "找不到解释器：$PY" }
Set-Location -LiteralPath $Here

& $PY -X utf8 -c "import torch;print('[check] torch',torch.__version__,'cuda',torch.cuda.is_available())"

switch ($Step) {
    "base"  { & $PY -X utf8 "build_base.py" }
    "train" { & $PY -X utf8 "train_sft.py" --epochs $Epochs }
    "eval"  { & $PY -X utf8 "eval_sft.py" }
    "auto" {
        if (-not (Test-Path -LiteralPath (Join-Path $Here "base_sft.pt"))) { & $PY -X utf8 "build_base.py" }
        & $PY -X utf8 "train_sft.py" --epochs $Epochs
        & $PY -X utf8 "eval_sft.py"
    }
}

if ($LASTEXITCODE -ne 0) { Write-Host "步骤 $Step 失败，退出码 $LASTEXITCODE" -ForegroundColor Red; exit $LASTEXITCODE }
Write-Host "步骤 $Step 完成" -ForegroundColor Green
