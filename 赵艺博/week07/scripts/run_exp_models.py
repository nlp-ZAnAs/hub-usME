"""一键：规则 + BERT/RoBERTa Linear 换模型对照（XPU）"""
import subprocess, sys
from pathlib import Path
CODE = Path(r"d:\BaiduNetdiskDownload\Agent开发学习资料\第七周 抽取任务\week7-信息抽取\week7-信息抽取\code")
PY = sys.executable
def run(args):
    print(">>>", " ".join(args), flush=True)
    subprocess.run([PY, *args], cwd=str(CODE), check=False)
run(["rule/run_rule.py"])
run(["bert/train.py","--device","xpu","--decoder","linear","--model","bert-base-chinese","--epochs","3","--run_name","hw_bert_linear_256"])
run(["bert/train.py","--device","xpu","--decoder","linear","--model","chinese-roberta-wwm-ext","--epochs","3","--run_name","hw_roberta_linear_256"])
print("DONE")
