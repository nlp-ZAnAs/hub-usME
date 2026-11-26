"""按顺序复现全部实验：下载 → 数据准备 → 规则 → 序列标注 → 本地 LLM → DeepSeek API → 汇总 → 测试。

--skip-existing 按「结果文件已存在」跳过整个方法（不是 epoch 级断点续训）；
没有 DEEPSEEK_API_KEY 时跳过 API 两组并在汇总表中标为未运行。
"""

import argparse
import subprocess
import sys

from common import ROOT

STEPS = [
    (["rule"], ["run_rule.py"]),
    (["bilstm_crf"], ["run_sequence.py", "--method", "bilstm_crf"]),
    (["roberta_linear", "roberta_linear_constrained"],
     ["run_sequence.py", "--method", "roberta_linear"]),
    (["roberta_crf", "roberta_crf_constrained"], ["run_sequence.py", "--method", "roberta_crf"]),
    (["qwen_zero"], ["run_llm_local.py", "--method", "qwen_zero"]),
    (["qwen_few"], ["run_llm_local.py", "--method", "qwen_few"]),
    (["qwen_lora"], ["run_llm_local.py", "--method", "qwen_lora"]),
]


def run(args):
    print("$", " ".join(args), flush=True)
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True)


def has_api_key():
    from run_api import find_api_key
    return find_api_key() is not None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true", help="不下载，要求本地资源已就绪")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--skip-api", action="store_true")
    args = parser.parse_args()
    if not args.offline:
        run(["download_assets.py"])
    if not (ROOT / "data" / "manifest.json").exists():
        run(["prepare_data.py"])
    for outputs, command in STEPS:
        if args.skip_existing and all((ROOT / "results" / f"{m}.json").exists() for m in outputs):
            print("skip", outputs, flush=True)
            continue
        run(command)
    if args.skip_api or not has_api_key():
        print("跳过 DeepSeek API（未配置 key 或指定了 --skip-api）", flush=True)
    else:
        for mode in ["zero", "few"]:
            # API 结果按样本缓存，重跑只补请求缺失样本，不会重复计费。
            run(["run_api.py", "--mode", mode])
    run(["summarize.py"])
    run(["-m", "unittest", "-v", "test_core.py"])


if __name__ == "__main__":
    main()
