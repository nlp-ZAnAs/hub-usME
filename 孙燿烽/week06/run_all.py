"""一条命令复现；已完成且数据/配置/逐条指标校验通过的结果可以跳过。"""

import argparse
import subprocess
import sys

from common import METHODS, ROOT, load_data, read_json
from summarize import validate_result


def run(*args):
    print("+", " ".join(args), flush=True)
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--offline", action="store_true", help="不联网下载，要求本地数据和模型已齐备")
    args = parser.parse_args()
    if not args.offline:
        run("download_assets.py")
    if not (ROOT / "data" / "manifest.json").exists():
        run("prepare_data.py")
    splits = load_data()

    def done(method):
        path = ROOT / "results" / (method + ".json")
        if not args.skip_existing or not path.exists():
            return False
        validate_result(read_json(path), splits)
        print("Verified existing result:", method, flush=True)
        return True

    if not all(done(method) for method in list(METHODS)[:5]):
        run("run_baselines.py")
    for method in ["rbt3", "qwen_zero", "qwen_lora", "qwen_full"]:
        if not done(method):
            run("run_neural.py", "--method", method)
    run("summarize.py")


if __name__ == "__main__":
    main()
