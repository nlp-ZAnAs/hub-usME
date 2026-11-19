"""下载公开数据与预训练权重；首次锁定版本，之后使用同一 revision 和 SHA256。"""

import argparse
import json
import os
from urllib.request import Request, urlopen

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from common import ROOT, config, file_sha256, read_json, write_json

DATA_PATH = "datasets/ChnSentiCorp_htl_all/ChnSentiCorp_htl_all.csv"
REPO = "SophonPlus/ChineseNlpCorpus"


def fetch_json(url):
    with urlopen(Request(url, headers={"User-Agent": "week06-homework"}), timeout=60) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-only", action="store_true")
    args = parser.parse_args()
    cfg = config()
    path = ROOT / "assets_manifest.json"
    manifest = read_json(path) if path.exists() else {"models": {}}
    if "dataset" not in manifest:
        rev = fetch_json(f"https://api.github.com/repos/{REPO}/commits?path={DATA_PATH}&per_page=1")[0]["sha"]
        manifest["dataset"] = {
            "repo": REPO, "revision": rev,
            "url": f"https://raw.githubusercontent.com/{REPO}/{rev}/{DATA_PATH}",
        }
    raw = ROOT / ".cache" / "ChnSentiCorp_htl_all.csv"
    raw.parent.mkdir(parents=True, exist_ok=True)
    if not raw.exists():
        print("Downloading dataset", flush=True)
        with urlopen(manifest["dataset"]["url"], timeout=60) as response:
            raw.write_bytes(response.read())
    checksum = file_sha256(raw)
    expected = manifest["dataset"].get("sha256", checksum)
    if checksum != expected:
        raise ValueError("数据 SHA256 不一致")
    manifest["dataset"]["sha256"] = checksum
    write_json(path, manifest)
    if args.data_only:
        return

    from huggingface_hub import HfApi, snapshot_download
    for key in ["encoder_model", "llm_model"]:
        model_id = cfg[key]
        if model_id not in manifest["models"]:
            info = HfApi().model_info(model_id)
            manifest["models"][model_id] = {"revision": info.sha}
            write_json(path, manifest)
        destination = ROOT / ".cache" / model_id.split("/")[-1]
        patterns = ["*.json", "*.txt", "*.safetensors", "README.md"]
        if key == "encoder_model":
            patterns.append("pytorch_model.bin")
        print(f"Downloading {model_id}", flush=True)
        snapshot_download(model_id, revision=manifest["models"][model_id]["revision"],
                          local_dir=destination, allow_patterns=patterns, max_workers=4)
        checksums = {
            f.name: file_sha256(f) for f in sorted(destination.iterdir()) if f.is_file()
        }
        expected = manifest["models"][model_id].get("files_sha256")
        if expected is not None and checksums != expected:
            raise ValueError(f"{model_id}: 模型文件 SHA256 与锁定版本不一致")
        manifest["models"][model_id]["files_sha256"] = checksums
        write_json(path, manifest)
    print("Assets ready; exact revisions saved in assets_manifest.json", flush=True)


if __name__ == "__main__":
    main()
