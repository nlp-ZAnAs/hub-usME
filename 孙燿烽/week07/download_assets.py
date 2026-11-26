"""下载 CLUENER2020 官方压缩包与两个预训练模型；首次锁定版本，之后按同一 revision/SHA256 校验。"""

import argparse
import os
import zipfile
from urllib.request import Request, urlopen

from common import ROOT, config, file_sha256, read_json, write_json

DATASET_URL = "https://storage.googleapis.com/cluebenchmark/tasks/cluener_public.zip"


def download_dataset(manifest):
    target = ROOT / ".cache" / "cluener" / "cluener_public.zip"
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        print("Downloading CLUENER2020", flush=True)
        request = Request(DATASET_URL, headers={"User-Agent": "week07-homework"})
        with urlopen(request, timeout=120) as response:
            target.write_bytes(response.read())
    checksum = file_sha256(target)
    expected = manifest.setdefault("dataset", {"url": DATASET_URL}).get("sha256", checksum)
    if checksum != expected:
        raise ValueError("CLUENER 压缩包 SHA256 与锁定版本不一致")
    manifest["dataset"]["sha256"] = checksum
    extracted = target.parent / "extracted"
    with zipfile.ZipFile(target) as archive:
        names = {"train.json", "dev.json", "README.md"}
        if not names <= set(archive.namelist()):
            raise ValueError("压缩包缺少 train.json / dev.json")
        for name in names:
            archive.extract(name, extracted)
    manifest["dataset"]["files_sha256"] = {
        name: file_sha256(extracted / name) for name in sorted(names)}


def download_models(manifest, cfg):
    # 主站在本机网络下经常超时；镜像与主站内容同源，revision 照样锁定。
    os.environ.setdefault("HF_ENDPOINT", cfg["hf_endpoint"])
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from huggingface_hub import HfApi, snapshot_download

    models = manifest.setdefault("models", {})
    for key in ["encoder_model", "llm_model"]:
        model_id = cfg[key]
        if model_id not in models:
            models[model_id] = {"revision": HfApi().model_info(model_id).sha,
                                "endpoint": os.environ["HF_ENDPOINT"]}
            write_json(ROOT / "assets_manifest.json", manifest)
        destination = ROOT / ".cache" / model_id.split("/")[-1]
        patterns = ["*.json", "*.txt", "*.safetensors", "README.md"]
        if key == "encoder_model":
            patterns.append("pytorch_model.bin")
        print(f"Downloading {model_id}", flush=True)
        snapshot_download(model_id, revision=models[model_id]["revision"], local_dir=destination,
                          allow_patterns=patterns, max_workers=4)
        checksums = {f.name: file_sha256(f) for f in sorted(destination.iterdir()) if f.is_file()}
        expected = models[model_id].get("files_sha256")
        if expected is not None and checksums != expected:
            raise ValueError(f"{model_id}: 模型文件 SHA256 与锁定版本不一致")
        models[model_id]["files_sha256"] = checksums
        write_json(ROOT / "assets_manifest.json", manifest)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-only", action="store_true")
    args = parser.parse_args()
    path = ROOT / "assets_manifest.json"
    manifest = read_json(path) if path.exists() else {}
    download_dataset(manifest)
    write_json(path, manifest)
    if not args.data_only:
        download_models(manifest, config())
    print("Assets ready; exact versions saved in assets_manifest.json", flush=True)


if __name__ == "__main__":
    main()
