"""Fixed splits and data I/O for the course DuEE-Fin benchmark."""
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT.parent / "code" / "data" / "processed"
SEED = 42
TRIGGER = "@trigger"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def load_rows(path):
    with Path(path).open(encoding="utf-8-sig") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def save_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def text_key(row):
    return "".join(row["text"].split())


def row_key(row):
    return row["sent_id"]


def split(data_dir, n_test=300):
    files = {n: data_dir / n for n in (
        "event_schema.json", "train_extract.jsonl", "dev_extract_clean.jsonl",
        "train_detect.jsonl", "dev_detect_clean.jsonl")}
    schema = {k: v for k, v in read_json(files["event_schema.json"]).items()
              if not k.startswith("__")}
    train_raw = load_rows(files["train_extract.jsonl"])
    clean = {}
    for row in train_raw:
        clean.setdefault(text_key(row), row)
    train_all = list(clean.values())
    by_doc = defaultdict(list)
    for row in train_all:
        by_doc[row["id"]].append(row)
    docs = sorted(by_doc)
    random.Random(SEED).shuffle(docs)
    dev_ids = set(docs[:len(docs) // 10])
    train = [r for r in train_all if r["id"] not in dev_ids]
    dev = [r for r in train_all if r["id"] in dev_ids]
    blocked_ids = {r["id"] for r in train_raw}
    test_all = [r for r in load_rows(files["dev_extract_clean.jsonl"])
                if r["id"] not in blocked_ids and text_key(r) not in clean]
    if n_test < len(test_all):
        test, _ = train_test_split(test_all, train_size=n_test, random_state=SEED,
                                   stratify=[r["event"]["type"] for r in test_all])
    else:
        test = test_all
    counts = {name: len(rows) for name, rows in (("train", train), ("dev", dev), ("test", test))}
    manifest = {
        "seed": SEED, "source_train": len(train_raw), "unique_train": len(train_all),
        "eligible_test": len(test_all), "counts": counts,
        "file_sha256": {k: hashlib.sha256(v.read_bytes()).hexdigest() for k, v in files.items()},
        "sent_ids": {name: [row_key(r) for r in rows]
                     for name, rows in (("train", train), ("dev", dev), ("test", test))},
    }
    return schema, train, dev, test, dev_ids, manifest


def prediction(row, event, **extra):
    return {"id": row["id"], "sent_id": row_key(row), "events": [event] if event else [], **extra}


def source_span(text, start, end):
    return {"start": start, "end": end, "text": text[start:end]}
