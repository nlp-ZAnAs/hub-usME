"""Run rules, train BERT, and compare joint and two-stage extraction."""
import argparse
import hashlib
import time
from collections import Counter
from contextlib import nullcontext
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from torch.nn import functional as F
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from data import DEFAULT_DATA, ROOT, SEED, load_rows, prediction, save_json, save_rows, split, text_key
from evaluate_events import evaluate
from rule import build_rules, extract_rule
from tagger import BERT, Tagger, decode, encode, make_labels, restore_weights

BATCH = 16


def device_for(name):
    return torch.device("cuda" if name == "auto" and torch.cuda.is_available()
                        else "cpu" if name == "auto" else name)


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def precision(device):
    return (torch.autocast("cuda", dtype=torch.bfloat16)
            if device.type == "cuda" and torch.cuda.is_bf16_supported() else nullcontext())


def put_result(out, name, gold, pred, seconds, **meta):
    result = evaluate(gold, pred, verbose=False)
    result.update({"seconds": seconds, **meta})
    save_json(out / "results" / f"{name}.json", result)
    save_rows(out / "results" / f"{name}_pred.jsonl", pred)
    print(name, {k: result[k]["f1"] for k in ("trigger", "argument", "event_exact")}, flush=True)
    return result


@torch.inference_mode()
def infer(model, tokenizer, schema, rows, device, types=None):
    model.eval()
    result = []
    for start in range(0, len(rows), BATCH):
        batch = rows[start:start+BATCH]
        inputs, offsets = encode(tokenizer, batch, device)
        with precision(device):
            logits = model(inputs).float().cpu().numpy()
        for i, row in enumerate(batch):
            kind = None if types is None else types[start+i]
            event = decode(row["text"], offsets[i], logits[i], model.labels, schema, kind=kind)
            result.append(prediction(row, event))
    return result


def train_bert(base, labels, schema, train, dev, device, epochs, out):
    model = Tagger(base, labels).to(device)
    tokenizer = AutoTokenizer.from_pretrained(base, use_fast=True)
    label_ids = {tag: i for i, tag in enumerate(labels)}
    encoder_params = [p for p in model.encoder.parameters() if p.requires_grad]
    trainable = encoder_params + list(model.head.parameters())
    opt = torch.optim.AdamW([{"params": encoder_params, "lr": 3e-4},
                             {"params": model.head.parameters(), "lr": 5e-4}],
                            weight_decay=0.01, fused=device.type == "cuda")
    steps = ((len(train) + BATCH - 1) // BATCH) * epochs
    scheduler = get_linear_schedule_with_warmup(opt, steps // 10, steps)
    weight = torch.full((len(labels),), 2., device=device)
    weight[0] = 1.
    rng = np.random.default_rng(SEED)
    history, step_history = [], []
    best = -1
    print(f"BERT trainable parameters: {sum(p.numel() for p in trainable):,}", flush=True)
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(train))
        losses = []
        for start in range(0, len(order), BATCH):
            rows = [train[i] for i in order[start:start+BATCH]]
            inputs, targets = encode(tokenizer, rows, device, label_ids)
            opt.zero_grad(set_to_none=True)
            with precision(device):
                logits = model(inputs)
                loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten(), weight=weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
            opt.step()
            scheduler.step()
            losses.append(float(loss.detach()))
            if len(losses) % 25 == 0:
                item = {"epoch": epoch, "step": len(step_history) + 1,
                        "batch":len(losses), "mean_last_25_loss":float(np.mean(losses[-25:]))}
                step_history.append(item)
                print(item, flush=True)
        scores = evaluate(dev, infer(model, tokenizer, schema, dev, device), verbose=False)
        item = {"epoch": epoch, "train_loss": float(np.mean(losses)),
                "dev_argument_f1": scores["argument"]["f1"],
                "dev_trigger_f1": scores["trigger"]["f1"],
                "elapsed_seconds":time.perf_counter()-started}
        history.append(item)
        if item["dev_argument_f1"] > best:
            best = item["dev_argument_f1"]
            state = {k: v.detach().cpu() for k, v in model.named_parameters() if v.requires_grad}
            torch.save({"state":state, "labels":labels, "epoch":epoch}, out / "checkpoints/bert_adapter.pt")
        save_json(out / "results/training.json", history)
        save_rows(out / "results/steps.jsonl", step_history)
        print(item, flush=True)
    saved = torch.load(out/"checkpoints/bert_adapter.pt", map_location="cpu", weights_only=True)
    restore_weights(model, saved["state"])
    return model, tokenizer, saved["epoch"]


def tfidf():
    return TfidfVectorizer(analyzer="char", ngram_range=(2, 4), min_df=2,
                           max_features=70000, sublinear_tf=True, dtype=np.float32)


def train_detector(data_dir, out, dev_ids):
    rows = load_rows(data_dir / "train_detect.jsonl")
    training = [r for r in rows if r["id"] not in dev_ids]
    holdout = [r for r in rows if r["id"] in dev_ids]
    train_texts = {text_key(r) for r in training}
    holdout = [r for r in holdout if text_key(r) not in train_texts]
    classifier = make_pipeline(tfidf(), LogisticRegression(C=2, max_iter=400, class_weight="balanced"))
    classifier.fit([r["text"] for r in training], [r["has_event"] for r in training])
    probs = classifier.predict_proba([r["text"] for r in holdout])[:, 1]
    truth = [r["has_event"] for r in holdout]
    threshold = max((0.30, 0.40, 0.50, 0.60, 0.70),
                    key=lambda t: f1_score(truth, probs >= t))
    test = [r for r in load_rows(data_dir / "dev_detect_clean.jsonl") if text_key(r) not in train_texts]
    pred = classifier.predict_proba([r["text"] for r in test])[:, 1] >= threshold
    p, r, f, _ = precision_recall_fscore_support([r["has_event"] for r in test], pred,
                                              average="binary", zero_division=0)
    joblib.dump({"classifier":classifier, "threshold":threshold}, out / "checkpoints/detector.joblib", compress=3)
    save_json(out / "results/detection.json", {"train":len(training), "dev":len(holdout), "test":len(test),
              "threshold":threshold, "precision":p, "recall":r, "f1":f})
    print("detector", f, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    p.add_argument("--output", type=Path, default=ROOT)
    p.add_argument("--bert", default=BERT)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--n-test", type=int, default=300)
    p.add_argument("--device", choices=["auto","cuda","cpu"], default="auto")
    a = p.parse_args()
    device = device_for(a.device)
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    out = a.output
    (out/"checkpoints").mkdir(parents=True,exist_ok=True)
    schema, train, dev, test, dev_ids, manifest = split(a.data_dir, a.n_test)
    save_json(out/"results/splits.json", manifest)
    save_json(out/"checkpoints/schema.json", schema)
    rules = build_rules(train)
    save_json(out/"checkpoints/rules.json", rules)
    start=time.perf_counter()
    pred = [prediction(r, extract_rule(r["text"], schema, rules)) for r in test]
    put_result(out,"rule",test,pred,time.perf_counter()-start,train_rows=len(train),device="cpu")
    start=time.perf_counter()
    type_model = make_pipeline(tfidf(),LinearSVC(C=2, class_weight="balanced", random_state=SEED))
    type_model.fit([r["text"] for r in train],[r["event"]["type"] for r in train])
    type_seconds = time.perf_counter()-start
    joblib.dump(type_model,out/"checkpoints/type.joblib",compress=3)
    model, tok, epoch = train_bert(a.bert,make_labels(schema),schema,train,dev,device,a.epochs,out)
    save_json(out/"results/environment.json", {"torch":str(torch.__version__),"cuda":torch.version.cuda,
              "device":torch.cuda.get_device_name(device) if device.type=="cuda" else "CPU",
              "bert_initial_files": {name:hashlib.sha256((Path(a.bert)/name).read_bytes()).hexdigest()
              for name in ("config.json","pytorch_model.bin","tokenizer.json") if (Path(a.bert)/name).is_file()}})
    for name in ("bert_joint","pipeline"):
        synchronize(device)
        start=time.perf_counter()
        types = type_model.predict([r["text"] for r in test]) if name=="pipeline" else None
        pred = infer(model,tok,schema,test,device,types)
        synchronize(device)
        put_result(out,name,test,pred,time.perf_counter()-start,best_epoch=epoch,device=str(device),
                   type_train_seconds=type_seconds if types is not None else None,
                   stage1_type_accuracy=float(np.mean(types==[r["event"]["type"] for r in test]))
                   if types is not None else None)
    train_detector(a.data_dir,out,dev_ids)


if __name__ == "__main__":
    main()
