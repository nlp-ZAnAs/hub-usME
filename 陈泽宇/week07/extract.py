"""Extract events from one segment or route a full announcement through the detector."""
import argparse
import json
import re
from pathlib import Path

import joblib

from data import DEFAULT_DATA, ROOT, load_rows, read_json, save_json
from tagger import BERT, load_tagger
from train import device_for, infer


def segments(text):
    for match in re.finditer(r"[^。！？；\r\n]+[。！？；]?", text):
        raw = match[0]
        content = raw.strip()
        if content:
            yield match.start() + len(raw) - len(raw.lstrip()), content


def document_events(text, schema, model, tok, detector, classifier, device):
    parts = list(segments(text))
    if not parts:
        return {"segments": [], "events": []}
    probabilities = detector["classifier"].predict_proba([p[1] for p in parts])[:, 1]
    accepted = [i for i, p in enumerate(probabilities) if p >= detector["threshold"]]
    records = [{"start": offset, "end": offset + len(segment),
                "detector_score": round(float(probabilities[i]), 4),
                "passed_detection": i in accepted}
               for i,(offset,segment) in enumerate(parts)]
    if not accepted:
        return {"segments": records, "events": []}
    rows = [{"id": "document", "sent_id": str(i), "text": parts[i][1]} for i in accepted]
    preds = infer(model, tok, schema, rows, device, classifier.predict([r["text"] for r in rows]))
    events = []
    seen = set()
    for i, pred in zip(accepted, preds, strict=True):
        offset = parts[i][0]
        records[i]["found_event"] = bool(pred["events"])
        records[i]["needs_review"] = not pred["events"] or not pred["events"][0]["args"]
        for ev in pred["events"]:
            key = (ev["type"], ev["trigger"]["text"],
                   tuple(sorted((a["role"], a["text"]) for a in ev["args"])))
            if key in seen:
                continue
            seen.add(key)
            for span in [ev["trigger"], *ev["args"]]:
                span["start"] += offset
                span["end"] += offset
                assert text[span["start"]:span["end"]] == span["text"]
            events.append({**ev, "segment_index":i, "needs_review":records[i]["needs_review"]})
    return {"segments":records, "events":events}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument("--text",help="One sentence with at most one event")
    group.add_argument("--file",type=Path,help="UTF-8 announcement")
    group.add_argument("--demo",action="store_true",help="Run the first three course dev documents")
    p.add_argument("--data-dir",type=Path,default=DEFAULT_DATA)
    p.add_argument("--bert",default=BERT)
    p.add_argument("--artifacts",type=Path,default=ROOT)
    p.add_argument("--out",type=Path)
    p.add_argument("--device",choices=["auto","cuda","cpu"],default="auto")
    a=p.parse_args()
    ckpt=a.artifacts/"checkpoints"
    device=device_for(a.device)
    schema=read_json(ckpt/"schema.json")
    model,tok=load_tagger(a.bert,ckpt/"bert_adapter.pt",device)
    classifier=joblib.load(ckpt/"type.joblib")
    if a.text:
        row={"id":"input","sent_id":"input","text":a.text}
        result=infer(model,tok,schema,[row],device,classifier.predict([a.text]))[0]
    else:
        detector=joblib.load(ckpt/"detector.joblib")
        if a.file:
            result=document_events(a.file.read_text(encoding="utf-8-sig"),schema,model,tok,detector,classifier,device)
        else:
            docs=load_rows(a.data_dir/"dev_docs.jsonl")[:3]
            result=[{"id":row["id"],"title":row["title"],
                     **document_events(row["text"],schema,model,tok,detector,classifier,device)}
                    for row in docs]
    if a.out:
        save_json(a.out,result)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
