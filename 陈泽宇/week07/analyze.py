"""Recompute the comparison, verify spans and write a ten-case analysis."""
import argparse
import csv
from collections import Counter
from pathlib import Path

from data import DEFAULT_DATA, ROOT, load_rows, read_json, save_json, split
from evaluate_events import evaluate

METHODS = ("rule", "bert_joint", "pipeline", "qwen_zero", "qwen_few")


def error_kind(gold, pred):
    if pred is None:
        return "漏抽事件"
    if gold["type"] != pred["type"]:
        return "触发词歧义或类型误判"
    if gold["trigger"]["text"] != pred["trigger"]["text"]:
        return "触发词边界"
    g = {(a["role"], a["text"]) for a in gold["args"]}
    p = {(a["role"], a["text"]) for a in pred["args"]}
    if g == p:
        return "完全匹配"
    wrong = [(r1, r2) for r1, v1 in g-p for r2, v2 in p-g if v1==v2 and r1!=r2]
    if wrong:
        return "角色串味"
    if any(r1==r2 and v1!=v2 and (v1 in v2 or v2 in v1)
           for r1,v1 in g-p for r2,v2 in p-g):
        return "论元边界"
    return "论元漏抽或多抽"


def describe(event):
    if event is None:
        return "空"
    arguments = "；".join(f"{arg['role']}＝{arg['text']}" for arg in event["args"])
    return f"{event['type']}｜{event['trigger']['text']}｜{arguments or '无论元'}"


def explain(out, test, predictions, metrics):
    best = max(METHODS, key=lambda name: metrics[name]["argument"]["f1"])
    buckets = {}
    for name in METHODS:
        for row, result in zip(test,predictions[name],strict=True):
            event = result["events"][0] if result["events"] else None
            group=error_kind(row["event"],event)
            if group!="完全匹配":
                buckets.setdefault(group,[]).append((name,row,event))
    sample = []
    used = set()
    # Cycle through error categories, preferring a different method per pass.
    for turn in range(10):
        for group, items in buckets.items():
            candidates = [entry for entry in items if entry[1]["sent_id"] not in used]
            if not candidates:
                continue
            preferred = METHODS[turn % len(METHODS)]
            name,row,event = next((e for e in candidates if e[0] == preferred), candidates[0])
            sample.append((group,name,row,event))
            used.add(row["sent_id"])
            if len(sample) == 10:
                break
        if len(sample) == 10:
            break
    notes = [
        "# 十条失败样本",
        "",
        "按类型误判、触发词边界、角色串味、论元边界及漏抽或多抽归类。"
        "为覆盖不同现象，从五种方法的同一评估集选取案例，分类规则优先记录类型与触发词问题。",
        "",
    ]
    for i,(group,name,row,event) in enumerate(sample,start=1):
        notes.extend([f"## {i}. {group}（{name}）","",
                      f"sent_id `{row['sent_id']}`","",
                      f"- 金标　{describe(row['event'])}",
                      f"- 预测　{describe(event)}",""])
    (out/"results"/"badcases.md").write_text("\n".join(notes),encoding="utf-8")
    for name in METHODS:
        evs=[p["events"][0] if p["events"] else None for p in predictions[name]]
        metrics[name]["errors"]=dict(Counter(error_kind(r["event"],e) for r,e in zip(test,evs,strict=True)))
    facts = {"best_argument_method":best, "methods":metrics}
    save_json(out/"results"/"comparison.json",facts)
    with (out/"results"/"comparison.csv").open("w",encoding="utf-8",newline="") as f:
        writer=csv.writer(f)
        writer.writerow(["method","trigger_f1","argument_f1","event_exact_f1","type_macro_f1","ms_per_item"])
        for name in METHODS:
            r=metrics[name]
            writer.writerow([name,r["trigger"]["f1"],r["argument"]["f1"],
                             r["event_exact"]["f1"],r["type_macro_f1"],r["seconds"]*1000/len(test)])
    return best


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir",type=Path,default=DEFAULT_DATA)
    p.add_argument("--output",type=Path,default=ROOT)
    a=p.parse_args()
    manifest=read_json(a.output/"results/splits.json")
    schema,train,dev,test,_,actual=split(a.data_dir,manifest["counts"]["test"])
    assert manifest==actual
    ids=[{r["id"] for r in rows} for rows in (train,dev,test)]
    assert not (ids[0]&ids[1] or ids[1]&ids[2] or ids[0]&ids[2])
    metrics={}
    predictions={}
    for name in METHODS:
        rescored=load_rows(a.output/"results"/f"{name}_pred.jsonl")
        assert [r["sent_id"] for r in rescored]==[r["sent_id"] for r in test]
        target=read_json(a.output/"results"/f"{name}.json")
        for row, pr in zip(test,rescored,strict=True):
            for event in pr["events"]:
                assert event["type"] in schema
                for span in [event["trigger"],*event["args"]]:
                    assert row["text"][span["start"]:span["end"]]==span["text"]
                assert all(arg["role"] in schema[event["type"]] for arg in event["args"])
        scored=evaluate(test,rescored,verbose=False)
        assert all(scored[k]==target[k] for k in scored)
        predictions[name]=rescored
        metrics[name]=target
    print("Verified scores and text offsets. Highest argument F1:",explain(a.output,test,predictions,metrics))


if __name__=="__main__":
    main()
