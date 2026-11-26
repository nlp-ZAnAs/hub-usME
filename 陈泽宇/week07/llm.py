"""Local Qwen extraction with a common prompt and source-grounded validation."""
import argparse
import json
import re
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from data import DEFAULT_DATA, ROOT, prediction, read_json, source_span, split
from train import device_for, put_result, synchronize

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
MAX_NEW = 192
BATCH = 16


def system_prompt(schema):
    lines = [f"{kind}：{'、'.join(roles)}" for kind, roles in schema.items()]
    return ("从中文金融公告中抽取一个事件。类型和角色只能用下表中的名称。"
            "触发词和论元必须逐字复制原文，原文没有就省略。"
            '只返回一个JSON对象，格式为{"type":"事件类型","trigger":"原文触发词",'
            '"args":[["角色","原文论元"]]}。无事件时返回{}。\n' + "\n".join(lines))


def example_answer(event):
    return json.dumps({"type":event["type"],"trigger":event["trigger"]["text"],
                       "args":[[a["role"],a["text"]] for a in event["args"]]},
                      ensure_ascii=False,separators=(",",":"))


def prompt(schema, text, examples=()):
    messages = [{"role":"system","content":system_prompt(schema)}]
    for row in examples:
        messages.extend(({"role":"user","content":row["text"]},
                         {"role":"assistant","content":example_answer(row["event"])}))
    return [*messages, {"role":"user","content":text}]


def parse(text, raw, schema):
    """Parse a single object and check every extracted substring and role."""
    s = raw.strip()
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", s, re.S | re.I)
    if match:
        s = match[1]
    try:
        obj, end = json.JSONDecoder().raw_decode(s)
    except ValueError:
        return None, "invalid_json", 0
    if not isinstance(obj, dict) or s[end:].strip():
        return None, "invalid_json", 0
    if not obj:
        return None, "empty", 0
    kind = obj.get("type")
    trigger = obj.get("trigger")
    args = obj.get("args")
    if not isinstance(kind, str) or kind not in schema or not isinstance(args, list):
        return None, "invalid_schema", 0
    start = text.find(trigger) if isinstance(trigger, str) and trigger else -1
    if start < 0:
        return None, "trigger_outside_source", 0
    placed = source_span(text, start, start + len(trigger))
    kept, seen, dropped = [], set(), 0
    for arg in args:
        if not (isinstance(arg, list) and len(arg)==2 and all(isinstance(v,str) for v in arg)):
            dropped += 1
            continue
        role, value = arg
        pos = text.find(value) if value else -1
        key = (role, value)
        if role not in schema[kind] or pos < 0 or key in seen:
            dropped += 1
            continue
        seen.add(key)
        kept.append({**source_span(text,pos,pos+len(value)),"role":role})
    return {"type":kind,"trigger":placed,"args":kept}, "ok", dropped


@torch.inference_mode()
def run(model, tokenizer, device, schema, train, test, out, n_shots, vectorizer):
    if n_shots:
        bank = vectorizer.transform([row["text"] for row in train])
        queries = vectorizer.transform([row["text"] for row in test])
        # Fit the vocabulary and IDF exclusively on training; select two examples at inference.
        sims = (queries @ bank.T).toarray()
        indices = np.argsort(-sims, axis=1, kind="stable")[:, :n_shots]
    else:
        indices = [()] * len(test)
    pred = []
    finished = 0
    max_prompt = 0
    synchronize(device)
    started = time.perf_counter()
    eot = tokenizer.convert_tokens_to_ids("<|im_end|>")
    for i in range(0,len(test),BATCH):
        batch = test[i:i+BATCH]
        prompts = []
        for row, picks in zip(batch,indices[i:i+BATCH],strict=True):
            msgs = prompt(schema,row["text"],[train[int(k)] for k in picks])
            prompts.append(tokenizer.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True))
        enc = tokenizer(prompts,add_special_tokens=False,padding=True,return_tensors="pt").to(device)
        width = enc["input_ids"].shape[1]
        max_prompt = max(max_prompt, width)
        sequences = model.generate(**enc,do_sample=False,max_new_tokens=MAX_NEW,pad_token_id=tokenizer.pad_token_id,
                                   eos_token_id=[eot,tokenizer.eos_token_id],temperature=None,top_p=None,top_k=None)
        for row, seq, picks in zip(batch,sequences[:,width:],indices[i:i+BATCH],strict=True):
            raw = tokenizer.decode(seq,skip_special_tokens=True)
            if eot in seq.tolist() or tokenizer.eos_token_id in seq.tolist():
                finished += 1
            event, status, dropped = parse(row["text"],raw,schema)
            pred.append(prediction(row,event,raw=raw,status=status,dropped_args=dropped,
                                   example_ids=[train[int(k)]["sent_id"] for k in picks]))
        print(f"{n_shots}-shot {len(pred)}/{len(test)}",flush=True)
    synchronize(device)
    name = "qwen_zero" if n_shots==0 else "qwen_few"
    from collections import Counter
    put_result(out,name,test,pred,time.perf_counter()-started,model=MODEL,
               revision=REVISION,n_shots=n_shots,max_prompt_tokens=max_prompt,
               max_new_tokens=MAX_NEW,device=str(device),ended=finished,
               statuses=dict(Counter(r["status"] for r in pred)),
               dropped_args=sum(r["dropped_args"] for r in pred))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir",type=Path,default=DEFAULT_DATA)
    p.add_argument("--output",type=Path,default=ROOT)
    p.add_argument("--qwen",default=MODEL)
    p.add_argument("--device",choices=["auto","cuda","cpu"],default="auto")
    p.add_argument("--n-test",type=int,default=300)
    a=p.parse_args()
    schema,train,_,test,_,manifest=split(a.data_dir,a.n_test)
    if manifest != read_json(a.output/"results/splits.json"):
        raise ValueError("Data split differs from the discriminative experiments")
    device=device_for(a.device)
    kwargs = {"revision": REVISION} if a.qwen == MODEL else {}
    tokenizer=AutoTokenizer.from_pretrained(a.qwen,padding_side="left",**kwargs)
    dtype=torch.bfloat16 if device.type=="cuda" and torch.cuda.is_bf16_supported() else torch.float32
    model=AutoModelForCausalLM.from_pretrained(a.qwen,torch_dtype=dtype,attn_implementation="sdpa",**kwargs).to(device).eval()
    vectorizer=joblib.load(a.output/"checkpoints/type.joblib")[0]
    for n in (0,2):
        run(model,tokenizer,device,schema,train,test,a.output,n,vectorizer)


if __name__=="__main__":
    main()
