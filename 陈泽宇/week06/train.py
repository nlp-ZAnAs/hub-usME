"""Reproducible TNEWS classification comparison. Run from any working directory."""
import argparse
import gc
import hashlib
import json
import random
import time
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer

from lm_utils import collate, completion_logits, generate, load_base, prompt_ids, select_device

ROOT = Path(__file__).resolve().parent
LABELS = ["故事", "文化", "娱乐", "体育", "财经", "房产", "汽车", "教育", "科技", "军事",
          "旅游", "国际", "证券", "农业", "电竞"]
REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def prepare(source):
    raw_train, raw_test = (read(source / f"{name}.json") for name in ("train", "val"))
    # Exclude all conflicting duplicate titles, not just one label.
    groups = {}
    for row in raw_train:
        key = " ".join(row["sentence"].split())
        groups.setdefault(key, []).append(row)
    clean = [dict(rows[0], sentence=key) for key, rows in groups.items()
             if len({r["label"] for r in rows}) == 1]
    selected, _ = train_test_split(clean, train_size=7000, stratify=[r["label"] for r in clean], random_state=42)
    train, dev = train_test_split(selected, test_size=1000, stratify=[r["label"] for r in selected], random_state=42)
    seen, test = set(groups), []
    for row in raw_test:
        key = " ".join(row["sentence"].split())
        if key not in seen:
            test.append(dict(row, sentence=key))
            seen.add(key)
    test, _ = train_test_split(test, train_size=1500, stratify=[r["label"] for r in test], random_state=42)
    data = {"train": train, "dev": dev, "test": test}
    for name, rows in data.items():
        write(ROOT / "data" / f"{name}.json", rows)
    write(ROOT / "results" / "data_stats.json", {
        "source_train":len(raw_train), "source_val":len(raw_test),
        "train_unique":len(clean), "train_duplicate_removed":len(raw_train)-len(clean),
        "splits": {n:{"rows":len(r),"counts":dict(Counter(x["label"] for x in r)),
                      "sha256":hashlib.sha256((ROOT/"data"/f"{n}.json").read_bytes()).hexdigest()}
                   for n,r in data.items()},
    })


def report(name, truth, pred, seconds, history=None, raw=None, train_seconds=0):
    value = {"method":name, "accuracy":accuracy_score(truth,pred),
             "macro_f1":f1_score(truth,pred,labels=list(range(15)),average="macro",zero_division=0),
             "invalid":sum(p not in range(15) for p in pred),
             "eval_seconds":seconds,"train_seconds":train_seconds,
             "per_class":classification_report(truth,pred,labels=list(range(15)),
                                                target_names=LABELS,output_dict=True,zero_division=0),
             # Extra row/column 15 represents invalid output.
             "confusion":confusion_matrix(truth,[p if p>=0 else 15 for p in pred],labels=list(range(16))).tolist(),
             "predictions":pred,"truth":truth,"history":history or [],"raw":raw}
    write(ROOT/"results"/f"{name}.json",value)
    print(name, f"accuracy={value['accuracy']:.4f} macro_f1={value['macro_f1']:.4f}",flush=True)


def classical(data):
    train, dev, test = (data[n] for n in ("train","dev","test"))
    texts=lambda rows:[r["sentence"] for r in rows]
    labels=lambda rows:[r["label"] for r in rows]
    truth=labels(test)
    fallback=Counter(labels(train)).most_common(1)[0][0]
    report("majority",truth,[fallback]*len(test),0)
    t=time.perf_counter()
    rules=[next((i for i,n in enumerate(LABELS) if n in r["sentence"]),fallback) for r in test]
    report("keyword",truth,rules,time.perf_counter()-t)
    for kind in ("svm","logistic"):
        started=time.perf_counter()
        best=-1
        history=[]
        for c in (0.5,2.0,8.0):
            estimator=(LinearSVC(C=c,class_weight="balanced",random_state=42)
                       if kind=="svm" else LogisticRegression(C=c,class_weight="balanced",max_iter=600,random_state=42))
            model=make_pipeline(TfidfVectorizer(analyzer="char",ngram_range=(1,3),min_df=2,
                                                max_features=60000,sublinear_tf=True),estimator)
            model.fit(texts(train),labels(train))
            score=f1_score(labels(dev),model.predict(texts(dev)),labels=list(range(15)),average="macro",zero_division=0)
            history.append({"C":c,"dev_macro_f1":score})
            if score>best:
                best=score
                joblib.dump(model,ROOT/"checkpoints"/f"{kind}.joblib")
        train_seconds=time.perf_counter()-started
        model=joblib.load(ROOT/"checkpoints"/f"{kind}.joblib")
        t=time.perf_counter()
        pred=model.predict(texts(test)).tolist()
        report(kind,truth,pred,time.perf_counter()-t,history,train_seconds=train_seconds)


@torch.inference_mode()
def bert_features(model, tokenizer, rows, pooling, device):
    output=[]
    for start in range(0,len(rows),64):
        batch=tokenizer([r["sentence"] for r in rows[start:start+64]],padding=True,truncation=True,
                        max_length=64,return_tensors="pt").to(device)
        hidden=model(**batch, return_dict=True).last_hidden_state.float()
        mask=batch["attention_mask"].unsqueeze(-1)
        feat=hidden[:,0] if pooling=="cls" else (hidden*mask).sum(1)/mask.sum(1)
        output.append(feat.cpu().numpy())
    return np.concatenate(output)


def bert_probes(data, base, device):
    tokenizer=AutoTokenizer.from_pretrained(str(base))
    model=AutoModel.from_pretrained(str(base),torch_dtype=torch.bfloat16 if device.type=="cuda" else torch.float32).to(device).eval()
    for pool in ("cls","mean"):
        started=time.perf_counter()
        features={n:bert_features(model,tokenizer,rows,pool,device) for n,rows in data.items()}
        history=[]
        best=-1
        for c in (0.1,1.,10.):
            clf=make_pipeline(StandardScaler(), LogisticRegression(C=c,class_weight="balanced",max_iter=2500,tol=0.001,random_state=42))
            clf.fit(features["train"],[r["label"] for r in data["train"]])
            score=f1_score([r["label"] for r in data["dev"]],clf.predict(features["dev"]),
                           labels=list(range(15)),average="macro",zero_division=0)
            history.append({"C":c,"dev_macro_f1":score})
            if score>best:
                best=score
                joblib.dump(clf,ROOT/"checkpoints"/f"bert_{pool}.joblib")
        clf=joblib.load(ROOT/"checkpoints"/f"bert_{pool}.joblib")
        pred=clf.predict(features["test"]).tolist()
        report(f"bert_{pool}",[r["label"] for r in data["test"]],pred,0,history,
               train_seconds=time.perf_counter()-started)
    del model
    gc.collect()
    torch.cuda.empty_cache()


def parse_label(text):
    hits=[i for i,name in enumerate(LABELS) if name in text]
    return hits[0] if len(hits)==1 else -1

def instruction(text):
    return "将新闻标题分为以下一个类别，只输出类别名称。类别："+"、".join(LABELS)+"。\n标题："+text


def llm_rows(rows,tok):
    from lm_utils import Example
    output=[]
    for row in rows:
        prompt=prompt_ids(tok,instruction(row["sentence"]))
        answer=tok.encode(LABELS[row["label"]],add_special_tokens=False)+[tok.convert_tokens_to_ids("<|im_end|>")]
        output.append(Example(instruction(row["sentence"]),LABELS[row["label"]],prompt+answer,[-100]*len(prompt)+answer,len(answer)))
    return output


@torch.inference_mode()
def llm_predict(model,tok,rows,device):
    raw=[]
    for i in range(0,len(rows),32):
        raw.extend(generate(model,tok,[instruction(r["sentence"]) for r in rows[i:i+32]],device,8))
    # Accept one unambiguous label even when extra text surrounds it.
    pred=[parse_label(r["text"]) for r in raw]
    return pred,raw


@torch.inference_mode()
def dev_loss(model,tok,rows,device):
    model.eval()
    total,count=0.,0
    for i in range(0,len(rows),8):
        x,m,y=collate(rows[i:i+8],tok.pad_token_id,device)
        logits,target=completion_logits(model,x,m,y)
        total+=F.cross_entropy(logits.float(),target,reduction="sum").item()
        count+=len(target)
    return total/count


def qwen(data,base,device):
    torch.manual_seed(42)
    cfg={"base_model":"Qwen/Qwen2.5-0.5B","revision":REVISION}
    model,tok=load_base(cfg,base,device)
    t=time.perf_counter()
    pred,raw=llm_predict(model,tok,data["test"],device)
    report("qwen_zero",[r["label"] for r in data["test"]],pred,time.perf_counter()-t,raw=raw)
    model=get_peft_model(model,LoraConfig(task_type="CAUSAL_LM",r=8,lora_alpha=16,lora_dropout=0.05,
                                         target_modules=["q_proj","v_proj"]))
    params=[p for p in model.parameters() if p.requires_grad]
    optim=torch.optim.AdamW(params,lr=0.0002,weight_decay=0.01)
    train=llm_rows(data["train"],tok)
    dev=llm_rows(data["dev"],tok)
    rng=random.Random(42)
    history=[]
    best=float("inf")
    started=time.perf_counter()
    for epoch in range(1,4):
        model.train()
        rng.shuffle(train)
        loss_total,count=0.,0
        for start in range(0,len(train),32):
            group=train[start:start+32]
            tokens=sum(row.supervised for row in group)
            optim.zero_grad(set_to_none=True)
            for off in range(0,len(group),8):
                x,m,y=collate(group[off:off+8],tok.pad_token_id,device)
                logits,target=completion_logits(model,x,m,y)
                loss=F.cross_entropy(logits.float(),target,reduction="sum")
                (loss/tokens).backward()
                loss_total+=loss.detach().item()
                count+=len(target)
            torch.nn.utils.clip_grad_norm_(params,1.)
            optim.step()
        val=dev_loss(model,tok,dev,device)
        history.append({"epoch":epoch,"train_nll":loss_total/count,"dev_nll":val})
        print(history[-1],flush=True)
        if val<best:
            best=val
            model.save_pretrained(ROOT/"checkpoints"/"qwen_adapter")
    elapsed=time.perf_counter()-started
    del model,optim,params
    gc.collect()
    torch.cuda.empty_cache()
    model,_=load_base(cfg,base,device)
    model=PeftModel.from_pretrained(model,ROOT/"checkpoints"/"qwen_adapter")
    # Make the saved adapter portable across machines.
    path=ROOT/"checkpoints"/"qwen_adapter"/"adapter_config.json"
    ac=read(path); ac.update(base_model_name_or_path=cfg["base_model"],revision=REVISION); write(path,ac)
    t=time.perf_counter()
    pred,raw=llm_predict(model,tok,data["test"],device)
    report("qwen_lora",[r["label"] for r in data["test"]],pred,time.perf_counter()-t,history,raw,elapsed)




def bert_finetune(data, base, device):
    """Train a mean-pooled encoder classifier with LoRA on attention query/value."""
    torch.manual_seed(42)
    tok=AutoTokenizer.from_pretrained(str(base))
    encoder=AutoModel.from_pretrained(str(base),torch_dtype=torch.bfloat16 if device.type=="cuda" else torch.float32)
    encoder=get_peft_model(encoder,LoraConfig(task_type="FEATURE_EXTRACTION",r=8,lora_alpha=16,
                                             lora_dropout=0.05,target_modules=["query","value"])).to(device)
    head=torch.nn.Linear(encoder.config.hidden_size,15).to(device)
    params=[p for p in encoder.parameters() if p.requires_grad]+list(head.parameters())
    opt=torch.optim.AdamW(params,lr=0.0002,weight_decay=0.01)
    counts=Counter(r["label"] for r in data["train"])
    weights=torch.tensor([len(data["train"])/(15*counts[i]) for i in range(15)],device=device)
    rng=random.Random(42)
    history=[]
    best=-1
    started=time.perf_counter()

    def forward(rows):
        inputs=tok([r["sentence"] for r in rows],padding=True,truncation=True,max_length=64,
                   return_tensors="pt").to(device)
        hidden=encoder(**inputs,return_dict=True).last_hidden_state.float()
        mask=inputs["attention_mask"].unsqueeze(-1)
        return head((hidden*mask).sum(1)/mask.sum(1))

    for epoch in range(1,4):
        rows=list(data["train"]); rng.shuffle(rows)
        encoder.train(); head.train()
        loss_sum=0.
        for start in range(0,len(rows),32):
            batch=rows[start:start+32]
            opt.zero_grad(set_to_none=True)
            logits=forward(batch)
            target=torch.tensor([r["label"] for r in batch],device=device)
            loss=F.cross_entropy(logits,target,weight=weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params,1.)
            opt.step()
            loss_sum+=loss.item()*len(batch)
        encoder.eval(); head.eval()
        pred=[]
        with torch.inference_mode():
            for start in range(0,len(data["dev"]),64):
                pred.extend(forward(data["dev"][start:start+64]).argmax(-1).tolist())
        score=f1_score([r["label"] for r in data["dev"]],pred,labels=list(range(15)),average="macro",zero_division=0)
        history.append({"epoch":epoch,"train_weighted_ce":loss_sum/len(rows),"dev_macro_f1":score})
        print("bert_lora",history[-1],flush=True)
        if score>best:
            best=score
            encoder.save_pretrained(ROOT/"checkpoints"/"bert_adapter")
            torch.save(head.state_dict(),ROOT/"checkpoints"/"bert_head.pt")
    train_seconds=time.perf_counter()-started
    del encoder,opt,params
    gc.collect(); torch.cuda.empty_cache()
    encoder=PeftModel.from_pretrained(AutoModel.from_pretrained(str(base)),ROOT/"checkpoints"/"bert_adapter").to(device).eval()
    head.load_state_dict(torch.load(ROOT/"checkpoints"/"bert_head.pt",map_location=device,weights_only=True))
    head.eval()
    t=time.perf_counter(); pred=[]
    with torch.inference_mode():
        for start in range(0,len(data["test"]),64):
            pred.extend(forward(data["test"][start:start+64]).argmax(-1).tolist())
    report("bert_lora",[r["label"] for r in data["test"]],pred,time.perf_counter()-t,history,train_seconds=train_seconds)
    path=ROOT/"checkpoints"/"bert_adapter"/"adapter_config.json"
    config=read(path); config["base_model_name_or_path"]="google-bert/bert-base-chinese"; write(path,config)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",type=Path,help="Teacher code/data directory, used only to recreate the saved subsets")
    p.add_argument("--bert",default="google-bert/bert-base-chinese")
    p.add_argument("--qwen",default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--stage",choices=["all","classical","bert","qwen"],default="all")
    args=p.parse_args()
    for n in ("data","results","checkpoints"): (ROOT/n).mkdir(exist_ok=True)
    write(ROOT/"results"/"environment.json", {"torch":str(torch.__version__),"cuda":torch.version.cuda,"gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU","seed":42})
    torch.manual_seed(42); np.random.seed(42); torch.set_num_threads(4)
    if args.source: prepare(args.source)
    data={n:read(ROOT/"data"/f"{n}.json") for n in ("train","dev","test")}
    device=select_device("auto")
    if args.stage in ("all","classical"): classical(data)
    if args.stage in ("all","bert"):
        bert_probes(data,args.bert,device)
        bert_finetune(data,args.bert,device)
    if args.stage in ("all","qwen"): qwen(data,args.qwen,device)


if __name__=="__main__":
    main()
