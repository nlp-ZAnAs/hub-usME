"""Predict one news title using any saved comparison method."""
import argparse
import json
from pathlib import Path

import joblib
from peft import PeftModel
from transformers import AutoModel, AutoTokenizer
import torch

from train import LABELS, ROOT, REVISION, bert_features, instruction, parse_label
from lm_utils import generate, load_base, select_device


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text",required=True)
    parser.add_argument("--method",choices=["svm","logistic","bert_cls","bert_mean","qwen_zero","qwen_lora","bert_lora"],default="svm")
    parser.add_argument("--bert",default="google-bert/bert-base-chinese")
    parser.add_argument("--qwen",default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--device",choices=["auto","cpu","cuda"],default="auto")
    args=parser.parse_args()
    device=select_device(args.device)
    raw=None
    if args.method in ("svm","logistic"):
        pred=int(joblib.load(ROOT/"checkpoints"/f"{args.method}.joblib").predict([args.text])[0])
    elif args.method=="bert_lora":
        tokenizer=AutoTokenizer.from_pretrained(args.bert)
        model=PeftModel.from_pretrained(AutoModel.from_pretrained(args.bert),ROOT/"checkpoints"/"bert_adapter").to(device).eval()
        features=bert_features(model,tokenizer,[{"sentence":args.text}],"mean",device)
        head=torch.nn.Linear(model.config.hidden_size,15).to(device)
        head.load_state_dict(torch.load(ROOT/"checkpoints"/"bert_head.pt",map_location=device,weights_only=True))
        head.eval()
        with torch.inference_mode():
            pred=int(head(torch.tensor(features,device=device)).argmax(-1).item())
    elif args.method.startswith("bert"):
        tokenizer=AutoTokenizer.from_pretrained(args.bert)
        model=AutoModel.from_pretrained(args.bert).to(device).eval()
        features=bert_features(model,tokenizer,[{"sentence":args.text}],args.method.split("_")[1],device)
        pred=int(joblib.load(ROOT/"checkpoints"/f"{args.method}.joblib").predict(features)[0])
    else:
        model,tokenizer=load_base({"base_model":"Qwen/Qwen2.5-0.5B","revision":REVISION},args.qwen,device)
        if args.method=="qwen_lora":
            model=PeftModel.from_pretrained(model,ROOT/"checkpoints"/"qwen_adapter")
        raw=generate(model,tokenizer,[instruction(args.text)],device,8)[0]["text"]
        pred=parse_label(raw)
    print(json.dumps({"text":args.text,"method":args.method,"label_id":pred,
                      "label":LABELS[pred] if pred>=0 else "解析失败","raw":raw},ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
