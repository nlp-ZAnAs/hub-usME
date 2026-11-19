"""使用已训练模型分类一条评论；示例见 README.md。"""

import argparse
import json

import joblib
import torch
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

from common import LABELS, ROOT, config
from prepare_data import normalize
from run_baselines import rule_predict
from run_neural import checkpoint_path, llm_predict, model_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True)
    parser.add_argument("--method", default="tfidf_svm",
                        choices=["rule", "tfidf_nb", "tfidf_lr", "tfidf_svm", "rbt3",
                                 "qwen_zero", "qwen_lora", "qwen_full"])
    args = parser.parse_args()
    cfg = config()
    original = normalize(args.text)
    text = original[:cfg["max_chars"]]
    if not text:
        parser.error("评论不能为空")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    raw = None
    if args.method == "rule":
        # 固定训练集多数类为正面；这是该已发布实验的规则兜底。
        prediction = rule_predict(text, fallback=1)
    elif args.method.startswith("tfidf_"):
        model = joblib.load(ROOT / "checkpoints" / (args.method + ".joblib"))
        prediction = int(model.predict([text])[0])
    elif args.method == "rbt3":
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path("rbt3"), local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint_path("rbt3"), local_files_only=True).to(device).eval()
        inputs = tokenizer(text, return_tensors="pt").to(device)
        with torch.inference_mode(), torch.autocast(device_type=device, dtype=torch.bfloat16,
                                                    enabled=device == "cuda"):
            prediction = int(model(**inputs).logits.argmax(-1).item())
    else:
        source = (checkpoint_path("qwen_full") if args.method == "qwen_full"
                  else model_path(cfg["llm_model"]))
        tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            source, dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            local_files_only=True, attn_implementation="sdpa").to(device)
        if args.method == "qwen_lora":
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, checkpoint_path("qwen_lora"))
        preds, answers, _ = llm_predict(model, tokenizer, [{"text": text}], cfg, device)
        prediction, raw = preds[0], answers[0]
    print(json.dumps({"method": args.method, "text": text,
                      "truncated": len(original) > len(text), "label_id": prediction,
                      "label": LABELS[prediction] if prediction >= 0 else "解析失败",
                      "raw": raw}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
