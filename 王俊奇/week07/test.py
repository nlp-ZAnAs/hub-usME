
from collections import defaultdict
import json
import pathlib
import random
import re
from typing import Any
from seqeval.metrics import (
    f1_score, precision_score, recall_score,
    classification_report as seqeval_report,
)


# from tokenizers import Tokenizer
# from transformers import BertTokenizer
# from torchcrf import CRF

# ENTITY_TYPES = json.loads(pathlib.Path("data/cluener/label.json").read_text("utf-8"))
# def build_label_schema() -> tuple[list[str], dict[str, int], dict[int, str]]:
#     """构建 BIO 标签体系，返回 (labels, label2id, id2label)。"""
#     labels = ["O"]
#     for etype in ENTITY_TYPES:
#         labels.append(f"B-{etype}")
#         labels.append(f"I-{etype}")

#     label2id = {lbl: i for i, lbl in enumerate(labels)}
#     id2label = {i: lbl for lbl, i in label2id.items()}
#     return labels, label2id, id2label

# labels, label2id, id2label = build_label_schema()

# print(labels, label2id, id2label)

# train_data = {
#         "text": "彭小军认为，国内银行现在走的是台湾的发卡模式... say i am playing",
#         "label": {
#             "address": {
#                 "台湾": [[15, 16]]
#             },
#             "name": {
#                 "彭小军": [[0, 2]]
#             }
#         }
#     }

# def span_to_bio(text: str, label_dict: dict[str,Any], label2id: dict[str,Any]) -> list[int]:
#     """将 cluener2020 的 span 格式标注转为逐字符 BIO 标签 id 列表。

#     教学要点：先全部初始化为 O，再按 span 位置填入 B/I。
#     若存在嵌套实体（本数据集极少），外层实体覆盖内层。
#     """
#     n = len(text)
#     bio = ["O"] * n

#     if not label_dict:
#         return [label2id[t] for t in bio]

#     for etype, spans in label_dict.items():
#         b_tag = f"B-{etype}"
#         i_tag = f"I-{etype}"
#         for surface, positions in spans.items():
#             for start, end in positions:
#                 if start >= n or end >= n:
#                     continue
#                 bio[start] = b_tag
#                 for idx in range(start + 1, end + 1):
#                     bio[idx] = i_tag

#     return [label2id.get(t, 0) for t in bio]
# char_labels = span_to_bio(train_data["text"],train_data["label"],label2id)
# print(char_labels)
# chars = list(train_data["text"])
# tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
# encoding = tokenizer(
#             chars,
#             is_split_into_words=True,
#             max_length=64,
#             truncation=True,
#             padding="max_length",
#             return_tensors="pt",
#         )
# word_ids = encoding.word_ids(batch_index=0)
# print(word_ids)
# aligned_labels = []
# prev_word_id = None
# for wid in word_ids:
#     if wid is None:
#         aligned_labels.append(-100)
#     elif wid != prev_word_id:
#         # 首次出现这个字符索引：使用 BIO 标签
#         if wid < len(char_labels):
#             aligned_labels.append(char_labels[wid])
#         else:
#             aligned_labels.append(-100)
#         prev_word_id = wid
#     else:
#         # 同一字符的后续子词（中文通常不会出现，但保留正确处理）
#         aligned_labels.append(-100)
# print(aligned_labels)


ENTITY_TYPE_ZH = {
    "address": "地址", "book": "书名", "company": "公司",
    "game": "游戏", "government": "政府机构", "movie": "影视作品",
    "name": "人名", "organization": "组织机构", "position": "职位",
    "scene": "景点/场所",
}

ENTITY_TYPES_EN = list(ENTITY_TYPE_ZH.keys())
# def gold_spans_from_record(record: dict) -> set[tuple[str, str, int, int]]:
#     """提取 gold spans，格式：{(text, type, start, end)}。"""
#     spans = set()
#     for etype, surfaces in (record.get("label") or {}).items():
#         for surface, positions in surfaces.items():
#             for start, end in positions:
#                 spans.add((surface, etype, start, end))
#     return spans
# def sample_recods(n:int, seed = 42):
#     # 遍历数据表中的每个label，将每个label作为key,存入每条数据。
#     # 每类数据的个数是n//10
#     # 遍历类型数组，从每个类型的数据数组中找到未选中的数据作为候选数据，将候选数据sample出来
#     # 每个选中的数据都存入set,确保不会重复选中
#     # 可能总数不能整除，找到所有未选中数据，打乱后，重新选择。将数据补齐到n
#     random.seed(seed)
#     label_dic = defaultdict(list)
#     all_datas = json.loads(pathlib.Path("data/cluener/validation.json").read_text("utf-8"))
#     for data in all_datas:
#         for label in data['label'] :
#             label_dic[label].append(data)
#     selected_list = []
#     selected_set = set()
#     num_per_type = n // len(ENTITY_TYPES_EN)
#     for label in ENTITY_TYPES_EN: 
#         houxuan_list = [item for item in label_dic[label] if not id(item) in selected_set]
#         selected_houxuan_list = random.sample(houxuan_list,num_per_type)
#         selected_list.extend(selected_houxuan_list)
#         selected_set.update([id(item) for item in selected_houxuan_list])
#     rest_num = n % len(ENTITY_TYPES_EN)
#     rest_list = [item for item in all_datas if not id(item)  in selected_set]
#     selected_list.extend(random.sample(rest_list, len(rest_list))[::-1][:rest_num])
#     return selected_list
# a = gold_spans_from_record(sample_recods(15)[0])
# print(a)


# def pred_spans

# pred1 = [set(),{ ('《球》', 'game', 11, 13)},{('渣打银行', 'company', 12, 15)}]
# pred2 = [{('发言人', 'position', 11, 13), ('花园球馆', 'scene', 6, 9)},set(),{('渣打银行', 'company', 12, 15)}]
# gold = [{('花园球馆', 'address', 6, 9)},{('大众软件', 'book', 0, 3), ('《球》', 'game', 11, 13)},{('渣打银行', 'company', 12, 15)}]


# def compute_span_f1(all_golds, all_preds):
#     """计算 span-level 精确率、召回率、F1。"""
#     tp = sum(len(g & p) for g, p in zip(all_golds, all_preds))
#     pred_total = sum(len(p) for p in all_preds)
#     gold_total = sum(len(g) for g in all_golds)
#     p = tp / pred_total if pred_total else 0.0
#     r = tp / gold_total if gold_total else 0.0
#     f1 = 2 * p * r / (p + r) if (p + r) else 0.0
#     return {"precision": p, "recall": r, "f1": f1, "tp": tp, "pred_total": pred_total, "gold_total": gold_total}
# a1 = compute_span_f1(gold,pred1)
# a2 = compute_span_f1(gold,pred2)
# a3 = compute_span_f1(gold,gold)
# print(a1)
# print(a2)
# print(a3)

# for strs in ["train","test","validation"] :
#     arr = json.loads(pathlib.Path(f"data/peoples_daily/{strs}.json").read_text("utf-8"))
#     print(strs,"数据集长度：",len(arr))



# def format_daily_data(records):
#     results = []
#     for record in records:
#         lable = defaultdict(dict)
#         result = {"text":"".join(record["tokens"]), "label":lable}
#         arr=[]
#         label_name = ""
#         for index,item  in enumerate(record["ner_tags"]):
#             if not len(arr) :
#                 if item in ["B-PER","B-ORG","B-LOC"] :
#                     arr.append(index)
#                     match item:
#                         case "B-PER":
#                             label_name = "name"
#                         case "B-ORG":
#                             label_name = "organization"
#                         case _:
#                             label_name = "address"
#             else:
#                 if item in ["I-PER","I-ORG","I-LOC"] :
#                     arr.append(index)
#                 elif item =="O":
#                     ner_name = result["text"][arr[0]:index]
#                     if not ner_name in lable[label_name] :
#                                     lable[label_name][ner_name] = []
#                     lable[label_name][ner_name].append([arr[0],arr[-1]])            
#                     arr=[]
#                     label_name=""

#         results.append(result)
#     return results

# daily_data = json.loads(pathlib.Path("data/peoples_daily/train.json").read_text("utf-8"))
# obj = format_daily_data(daily_data)
# print(obj[0])



a = [['O', 'O', 'O', 'O', 'O', 'B-organization', 'I-organization', 'I-organization', 'I-organization', 'I-organization', 'O', 'B-organization', 'I-organization', 'I-organization', 'I-organization', 'I-organization', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']]
b = [['O', 'B-organization', 'B-organization', 'O', 'O', 'B-organization', 'I-organization', 'I-organization', 'I-organization', 'I-organization', 'I-organization', 'B-organization', 'I-organization', 'I-organization', 'I-organization', 'I-organization', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']]
r1 = precision_score(a,b)
r2 = recall_score(a,b)
r3 = f1_score(a,b)
print(r1,r2,r3)