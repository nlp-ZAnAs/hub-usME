from datasets.arrow_dataset import Dataset


from datasets.dataset_dict import DatasetDict


import json
import os
import pathlib

from datasets import load_dataset

os.makedirs(r"data\cluener",exist_ok = True)
label_list = set()
def download_cluener():
    ds = load_dataset("xusenlin/clue-ner")
    save_json(ds,"train")
    save_json(ds,"test")
    save_json(ds,"validation")
    pathlib.Path(r"data\cluener\label.json").write_text(json.dumps(list(label_list)),"utf-8")

def save_json(ds,type):
    arr = []
    for index,item in enumerate(ds[type]):
        label_data= {}
        for entity in item['entities'] :
            label_name = entity["label"]
            entity_name = entity["entity"]
            if type == "train":
                label_list.add(entity["label"])
            if label_name in label_data:
                if entity_name in label_data[label_name] :
                    label_data[label_name][entity_name].append([entity["start_offset"], entity["end_offset"] - 1])
                else :
                    label_data[label_name][entity_name] = [[entity["start_offset"], entity["end_offset"] - 1]]
            else :
                label_data[label_name]={entity_name: [[entity["start_offset"], entity["end_offset"] - 1]]}
        arr.append({"text":item["text"],"label":label_data})
    pathlib.Path(fr"data\cluener\{type}.json").write_text(json.dumps(arr,indent=2,ensure_ascii=False),encoding="utf-8")
download_cluener()
