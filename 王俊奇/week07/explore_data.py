

from collections import Counter
import json
import os
import pathlib
from matplotlib import pyplot as plt
import seaborn

ROOT = os.path.dirname(__file__)
figures_url = ROOT + "/outputs/figures"

import matplotlib.font_manager as fm


plt.rcParams["font.family"] = "Microsoft YaHei"
def get_data(type):
    str = pathlib.Path(ROOT + rf"/data/peoples_daily/{type}.json").read_text("utf-8")
    return json.loads(str)


def parse_text_len():
    train_data = get_data("train")
    len_list = [len(item["tokens"]) for item in train_data]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.hist(len_list, bins=40, color="#4C72B0", edgecolor="white", linewidth=0.4)
    ax.set_title("文本长度分布图")
    ax.set_xlabel("文本长度")
    ax.set_ylabel("样本个数")
    ax.tick_params(axis="x", rotation=30)
    ax.axvline(64,color="red",linestyle="--", label="64字符")
    ax.axvline(128,color="red",linestyle="--", label="128字符")
    plt.legend()
    plt.tight_layout()
    os.makedirs(figures_url,exist_ok= True)
    plt.savefig(figures_url + "/文本长度分布图.png")
    plt.close()

def parse_label_distribute():

    et_label = {
        "B-LOC": "地址", 
        "B-PER": "人名", "B-ORG": "组织"
    }
    train_data = get_data("train")
    c = Counter([])
    for item in train_data:
        for label_name in item["ner_tags"]:
            if label_name in et_label:
                c.update([label_name])
    _,ax = plt.subplots(figsize=(12, 5))
    colors = seaborn.color_palette("husl", len(c.keys()))
    bars = ax.bar([f"{i}\n({et_label[i]})" for i in c.keys()],c.values(),color = colors,edgecolor="white", linewidth=0.5)
    for bar, cnt in zip(bars, c.values()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                str(cnt), ha="center", va="bottom", fontsize=8)
    ax.set_title("标签分布图")
    ax.set_ylabel("标签个数")
    ax.set_xlabel("标签名称")

    # plt.legend()
    plt.tight_layout()
    os.makedirs(figures_url,exist_ok= True)
    plt.savefig(figures_url + "/标签分布图.png")
    plt.close()
def parse_entity_len_distribute():
    train_data = get_data("train")
    entity_len_arr = []
    for item in train_data:
        for label_name in item["ner_tags"]:
            start = 0
            try :
                for tentity_name in item["label"][label_name]:
                    entity_len_arr.append(len(tentity_name))
                    # if len(tentity_name) > 30:
                    #     print(item["text"])
            except KeyError as e:
                print(e)
    c = Counter(entity_len_arr)
    top = c.most_common()
    labels, counts = zip(*top)
    _,ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(labels, counts,edgecolor="white", linewidth=0.5)
    for bar,count in zip(bars,counts) :
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50, str(count),ha="center", va="bottom",fontsize=8,)
    ax.set_xticks(labels) 
    ax.set_title("实体长度分布图")
    ax.set_xlabel("实体长度")
    ax.set_ylabel("实体样本个数")
    plt.tight_layout()
    os.makedirs(figures_url,exist_ok= True)
    plt.savefig(figures_url + "/实体字符长度分布图.png")
    plt.close()

parse_text_len()
parse_label_distribute()
parse_entity_len_distribute()