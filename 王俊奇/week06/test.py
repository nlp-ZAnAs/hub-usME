# def get_data(file_path:str):
#     label_list = []
#     with open(file_path,"r",encoding="utf-8") as f:
#         for index,str in enumerate(f.readlines()):
#             if index == 0:
#                 continue
#             label = str.split("\t")[-1].strip()
#             # print(str.split("\t"))
#             label_list.append(label)
#     label_set = set(label_list)
#     print(len(label_set))
#     print(len(label_list))
#     label2id = {}
#     id2label = {}
#     for id,label in enumerate(label_set):
#         label2id[label]=id
#         id2label[id]=label
#     print(label2id)
#     print(id2label)
# get_data(r"G:\大模型课程\尚硅谷\尚硅谷大模型250319线上同步\10_尚硅谷大模型项目之智能商品发布\4.视频\day_02\day_02\product-classify-bert\data\raw\valid.txt")
from collections import Counter
import json


label2id = {'水果': 0, '运动': 1, '蔬菜': 2, '家居': 3, '粮油速食': 4, '休闲食品': 5, '香烟': 6, '美妆': 7, '家电': 8, '个护': 9, '餐饮': 10, '礼品': 11, '鞋靴箱包': 12, '宠物': 13, '酒饮冲调': 14, '办公': 15, '健康': 16, '书籍': 17, '肉禽蛋': 18, '水产': 19, '3C数码': 20, '玩具': 21, '钟表配饰': 22, '乳品': 23, '汽车用品': 24, '母婴': 25, '服饰内衣': 26, '鲜花绿植': 27, '健康食品': 28, '清洁': 29}
id2label = {0: '水果', 1: '运动', 2: '蔬菜', 3: '家居', 4: '粮油速食', 5: '休闲食品', 6: '香烟', 7: '美妆', 8: '家电', 9: '个护', 10: '餐饮', 11: '礼品', 12: '鞋靴箱包', 13: '宠物', 14: '酒饮冲调', 15: '办公', 16: '健康', 17: '书籍', 18: '肉禽蛋', 19: '水产', 20: '3C数码', 21: '玩具', 22: '钟表配饰', 23: '乳品', 24: '汽车用品', 25: '母婴', 26: '服饰内衣', 27: '鲜花绿植', 28: '健康食品', 29: '清洁'}
LABELS = ['水果', '运动', '蔬菜', '家居', '粮油速食', '休闲食品', '香烟', '美妆', '家电', '个护', '餐饮', '礼品', '鞋靴箱包', '宠物', '酒饮 冲调', '办公', '健康', '书籍', '肉禽蛋', '水产', '3C数码', '玩具', '钟表配饰', '乳品', '汽车用品', '母婴', '服饰内衣', '鲜花绿植', '健康食品', '清洁']

def create_file(file_path,out_path):
    item_list=[]
    with open(file_path,"r",encoding="utf-8") as f:
        for index,str in enumerate(f.readlines()):
            if index == 0:
                continue
            label = str.split("\t")[0].strip()
            product = str.split("\t")[-1].strip()
            # print(str.split("\t"))
            item_list.append({"idx":index-1,"sentence":product,"label":label2id[label]})
    with open(out_path,"w",encoding="utf-8") as out_f:
        json.dump(item_list,out_f,ensure_ascii=False,indent=2)
# create_file(r"G:\大模型课程\尚硅谷\尚硅谷大模型250319线上同步\10_尚硅谷大模型项目之智能商品发布\4.视频\day_02\day_02\product-classify-bert\data\raw\train.txt","train.json")

def get_counter(file_path ="train.json"):
    with open(file_path,"r",encoding="utf-8") as f:
            item_list = json.loads(f.read())
            c = Counter(item["label"] for item in item_list)
            res = c.most_common()
            print(res)
# get_counter()
# get_counter("valid.json")
