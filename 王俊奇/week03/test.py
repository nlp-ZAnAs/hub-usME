from collections import OrderedDict
import csv
import os
import pathlib


path = os.path.join( os.getcwd(),"data","langid","v12","dev.csv")
voc_path = os.path.join( os.getcwd(),"data","langid","vocab","vocab_v8.txt")
print(path)
path = pathlib.Path(path)
print(path)
rows = []
with open(path, encoding="utf-8") as f:
    for r in csv.DictReader(f):
        rows.append((r["text"], r["label"]))
s = {label for _,label in rows}
print(s)


def load_vocab_file(path):
    """
    读已经生成好的字符表（每行「字符 \\t id \\t 出现次数」，以 # 开头的是注释）。

    课堂用的就是 data/langid/vocab/vocab_v8.txt，它是从 v8 训练集统计出来的。
    → 换了训练集、加了语种，这张表就得重新生成（用上面的 build_char_vocab）。
    """
    vocab = OrderedDict()
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            ch, idx, _freq = line.rstrip("\n").split("\t")
            vocab[ch] = int(idx)
    return vocab
dict = load_vocab_file(voc_path)
print(dict)