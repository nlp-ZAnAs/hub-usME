import json
from pathlib import Path                                                                                                                                              
from sklearn.model_selection import train_test_split

HERE =  Path(__file__).resolve().parent.parent
DATA_DIR = HERE.joinpath("data")     
  
  
def split_and_build(full, train_file, val_file):                                                                            
    with open(full) as f:                                                                                                                                
        rows = [json.loads(line) for line in f]
                                                                                                                                                                
    train, val = train_test_split(rows, test_size=0.1, random_state=42)
    tr_samples = build_sample(train, "train")                              
    val_samples = build_sample(val, "val")
    
    with open(train_file, 'w') as f:
        f.write(json.dumps(tr_samples, ensure_ascii=False, indent=2))
                                                                                                                                                                
    with open(val_file, 'w') as f:
        f.write(json.dumps(val_samples, ensure_ascii=False, indent=2))
            
# 0 - human_answer, 1 - ChatGPT_answer  
def build_sample(rows, type):
    idx = 0
    samples = []
    label_0 = 0
    label_1 = 0
    for row in rows:                                                                                                                                            
        for ans in row["human_answers"]:
            if not ans:
                continue                                                                                                                    
            samples.append({"idx": idx, "text": ans, "label": 0}) 
            idx += 1
            label_0 += 1                                                                         
        for ans in row["chatgpt_answers"]:                                                                                                                      
            samples.append({"idx": idx, "text": ans, "label": 1}) 
            idx += 1
            label_1 += 1
    print("{} 样本数: {}, label_0: {}, label_1: {}".format(type, len(samples), label_0, label_1))
    return samples
            
            
                
if __name__ == '__main__':
    full_data = DATA_DIR.joinpath("all.jsonl")
    train_data = DATA_DIR.joinpath( 'train.json')
    val_data = DATA_DIR.joinpath( 'val.json')
    split_and_build(full_data, train_data, val_data)