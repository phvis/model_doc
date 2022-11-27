import os
import sys
import json

def trans(input_fp, output_fp, label_fp=None):
    label_map = dict()
    with open(input_fp, "r") as fin, open(output_fp, "w") as fout:
        lines = fin.readlines()
        for line in lines:
            info = json.loads(line)
            s_list = [s for s in info["data"]]
            label_list = ["O"] * len(s_list)
            for label_info in info["label"]:
                label_list[label_info[0]] = "B-" + label_info[2]
                for idx in range(label_info[0]+1, label_info[1]):
                    label_list[idx] = "I-" + label_info[2]
            
                if "B-" + label_info[2] not in label_map:
                    label_map["B-" + label_info[2]] = len(label_map)
                    label_map["I-" + label_info[2]] = len(label_map)
        
        fout.write("\002".join(s_list) + "\t" + "\002".join(label_list) + "\n")
    
    label_map["O"] = len(label_map)
    
    if label_fp is None:
        return
    with open(label_fp, "w") as fout:
        s = json.dumps(label_map)
        fout.write(s)


if __name__ == "__main__":
    input_fp = "./all.jsonl"
    output_fp = "./all.txt"
    label_fp = "./label.txt"
    trans(input_fp, output_fp, label_fp)