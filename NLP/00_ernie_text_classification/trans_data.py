import os
import sys
import json

def trans(input_fp, output_fp, label_fp=None):
    label_set = set()
    with open(input_fp, "r") as fin, open(output_fp, "w") as fout:
        lines = fin.readlines()
        for line in lines:
            info = json.loads(line)
            fout.write(info["data"] + "\t" + info["label"][0] + "\n")
            label_set.add(info["label"][0])
    
    if label_fp is None:
        return
    with open(label_fp, "w") as fout:
        for key in label_set:
            fout.write(key + "\n")


if __name__ == "__main__":
    input_fp = "./all.jsonl"
    output_fp = "./all.txt"
    label_fp = "./label.txt"
    trans(input_fp, output_fp, label_fp)