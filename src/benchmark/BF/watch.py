import json
import sys

import numpy as np
from tqdm import tqdm
sys.path.append("/data/swf/Assertion/src/")
from utils.tokenizer import Tokenizer

from benchmark.JCSD.bleu import bleu
from transformers import T5ForConditionalGeneration, AutoTokenizer

if __name__ == "__main__":
    correct, one_correct = 0, 0
    with open("/data/swf/Assertion/Dataset/BFs/unix_10_test.json.rank.jsonl", "r") as f:
        lines = f.readlines()
        for index, line in tqdm(enumerate(lines)):
            line = json.loads(line)
            gold = line["fix"]
            generations = line["candidate"]
            one_generation = line["generation"]
            for i in range(10):
                generation = generations[i]
                if Tokenizer.Recoder_whether_equally(expected=gold.strip(), actual=generation.strip()):
                    correct += 1
                    break
            if Tokenizer.Recoder_whether_equally(expected=gold.strip(), actual=one_generation.strip()):
                one_correct += 1

    print(correct / len(lines))
    print(one_correct / len(lines))

    

# model_name = "./codet5-base"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# token_lenghs = []
# with open("/data/swf/Assertion/Dataset/ConCode/data_concode_test.json","r") as f:
#     lines = f.readlines()
#     for index, line in enumerate(lines):
#         line = json.loads(line)
#         gold = line["code"]
#         # 获取token后的长度
#         token_lengh = tokenizer(gold)['input_ids']

#         token_lenghs.append(len(token_lengh))

# token_lenghs = np.array(token_lenghs)
# print(np.mean(token_lenghs))
# print(np.max(token_lenghs))
# print(np.min(token_lenghs))
# print(np.percentile(token_lenghs, 90))
        