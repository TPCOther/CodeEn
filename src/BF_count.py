import json
import re
import numpy as np
import sys
from benchmark.MCMD.bleu import bleuFromMaps, computeMaps1
from utils.tokenizer import Tokenizer
from difflib import SequenceMatcher
from suffix_trees import STree
import editdistance

sys.path.append("./")
from tqdm import tqdm

def calc_lcs(expected_orig, actual_orig):
    """
    https://stackoverflow.com/questions/18715688/find-common-substring-between-two-strings
    """
    try:
        input_list = [expected_orig, actual_orig]
        st = STree.STree(input_list)
        longest_lcs = st.lcs()

    except RecursionError as e:
        print(e)
        print(f"error in calc_lcs for {expected_orig} and {actual_orig}")
        match = SequenceMatcher(None, expected_orig, actual_orig)\
            .find_longest_match(0, len(expected_orig), 0, len(actual_orig))
        longest_lcs = expected_orig[match.a:match.a + match.size]


    return longest_lcs

def edit_distance(expected_orig, actual_orig):
    return editdistance.eval(expected_orig, actual_orig)

def repair_metric():
    file_dir = "/home/yanmeng/huangnaiqi/Assertion/outputs/BFs/PPL+repair/CodeT5_gpt-4o-2024-08-06_bf_mode.jarcard_5_results_v2.jsonl"

    before_hypothesis, after_hypothesis, references = [], [], []
    before_correct_count, after_correct_count = 0, 0
    ED, before_ED = 0, 0
    cls, before_cls = 0, 0
    all_count = 0
    with open(file_dir, "r") as f:
        lines = f.readlines()
        for index, line in tqdm(enumerate(lines)):
            line = json.loads(line)

            gold = line["fix"]

            generation = line["generation"]

            references.append(gold)

            after_hypothesis.append(generation)

            after_flag = Tokenizer.Recoder_whether_equally(
                expected=gold.strip(), actual=generation.strip())
            # after_flag = gold.strip() == generation.strip()

            if line['repair']:
                old_generation = line["BFs_generation"]
                before_hypothesis.append(old_generation)
                before_flag = Tokenizer.Recoder_whether_equally(
                    expected=gold.strip(), actual=old_generation.strip())
                # before_flag = gold.strip() == old_generation.strip()
                tokenize_old_generation = Tokenizer.Tokenize_code(old_generation)
                tokenize_gold = Tokenizer.Tokenize_code(gold)
                tokenize_old_generation = " ".join(tokenize_old_generation)
                tokenize_gold = " ".join(tokenize_gold)
                
                before_ED += edit_distance(tokenize_gold, tokenize_old_generation)
                before_cls += len(calc_lcs(gold, old_generation)) / len(gold)
                
                if before_flag:
                    before_correct_count += 1
                # if before_flag and not after_flag:
                #     print("Before: ", old_generation)
                #     print("After: ", generation)
            else:
                before_hypothesis.append(generation)
                
                tokenize_generation = Tokenizer.Tokenize_code(generation)
                tokenize_gold = Tokenizer.Tokenize_code(gold)
                tokenize_generation = " ".join(tokenize_generation)
                tokenize_gold = " ".join(tokenize_gold)
                before_ED += edit_distance(tokenize_gold, tokenize_generation)
                before_cls += len(calc_lcs(gold, generation)) / len(gold)
                
                if after_flag:
                    before_correct_count += 1

            if after_flag:
                after_correct_count += 1

            tokenized_gold = Tokenizer.Tokenize_code(gold)
            tokenized_generation = Tokenizer.Tokenize_code(generation)
            tokenized_gold = " ".join(tokenized_gold)
            tokenized_generation = " ".join(tokenized_generation)
            
            cls += len(calc_lcs(gold, generation)) / len(gold)

            ED += edit_distance(tokenized_gold, tokenized_generation)
            all_count += 1

    print("Before accuracy: ", before_correct_count / len(before_hypothesis))
    print("After accuracy: ", after_correct_count / len(after_hypothesis))

    before_hypothesis = [
        re.sub('[0-9]+\t', '', v, 1).strip() for v in before_hypothesis
    ]

    references = [re.sub('[0-9]+\t', '', v, 1).strip() for v in references]

    res = [f"{k}\t{v}" for k, v in enumerate(before_hypothesis)]
    gts = [f"{k}\t{v}" for k, v in enumerate(references)]

    print("Before:")

    (goldMap, predictionMap) = computeMaps1(res, gts)
    print(bleuFromMaps(goldMap, predictionMap)[0])

    print("===============================================")

    print("After:")

    after_hypothesis = [
        re.sub('[0-9]+\t', '', v, 1).strip() for v in after_hypothesis
    ]

    references = [re.sub('[0-9]+\t', '', v, 1).strip() for v in references]

    res = [f"{k}\t{v}" for k, v in enumerate(after_hypothesis)]
    gts = [f"{k}\t{v}" for k, v in enumerate(references)]

    (goldMap, predictionMap) = computeMaps1(res, gts)
    print(bleuFromMaps(goldMap, predictionMap)[0])
    print("Average LCS: ", cls / all_count)
    print("Average ED: ", ED / all_count)
    print("Average before ED: ", before_ED / all_count)
    print("Average before LCS: ", before_cls / all_count)


def gen_metric():

    file_dir = "/home/yanmeng/huangnaiqi/Assertion/outputs/BFs/PPL+repair/#gpt-3.5-turbo-0125_bf_mode.jarcard_5_gen_catch.jsonl"

    
    after_hypothesis, after_correct_count = [], 0
    with open(file_dir, "r") as f:
        lines = f.readlines()
        for index, line in tqdm(enumerate(lines)):
            line = json.loads(line)

            gold = line["fix"]

            generation = line["best_repair"]

            after_hypothesis.append(generation)

            after_flag = Tokenizer.Recoder_whether_equally(
                expected=gold.strip(), actual=generation.strip())
            # after_flag = gold.strip() == generation.strip()

            if after_flag:
                after_correct_count += 1

    print("accuracy: ", after_correct_count / len(after_hypothesis))



if __name__ == "__main__":
    repair_metric()
    # gen_metric()