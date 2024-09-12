import json
import re
import numpy as np
import sys
from benchmark.MCMD.bleu import bleuFromMaps, computeMaps1
from difflib import get_close_matches
from difflib import SequenceMatcher
from suffix_trees import STree
from utils.tokenizer import Tokenizer
import editdistance

sys.path.append("./")
from tqdm import tqdm

def is_exact_match(generation, actual):
    expected = ''
    for line in generation.split('\n'):
        expected += re.sub(' +', ' ', line.strip())
    
    codex = ''
    for line in actual.split('\n'):
        codex += re.sub(' +', ' ', line.strip())

    return expected == codex or expected in codex


def is_match(expected_orig, actual_orig, buggy_code, warning_line):
    if is_exact_match(expected_orig, actual_orig):
        return True
    
    splitted_buggy_code = buggy_code.split('\n')
    splitted_expected_code = expected_orig.split('\n')
    splitted_codex_code = actual_orig.split('\n')

    expected = ''
    for line in expected_orig.split('\n'):
        expected += re.sub(' +', ' ', line.strip())
    
    actual = ''
    for line in actual_orig.split('\n'):
        actual += re.sub(' +', ' ', line.strip())

    if len(splitted_buggy_code) > len(splitted_expected_code):
        closest_match_expected = get_close_matches(warning_line, splitted_expected_code, n=1)
        if not len(closest_match_expected): ## The warning line was completely removed
            if warning_line not in expected and warning_line not in actual:
                return True
        else: ## The warning line exists but is modified
            closest_match_codex = get_close_matches(warning_line, splitted_codex_code, n=1)
            if len(closest_match_codex) == len(closest_match_expected) == 1:
                if closest_match_expected[0] \
                    .replace(' ', '') \
                    .replace('==', '===') \
                    .replace('(', '').replace(')', '') == closest_match_codex[0] \
                    .replace(' ', '') \
                    .replace('==', '===') \
                    .replace('(', '').replace(')', ''):
                    return True
    
    elif len(splitted_buggy_code) == len(splitted_expected_code) == len(splitted_codex_code):
        i = 0
        buggy_line_num = -1
        for line in splitted_buggy_code:
            if line.strip() == warning_line:
                buggy_line_num = i
            i += 1
        
        if buggy_line_num != -1 and splitted_codex_code[buggy_line_num] \
            .replace(' ', '') \
            .replace('==', '===') \
            .replace('(', '').replace(')', '') == splitted_expected_code[buggy_line_num] \
            .replace(' ', '') \
            .replace('==', '===') \
            .replace('(', '').replace(')', ''):
            return True 

    return False

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
    file_dir = "./outputs/NBFs/#CodeGen_DeepSeek-Coder-V2-Lite-Instruct_bf_mode.jarcard_15_results_v2.jsonl"

    before_hypothesis, after_hypothesis, references = [], [], []
    before_correct_count, after_correct_count = 0, 0
    after_cls, before_cls = 0, 0
    before_ED, after_ED = 0, 0
    with open(file_dir, "r") as f:
        lines = f.readlines()
        for index, line in tqdm(enumerate(lines)):
            line = json.loads(line)

            gold = line["fix"]

            generation = line["generation"]

            references.append(gold)

            after_hypothesis.append(generation)

            after_flag = gold.strip() == generation.strip()

            after_cls += len(calc_lcs(gold, generation)) / len(gold)

            after_ED += edit_distance(gold, generation)
            
            if line['repair']:
                old_generation = line["NBFs_generation"]
                before_hypothesis.append(old_generation)
                before_flag = gold.strip() == old_generation.strip()
                before_cls += len(calc_lcs(gold, old_generation)) / len(gold)
                before_ED += edit_distance(gold, old_generation)
                if before_flag:
                    before_correct_count += 1
                # if before_flag and not after_flag:
                #     print("Before: ", old_generation)
                #     print("After: ", generation)
            else:
                before_hypothesis.append(generation)
                if after_flag:
                    before_correct_count += 1

            if after_flag:
                after_correct_count += 1

    print(len(before_hypothesis), len(after_hypothesis), len(references))
    print("Before accuracy: ", before_correct_count / len(before_hypothesis))
    print("After accuracy: ", after_correct_count / len(after_hypothesis))
    print("Before cls: ", before_cls / len(before_hypothesis))
    print("After cls: ", after_cls / len(after_hypothesis))
    print("Before ED: ", before_ED / len(before_hypothesis))
    print("After ED: ", after_ED / len(after_hypothesis))

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


def gen_metric():

    file_dir = "./outputs/NBFs/CodeGen_DeepSeek-Coder-V2-Lite-Instruct_bf_mode.jarcard_15_catch_v2.jsonl"

    
    after_hypothesis, after_correct_count = [], 0
    with open(file_dir, "r") as f:
        lines = f.readlines()
        for index, line in tqdm(enumerate(lines)):
            line = json.loads(line)

            gold = line["fix"]

            generation = line["best_repair"]

            after_hypothesis.append(generation)

            after_flag = gold.strip() == generation.strip()

            if after_flag:
                after_correct_count += 1

    print("accuracy: ", after_correct_count / len(after_hypothesis))



if __name__ == "__main__":
    repair_metric()
    # gen_metric()