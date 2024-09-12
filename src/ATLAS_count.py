import json

from tqdm import tqdm
from utils.javatokenizer.tokenizer import tokenize_java_code_o, tokenize_java_code_origin, tokenize_java_code_raw, tokenize_string_literal

from utils.tokenizer import Tokenizer
from difflib import SequenceMatcher
from suffix_trees import STree
import editdistance

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


def repair_accuracy():
    file_dir = "./outputs/ATLAS_LOU/CodeT5_gpt-3.5-turbo-0125_atlas_mode.jarcard_5_results_v2.jsonl"

    all_correct_count, all_count, repair_correct_count, repair_all = 0, 0, 0, 0
    repair_error, still_correct, still_error, repair_correct = 0, 0, 0, 0
    cls = 0
    ED = 0
    repair_ED, before_ED = 0, 0
    with open(file_dir, "r") as f:
        lines = f.readlines()
        for index, line in tqdm(enumerate(lines)):
            line = json.loads(line)

            gold = line["gold"]

            generation = line["generation"]

            after_flag = Tokenizer.whether_equally(expected=gold.strip(), actual=generation.strip())

            if line['repair']:
                repair_all += 1

                old_generation = line["old_generation"]

                tokenize_old_generation = Tokenizer.Tokenize_code(old_generation)
                tokenize_gold = Tokenizer.Tokenize_code(gold)
                tokenize_old_generation = " ".join(tokenize_old_generation)
                tokenize_gold = " ".join(tokenize_gold)
                

                repair_ED += edit_distance(tokenize_gold, tokenize_old_generation)
                before_ED += edit_distance(tokenize_gold, tokenize_old_generation)
            
            else:
                generation = line["generation"]

                tokenize_generation = Tokenizer.Tokenize_code(generation)
                tokenize_gold = Tokenizer.Tokenize_code(gold)
                tokenize_generation = " ".join(tokenize_generation)
                tokenize_gold = " ".join(tokenize_gold)

                before_ED += edit_distance(tokenize_gold, tokenize_generation)

            #     before_flag = Tokenizer.whether_equally( expected= gold.strip(), actual=old_generation.strip())

            #     if before_flag == True and after_flag == False:
            #         repair_error += 1
            #     if before_flag == True and after_flag == True:
            #         still_correct += 1
            #     if before_flag == False and after_flag == False:
            #         still_error += 1
            #     if before_flag == False and after_flag == True:
            #         repair_correct += 1

            if after_flag:
                all_correct_count += 1

                # if line['repair']:
                #     repair_correct_count += 1
            gold = Tokenizer.Tokenize_code(gold)
            generation = Tokenizer.Tokenize_code(generation)
            gold = " ".join(gold)
            generation = " ".join(generation)
            
            cls += len(calc_lcs(gold, generation)) / len(gold)

            ED += edit_distance(gold, generation)


            all_count += 1

    print("all_count: ", all_count)
    print("Total accuracy: ", all_correct_count / all_count)
    print("Average LCS: ", cls / all_count)
    print("Average ED: ", ED / all_count)
    print("Average repair ED: ", repair_ED / repair_all)
    print("Average before ED: ", before_ED / all_count)
    print("repair_all: ", repair_all, "all_count: ", all_count)

    # print("repair_all: ", repair_all)
    # print("repair_correct_count: ", repair_correct_count)
    # print("Repair accuracy: ", repair_correct_count / repair_all)

    # print("repair_error: ", repair_error)
    # print("repair_correct: ", repair_correct)
    # print("still_correct: ", still_correct)
    # print("still_error: ", still_error)
    # print("repair_total", (repair_error + still_correct + still_error + repair_correct))


def gen_accuracy():
    file_dir = "./outputs/ATLAS_LOU/CodeGen_gpt-3.5-turbo-0125_atlas_mode.jarcard_5_rank_results_v2.jsonl"
    all_correct_count, all_count = 0, 0
    with open(file_dir, "r") as f:
        lines = f.readlines()
        for index, line in tqdm(enumerate(lines)):
            line = json.loads(line)

            gold = line["gold"]

            generation = line["best_repair"]

            after_flag = Tokenizer.whether_equally(expected=gold.strip(),
                                                   actual=generation.strip())

            if after_flag:
                all_correct_count += 1

            all_count += 1

    print("all_count: ", all_count)
    print("Total accuracy: ", all_correct_count / all_count)


if __name__ == "__main__":
    # gen_accuracy()
    repair_accuracy()