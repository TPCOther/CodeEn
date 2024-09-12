import json
import re
import numpy as np
import sys
from eval.bleu.bleu import Bleu
from eval.cider.cider import Cider
from utils.tokenizer import Tokenizer
from eval.meteor.meteor import Meteor
from eval.rouge.rouge import Rouge
sys.path.append("./")
from tqdm import tqdm
from benchmark.JCSD.bleu import bleu, bleuFromMaps, computeMaps, computeMaps1
from benchmark.JCSD.bleu import bleu
import editdistance

def edit_distance(expected_orig, actual_orig):
    return editdistance.eval(expected_orig, actual_orig)

def repair_metric():

    file_dir = "./outputs/JCSD/CodeT5_gpt-3.5-turbo-0125_comment_mode.jarcard_5_results_v2.jsonl"

    before_hypothesis, after_hypothesis, references = [], [], []
    before_mul_blue, after_mul_blue = 0, 0
    before_ED, repair_ED = 0, 0
    repair_all = 0
    with open(file_dir, "r") as f:
        lines = f.readlines()
        for index, line in tqdm(enumerate(lines)):
            line = json.loads(line)

            gold = line["comment"]

            generation = line["generation"]

            references.append(gold)
            after_hypothesis.append(generation)

            after_mul_blue += bleu([gold], generation)[0]

            if line['repair']:
                old_generation = line["old_generation"]
                before_hypothesis.append(old_generation)
                before_mul_blue += bleu([gold], old_generation)[0]

                repair_all += 1

                tokenize_old_generation = " ".join(Tokenizer.Tokenize_code(old_generation))
                tokenize_gold = " ".join(Tokenizer.Tokenize_code(gold))
                repair_ED += edit_distance(tokenize_gold, tokenize_old_generation)
                before_ED += edit_distance(tokenize_gold, tokenize_old_generation)
            else:
                before_hypothesis.append(generation)
                before_mul_blue += bleu([gold], generation)[0]

                tokenize_generation = " ".join(Tokenizer.Tokenize_code(generation))
                tokenize_gold = " ".join(Tokenizer.Tokenize_code(gold))
                before_ED += edit_distance(tokenize_gold, tokenize_generation)

    res = {
        k: [re.sub('[0-9]+\t', '', v, 1).strip().lower()]
        for k, v in enumerate(before_hypothesis)
    }
    gts = {
        k: [re.sub('[0-9]+\t', '', v, 1).strip().lower()]
        for k, v in enumerate(references)
    }    


    print("Before ED:", before_ED / len(lines))
    print("Repair ED:", repair_ED / repair_all)
    print(len(lines), repair_all, before_ED, repair_ED)

    print("Before:")
    print ("Muti BLEU", before_mul_blue / len(lines))

    score_Bleu, scores_Bleu = Bleu(4).compute_score(gts, res)

    print("Corpus-level Bleu_1: ", score_Bleu[0])
    print("Corpus-level Bleu_2: ", score_Bleu[1])
    print("Corpus-level Bleu_3: ", score_Bleu[2])
    print("Corpus-level Bleu_4: ", score_Bleu[3])
    print("Sentence-level Bleu_1: ", np.mean(scores_Bleu[0]))
    print("Sentence-level Bleu_2: ", np.mean(scores_Bleu[1]))
    print("Sentence-level Bleu_3: ", np.mean(scores_Bleu[2]))
    print("Sentence-level Bleu_4: ", np.mean(scores_Bleu[3]))

    score_Meteor, scores_Meteor = Meteor().compute_score(gts, res)
    print("Meteor: ", score_Meteor),

    score_Rouge, scores_Rouge = Rouge().compute_score(gts, res)
    print("Rouge: ", score_Rouge)

    score_Cider, scores_Cider = Cider().compute_score(gts, res)
    print("Cider: ", score_Cider),

    print("===============================================")

    print("After:")

    res = {
        k: [re.sub('[0-9]+\t', '', v, 1).strip().lower()]
        for k, v in enumerate(after_hypothesis)
    }
    gts = {
        k: [re.sub('[0-9]+\t', '', v, 1).strip().lower()]
        for k, v in enumerate(references)
    }

    score_Bleu, scores_Bleu = Bleu(4).compute_score(gts, res)

    print("Mutiple BLEU: ", after_mul_blue / len(lines))

    print("Corpus-level Bleu_1: ", score_Bleu[0])
    print("Corpus-level Bleu_2: ", score_Bleu[1])
    print("Corpus-level Bleu_3: ", score_Bleu[2])
    print("Corpus-level Bleu_4: ", score_Bleu[3])
    print("Sentence-level Bleu_1: ", np.mean(scores_Bleu[0]))
    print("Sentence-level Bleu_2: ", np.mean(scores_Bleu[1]))
    print("Sentence-level Bleu_3: ", np.mean(scores_Bleu[2]))
    print("Sentence-level Bleu_4: ", np.mean(scores_Bleu[3]))

    score_Meteor, scores_Meteor = Meteor().compute_score(gts, res)
    print("Meteor: ", score_Meteor),

    score_Rouge, scores_Rouge = Rouge().compute_score(gts, res)
    print("Rouge: ", score_Rouge)

    score_Cider, scores_Cider = Cider().compute_score(gts, res)
    print("Cider: ", score_Cider),

def gen_metric():

    file_dir = "./outputs/JCSD/DeepSeek-Coder-V2-Lite-Instruct_comment_mode.jarcard_5_gen_catch.jsonl"

    after_hypothesis, references = [], []
    before_mul_blue = 0
    with open(file_dir, "r") as f:
        lines = f.readlines()
        for index, line in tqdm(enumerate(lines)):
            line = json.loads(line)

            gold = line["comment"]

            generation = line["best_repair"]

            references.append(gold)
            after_hypothesis.append(generation)

            before_mul_blue += bleu([gold], generation)[0]


    res = {
        k: [re.sub('[0-9]+\t', '', v, 1).strip().lower()]
        for k, v in enumerate(after_hypothesis)
    }
    gts = {
        k: [re.sub('[0-9]+\t', '', v, 1).strip().lower()]
        for k, v in enumerate(references)
    }

    print("Before:")
    print ("Muti BLEU", before_mul_blue / len(lines))    

    score_Bleu, scores_Bleu = Bleu(4).compute_score(gts, res)

    print("Corpus-level Bleu_1: ", score_Bleu[0])
    print("Corpus-level Bleu_2: ", score_Bleu[1])
    print("Corpus-level Bleu_3: ", score_Bleu[2])
    print("Corpus-level Bleu_4: ", score_Bleu[3])
    print("Sentence-level Bleu_1: ", np.mean(scores_Bleu[0]))
    print("Sentence-level Bleu_2: ", np.mean(scores_Bleu[1]))
    print("Sentence-level Bleu_3: ", np.mean(scores_Bleu[2]))
    print("Sentence-level Bleu_4: ", np.mean(scores_Bleu[3]))

    # score_Meteor, scores_Meteor = Meteor().compute_score(gts, res)
    # print("Meteor: ", score_Meteor),

    # score_Rouge, scores_Rouge = Rouge().compute_score(gts, res)
    # print("Rouge: ", score_Rouge)

    # score_Cider, scores_Cider = Cider().compute_score(gts, res)
    # print("Cider: ", score_Cider),

if __name__ == "__main__":
    repair_metric()
    # gen_metric()