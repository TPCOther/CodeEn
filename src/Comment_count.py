import argparse
import json
import random
import re
import numpy as np
import sys

import torch
from eval.bleu.bleu import Bleu
from eval.cider.cider import Cider
from utils.tokenizer import Tokenizer
from eval.meteor.meteor import Meteor
from eval.rouge.rouge import Rouge
sys.path.append("/data/swf/Assertion")
from tqdm import tqdm
from benchmark.JCSD.bleu import bleu, bleuFromMaps, computeMaps, computeMaps1
from benchmark.JCSD.bleu import bleu
import editdistance
from transformers import AutoTokenizer
from bert_score import score
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F

def edit_distance(expected_orig, actual_orig):
    return editdistance.eval(expected_orig, actual_orig)

model = SentenceTransformer("all-roberta-large-v1")

def repair_metric(file_dir=None):
    if file_dir is None:
        file_dir = "/home/yanmeng/huangnaiqi/Assertion/outputs/JCSD_4o/CodeT5_gpt-4o-2024-08-06_comment_mode.jarcard_5_results_v2.jsonl"

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

    lres = [x[0] for x in res.values()]
    lgts = [x[0] for x in gts.values()]
    
    gts_embedding = model.encode(lgts)
    res_embedding = model.encode(lres)
    gts_embedding = torch.tensor(gts_embedding).to("cuda")
    res_embedding = torch.tensor(res_embedding).to("cuda")
    scores = F.cosine_similarity(gts_embedding, res_embedding, dim=1)
    print("Similarity: ", torch.mean(scores))
    P, R, F1 = score(lgts, lres, lang='en', verbose=True, rescale_with_baseline=True)
    print("F1 Score:", F1.mean())
    
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

    lres = [x[0] for x in res.values()]
    lgts = [x[0] for x in gts.values()]
    
    gts_embedding = model.encode(lgts)
    res_embedding = model.encode(lres)
    gts_embedding = torch.tensor(gts_embedding).to("cuda")
    res_embedding = torch.tensor(res_embedding).to("cuda")
    print("gts_embedding: ", gts_embedding.shape)
    print("res_embedding: ", res_embedding.shape)
    scores = F.cosine_similarity(gts_embedding, res_embedding, dim=1)
    print("score: ", scores.shape)
    print("Similarity: ", torch.mean(scores))
    P, R, F1 = score(lgts, lres, lang='en', verbose=True, rescale_with_baseline=True)
    print("F1 Score:", F1.mean())
    
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

    file_dir = "/home/yanmeng/huangnaiqi/Assertion/outputs/JCSD_agent/DeepSeek-Coder-V2-Lite-Instruct_comment_mode.jarcard_5_gen_catch.jsonl"

    after_hypothesis, references = [], []
    before_mul_blue = 0
    token_usage = 0
    fix_time = 0
    with open(file_dir, "r") as f:
        lines = f.readlines()
        for index, line in tqdm(enumerate(lines)):
            line = json.loads(line)

            gold = line["comment"]

            generation = line["best_repair"]
            
            token_usage += line["token_usage"]
            fix_time += line["fix_times"]

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
    
    print("Average token usage: ", token_usage / len(lines))
    print("Average fix time: ", fix_time / len(lines))

    # score_Meteor, scores_Meteor = Meteor().compute_score(gts, res)
    # print("Meteor: ", score_Meteor),

    # score_Rouge, scores_Rouge = Rouge().compute_score(gts, res)
    # print("Rouge: ", score_Rouge)

    # score_Cider, scores_Cider = Cider().compute_score(gts, res)
    # print("Cider: ", score_Cider),

def cal_token_usage():
    
    file_dir = "/home/yanmeng/huangnaiqi/Assertion/outputs/JCSD/CodeT5_Meta-Llama-3.1-8B-Instruct_comment_mode.jarcard_5_results_v2.jsonl"
    file_gen = "/home/yanmeng/huangnaiqi/Assertion/outputs/JCSD/CodeT5_Meta-Llama-3.1-8B-Instruct_comment_mode.jarcard_5_catch_v2.jsonl"
    file_rank = "/home/yanmeng/huangnaiqi/Assertion/outputs/JCSD/Meta-Llama-3.1-8B-Instruct_comment_mode.jarcard_5_gen_catch.jsonl"
    
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct", trust_remote_code=True)
    
    gen_data = {}
    rank_data = {}
    with open(file_gen, "r") as f:
        for line in f:
            line = json.loads(line)
            gen_data[line["code"]] = line
    with open(file_rank, "r") as f:
        for line in f:
            line = json.loads(line)
            rank_data[line["code"]] = line

    after_hypothesis, references = [], []
    before_mul_blue = 0
    token_usage = 0
    fix_time = 0
    with open(file_dir, "r") as f:
        lines = f.readlines()
        for index, line in tqdm(enumerate(lines)):
            line = json.loads(line)

            gold = line["comment"]

            generation = line["generation"]
            
            if line["repair"]:
                token_usage += len(tokenizer.apply_chat_template(gen_data[line["code"]]["prompt"]))
                token_usage += len(tokenizer.apply_chat_template(rank_data[line["code"]]["prompt"]))
                

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
    
    print("Average token usage: ", token_usage / len(lines))
    print("Average fix time: ", fix_time / len(lines))

def select_metric(file_dir=None):
    if file_dir is None:
        file_dir = "/home/yanmeng/huangnaiqi/Assertion/outputs/JCSD/CodeT5_gpt-3.5-turbo-0125_comment_mode.jarcard_5_results_v2.jsonl"

    before_hypothesis, after_hypothesis, references = [], [], []
    before_mul_blue, after_mul_blue = 0, 0
    repair_count = 0
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
                repair_count += 1
                old_generation = line["old_generation"]
                before_hypothesis.append(old_generation)
                before_mul_blue += bleu([gold], old_generation)[0]

    print("Muti BLEU", before_mul_blue / repair_count)

def rd_metric(file_dir=None):
    if file_dir is None:
        file_dir = "/home/yanmeng/huangnaiqi/Assertion/outputs/JCSD/CodeT5_gpt-3.5-turbo-0125_comment_mode.jarcard_5_results_v2.jsonl"

    gen_file_dir = file_dir.replace("results", "gen_results")
    rank_file_dir = file_dir.replace("results", "rank_results")
    before_hypothesis, after_hypothesis, references = [], [], []
    before_mul_blue, after_mul_blue = 0, 0
    before_ED, repair_ED = 0, 0
    repair_all = 0
    with open(file_dir, "r") as f, open(gen_file_dir, "r") as g, open(rank_file_dir, "r") as r:
        lines = f.readlines()
        rlines = r.readlines()
        glines = g.readlines()
        for index, line in tqdm(enumerate(zip(lines, glines, rlines))):
            gline = json.loads(line[1])
            rline = json.loads(line[2])
            line = json.loads(line[0])
            rd = random.randint(0, 1)

            gold = line["comment"]

            if line["repair"]:
                generation = gline["generation"] if rd else rline["generation"]
            else:
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

    # score_Meteor, scores_Meteor = Meteor().compute_score(gts, res)
    # print("Meteor: ", score_Meteor),

    # score_Rouge, scores_Rouge = Rouge().compute_score(gts, res)
    # print("Rouge: ", score_Rouge)

    # score_Cider, scores_Cider = Cider().compute_score(gts, res)
    # print("Cider: ", score_Cider),

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

    # score_Meteor, scores_Meteor = Meteor().compute_score(gts, res)
    # print("Meteor: ", score_Meteor),

    # score_Rouge, scores_Rouge = Rouge().compute_score(gts, res)
    # print("Rouge: ", score_Rouge)

    # score_Cider, scores_Cider = Cider().compute_score(gts, res)
    # print("Cider: ", score_Cider),

    

if __name__ == "__main__":
    random.seed(42)
    argsparse = argparse.ArgumentParser()
    argsparse.add_argument("--file_dir", type=str, default=None)
    args = argsparse.parse_args()
    # select_metric(args.file_dir)
    repair_metric(args.file_dir)
    # rd_metric(args.file_dir)
    # gen_metric()
    # cal_token_usage()