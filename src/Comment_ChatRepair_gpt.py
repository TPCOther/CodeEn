import hashlib
import json
import math
from multiprocessing import Pool
import os
import re
import sys
import time
from typing import List
import jsonlines
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, IsolationForest
from sklearn.metrics import precision_score
from sklearn.neighbors import KernelDensity
import tiktoken as tt
import torch
from tqdm import tqdm
from Comment_prompt_construction import OpenAIClientSingleton, comment_gen_demo_prompt, comment_gen_system_prompt, comment_gen_user_prompt, comment_get_zero_shot_prompt, comment_get_zero_shot_prompt_repair, comment_repair_demo_prompt
from benchmark.JCSD.bleu import bleu
from benchmark.JCSD.ms_evaluate import read_examples_, unix_convert_examples_to_features, generate_prompt, INS
from utils import utils
from rank_bm25 import BM25Okapi
import concurrent
from utils.javatokenizer.tokenizer import tokenize_java_code_o
from utils.tokenizer import Tokenizer
from Comment_models import *
from dataset.comment_dataset import CommentDataset, CommentPrompt, build_comment_promp_jarcard, build_comment_prompt_bm25, build_vocab, process_query, process_query_vocab
from transformers import T5ForConditionalGeneration, AutoTokenizer, AutoModelForCausalLM, GenerationConfig,RobertaConfig, RobertaModel
from benchmark.JCSD.UnixCoder import Seq2Seq

# Usage
client = OpenAIClientSingleton.get_client()


def load_bm_25(bm_25_cache_dict, test_methods, training_data_with_length):
    '''
    BM25是信息索引领域用来计算query与文档相似度得分的经典算法。
    '''
    start_time = time.time()
    how_many_md5hash_conflicts = 0

    for dp in training_data_with_length:

        # tokenized_test_method = dp.datapoint.test_method.split(" ")
        tokenized_test_method = Tokenizer.Tokenize_code(dp.datapoint.code)

        md5hash = hashlib.md5(
            " ".join(tokenized_test_method).encode('utf-8')).hexdigest()

        if md5hash in bm_25_cache_dict:
            how_many_md5hash_conflicts += 1
        else:
            bm_25_cache_dict[md5hash] = dp
        # test_methods.append(dp.datapoint.test_method)
        test_methods.append(" ".join(tokenized_test_method))

    print("how_many_md5hash_conflicts: ", how_many_md5hash_conflicts)

    bm25 = BM25Okapi(test_methods)

    end_time = time.time()
    print("load_bm_25: ", end_time - start_time)
    print("The size of the bm25 cache is {} bytes".format(
        sys.getsizeof(bm_25_cache_dict)))
    print(f"total entries: {len(bm_25_cache_dict.keys())}")
    return bm25



def load_jarcard(jarcard_cache_dict, test_methods, training_data_with_length):
    '''
    BM25是信息索引领域用来计算query与文档相似度得分的经典算法。
    '''
    how_many_md5hash_conflicts = 0

    for dp in tqdm(training_data_with_length):

        # tokenized_test_method = dp.datapoint.test_method.split(" ")
        tokenized_test_method = Tokenizer.Tokenize_code(dp.datapoint.code)

        md5hash = hashlib.md5(
            " ".join(tokenized_test_method).encode('utf-8')).hexdigest()

        if md5hash in jarcard_cache_dict:
            how_many_md5hash_conflicts += 1
        else:
            jarcard_cache_dict[md5hash] = dp

        test_methods.append(" ".join(tokenized_test_method))

    print("how_many_md5hash_conflicts: ", how_many_md5hash_conflicts)

    print("The size of the jarcard cache is {} bytes".format(
        sys.getsizeof(jarcard_cache_dict)))

    print(f"total entries: {len(jarcard_cache_dict.keys())}")



def extractResultTemplate(text, templates):
    # this pattern is for ChatGPT
    # pattern = re.compile('<Repair> <Event\d> (.+) <Repair>')
    start, end = templates
    pattern = re.compile(f'{start}(.+){end}', re.DOTALL)
    # findall return a list
    result = pattern.findall(text)
    if (len(result)): return result[0].strip()
    else: return ""


def get_gpt_result(prompt, line_index, templates, mode, model, n, temperture,
                   t: Comment_Datapoint):
    re_id, max_temps = 0, 5
    while True:
        try:
            response = OpenAIClientSingleton.get_response(
                messages=prompt,
                temperature=temperture,
                model=model,
                n=n,
                logprobs=True)

        except Exception as e:
            print(e)
            print("Request busy, code comment {} is now waiting ...".format(
                line_index))
            re_id += 1
            if re_id < max_temps:
                if "repetitive patterns" in str(e):
                    for a in prompt:
                        # 删除第一次出现user以及assistant的对话
                        if a['role'] == 'user':
                            prompt.remove(a)
                            break
                    for a in prompt:
                        if a['role'] == 'assistant':
                            prompt.remove(a)
                            break
                time.sleep(1)
            else:
                repaired_comment = []
                print("Comment {} is failed to repair".format(line_index))
                raise Exception("Chat API is busy")
        else:
            repaired_comment = []
            ppls = []
            for choice_index, choice in enumerate(response.choices):
                result = extractResultTemplate(choice.message.content,
                                               templates[mode])

                if result != "":
                    repaired_comment.append(result)

                    token_logprobs = [
                        (chattokenlogpro.token, chattokenlogpro.logprob)
                        for chattokenlogpro in choice.logprobs.content
                    ]
                    token_logprobs = token_logprobs[3:-3]
                    token_probs = [
                        math.exp(logp[1]) for logp in token_logprobs
                    ]
                    token_probs = torch.tensor(token_probs)
                    log_single_seq_probs = -torch.mean(torch.log(token_probs))
                    perplexity = torch.exp(log_single_seq_probs)
                    ppls.append(perplexity.item())     
                                   
            if len(repaired_comment) != 0:
                break
            elif len(repaired_comment) == 0 and re_id < max_temps:
                time.sleep(1)
                re_id += 1
            elif len(repaired_comment) == 0 and re_id >= max_temps:
                print("Comment {} is failed to repair".format(line_index))
                break
    if len(repaired_comment) == 0:
        best_repair = t.prediction
        best_ppl = 1000000
    else:
        if mode == comment_mode.zero_shot:
            best_repair = repaired_comment[0]
            best_ppl = ppls[0]
        else:
            best_repair = repaired_comment[0]
            best_ppl = ppls[0]

    result = {
        "index": str(line_index),
        "prompt": prompt,
        "comment": t.nl,
        "code": t.code,
        "best_repair": best_repair,
        "best_ppl": best_ppl,
    }

    return result, line_index


class ModelTester():

    def __init__(self, train_file_path, valid_file_path, test_file_name, result_dir,
                 model_under_test, demonstration_number):
        self.train_file_path = train_file_path
        self.valid_file_path = valid_file_path
        self.test_file_name = test_file_name
        self.result_dir = result_dir
        self.model_under_test = model_under_test
        self.data_set_path = self.result_dir + self.model_under_test
        self.demonstration_number = demonstration_number

        # if os.path.exists(self.data_set_path) == False:
        #     raise Exception("The model under test does not exist")

        #加载各类数据
        self.training_set = CommentDataset(self.train_file_path, mode="train")
        self.test_set = CommentDataset(self.test_file_name, mode="test")
        self.valid_set = CommentDataset(self.valid_file_path, mode="valid")

        self.test_objects = self.test_set.objects
        self.valid_objects = self.valid_set.objects

        print("Loading data ...")
        print("train_datas: ", len(self.training_set), train_file_path)
        print("test_datas: ", len(self.test_set), test_file_name)
        print("valid_datas: ", len(self.valid_set), valid_file_path)


        print("Parsing data ...")
        self.training_data: list[Comment_Datapoint] = self.training_set.parse()
        self.test_data: list[Comment_Datapoint] = self.test_set.parse()
        self.valid_data: list[Comment_Datapoint] = self.valid_set.parse()

        self.client = OpenAIClientSingleton.get_client()  # 获取openai client、

    def generate_final_result(self, model, mode, prediction_mode = "combined", **kwargs):

        if mode in [
                comment_mode.zero_shot,
                comment_mode.zero_shot_repair,
                comment_mode.bm_25,
                comment_mode.jarcard
        ]:
            output_file_name = self.data_set_path + f"_{model}_{mode}_{self.demonstration_number}_results_v2.jsonl"

        if prediction_mode == "rank":
            output_file_name = output_file_name.replace("results", "rank_results")
        elif prediction_mode == "gen":
            output_file_name = output_file_name.replace("results", "gen_results")

        with jsonlines.open(output_file_name, mode='w') as writer:
            for t in self.keep_test_data:
                object = {
                    "comment": t.nl,
                    "code": t.code,
                    "generation": t.prediction,
                    "repair": False
                }
                writer.write(object)
        if mode in [
                comment_mode.zero_shot,
                comment_mode.zero_shot_repair,
        ]:
            with jsonlines.open(output_file_name, mode='a') as writer:
                for index, t in zip(self.sorted_index_desc_select,
                                    self.selected_test_data):
                    object = self.catch[index]
                    save_object = {
                        "comment": object['comment'],
                        "code": object['code'],
                        "generation": object['best_repair'],
                        "old_generation": t.prediction,
                        "repair": True
                    }
                    writer.write(save_object)
        else:
            if prediction_mode == "combined":
                with jsonlines.open(output_file_name, mode='a') as writer:
                    for index, t in zip(self.sorted_index_desc_select,
                                        self.selected_test_data):
                        rank_object = self.catch[index]
                        gen_object = self.gen_catch[index]
                        save_object = {}
                        save_object['comment'] = rank_object['comment']
                        save_object['code'] = rank_object['code']

                        if gen_object["best_ppl"] < rank_object["best_ppl"]:
                            save_object["generation"] = gen_object["best_repair"]
                            save_object["best_choice"] = "gen"
                        else:
                            save_object["generation"] = rank_object["best_repair"]
                            save_object["best_choice"] = "rank"
                        
                        save_object["old_generation"] = t.prediction
                        save_object["repair"] = True
                        writer.write(save_object)

            elif prediction_mode == "rank":
                with jsonlines.open(output_file_name, mode='a') as writer:
                    for index, t in zip(self.sorted_index_desc_select,
                                        self.selected_test_data):
                        rank_object = self.catch[index]
                        save_object = {}
                        save_object['comment'] = rank_object['comment']
                        save_object['code'] = rank_object['code']
                        save_object["generation"] = rank_object["best_repair"]
                        save_object["old_generation"] = t.prediction
                        save_object["repair"] = True
                        writer.write(save_object)
            elif prediction_mode == "gen":
                with jsonlines.open(output_file_name, mode='a') as writer:
                    for index, t in zip(self.sorted_index_desc_select,
                                        self.selected_test_data):
                        gen_object = self.gen_catch[index]
                        save_object = {}
                        save_object['comment'] = gen_object['comment']
                        save_object['code'] = gen_object['code']
                        save_object["generation"] = gen_object["best_repair"]
                        save_object["old_generation"] = t.prediction
                        save_object["repair"] = True
                        writer.write(save_object)

        print(f"output: jsonl file: {os.path.abspath(output_file_name)}")

    def read_catch(self, model, mode, **kwargs):
        '''
        读取jsonl文件
        '''
        self.catch = {}
        if mode in [
                comment_mode.zero_shot,
                comment_mode.zero_shot_repair,
                comment_mode.bm_25, 
                comment_mode.jarcard
        ]:
            self.catch_name = self.result_dir + f"{self.model_under_test}_{model}_{mode}_{self.demonstration_number}_catch_v2.jsonl"
        print("read catch", self.catch_name)
        if os.path.exists(self.catch_name) == True:
            with jsonlines.open(self.catch_name) as reader:
                for obj in tqdm(reader):
                    index = obj['index']
                    index = int(index)
                    self.catch[index] = obj

    def read_gen_catch(self, model, mode, **kwargs):
        '''
        读取jsonl文件
        '''
        self.gen_catch = {}
        if mode in [
                comment_mode.zero_shot,
                comment_mode.zero_shot_repair,
                comment_mode.bm_25, 
                comment_mode.jarcard
        ]:
            self.gen_catch_name = self.result_dir + f"{model}_{mode}_{self.demonstration_number}_gen_catch.jsonl"

        if os.path.exists(self.gen_catch_name) == True:
            with jsonlines.open(self.gen_catch_name) as reader:
                for obj in reader:
                    index = obj['index']
                    index = int(index)
                    self.gen_catch[index] = obj


    def get_hidden_state(self, model, tokenizer, device, dataset_type, model_under_test, is_dropout=False):
        '''
        获取hidden state
        '''
        if is_dropout:
            model.train()
        else:
            model.eval()  

        global batch_size
        if dataset_type == "valid":
            datasets = self.valid_set.source_dataset
            objects = self.valid_objects
        elif dataset_type == "test":
            datasets = self.test_set.source_dataset
            objects = self.test_objects
        elif dataset_type == "train":
            datasets = self.training_set.source_dataset
            objects = self.training_set.objects
        
        hidden_states_result = []
    
        for i in tqdm(range(0, len(datasets), batch_size)):
            model_input = datasets[i:i + batch_size]
            model_objects = objects[i:i + batch_size]

            if model_under_test == "CodeT5":
                inputs = tokenizer(model_input,
                        max_length=512,
                        padding=True,
                        truncation=True,
                        return_tensors="pt").to(device)
               
                with torch.no_grad():
                    encoder_outputs = model.encoder(**inputs)
                    hidden_states = encoder_outputs['last_hidden_state']
                    eos_mask = inputs.input_ids.eq(tokenizer.eos_token_id)
                    #eos_mask查找每行中的eos token的索引位置
                    eos_mask = eos_mask.cumsum(1).cumsum(1).eq(1)

                if len(torch.unique(eos_mask.sum(1))) > 1:
                    with open("tokenizer_debug.txt", 'w') as file:
                        for input_id in inputs.input_ids:
                            decoded_text = tokenizer.decode(input_id, skip_special_tokens=False)
                            file.write("Decoded Text: " + decoded_text + "\n")
                        raise ValueError("All examples must have the same number of <eos> tokens.")
                    
                vec = hidden_states[eos_mask, :].view(hidden_states.size(0), -1,
                                                hidden_states.size(-1))[:, -1, :]

                hidden_states_result.append(vec.detach().cpu())
            
            elif model_under_test == "CodeGen":
                input_data = [
                    generate_prompt(INS, input)
                    for input in model_input
                ]
                inputs = tokenizer(input_data,
                                    max_length=1920,
                                    truncation=True,
                                    padding=True,
                                    return_tensors="pt").to(device)  
                
                with torch.no_grad():
                    outputs = model(**inputs, output_hidden_states=True)

                hidden_states = outputs.hidden_states[-1]

                vec = hidden_states[:, 0, :]

                hidden_states_result.append(vec.detach().cpu())

            if model_under_test == "Unixcoder":
                
                examples = read_examples_(model_objects)

                features = unix_convert_examples_to_features(
                    examples,
                    tokenizer,
                    max_source_length=512,
                    max_target_length=128)

                source_ids = torch.tensor([f.source_ids for f in features],
                                        dtype=torch.long).to(device)
                attention_mask = source_ids.ne(tokenizer.pad_token_id)

                with torch.no_grad():

                    vec = model.encoder(input_ids=source_ids, attention_mask=attention_mask)[0]

                    vec = (vec*source_ids.ne(1)[:,:,None]).sum(1)/source_ids.ne(1).sum(-1)[:,None]

                    hidden_states_result.append(vec.detach().cpu())
        
        model.eval() # 关闭dropout

        print(hidden_states_result[0].shape)

        hidden_states_embedding = np.vstack(hidden_states_result)       

        return hidden_states_embedding


    def metric_1(self, dataset_type):

        print(f"Starting {dataset_type} metric_1 ...")

        if dataset_type == "valid":
            valid_ppls = np.array(self.get_valid_ppls())
            valid_ppls = (valid_ppls - np.min(valid_ppls)) / (np.max(valid_ppls) -
                                                 np.min(valid_ppls))
            return valid_ppls

        elif dataset_type == "test":
            test_ppls = np.array(self.get_test_ppls())
            #如果test_ppls中存在nan值，将nan值替换为test_ppls的最大值
            if np.isnan(test_ppls).any():
                test_ppls[np.isnan(test_ppls)] = np.nanmax(test_ppls)
                

            # 对test_ppls进行正则化，使得其值在[0, 1]之间
            test_ppls = (test_ppls - np.min(test_ppls)) / (np.max(test_ppls) -
                                                        np.min(test_ppls))
            
            if np.isnan(test_ppls).any():
                nan_index = np.argwhere(np.isnan(test_ppls))
                print("nan_index", nan_index)
            
            return test_ppls

    def metric_4(self, dataset_type, model_under_test, times=5, tokenizer=None, model=None, device=None):
        '''
        根据droup out 来生成多个结果，并计算不一致性
        '''
        print(f"Starting {dataset_type} metric_4 ...")
        import torch.nn.functional as F
        def cal_cosin(pred):
            num_generations = len(pred)
            cosin_dvar = []
            for i in range(len(pred[0])):
                cosin = []
                for j in range(0, num_generations-1):
                    for k in range(j+1, num_generations):
                        if (j != k):
                            obj1_embdding = pred[j][i]
                            obj2_embdding = pred[k][i]
                            # torch计算余弦相似度
                            sim = F.cosine_similarity(torch.tensor(obj1_embdding), torch.tensor(obj2_embdding), dim=0)
                            cosin.append(sim)
                cosin_dvar.append(np.min(cosin) + np.mean(cosin))

            return cosin_dvar
        
        decode_embedds = []
        for _ in range(times):
            decode_embedds.append(self.get_hidden_state(model, tokenizer, device, dataset_type, model_under_test, is_dropout=True))
                    
        model.eval()
        hidden_vars = cal_cosin(decode_embedds)

        hidden_vars = np.array(hidden_vars)
        hidden_vars = (hidden_vars - np.min(hidden_vars)) / (np.max(hidden_vars) -
                                                np.min(hidden_vars))
        
        return hidden_vars    
    

    def metric_2(self, dataset_type):
        '''
        根据相似度来计算比较相应生成的相似度
        '''
        print(f"Starting {dataset_type} metric_2 ...")

        if dataset_type == "valid":
            datasets = self.valid_data
        elif dataset_type == "test":
            datasets = self.test_data

        jarcard_cache_dict, how_many_md5hash_conflicts, test_methods = {} , 0, []
        tokenized_dict = {}

        for dp in self.training_data:

            # tokenized_test_method = dp.datapoint.test_method.split(" ")
            tokenized_test_method = Tokenizer.Tokenize_code(dp.code)

            md5hash = hashlib.md5(
                " ".join(tokenized_test_method).encode('utf-8')).hexdigest()

            if md5hash in jarcard_cache_dict:
                how_many_md5hash_conflicts += 1
            else:
                jarcard_cache_dict[md5hash] = dp
                tokenized_dict[md5hash] = " ".join(tokenized_test_method)

            test_methods.append(" ".join(tokenized_test_method))

        print("how_many_md5hash_conflicts: ", how_many_md5hash_conflicts)
        
        if dataset_type == "valid":
            bm_catch_name = self.result_dir + f"{dataset_type}_jarcard_neighbor_metric3.jsonl"
            if os.path.exists(bm_catch_name) == True:
                bm_catch_neighbor = []
                if os.path.exists(bm_catch_name) != False:
                    with jsonlines.open(bm_catch_name) as reader:
                        for obj in reader:
                            hash_id = [ element[1] for element in obj]
                            bm_catch_neighbor.append(hash_id)
        else:
            bm_catch_name = self.result_dir + f"jarcard_neighbor.jsonl"
            if os.path.exists(bm_catch_name) == True:
                bm_catch_neighbor = []
                bm_catch_neighbor_map = {}
                with jsonlines.open(bm_catch_name) as reader:
                    for obj in reader:
                        index = obj['index']
                        index = int(index)
                        bm_catch_neighbor_map[index] = obj
                for t in range(len(datasets)):
                    bm_catch_neighbor.append(bm_catch_neighbor_map[t]["results_top_n"])
           

        # 如果命名空间不存在bm_catch_neighbor这个变量
                    
        if "bm_catch_neighbor" not in locals():
            bm_catch_neighbor = []

            print("Building vocab ...")
            md5_hash, vectorizer, candidate_set = build_vocab(test_methods)
            print("Builded")


            for index in tqdm(range(len(datasets))):
                t = datasets[index]
                retrive_results = process_query_vocab(vectorizer, candidate_set, t.code, md5_hash)

                assert len(retrive_results) == len(test_methods)
                # 得分从大到小排序
                retrive_results.sort(key=lambda x: x[0], reverse=True)
                
                top_results = retrive_results[:500]
                results_top_n = [x[1] for x in top_results]
                filtered_query = [tokenized_dict[md5] for md5 in results_top_n]
                top_results = process_query(t.code, filtered_query)
                top_results.sort(key=lambda x: x[0], reverse=True)
                retrive_results = top_results

                # 得分从大到小排序
                retrive_results.sort(key=lambda x: x[0], reverse=True)
                # 保存结果
                if dataset_type == "valid":
                    result = retrive_results[:10]
                    bm_catch_neighbor.append([element[1] for element in result])
                    with jsonlines.open(bm_catch_name, mode='a') as writer:
                        writer.write(retrive_results[:10])
                else:
                    top_retrive_results = retrive_results[:80]
                    results_top_n = [x[1] for x in top_retrive_results]

                    bottom_retrive_results = retrive_results[:-81:-1]
                    results_bottom_n = [x[1] for x in bottom_retrive_results]
                    with jsonlines.open(bm_catch_name, mode='a') as writer:
                        writer.write({"index": str(index), "results_top_n": results_top_n, "results_bottom_n": results_bottom_n})
                    bm_catch_neighbor.append(results_top_n)


        jarcard_sim = []

        for t in tqdm(datasets):
            retrive_results = bm_catch_neighbor.pop(0)

            code_sim = utils.getJaccard(Tokenizer.Tokenize_code(dp.code.strip()),
                                            Tokenizer.Tokenize_code(jarcard_cache_dict[retrive_results[0]].code.strip()))
            nl_not_sim = 1.0-utils.getJaccard(Tokenizer.Tokenize_code(t.nl.strip()), 
                                          Tokenizer.Tokenize_code(jarcard_cache_dict[retrive_results[0]].nl.strip()))
            
            jarcard_sim.append(code_sim * nl_not_sim)

        jarcard_sim = np.array(jarcard_sim)
        jarcard_sim = (jarcard_sim - np.min(jarcard_sim)) / (np.max(jarcard_sim) -
                                                    np.min(jarcard_sim))
        
        return jarcard_sim


    def metric_3(self, dataset_type, hidden_states, train_hidden_states):
        print(f"Starting {dataset_type} metric_3 ...")
        import torch.nn.functional as F
        y = torch.tensor(train_hidden_states, dtype=torch.float32).unsqueeze(0).to("cuda:1")
        chuck = 5000
        data_lists = torch.chunk(torch.tensor(hidden_states, dtype=torch.float32).unsqueeze(1), chuck, dim=0)
        values, indices = [], []
        for data in tqdm(data_lists):
            data = data.to("cuda:1")
            cosion_sim = F.cosine_similarity(data, y, dim=-1)
            value, indice = torch.topk(cosion_sim, k=int(1),
                                    dim=-1,
                                    largest=True,
                                    sorted=True)
            values.append(value.data.cpu())
            indices.append(indice.data.cpu())
            del data
        values, indices = torch.cat(values, dim=0), torch.cat(indices, dim=0)

        seman_sim_scores = []

        if dataset_type == "test":
            datasets = self.test_data
        elif dataset_type == "valid":
            datasets = self.valid_data

        for index, t in tqdm(enumerate(datasets)):
            indice = indices[index].item()
            code_sim = values[index].item()

            nl_not_sim = 1.0-utils.getJaccard(Tokenizer.Tokenize_code(t.nl.strip()), 
                                          Tokenizer.Tokenize_code(self.training_data[indice].nl.strip()))
            
            seman_sim_scores.append(code_sim * nl_not_sim)

        seman_sim_scores = np.array(seman_sim_scores)
        seman_sim_scores = (seman_sim_scores - np.min(seman_sim_scores)) / (np.max(seman_sim_scores) -
                                                    np.min(seman_sim_scores))
        
        return seman_sim_scores
        




    def priorization(self, method = "OFTOR"):
        '''
        对于Testing Data, 按照ppl进行降序排序，取前keep_ratio%的数据
        '''

        if method == "OFTOR":
            # 先分别计算在valid上的指标并训练一个模型
                   
            device = torch.device(
            f"cuda:{1}" if torch.cuda.is_available() else "cpu")

            if self.model_under_test == "CodeT5":
                tokenizer = AutoTokenizer.from_pretrained("./outputs/JCSD/CodeT5/checkpoint", device_map=device)
                model = T5ForConditionalGeneration.from_pretrained("./outputs/JCSD/CodeT5/checkpoint").to(device)

            elif self.model_under_test == 'CodeGen':
                tokenizer = AutoTokenizer.from_pretrained("./outputs/JCSD/CodeGen/checkpoint",
                                                        padding_side="left", device_map=device)
                model = AutoModelForCausalLM.from_pretrained("./outputs/JCSD/CodeGen/checkpoint").to(device)

                generation_config = GenerationConfig(
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                )
                        
            elif model_under_test == "Unixcoder":
                tokenizer = AutoTokenizer.from_pretrained(
                    "microsoft/unixcoder-base", device_map=device)
                
                config = RobertaConfig.from_pretrained(
                    "microsoft/unixcoder-base"
                )

                config.is_decoder = True
                encoder = RobertaModel.from_pretrained(
                    "microsoft/unixcoder-base",
                    config=config)

                model = Seq2Seq(encoder=encoder,
                                decoder=encoder,
                                config=config,
                                beam_size=BEAM_SIZE,
                                max_length=128,
                                sos_id=tokenizer.convert_tokens_to_ids(["<mask0>"])[0],
                                eos_id=tokenizer.sep_token_id)

                model.load_state_dict(torch.load(os.path.join("./outputs/JCSD/Unixcoder/checkpoint/pytorch_model.bin")),
                                    strict=False)
                model.to(device)

            valid_ppls = self.metric_1("valid")

            test_ppls = self.metric_1("test")

            valid_neighor_sim = self.metric_2("valid")

            test_neighor_sim = self.metric_2("test")            

            hidden_state_file_path = self.result_dir + f"{model_under_test}_train_hidden_states.npy"
            valid_hidden_states_file_path = self.result_dir + f"{model_under_test}_valid_hidden_states.npy"
            test_hidden_states_file_path = self.result_dir + f"{model_under_test}_test_hidden_states.npy"
            
            if os.path.exists(hidden_state_file_path) == False:

                train_hidden_states = self.get_hidden_state(model, tokenizer, device, "train", model_under_test)
               
                train_hidden_states = train_hidden_states.reshape(-1, train_hidden_states.shape[-1])

                np.save(hidden_state_file_path, train_hidden_states)
            else:
                train_hidden_states = np.load(hidden_state_file_path)

            if os.path.exists(valid_hidden_states_file_path) == False:
                valid_hidden_states = self.get_hidden_state(model, tokenizer, device, "valid", model_under_test, is_dropout=False)
                print(valid_hidden_states_file_path)
                np.save(valid_hidden_states_file_path, valid_hidden_states)
            else:
                valid_hidden_states = np.load(valid_hidden_states_file_path)

            valid_sem_sim = self.metric_3("valid", valid_hidden_states, train_hidden_states)

            if os.path.exists(test_hidden_states_file_path) == False:
                test_hidden_states = self.get_hidden_state(model, tokenizer, device, "test", model_under_test, is_dropout=False)
                np.save(test_hidden_states_file_path, test_hidden_states)
            else:
                test_hidden_states = np.load(test_hidden_states_file_path)

            test_sem_sim = self.metric_3("test", test_hidden_states, train_hidden_states)

            # valid_hidden_vars = self.metric_4("valid", model_under_test, times=10, tokenizer=tokenizer, model=model, device=device)

            # test_hidden_vars = self.metric_4("test", model_under_test, times=10, tokenizer=tokenizer, model=model, device=device)


            # X = np.vstack((valid_ppls, valid_neighor_sim, valid_sem_sim, valid_hidden_vars)).T
            X = np.vstack((valid_ppls, valid_neighor_sim, valid_sem_sim)).T
            print(X.shape)
            y = np.array([0 if t['BLEU'] >= 0.4 else 1 for t in self.valid_objects])
            # clf = LogisticRegression(random_state=0, max_iter=500).fit(X, y)
            # clf = RandomForestClassifier(random_state=0).fit(X,y)
            clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
                                             max_depth=1, random_state=0).fit(X, y)
            #基于构建的clf去预测测试集
            # X = np.vstack((test_ppls, test_neighor_sim, test_sem_sim, test_hidden_vars)).T
            X = np.vstack((test_ppls, test_neighor_sim, test_sem_sim)).T
          
            wrong_prob= clf.predict_proba(X)[:, 1]
            #获取wrong_prob概率大于0.4数据索引
            sorted_index_desc_select = np.where(wrong_prob > 0.45)[0]

        elif method == "PPL":
            valid_ppls = self.metric_1("valid")

            test_ppls = self.metric_1("test")

            X = valid_ppls.reshape(-1, 1)
            y = np.array([0 if t['BLEU'] >= 0.4 else 1 for t in self.valid_objects])
            clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
                                             max_depth=1, random_state=0).fit(X, y)
            #基于构建的clf去预测测试集
            X = test_ppls.reshape(-1, 1)
            wrong_prob= clf.predict_proba(X)[:, 1]
            #获取wrong_prob概率大于0.4数据索引
            sorted_index_desc_select = np.where(wrong_prob > -1)[0]

        print("Finish calculating metrics ...")

        #排序sorted_index_desc_select中的元素, 使得和后续的self.test_data, self.test_objects一一对应
        self.sorted_index_desc_select = sorted_index_desc_select
        self.selected_test_data, self.keep_test_data, self.selected_objects = [], [], []
        self.sorted_index_desc_select.sort()
        for index in range(len(self.test_data)):
            if index in self.sorted_index_desc_select:
                self.selected_test_data.append(self.test_data[index])
                self.selected_objects.append(self.test_objects[index])
            else:
                self.keep_test_data.append(self.test_data[index])

    def get_metrics(self):
        '''
        统计所选择的selected test data的metrics
        '''
        one_blue, beam_blue = 0, 0
        prediction = np.ones(len(self.selected_test_data))
        ground_truth = np.array(
            [0 if t['BLEU']>=0.4 else 1 for t in self.selected_objects])
        all_ground_truth = np.array(
            [0 if t['BLEU']>=0.4 else 1 for t in self.test_objects])    
        precision = precision_score(ground_truth, prediction)
        recall = sum(ground_truth) / sum(all_ground_truth)

        for line in self.selected_test_data:
            gold = line.nl
            prediction = line.prediction
            candidates = line.candidate
            one_blue += bleu([gold], prediction)[0]
            best_blue = 0
            for i in range(len(candidates)):
                generation = candidates[i]
                currect_blue = bleu([gold], generation)[0]
                if currect_blue > best_blue:
                    best_blue = currect_blue
            beam_blue += best_blue

        return one_blue/len(self.selected_test_data), beam_blue/len(self.selected_test_data),precision, recall

        

    def get_test_ppls(self):
        '''
        获取test_data中每个datapoint的ppl
        '''
        return np.array(self.test_set.getPPL())
    
    def get_valid_ppls(self):

        return np.array(self.valid_set.getPPL())

    def get_client(self):
        return self.client

    def prompt_chat(self,
                    prompts,
                    filtered_sorted_index_desc_select,
                    filtered_selected_test_data,
                    model="gpt-3.5-turbo",
                    temperture=0,
                    n=1,
                    mode=comment_mode.zero_shot,
                    rank = True,
                    **kwargs):
        '''
        prompts: 一个二维列表，每个列表中的元素为一个prompt
        '''

        templates = {
            comment_mode.zero_shot:
            ("<Generation>", "</Generation>"),
            comment_mode.zero_shot_repair:
            ("<Repair>", "</Repair>"),
            comment_mode.bm_25: ("<Repair>", "</Repair>"),
            comment_mode.jarcard: ("<Repair>", "</Repair>")
        }
        responses = []
        batch_thread = 16

        if rank == True:
            rank_write = jsonlines.open(self.catch_name, mode='a')
        else:
            gen_write = jsonlines.open(self.gen_catch_name, mode='a')

        for line_index, prompt, t in tqdm(
                zip(filtered_sorted_index_desc_select, prompts,
                    filtered_selected_test_data)):
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(get_gpt_result, prompt,
                                            line_index, templates, mode,
                                            model, n, temperture, t)
                responses.append(future)
                if len(responses) == batch_thread:
                    for future in concurrent.futures.as_completed(
                            responses):
                        result, line_index = future.result(
                        )
                        if rank == True:
                            self.catch[line_index] = result
                            rank_write.write(result)
                        else:
                            self.gen_catch[line_index] = result
                            gen_write.write(result)
                    responses = []

        if len(responses) != 0:
            for future in concurrent.futures.as_completed(responses):
                result, line_index = future.result()
                if rank == True:
                    self.catch[line_index] = result
                    rank_write.write(result)
                else:
                    self.gen_catch[line_index] = result
                    gen_write.write(result)
            responses = []

        if rank == True:
            rank_write.close()
        else:
            gen_write.close()        

    def repair(self,
               model="gpt-3.5-turbo",
               mode=comment_mode.zero_shot,
               n=1,
               temperture=0,
               **kwargs):
        comment_prompts: list[List] = []
        print("Current mode is {}".format(mode))

        # 读取之前已经修复过的数据
        self.read_catch(model=model, mode=mode, **kwargs)

        self.read_gen_catch(model=model, mode=mode, **kwargs)

        filtered_sorted_index_desc_select, filtered_selected_test_data = [], []

        if mode in [
                comment_mode.bm_25, comment_mode.jarcard
        ]:
            #判断kwargs是否有with_commands属性
            if "with_commands" not in kwargs:
                raise Exception(
                    "with_commands must be specified for this mode")
            # 对于Training Data, 需要获取
            training_data_with_length: list[
                comment_datapoint_with_demo_length] = []

            print("Calculating token count for training data....")
            for datapoint in tqdm(self.training_data):
                token_count = utils.count_codex_tokens(
                    comment_repair_demo_prompt(
                        datapoint,
                        with_commands=kwargs["with_commands"],
                        is_str=True))
                training_data_with_length.append(
                    comment_datapoint_with_demo_length(
                        datapoint, token_count))

            ####################################################################################

            if mode == comment_mode.bm_25:
                print("Loading BM25 ...")
                bm_25_cache_dict = {}
                test_methods = []
                bm25 = load_bm_25(bm_25_cache_dict, test_methods,
                                  training_data_with_length)
                bm_catch_name = self.result_dir + f"bm_25_neighbor.jsonl"
            elif mode == comment_mode.jarcard:
                print("Loading Jarcard ...")
                jarcard_cache_dict = {}
                test_methods = []
                load_jarcard(jarcard_cache_dict, test_methods,
                             training_data_with_length)
                bm_catch_name = self.result_dir + f"jarcard_neighbor.jsonl"

            bm_catch_neighbor = {}
            if os.path.exists(bm_catch_name) != False:
                with jsonlines.open(bm_catch_name) as reader:
                    for obj in reader:
                        index = obj['index']
                        index = int(index)
                        bm_catch_neighbor[index] = obj

            ###################################################################################
                        
            print("Starting constructing gen prompts ...")

            for line_index, t in tqdm(
                    zip(self.sorted_index_desc_select,
                        self.selected_test_data)):
                if line_index in self.gen_catch:
                    continue

                if mode == comment_mode.bm_25:
                    ap: CommentPrompt = build_comment_prompt_bm25(
                        line_index=line_index,
                        bm_catch_neighbor=bm_catch_neighbor,
                        training_data=self.training_data,
                        bm25=bm25,
                        test_methods=test_methods,
                        bm_25_cache_dict=bm_25_cache_dict,
                        inference=t,
                        with_commands=kwargs["with_commands"],
                        demonstration_number=self.demonstration_number)
                elif mode == comment_mode.jarcard:
                    ap: CommentPrompt = build_comment_promp_jarcard(
                        line_index=line_index,
                        bm_catch_neighbor=bm_catch_neighbor,
                        training_data=self.training_data,
                        test_methods=test_methods,
                        jarcard_cache_dict=jarcard_cache_dict,
                        inference=t,
                        with_commands=kwargs["with_commands"],
                        demonstration_number=self.demonstration_number)

                generation_prompt: List = construct_generation_prompt(
                    ap.demonstration_records, t)
                comment_prompts.append(generation_prompt)
                filtered_sorted_index_desc_select.append(line_index)
                filtered_selected_test_data.append(t)

            self.prompt_chat(comment_prompts,
                             filtered_sorted_index_desc_select,
                             filtered_selected_test_data,
                             model,
                             temperture,
                             n,
                             mode=mode,
                             rank=False,
                             **kwargs)
            # 保存bm_25_neighbor
            with jsonlines.open(bm_catch_name, mode='w') as writer:
                for key in bm_catch_neighbor.keys():
                    writer.write(bm_catch_neighbor[key])

            print("Starting constructing rank prompts ...")
            comment_prompts = []
            filtered_sorted_index_desc_select = []
            filtered_selected_test_data = []
            
            for line_index, t in tqdm(
                    zip(self.sorted_index_desc_select,
                        self.selected_test_data)):
                if line_index in self.catch:
                    continue

                if mode == comment_mode.bm_25:
                    ap: CommentPrompt = build_comment_prompt_bm25(
                        line_index=line_index,
                        bm_catch_neighbor=bm_catch_neighbor,
                        training_data=self.training_data,
                        bm25=bm25,
                        test_methods=test_methods,
                        bm_25_cache_dict=bm_25_cache_dict,
                        inference=t,
                        with_commands=kwargs["with_commands"],
                        demonstration_number=self.demonstration_number)
                elif mode == comment_mode.jarcard:
                    ap: CommentPrompt = build_comment_promp_jarcard(
                        line_index=line_index,
                        bm_catch_neighbor=bm_catch_neighbor,
                        training_data=self.training_data,
                        test_methods=test_methods,
                        jarcard_cache_dict=jarcard_cache_dict,
                        inference=t,
                        with_commands=kwargs["with_commands"],
                        demonstration_number=self.demonstration_number)

                comment_prompt: List = ap.construct_prompt()

                comment_prompts.append(comment_prompt)
                filtered_sorted_index_desc_select.append(line_index)
                filtered_selected_test_data.append(t)

            assert len(comment_prompts) == len(
                filtered_sorted_index_desc_select) == len(
                    filtered_selected_test_data)

        if mode == comment_mode.zero_shot:
            print("Starting constructing prompts ...")
            for line_index, t in tqdm(
                    zip(self.sorted_index_desc_select,
                        self.selected_test_data)):
                if line_index in self.catch:
                    continue
                comment_prompt = comment_get_zero_shot_prompt(code=t.code)
                comment_prompts.append(comment_prompt)
                filtered_sorted_index_desc_select.append(line_index)
                filtered_selected_test_data.append(t)

        if mode == comment_mode.zero_shot_repair:
            print("Starting constructing prompts ...")
            for line_index, t in tqdm(
                    zip(self.sorted_index_desc_select,
                        self.selected_test_data)):
                if line_index in self.catch:
                    continue
                comment_prompt = comment_get_zero_shot_prompt_repair(
                    code = t.code,
                    candidates=t.candidate)
                comment_prompts.append(comment_prompt)
                filtered_sorted_index_desc_select.append(line_index)
                filtered_selected_test_data.append(t)

        print(
            "Starting calling Chat API ... Total number of prompts: {}".format(
                len(comment_prompts)))
        
        self.prompt_chat(comment_prompts,
                         filtered_sorted_index_desc_select,
                         filtered_selected_test_data,
                         model,
                         temperture,
                         n,
                         mode=mode,
                         rank=True,
                         **kwargs)

def construct_generation_prompt(demonstration_records,inference: 'Comment_Datapoint'):

    system_prompt = comment_gen_system_prompt()

    prompt = [{
        "role": "system",
        "content": system_prompt
    }]

    demonstrations = []
    for i in range(len(demonstration_records) - 1, -1, -1):
        record = demonstration_records[i]
        demo = comment_gen_demo_prompt(record, True)
        if record.code != '':
            demonstrations.append(demo)

    query = comment_gen_user_prompt(inference.code)

    for demo in demonstrations:
        for d in demo:
            prompt.append({"role": "user", "content": d[0]})
            prompt.append({"role": "assistant", "content": d[1]})

    prompt.append({"role": "user", "content": query})

    return prompt


def setseed(seed):
    np.random.seed(seed)
    import random
    random.seed(seed)
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    chat_model = "gpt-3.5-turbo-0125"
    repair_mode = comment_mode.jarcard
    demonstration_number = 30
    BEAM_SIZE = 10
    seed = 42
    setseed(seed)
    # map_dic = {"CodeT5": "t5", "Unixcoder": "unix", "CodeGen": "codegen"}
    result_dir = "./outputs/JCSD/"
    map_dic = {"Unixcoder": "unix"}

    for model_under_test, abstract in map_dic.items():
        train_file_path = f"./Dataset/JCSD/{abstract}_{BEAM_SIZE}_train.json.rank.jsonl"
        valid_file_path = f"./Dataset/JCSD/{abstract}_{BEAM_SIZE}_valid.json.rank.jsonl"
        test_file_name = f"./Dataset/JCSD/{abstract}_{BEAM_SIZE}_test.json.rank.jsonl"

        if model_under_test == "CodeT5":
            batch_size = 128
            
        elif model_under_test =="Unixcoder":
            batch_size = 128
        elif model_under_test == "CodeGen":
            batch_size = 8


        # 用一行语句打印上述的所有参数
        print(f"train_file_path: {train_file_path}, test_file_name: {test_file_name}, result_dir: {result_dir}, model_under_test: {model_under_test}, demonstration_number: {demonstration_number}, repair_mode: {repair_mode}, chat_model: {chat_model}, BEAM_SIZE: {BEAM_SIZE}, seed: {seed}")

        #################################################################################################################




        experiment = ModelTester(train_file_path=train_file_path,
                                valid_file_path=valid_file_path,
                                test_file_name=test_file_name,
                                result_dir=result_dir,
                                model_under_test=model_under_test,
                                demonstration_number=demonstration_number)



        experiment.priorization(method="OFTOR")

        one_bleu, beam_bleu, pre, recall = experiment.get_metrics()
        print("The Selected Average BLEU: ", one_bleu*100)
        print("The Beam Average BLEU: ", beam_bleu*100)
        print("pre: ", pre)
        print("recall: ", recall)

        #gpt-3.5-turbo
        experiment.repair(model=chat_model,
                        mode=repair_mode,
                        n=1,
                        temperture=0,
                        with_commands=True)

        # prediction_model: combined, rank, gen
        # for prediction_model in ["combined", "rank", "gen"]:
        for prediction_model in ["combined", "rank", "gen"]:
            experiment.generate_final_result(model=chat_model, mode=repair_mode, prediction_mode=prediction_model)
