import sys
import time
sys.path.append("/home/yanmeng/huangnaiqi/Assertion/src/")

from benchmark.JCSD.bleu import bleu
import argparse
import json
import jsonlines
import numpy as np
from sklearn.metrics import precision_score
import torch
from transformers import T5ForConditionalGeneration, AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from tqdm import tqdm
from datasets import load_dataset
import os
import re
from benchmark.ATLAS.UnixCoder import Seq2Seq
from benchmark.ATLAS.UnixCoder_run import Example, InputFeatures
from utils.tokenizer import Tokenizer
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
              RobertaConfig, RobertaModel, RobertaTokenizer)

beam_time = 0.0
PPL_time = 0.0
total_time = 0.0
tokenizer_time = 0.0


# 讲utils的路径加入到环境变量中
INS = "Summarization task: give you a function in java, try to summaries code into one brief sentence"

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def generate_prompt(instruction, input=None):
    return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:"""

def read_examples(dataset: dict):
    """Read examples from filename."""
    examples = []
    focal_test = dataset['code']
    gold = dataset['comment']
    assert len(focal_test) == len(gold)
    for idx, obj in enumerate(zip(focal_test, gold)):
        code, nl = obj
        code = code.replace('\n', ' ').strip()
        nl = nl.replace('\n', '').strip()
        examples.append(Example(
            idx=idx,
            source=code,
            target=nl,
        ))

    return examples


def read_examples_(dataset: dict):
    """Read examples from filename."""
    examples = []
    for idx, obj in enumerate(dataset):
        code, nl = obj['code'], obj['comment']
        code = code.replace('\n', ' ').strip()
        nl = nl.replace('\n', '').strip()
        examples.append(Example(
            idx=idx,
            source=code,
            target=nl,
        ))
    return examples


def unix_convert_examples_to_features(examples, tokenizer, max_source_length, max_target_length):
    """convert examples to token ids"""
    features = []
    for example_index, example in enumerate(examples):
        #source
        source_tokens = tokenizer.tokenize(example.source)[:max_source_length-5]
        source_tokens = [tokenizer.cls_token,"<encoder-decoder>",tokenizer.sep_token,"<mask0>"]+source_tokens+[tokenizer.sep_token]
        source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
        padding_length = max_source_length - len(source_ids)
        source_ids += [tokenizer.pad_token_id]*padding_length


        target_tokens = tokenizer.tokenize("None")

        target_tokens = ["<mask0>"] + target_tokens + [tokenizer.sep_token]
        target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
        padding_length = max_target_length - len(target_ids)
        target_ids += [tokenizer.pad_token_id] * padding_length

        features.append(
            InputFeatures(
                 example_index,
                 source_ids,
                 target_ids,
            )
        )
    return features


def read_dataset(file_path):
    # 加载数据集
    datas = []
    with open(file_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            f_t_object = json.loads(line)
            datas.append(f_t_object)
    return datas


def get_model_perplexity(scores,
                         output_ids,
                         eos_token_id,
                         model_type="encoder-decoder"):
    softmax = torch.nn.functional.softmax
    probs = [softmax(scores, dim=-1) for scores in scores]
    probs_stacked = torch.stack(probs)
    # batch_size sequece_length vacab_size
    # t5：torch.Size([49, 32, 32100]), output_ids:torch.Size([32, 49])
    # t5: torch.Size([17, 2, 50296]), output_ids:torch.Size([2, 514])
    probs_transposed = probs_stacked.transpose(0, 1)
    # pytroch 根据ouput_ids的id索引 得到probs_transposed对应token的概率
    # batch_size sequece_length
    if model_type == "encoder-decoder":
        output_ids = output_ids[:, :probs_transposed.shape[1]]
    elif model_type == "decoder":
        output_ids = output_ids[:, -probs_transposed.shape[1]:]

    probs_of_output_ids = torch.gather(probs_transposed, 2,
                                       output_ids.unsqueeze(-1)).squeeze(-1)

    perplexity_result = []
    for seq_i in range(len(probs_of_output_ids)):
        single_seq_probs = probs_of_output_ids[seq_i]
        single_seq_output_ids = output_ids[seq_i]
        # 定位到eos_token_id的位置
        eos_token_id_pos = np.where(
            single_seq_output_ids.cpu() == eos_token_id)[0]
        if len(eos_token_id_pos) != 0:
            single_seq_probs = single_seq_probs[:eos_token_id_pos[0]]
        log_single_seq_probs = -torch.mean(torch.log(single_seq_probs))
        perplexity = torch.exp(log_single_seq_probs)
        perplexity_result.append(perplexity.item())

    return perplexity_result


def main(model_name, cuda, mode: str, BEAM_SIZE=5, model_type="t5"):
    global beam_time, total_time, PPL_time, tokenizer_time

    if model_type in ["t5", "unix"]:
        batch_size = 16
    elif model_type == "codegen":
        batch_size = 1

    max_input_length = 512
    max_output_length = 128

    device = torch.device(
        f"cuda:{cuda}" if torch.cuda.is_available() else "cpu")

    if model_type == "t5":
        tokenizer = AutoTokenizer.from_pretrained(model_name,
                                                  device_map=device)
        model = T5ForConditionalGeneration.from_pretrained(model_name).to(
            device)
    elif model_type == 'codegen':
        tokenizer = AutoTokenizer.from_pretrained(model_name,
                                                  device_map=device,
                                                  padding_side="left")
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

        generation_config = GenerationConfig(
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        model.eval()

    elif model_type == "unix":
        tokenizer = AutoTokenizer.from_pretrained(
            "/data/swf/Assertion/Unixcoder-base/snapshots/727b99cf2a9cab12f417ac638ecd7c9242e896bf",
            device_map=device)
        config = RobertaConfig.from_pretrained(
            "/data/swf/Assertion/Unixcoder-base/snapshots/727b99cf2a9cab12f417ac638ecd7c9242e896bf"
        )
        config.is_decoder = True
        encoder = RobertaModel.from_pretrained(
            "/data/swf/Assertion/Unixcoder-base/snapshots/727b99cf2a9cab12f417ac638ecd7c9242e896bf",
            config=config)

        model = Seq2Seq(encoder=encoder,
                        decoder=encoder,
                        config=config,
                        beam_size=BEAM_SIZE,
                        max_length=max_output_length,
                        sos_id=tokenizer.convert_tokens_to_ids(["<mask0>"])[0],
                        eos_id=tokenizer.sep_token_id)

        model.load_state_dict(torch.load(os.path.join(model_name)),
                              strict=False)

        model.to(device)

        model.eval()

    config = {
        'train': "/home/yanmeng/huangnaiqi/Assertion/Dataset/JCSD/train.json",
        'test': "/home/yanmeng/huangnaiqi/Assertion/Dataset/JCSD/test.json",
        'valid': "/home/yanmeng/huangnaiqi/Assertion/Dataset/JCSD/valid.json",
    }

    dataset = load_dataset('json',
                           data_files={
                               'train': config['train'],
                               'test': config['test'],
                               'valid': config['valid'],
                           })

    object_lists = read_dataset(config[mode])

    dataset = dataset[mode]

    print("data_set_size", len(dataset))

    p, ppl, only = [], [], []

    for i in tqdm(range(0, len(dataset), batch_size)):

        if model_type == "t5":
            full_start_time = time.time()

            tokenizer_start_time = time.time()

            inputs = tokenizer(dataset[i:i + batch_size]["code"],
                               max_length=max_input_length,
                               padding=True,
                               truncation=True,
                               return_tensors="pt").to(device)
            
            tokenizer_end_time = time.time()
            tokenizer_time += tokenizer_end_time - tokenizer_start_time
            beam_time += tokenizer_end_time - tokenizer_start_time

            if mode == "train":
                outputs_ids = model.generate(**inputs,
                                             max_length=max_output_length,
                                             num_beams=BEAM_SIZE,
                                             num_return_sequences=BEAM_SIZE)
                outputs = tokenizer.batch_decode(outputs_ids,
                                                 skip_special_tokens=True)
                p.extend(outputs)

            elif mode in ["test", "valid"]:
                outputs = model.generate(**inputs,
                                         max_length=max_output_length,
                                         do_sample=False,
                                         output_scores=True,
                                         return_dict_in_generate=True)
                # 根据output的输出概率 计算困惑度
                outputs_scores = outputs.scores
                outputs_ids = outputs.sequences
                ppl_start_time = time.time()
                perplexity_result = get_model_perplexity(
                    outputs_scores, outputs_ids[:, 1:], tokenizer.eos_token_id)
                outputs_word = tokenizer.batch_decode(outputs_ids,
                                                      skip_special_tokens=True)
                ppl_end_time = time.time()
                PPL_time += ppl_end_time - ppl_start_time

                ppl.extend(perplexity_result)
                only.extend(outputs_word)

                start_time = time.time()
                outputs_ids = model.generate(**inputs,
                                             max_length=max_output_length,
                                             num_beams=BEAM_SIZE,
                                             num_return_sequences=BEAM_SIZE)

                outputs_word = tokenizer.batch_decode(outputs_ids,
                                                      skip_special_tokens=True)
                end_time = time.time()
                full_end_time = time.time()

                total_time += full_end_time - full_start_time
                beam_time += end_time - start_time

                p.extend(outputs_word)

        elif model_type == "codegen":

            input_data = [
                generate_prompt(INS, input)
                for input in dataset[i:i + batch_size]["code"]
            ]
            model_inputs = tokenizer(input_data,
                                     max_length=1920,
                                     truncation=True,
                                     padding=True,
                                     return_tensors="pt").to(device)

            if mode == "train":
                with torch.no_grad():
                    generated_ids = model.generate(
                        model_inputs.input_ids,
                        max_new_tokens=128,
                        num_beams=BEAM_SIZE,
                        num_return_sequences=BEAM_SIZE,
                        generation_config=generation_config)
                prediction = tokenizer.batch_decode(
                    generated_ids[:, model_inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True)
                p.extend([p for p in prediction])

            elif mode in ["test", "valid"]:
                with torch.no_grad():
                    outputs = model.generate(**model_inputs,
                                             max_new_tokens=128,
                                             do_sample=False,
                                             output_scores=True,
                                             return_dict_in_generate=True)
                    # 根据output的输出概率 计算困惑度
                    outputs_scores = outputs.scores
                    outputs_ids = outputs.sequences
                    perplexity_result = get_model_perplexity(
                        outputs_scores, outputs_ids,
                        tokenizer.eos_token_id,
                        model_type="decoder")
                    outputs_word = tokenizer.batch_decode(
                        outputs_ids[:, model_inputs["input_ids"].shape[1]:], skip_special_tokens=True)

                    ppl.extend(perplexity_result)
                    only.extend(outputs_word)

                    outputs_ids = model.generate(
                        **model_inputs,
                        max_new_tokens=128,
                        num_beams=BEAM_SIZE,
                        num_return_sequences=BEAM_SIZE)

                    outputs_word = tokenizer.batch_decode(
                        outputs_ids[:, model_inputs["input_ids"].shape[1]:],
                        skip_special_tokens=True)

                    p.extend(outputs_word)

        if model_type == "unix":

            train_examples = read_examples(dataset[i:i + batch_size])

            train_features = unix_convert_examples_to_features(
                train_examples,
                tokenizer,
                max_source_length=max_input_length,
                max_target_length=max_output_length)

            source_ids = torch.tensor([f.source_ids for f in train_features],
                                      dtype=torch.long).to(device)

            if mode == "train":
                with torch.no_grad():
                    preds = model(source_ids)
                    for pred in preds:
                        text = tokenizer.batch_decode(pred,
                                                      skip_special_tokens=True)
                        p.extend(text)

            elif mode in ["test", "valid"]:
                model.beam_size = 1
                with torch.no_grad():
                    outputs = model(source_ids, return_logits=True)

                    outputs_ids = outputs.sequences.squeeze(1)
                    text = tokenizer.batch_decode(outputs_ids,
                                                  skip_special_tokens=True)
                    only.extend(text)

                    # 根据output的输出概率 计算困惑度
                    outputs_scores = outputs.scores
                    outputs_scores = outputs_scores.transpose(0, 1)
                    perplexity_result = get_model_perplexity(
                        outputs_scores, outputs_ids, 0)
                    ppl.extend(perplexity_result)

                model.beam_size = BEAM_SIZE
                with torch.no_grad():
                    preds = model(source_ids)
                    for pred in preds:
                        text = tokenizer.batch_decode(pred,
                                                      skip_special_tokens=True)
                        p.extend(text)

    if mode == "train":
        # 以一个样本，对应Beam size为组
        assert len(
            p) == len(dataset) * BEAM_SIZE == len(object_lists) * BEAM_SIZE

        for i in range(len(object_lists)):

            inial = p[i * BEAM_SIZE:(i + 1) * BEAM_SIZE]

            object_lists[i]["generation"] = inial

    elif mode in ["test", "valid"]:
        BLUE_SCORE = 0
        for i, item in enumerate(
                zip(dataset[:]["comment"], object_lists, ppl, only)):
            gold, object, ppl_value, pre_name = item

            inial = p[i * BEAM_SIZE:(i + 1) * BEAM_SIZE]

            object["BLEU"] = bleu(refs=[gold], candidate=pre_name)[0]

            BLUE_SCORE += object["BLEU"]

            object["ppl"] = ppl_value

            object["generation"] = pre_name

            object["candidate"] = inial

        print("Average BLUE Score", BLUE_SCORE / len(ppl) * 100)

    file_path = config[mode]
    file_name = file_path.split("/")[-1].split(".jsonl")[0]
    file_prefix = "/".join(file_path.split("/")[:-1])

    file_path = file_prefix + "/" + model_type + "_" + str(
        BEAM_SIZE) + "_" + file_name + ".time.rank.jsonl"

    jsonlines.open(file_path, "w").write_all(object_lists)


if __name__ == "__main__":

    # main(
    #     model_name="/data/swf/Assertion/outputs/JCSD/Unixcoder/checkpoint-4/pytorch_model.bin",
    #     cuda=0,
    #     mode="valid",
    #     model_type="unix",
    #     BEAM_SIZE=10)

    main(model_name="/home/yanmeng/huangnaiqi/Assertion/outputs/JCSD/CodeT5/checkpoint-43570",
         cuda=0,
         mode="test",
         model_type="t5",
         BEAM_SIZE=10)
    
    print("beam_time", beam_time)
    print("PPL_time", PPL_time)
    print("tokenizer_time", tokenizer_time)
    print("total_time", total_time)

    # main(
    # model_name=
    # "/data/swf/Assertion/outputs/JCSD/CodeGen/checkpoint-26142",
    # cuda=0,
    # mode="valid",
    # model_type="codegen",
    # BEAM_SIZE=10)
