import sys
sys.path.append("/data/swf/Assertion/src/")
import argparse
import json
import jsonlines
import numpy as np
from sklearn.metrics import precision_score
import torch
from transformers import T5ForConditionalGeneration, AutoTokenizer
from tqdm import tqdm
from datasets import load_dataset
import os
import re
from benchmark.ATLAS.UnixCoder import Seq2Seq
from benchmark.ATLAS.UnixCoder_run import Example, InputFeatures
from utils.tokenizer import Tokenizer
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
              RobertaConfig, RobertaModel, RobertaTokenizer)
# 讲utils的路径加入到环境变量中

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

def read_examples(dataset: dict):
    """Read examples from filename."""
    examples = []
    focal_test = dataset['focal_test']
    gold = dataset['gold']
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


def get_model_perplexity(scores, output_ids, eos_token_id):
    softmax = torch.nn.functional.softmax
    probs = [softmax(scores, dim=-1) for scores in scores]
    probs_stacked = torch.stack(probs)
    # batch_size sequece_length vacab_size
    # t5：torch.Size([49, 32, 32100]), output_ids:torch.Size([32, 49])
    probs_transposed = probs_stacked.transpose(0, 1)
    # pytroch 根据ouput_ids的id索引 得到probs_transposed对应token的概率
    # batch_size sequece_length

    output_ids = output_ids[:, : probs_transposed.shape[1]]

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

# def get_model_gini(scores, output_ids):

#     def get_real_length(sequences):
#         max_len = sequences[0].shape[0]
#         real_length = []
#         for seq in sequences:
#             non_zero_indices = torch.nonzero(seq == 0)
#             first_zero_index = non_zero_indices[0] if len(non_zero_indices) > 0 else max_len
#             real_length.append(first_zero_index)
#         return torch.tensor(real_length).cpu()

#     seq_len = get_real_length(output_ids)
#     softmax = torch.nn.functional.softmax
#     probs = [softmax(scores, dim=-1) for scores in scores]
#     probs_stacked = torch.stack(probs)
#     # batch_size sequece_length vacab_size
#     probs_transposed = probs_stacked.transpose(0, 1)
#     # pytroch 根据ouput_ids的id索引 得到probs_transposed对应token的概率
#     # batch_size sequece_length

#     output = probs_transposed ** 2

#     outputs = []

#     for j in range(output.shape[0]):
#         osum = 1 - output[j][:seq_len[j]].sum(dim=-1)
#         outputs.append(osum.mean().item())


#     return outputs


def main(model_name, cuda, mode: str, BEAM_SIZE=5, model_type="t5"):

    if mode == "train":
        batch_size = 16
    elif mode == "test":
        batch_size = 32
    max_input_length = 512
    max_output_length = 128

    device = torch.device(
        f"cuda:{cuda}" if torch.cuda.is_available() else "cpu")

    if model_type == "t5":
        tokenizer = AutoTokenizer.from_pretrained(model_name,
                                                  device_map=device)
        model = T5ForConditionalGeneration.from_pretrained(model_name).to(
            device)
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

        model.load_state_dict(torch.load(os.path.join(model_name)))

        model.to(device)

        model.eval()

    config = {
        'train': "Dataset/old/train.raw.lou.jsonl",
        'test': "Dataset/old/test.raw.lou.jsonl",
        'valid': "Dataset/old/val.raw.lou.jsonl",
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

    p, ppl = [], []

    for i in tqdm(range(0, len(dataset), batch_size)):

        if model_type == "t5":

            inputs = tokenizer(dataset[i:i + batch_size]["focal_test"],
                               max_length=max_input_length,
                               padding=True,
                               truncation=True,
                               return_tensors="pt").to(device)

            if mode == "train":
                outputs_ids = model.generate(**inputs,
                                             max_length=max_output_length,
                                             num_beams=BEAM_SIZE,
                                             num_return_sequences=BEAM_SIZE)
                outputs = tokenizer.batch_decode(outputs_ids,
                                                 skip_special_tokens=True)
                p.extend(outputs)

            elif mode == "test":
                outputs = model.generate(**inputs,
                                         max_length=max_output_length,
                                         do_sample=False,
                                         output_scores=True,
                                         return_dict_in_generate=True)
                # 根据output的输出概率 计算困惑度
                outputs_scores = outputs.scores
                perplexity_result = get_model_perplexity(
                    outputs_scores, outputs_ids[:, 1:], tokenizer.eos_token_id)
                
                ppl.extend(perplexity_result)
                
                outputs_ids = model.generate(**inputs,
                                             max_length=max_output_length,
                                             num_beams=BEAM_SIZE,
                                             num_return_sequences=BEAM_SIZE)
                
                outputs_word = tokenizer.batch_decode(outputs_ids,
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

            elif mode == "test":
                model.beam_size = 1
                with torch.no_grad():
                    outputs = model(source_ids, return_logits=True)
                    outputs_ids = outputs.sequences.squeeze(1)
                    text = tokenizer.batch_decode(outputs_ids,
                                                  skip_special_tokens=True)
                    p.extend(text)

                    # 根据output的输出概率 计算困惑度
                    outputs_scores = outputs.scores

                    outputs_scores = outputs_scores.transpose(0, 1)

                    perplexity_result = get_model_perplexity(
                        outputs_scores, outputs_ids, 0)

                    ppl.extend(perplexity_result)

    if mode == "train":
        # 以一个样本，对应Beam size为组
        assert len(
            p) == len(dataset) * BEAM_SIZE == len(object_lists) * BEAM_SIZE

        for i in range(len(object_lists)):
            gold_sequcence = tokenizer.encode(object_lists[i]["gold"],
                                              max_length=max_output_length,
                                              truncation=True)
            gold_decoder = tokenizer.decode(gold_sequcence,
                                            skip_special_tokens=True)
            inial = list(set(p[i * BEAM_SIZE:(i + 1) * BEAM_SIZE]))
            after = []
            for item in inial:
                if not Tokenizer.whether_equally(item.strip(),
                                                 gold_decoder.strip()):
                    after.append(item)
            object_lists[i]["generation"] = after

    elif mode == "test":
        correct_count = 0
        assert len(p) == len(dataset) == len(object_lists) == len(ppl)
        for pre_name, gold, object, ppl_value in zip(p, dataset[:]["gold"],
                                                     object_lists, ppl):
            gold_sequcence = tokenizer.encode(gold,
                                              max_length=max_output_length,
                                              truncation=True)
            gold_decoder = tokenizer.decode(gold_sequcence,
                                            skip_special_tokens=True)
            if pre_name == gold_decoder:
                correct_count += 1
                object["is_correct"] = True
            else:
                object["is_correct"] = False

            object["ppl"] = ppl_value

            object["generation"] = pre_name

        print("Acc.", correct_count / len(p))

        # # 根据object['ppl']对object_lists进行从大到小排序
        # object_lists = sorted(object_lists,
        #                       key=lambda x: x['ppl'],
        #                       reverse=True)

        # selection_ratios = [0.025, 0.05, 0.075, 0.1, 0.2, 0.3, 0.4, 0.5]

        # for selection_ratio in selection_ratios:
        #     print("selection_ratio: ", selection_ratio)
        #     select_num = int(len(object_lists) * selection_ratio)
        #     select_object_lists = object_lists[:select_num]
        #     prediction = np.ones(len(select_object_lists))
        #     ground_truth = np.array(
        #         [0 if t['is_correct'] else 1 for t in select_object_lists])
        #     all_ground_truth = np.array(
        #         [0 if t['is_correct'] else 1 for t in object_lists])
        #     precision = precision_score(ground_truth, prediction)
        #     recall = sum(ground_truth) / sum(all_ground_truth)
        #     print("Precision: ", precision)
        #     print("Recall: ", recall)

    file_path = config[mode]
    file_name = file_path.split("/")[-1].split(".jsonl")[0]
    file_prefix = "/".join(file_path.split("/")[:-1])

    file_path = file_prefix+ "/"+ model_type + "_" + file_name + ".first_s.jsonl"

    jsonlines.open(file_path, "w").write_all(object_lists)


if __name__ == "__main__":
    main(
        model_name=
        "/data/swf/Assertion/outputs/ATLAS_LOU/Unixcoder/checkpoint-best-bleu/pytorch_model.bin",
        cuda=0,
        mode="test",
        model_type="unix")

    # main(model_name="/data/swf/Assertion/outputs/ATLAS_LOU/CodeT5/checkpoint-23514",
    #      cuda=0,
    #      mode="train",
    #      model_type="t5")
