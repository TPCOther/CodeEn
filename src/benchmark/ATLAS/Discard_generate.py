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

            if mode in ["train", "test"]:
                outputs_ids = model.generate(**inputs,
                                             max_length=max_output_length,
                                             num_beams=BEAM_SIZE,
                                             num_return_sequences=BEAM_SIZE)
                outputs = tokenizer.batch_decode(outputs_ids,
                                                 skip_special_tokens=True)
                p.extend(outputs)

        if model_type == "unix":

            train_examples = read_examples(dataset[i:i + batch_size])

            train_features = unix_convert_examples_to_features(
                train_examples,
                tokenizer,
                max_source_length=max_input_length,
                max_target_length=max_output_length)

            source_ids = torch.tensor([f.source_ids for f in train_features],
                                      dtype=torch.long).to(device)

            if mode in ["train", "test"]:
                with torch.no_grad():
                    preds = model(source_ids)
                    for pred in preds:
                        text = tokenizer.batch_decode(pred,
                                                      skip_special_tokens=True)
                        p.extend(text)

    if mode in ["train", "test"]:
        # 以一个样本，对应Beam size为组
        assert len(
            p) == len(dataset) * BEAM_SIZE == len(object_lists) * BEAM_SIZE

        for i in range(len(object_lists)):

            inial = p[i * BEAM_SIZE:(i + 1) * BEAM_SIZE]
           
            object_lists[i]["generation"] = inial



    file_path = config[mode]
    file_name = file_path.split("/")[-1].split(".jsonl")[0]
    file_prefix = "/".join(file_path.split("/")[:-1])

    file_path = file_prefix+ "/"+ model_type + "_" + file_name + ".rank.jsonl"

    jsonlines.open(file_path, "w").write_all(object_lists)


if __name__ == "__main__":
    # main(
    #     model_name=
    #     "/data/swf/Assertion/outputs/ATLAS_LOU/Unixcoder/checkpoint-best-bleu/pytorch_model.bin",
    #     cuda=0,
    #     mode="test",
    #     model_type="unix")
    types = ["test", "train"]
    for type in types:
        main(
            model_name=
            "/data/swf/Assertion/outputs/ATLAS_LOU/CodeT5/checkpoint-23514",
            cuda=0,
            mode=type,
            model_type="t5")

