import argparse
from transformers import PLBartForConditionalGeneration, PLBartTokenizer
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import DataCollatorForSeq2Seq
from datasets import load_dataset
import numpy as np
import re
import evaluate
import torch

parser = argparse.ArgumentParser()

parser.add_argument("--train_data_path", type=str, default="Dataset/old", )
parser.add_argument("--output_dir", type=str, default="./outputs/ATLAS_LOU/PLBART")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--data_seed", type=int, default=42)
parser.add_argument("--num_train_epochs", type=int, default=5)

args = parser.parse_args()

model_name = "plbart-base"
tokenizer = PLBartTokenizer.from_pretrained(model_name, src_lang="java", tgt_lang="en_XX")
model = PLBartForConditionalGeneration.from_pretrained(model_name)

dataset = load_dataset(
    'json',
    data_files={
        'train': args.train_data_path + "/train.raw.lou.jsonl",
        'test': args.train_data_path + "/test.raw.lou.jsonl",
        'valid': args.train_data_path + "/val.raw.lou.jsonl",
})

prefix = ""
max_input_length = 512
max_target_length = 128

def preprocess_example(example):

    focal_tests = example['focal_test']
    assertions = example['gold']

    inputs = [prefix + focal_test.strip() for focal_test in focal_tests]
    assertions = [assertion.strip() for assertion in assertions]

    model_inputs = tokenizer(inputs, max_length=max_input_length, padding="max_length", truncation=True)
    
    labels = tokenizer(assertions, max_length=max_target_length, padding="max_length", truncation=True).input_ids

    #将pad标签值0替换为-100，不在CrossEntropyLoss的计算范围内
    labels_with_ignore_index = []
    for labels_example in labels:
        labels_example = [label if label != 0 else -100 for label in labels_example]
        labels_with_ignore_index.append(labels_example)
    
    model_inputs["labels"] = labels_with_ignore_index

    return model_inputs

dataset["train"] = dataset["train"].train_test_split(test_size=0.5, seed=args.seed)["train"]
dataset = dataset.map(preprocess_example, batched=True)

exact_match_metric = evaluate.load("./evaluate/exact_match.py")

print("train_data_size", len(dataset["train"]))
print("valid_data_size", len(dataset["valid"]))
print("test_data_size", len(dataset["test"]))

data_collator = DataCollatorForSeq2Seq(tokenizer)
training_args = Seq2SeqTrainingArguments(
    output_dir = args.output_dir,
    evaluation_strategy="epoch",
    logging_strategy="steps",
    logging_steps=1000,
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    save_total_limit=10,
    num_train_epochs=args.num_train_epochs,
    seed=args.seed,
    data_seed=args.data_seed,
    predict_with_generate=True,
    load_best_model_at_end=True,
    metric_for_best_model="eval_exact_match",
)

def comput_metrics(eval_pred):
    predictions, labels = eval_pred
    decode_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decode_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    result = bleu.compute(predictions=decode_preds, references=decode_labels, use_effective_order=True, smooth_method="add-k")

    return result



trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["valid"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=comput_metrics
)

trainer.train()
trainer.save_model()