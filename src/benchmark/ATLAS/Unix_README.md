# Code Summarization

## Data Download

```bash
wget https://github.com/microsoft/CodeXGLUE/raw/main/Code-Text/code-to-text/dataset.zip
unzip dataset.zip
rm dataset.zip
cd dataset
wget https://zenodo.org/record/7857872/files/python.zip
wget https://zenodo.org/record/7857872/files/java.zip
wget https://zenodo.org/record/7857872/files/ruby.zip
wget https://zenodo.org/record/7857872/files/javascript.zip
wget https://zenodo.org/record/7857872/files/go.zip
wget https://zenodo.org/record/7857872/files/php.zip

unzip python.zip
unzip java.zip
unzip ruby.zip
unzip javascript.zip
unzip go.zip
unzip php.zip
rm *.zip
rm *.pkl

python preprocess.py
rm -r */final
cd ..
```

## Dependency 

- pip install torch
- pip install transformers

## Fine-Tune Setting

Here we provide fine-tune settings for code summarization, whose results are reported in the paper.

```shell
lang=python

# Training
CUDA_VISIBLE_DEVICES=1 python /data/swf/Assertion/src/benchmark/ATLAS/UnixCoder_run.py \
	--do_train \
	--do_eval \
	--model_name_or_path /data/swf/Assertion/Unixcoder-base/snapshots/727b99cf2a9cab12f417ac638ecd7c9242e896bf \
	--train_filename Dataset/old/train.raw.lou.jsonl \
	--dev_filename Dataset/old/val.raw.lou.jsonl \
	--output_dir /data/swf/Assertion/outputs/ATLAS_LOU/Unixcoder_new \
	--max_source_length 512 \
	--max_target_length 128 \
	--beam_size 1 \
	--train_batch_size 32 \
	--eval_batch_size 32 \
	--learning_rate 5e-5 \
   	--gradient_accumulation_steps 2 \
	--num_train_epochs 10 
	
# Evaluating	
CUDA_VISIBLE_DEVICES=1 python run.py \
	--do_test \
	--model_name_or_path /data/swf/Assertion/Unixcoder-base/snapshots/727b99cf2a9cab12f417ac638ecd7c9242e896bf \
	--test_filename ../data/JCSD/test.json \
	--output_dir /data/swf/Assertion/outputs/ATLAS_LOU/Unixcoder \
	--max_source_length 512 \
	--max_target_length 128 \
	--beam_size 1 \
	--eval_batch_size 32 \	

CUDA_VISIBLE_DEVICES=1 python ./roberta_stage_one.py \
    --model_name microsoft/unixcoder-base \
	--model_path ../UnixCoder/saved_models/pcsd_half/checkpoint-best-bleu/pytorch_model.bin \
    --select_data_path ../data/PCSD/train.json \
    --split_percentage 0.5 \
    --cache_name pcsd_half \
    --cuda 0
```

CUDA_VISIBLE_DEVICES=1 python ./roberta_stage_one.py \
    --model_name microsoft/unixcoder-base \
	--model_path ../UnixCoder/saved_models/jcsd_half_add/checkpoint-best-bleu/pytorch_model.bin \
    --select_data_path ../data/JCSD/train.json \
    --split_percentage 0.5 \
    --cache_name jcsd_half_add \
    --cuda 0

CUDA_VISIBLE_DEVICES=2 python run.py \
	--do_train \
	--do_eval \
	--model_name_or_path microsoft/unixcoder-base \
	--train_filename ../data/JCSD/split_50_train.json \
	--dev_filename ../data/PCSD/valid.json \
	--output_dir saved_models/jcsd_half \
	--max_source_length 512 \
	--max_target_length 128 \
	--beam_size 1 \
	--train_batch_size 32 \
	--eval_batch_size 32 \
	--learning_rate 5e-5 \
	--gradient_accumulation_steps 2 \
	--num_train_epochs 18 

CUDA_VISIBLE_DEVICES=2 python run.py \
	--do_test \
	--model_name_or_path microsoft/unixcoder-base \
	--test_filename ../data/JCSD/test.json \
	--output_dir saved_models/jcsd_half_0.025_normal \
	--max_source_length 512 \
	--max_target_length 128 \
	--beam_size 1 \
	--eval_batch_size 32	
CUDA_VISIBLE_DEVICES=2 python run.py \
	--do_test \
	--model_name_or_path microsoft/unixcoder-base \
	--test_filename ../data/JCSD/test.json \
	--output_dir saved_models/jcsd_half_0.025_random_x5 \
	--max_source_length 512 \
	--max_target_length 128 \
	--beam_size 1 \
	--eval_batch_size 32
CUDA_VISIBLE_DEVICES=2 python run.py \
	--do_test \
	--model_name_or_path microsoft/unixcoder-base \
	--test_filename ../data/JCSD/test.json \
	--output_dir saved_models/jcsd_half_0.025_Gini_x5 \
	--max_source_length 512 \
	--max_target_length 128 \
	--beam_size 1 \
	--eval_batch_size 32	