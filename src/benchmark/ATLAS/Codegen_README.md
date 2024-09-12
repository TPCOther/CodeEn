不添加数据不需要add_path
输入部分可能需要按需更改
```shell
python ./ms_evaluate.py \
    --model_name models/jcsd_half \
    --cuda 0

    CUDA_VISIBLE_DEVICES=0 python /data/swf/Assertion/src/benchmark/ATLAS/finetuned_codegen.py \
    --model_name_or_path "/data/swf/Assertion/codegen-base" \
    --data_path "Dataset/old/train.raw.lou.jsonl" \
    --valid_path "Dataset/old/val.raw.lou.jsonl" \
    --output_dir "/data/swf/Assertion/outputs/ATLAS_LOU/CodeGen" \
    --num_train_epochs 10 \
    --model_max_length 2048 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 16 \
    --evaluation_strategy "epoch" \
    --save_strategy "epoch" \
    --save_total_limit 10 \
    --learning_rate 5e-5 \
    --logging_steps 100 \
    --load_best_model_at_end True \
    --metric_for_best_model "eval_exact_match" \
    --gradient_checkpointing True \
    --fp16 True 
```