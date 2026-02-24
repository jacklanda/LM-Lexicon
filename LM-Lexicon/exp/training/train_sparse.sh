#!/bin/bash
# -*- coding: utf-8 -*-
#
# @author: Yang Liu <yangliu.real@gmail.com>
# @date: 2024/04/03

set -x

export WANDB_ENTITY=lm-lexicon
export WANDB_PROJECT=LM-Lexicon
export WANDB_ARTIFACT_LOCATION=logs/wandb
export WANDB_ARTIFACT_DIR=logs/wandb
export WANDB_CACHE_DIR=logs/wandb
export WANDB_CONFIG_DIR=logs/wandb
export https_proxy=http://127.0.0.1:7895 http_proxy=http://127.0.0.1:7895 all_proxy=socks5://127.0.0.1:7895

# Learning Rate
# 1e-6, 3e-6, 5e-6, 1e-5, 5e-5, 3e-4, 5e-4, 1e-3

#dataset="wordnet"
#dataset="oxford"
#dataset="wiki"
#dataset="slang"
#dataset="3D-EX"

for model_name in "/data2/checkpoints/moe/LM-Lexicon-4xLlama-3-8b"
do
    model_type="llama-moe"
    model_name_suffix=$(echo "${model_name}" | rev | cut -d'/' -f1 | rev)
    #for dataset in "wordnet"
    for dataset in "3D-EX"
    do
    #for dataset in "wordnet" "oxford" "wiki" "slang"
    #for dataset in "wiki"
        # Traning Note:
        # - 1e-6 for wordnet, 5e-6 for 3D-EX
        nohup ./run.sh \
            --do_train --do_eval --do_predict \
            --enable_distributed_training \
            --enable_deepspeed \
            --model_type ${model_type} \
            --model_name_or_path ${model_name} \
            --data_path_train /home/ivanfung/workspace/LM-Lexicon/sift/dataset/${dataset}/train.jsonl \
            --data_path_valid /home/ivanfung/workspace/LM-Lexicon/sift/dataset/${dataset}/valid.jsonl \
            --data_path_test /home/ivanfung/workspace/LM-Lexicon/sift/dataset/${dataset}/test.jsonl \
            --output_dir /data2/checkpoints/results/${model_name_suffix}-${dataset}-gate-ffn \
            --run_name ${model_name_suffix}-${dataset}-gate-ffn \
            --mask_term "False" \
            --train_on_input "False" \
            --seq2seq_training "False" \
            --trainable_layers "gate,ffn" \
            --batch_size 128 \
            --per_device_train_batch_size 32 \
            --per_device_eval_batch_size 128 \
            --epochs 3 \
            --learning_rate 1e-6 \
            --max_seq_len 128 \
            --eval_times 30 \
            --eval_sample_ratio 0.01 \
            --logging_steps 10 \
            --grad_accumulation_steps 1 > logs/${model_name_suffix}-${dataset}.log 2>&1 &

        pid=$!
        echo "Started training ${model_name_suffix} on ${dataset} with pid ${pid} ..."
        wait $pid
    done
done
