#!/bin/bash
# -*- coding: utf-8 -*-
#
# @author: Yang Liu <yangliu.real@gmail.com>
# @date: 2024/04/03

set -x

export WANDB_ENTITY=lm-lexicon
export WANDB_PROJECT=LM-Lexicon
export WANDB_ARTIFACT_LOCATION=/home/ivanfung/workspace/LM-Lexicon/sift/logs/wandb
export WANDB_ARTIFACT_DIR=/home/ivanfung/workspace/LM-Lexicon/sift/logs/wandb
export WANDB_CACHE_DIR=/home/ivanfung/workspace/LM-Lexicon/sift/logs/wandb
export WANDB_CONFIG_DIR=/home/ivanfung/workspace/LM-Lexicon/sift/logs/wandb

export https_proxy=http://127.0.0.1:7895 http_proxy=http://127.0.0.1:7895 all_proxy=socks5://127.0.0.1:7895

# Learning Rate
# 1e-6, 3e-6, 5e-6, 1e-5, 5e-5, 3e-4, 5e-4, 1e-3

#for model_name in "meta-llama/Llama-2-7b-chat-hf"
#for model_name in "meta-llama/Llama-2-13b-chat-hf"
#for model_name in "meta-llama/Llama-2-7b-chat-hf" "meta-llama/Llama-2-13b-chat-hf"
#for model_name in "meta-llama/Meta-Llama-3-8B-Instruct"
#for model_name in "meta-llama/Meta-Llama-3-8B" "meta-llama/Meta-Llama-3-8B-Instruct" "meta-llama/Llama-2-7b-hf" "meta-llama/Llama-2-7b-chat-hf"
for model_name in "meta-llama/Meta-Llama-3-8B"
do
    if [ "${model_name}" == "meta-llama/Meta-Llama-3-8B" ]; then
        model_type="llama-3"
    elif [ "${model_name}" == "meta-llama/Meta-Llama-3-8B-Instruct" ]; then
        model_type="llama-3"
    elif [ "${model_name}" == "meta-llama/Llama-2-7b-hf" ]; then
        model_type="llama-2"
    elif [ "${model_name}" == "meta-llama/Llama-2-7b-chat-hf" ]; then
        model_type="llama-2"
    else
        model_type="llama-3"
    fi
    model_name_suffix=$(echo "${model_name}" | rev | cut -d'/' -f1 | rev)
    #for dataset in "3D-EX"
    for dataset in "wordnet" "oxford" "wiki" "slang"
    #for dataset in "wordnet"
    do
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
            --output_dir /data2/checkpoints/results/${model_name_suffix}-${dataset} \
            --run_name ${model_name_suffix}-${dataset} \
            --mask_term "False" \
            --train_on_input "False" \
            --seq2seq_training "False" \
            --trainable_layers "all" \
            --batch_size 256 \
            --per_device_train_batch_size 64 \
            --per_device_eval_batch_size 128 \
            --epochs 5 \
            --learning_rate 1e-6 \
            --max_seq_len 128 \
            --eval_times 10 \
            --eval_sample_ratio 1.0 \
            --logging_steps 10 \
            --grad_accumulation_steps 1 > logs/${model_name_suffix}-${dataset}.log 2>&1 &

        pid=$!
        echo "Started training ${model_name_suffix} on ${dataset} with pid ${pid} ..."
        wait $pid
    done
done
