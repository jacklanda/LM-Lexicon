#!/bin/bash
# -*- coding: utf-8 -*-

set -x

export WANDB_ENTITY=definition-modeling
export WANDB_PROJECT=dm
export WANDB_ARTIFACT_LOCATION=logs/wandb
export WANDB_ARTIFACT_DIR=logs/wandb
export WANDB_CACHE_DIR=logs/wandb
export WANDB_CONFIG_DIR=logs/wandb

#export https_proxy=http://127.0.0.1:7890 http_proxy=http://127.0.0.1:7890 all_proxy=socks5://127.0.0.1:7890

# Learning Rate
# 1e-6, 3e-6, 5e-6, 1e-5, 5e-5, 3e-4, 5e-4, 1e-3

#for model_name in "meta-llama/Meta-Llama-3-8B-Instruct"
#for model_name in "meta-llama/Meta-Llama-3.1-8B"
#for model_name in "meta-llama/Meta-Llama-3-8B"
for model_name in "/share/nlp/share/plm/Qwen3-8B-Base"
do
    if [ "${model_name}" == "/share/nlp/share/plm/Qwen3-8B-Base" ]; then
        model_type="qwen3"
    elif [ "${model_name}" == "meta-llama/Meta-Llama-3-8B" ]; then
        model_type="llama-3"
    elif [ "${model_name}" == "meta-llama/Meta-Llama-3.1-8B" ]; then
        model_type="llama-3"
    elif [ "${model_name}" == "meta-llama/Meta-Llama-3-8B-Instruct" ]; then
        model_type="llama-3"
    elif [ "${model_name}" == "meta-llama/Meta-Llama-3.1-8B-Instruct" ]; then
        model_type="llama-3"
    fi
    model_name_suffix=$(echo "${model_name}" | rev | cut -d'/' -f1 | rev)
    #for dataset in "wordnet" "oxford" "wiki" "slang"
    #for dataset in "wordnet"
    for dataset in "oxford"
    #for dataset in "wiki"
    #for dataset in "slang"
    #for dataset in "3D-EX"
    do
        # Traning Note:
        # - 1e-6 for wordnet, 5e-6 for 3D-EX
        ./run.sh \
            --do_train --do_eval --do_predict \
            --enable_distributed_training \
            --enable_deepspeed \
            --model_type ${model_type} \
            --model_name_or_path ${model_name} \
            --data_path_train dataset/${dataset}/train.jsonl \
            --data_path_valid dataset/${dataset}/test.jsonl \
            --data_path_test dataset/${dataset}/test.jsonl \
            --output_dir output/${model_name_suffix}-${dataset} \
            --run_name ${model_name_suffix}-${dataset} \
            --mask_term "False" \
            --train_on_input "False" \
            --seq2seq_training "False" \
            --trainable_layers "all" \
            --batch_size 1024 \
            --per_device_train_batch_size 256 \
            --per_device_eval_batch_size 128 \
            --epochs 10 \
            --learning_rate 5e-6 \
            --max_seq_len 128 \
            --eval_times 30 \
            --eval_samples 8192 \
            --logging_steps 10 \
            --grad_accumulation_steps 1

        pid=$!
        echo "Started training ${model_name_suffix} on ${dataset} with pid ${pid} ..."
        wait $pid
    done
done
