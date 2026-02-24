#!/bin/bash

export WANDB_PROJECT=LM-Lexicon
export https_proxy=http://127.0.0.1:7895 http_proxy=http://127.0.0.1:7895 all_proxy=socks5://127.0.0.1:7896

model_type="llama"
model_name="meta-llama/Llama-2-13b-chat-hf"
model_name_suffix=$(echo "${model_name}" | rev | cut -d'/' -f1 | rev)

#nohup ./run.sh \
#--do_train \
#--enable_distributed_training \
#--enable_deepspeed \
#--model_type llama \
#--model_name_or_path meta-llama/Llama-2-7b-chat-hf \
#--data_path_train /home/ivanfung/workspace/LM-Lexicon/sift/dataset/train.jsonl \
#--data_path_valid /home/ivanfung/workspace/LM-Lexicon/sift/dataset/valid.jsonl \
#--output_dir /data2/checkpoints/results/Llama-2-7b-chat-hf-lm-lexicon \
#--batch_size 512 \
#--per_device_train_batch_size 128 \
#--per_device_eval_batch_size 1024 \
#--epochs 5 \
#--learning_rate 4e-5 \
#--max_seq_len 256 \
#--eval_steps 300 \
#--save_steps 300 \
#--logging_steps 10 \
#--grad_accumulation_steps 1 > log 2>&1 &

nohup ./run.sh \
    --do_train \
    --enable_distributed_training \
    --enable_deepspeed \
    --model_type ${model_type} \
    --model_name_or_path ${model_name} \
    --data_path_train /home/ivanfung/workspace/LM-Lexicon/sift/dataset/train.jsonl \
    --data_path_valid /home/ivanfung/workspace/LM-Lexicon/sift/dataset/valid.jsonl \
    --output_dir /data2/checkpoints/results/Llama-2-13b-chat-hf-lm-lexicon \
    --batch_size 256 \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 512 \
    --epochs 5 \
    --learning_rate 4e-5 \
    --max_seq_len 256 \
    --eval_steps 300 \
    --save_steps 300 \
    --logging_steps 10 \
    --grad_accumulation_steps 1 > logs/${model_name_suffix}.log 2>&1 &
