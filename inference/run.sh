#!/usr/bin/env bash

set -ex

python main.py \
  --task word-interpretation \
  --base_url "" \
  --api_key "" \
  --model /mnt/buffer/liuyang/Meta-Llama-3-4x8B-MoE-3D-EX-gate-ffn/Meta-Llama-3-4x8B-MoE-3D-EX/checkpoint-4452 \
  --prompt_path prompts/word-interpretation.txt \
  --example_path dataset/3D-EX/train.jsonl \
  --input_path dataset/3D-EX/test.8192.jsonl \
  --output_format json \
  --retrieval_policy random \
  --evaluate \
  --shot_num 0 \
  --max_query 8192 \
  --max_length 128 \
  --max_tokens 64 \
  --temperature 0.6 \
  --top_p 0.9 \
  --repetition_penalty 1.05 \
  --presence_penalty 0 \
  --frequency_penalty 0 \
  --num_return_sequences 1 \
  --run_local_model \
  --verbal

#for i in {1..128}
#for i in {118..128}
#do
    #python main.py \
        #--task word-interpretation \
        #--base_url "" \
        #--api_key "" \
        #--model /mnt/buffer/liuyang/lm-lexicon-dense-3d-ex-best-ckpt \
        #--prompt_path prompts/word-interpretation.txt \
        #--example_path dataset/3D-EX/train.jsonl \
        #--input_path dataset/3D-EX/test.jsonl \
        #--output_format json \
        #--retrieval_policy random \
        #--evaluate \
        #--shot_num 0 \
        #--max_query 64 \
        #--max_length 128 \
        #--max_tokens 64 \
        #--temperature 0.6 \
        #--top_p 0.9 \
        #--repetition_penalty 1.05 \
        #--presence_penalty 0 \
        #--frequency_penalty 0 \
        #--num_return_sequences ${i} \
        #--run_local_model
#done

#python main.py \
    #--task word-interpretation \
    #--base_url "" \
    #--api_key "" \
    #--model /mnt/buffer/liuyang/lm-lexicon-dense-oxford-best-ckpt \
    #--prompt_path prompts/word-interpretation.txt \
    #--example_path dataset/oxford/train.jsonl \
    #--input_path dataset/oxford/test.jsonl \
    #--output_format json \
    #--retrieval_policy random \
    #--evaluate \
    #--shot_num 0 \
    #--max_query 1000000 \
    #--max_length 128 \
    #--max_tokens 64 \
    #--temperature 0.6 \
    #--top_p 0.9 \
    #--repetition_penalty 1.05 \
    #--presence_penalty 0 \
    #--frequency_penalty 0 \
    #--num_return_sequences 32 \
    #--run_local_model \
    #--verbal
