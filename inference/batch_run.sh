#!/bin/bash

set -x

export CUDA_VISIBLE_DEVICES=0

# TASK
TASK="word-interpretation"

# MODEL
# claude-3-opus-20240229, gpt-4-turbo, gemini-1.5-pro-latest
#MODEL="gpt-4-turbo-2024-04-09"
#MODEL="gemini-1.5-pro-latest"
#MODEL="gpt-4o-2024-08-06"
#MODEL="claude-3-haiku-20240307"
#MODEL="claude-3-5-sonnet-20240620"
#MODEL="claude-3-opus-20240229"
#MODEL="meta-llama/Meta-Llama-3-8B-Instruct"
#MODEL="~/workspace/dm/training/output/checkpoint-5360-Slang-gate-ffn/checkpoint-5400"
#MODEL="gpt-4o-2024-11-20"
MODEL="claude-3-5-sonnet-20241022"
#MODEL="claude-3-opus-20240229"
#MODEL="gemini-1.5-pro-latest"
#MODEL="claude-3-haiku-20240307"
#MODEL="/scratch2/nlp/liuyang/lm-lexicon-dense-wordnet-best-ckpt"
DATASET="wordnet"
#DATASET="3D-EX"

# API KEY and BASE URL
#API_KEY="sk-Zsigo6XBRaIThEMAF7085712D9Bc4eF3A377B580023aD9Cb"
#API_KEY="qCyEEbxzmFy2V2SRGAs19ISUuGw6CgYi"
#BASE_URL="http://0.0.0.0:8888/v1"
API_KEY="sk-Zsigo6XBRaIThEMAF7085712D9Bc4eF3A377B580023aD9Cb"
BASE_URL="https://api.aigc369.com/v1"

#API_KEY="4d68cab5a8a3d8bc29784d280678027b"
#BASE_URL="https://api.tonggpt.mybigai.ac.cn/proxy/eastus2"

# IN-CONTEXT LEARNING POLICY
ICL_POLICY="random"
#ICL_POLICY="topk"


while [[ $# -gt 0 ]]; do
    case $1 in
        --task)
            TASK=$2
            shift # past argument
            shift # past value
            ;;
        --model)
            MODEL=$2
            shift # past argument
            shift # past value
            ;;
        --dataset)
            DATASET=$2
            shift # past argument
            shift # past value
            ;;
        --api_key)
            API_KEY=$2
            shift # past argument
            shift # past value
            ;;
        --base_url)
            BASE_URL=$2
            shift # past argument
            shift # past value
            ;;
        --icl_policy)
            ICL_POLICY=$2
            shift # past argument
            shift # past value
            ;;
    esac
done

for i in "32 64 128";
do
    #for dataset in "wordnet" "oxford" "wiki" "3D-EX";
    #for dataset in "oxford" "wiki" "3D-EX";
    #for dataset in "slang";
    #for dataset in "wordnet" "oxford" "wiki" "slang" "3D-EX";
    for dataset in "wordnet";
    #for dataset in "3D-EX";
    do
        /home/liuyang/app/anaconda3/envs/workspace/bin/python main.py \
            --task ${TASK} \
            --base_url ${BASE_URL} \
            --api_key ${API_KEY} \
            --model ${MODEL} \
            --prompt_path prompts/word-interpretation.txt \
            --example_path "dataset/${dataset}" \
            --input_path "dataset/${dataset}/test.jsonl" \
            --output_format json \
            --retrieval_policy ${ICL_POLICY} \
            --evaluate \
            --shot_num ${i} \
            --max_query 100 \
            --max_length 128 \
            --max_tokens 64 \
            --temperature 0.6 \
            --top_p 0.9 \
            --repetition_penalty 1.05 \
            --presence_penalty 0 \
            --frequency_penalty 0 \
            --num_return_sequences 1 \
            --verbal

        pid=$!
        echo "Finished inference ${MODEL} on ${dataset} with pid ${pid} ..."
        wait $pid
    done
done
