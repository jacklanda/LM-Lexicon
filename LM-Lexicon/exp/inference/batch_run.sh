#!/bin/zsh

set -x

TASK=""
# claude-3-opus-20240229, gpt-4-turbo, gemini-pro
MODEL=""
DATASET=""
API_KEY=""
# https://ai56.top/v1
BASE_URL="None"

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
    esac
done

rm -f "logs/${TASK}-${MODEL}-${DATASET}.log"

for i in "0" "1" "2" "4" "8" "16" "32" "64" "128" "256" "512" "1024"
do
    nohup python main.py \
        --task ${TASK} \
        --base_url ${BASE_URL} \
        --api_key ${API_KEY} \
        --model ${MODEL} \
        --proxy http://127.0.0.1:7895 \
        --prompt_path prompts/word-interpretation.txt \
        --example_path "dataset/${DATASET}/train.jsonl" \
        --input_path "dataset/${DATASET}/test.jsonl" \
        --output_format json \
        --evaluate \
        --shot_num ${i} \
        --max_query 300 \
        --max_tokens 128 \
        --temperature 0 \
        --presence_penalty 0 \
        --frequency_penalty 0 >> "logs/${TASK}-${MODEL}-${DATASET}.log" 2>&1 &

    pid=$!
    echo "Started inference ${MODEL} on ${DATASET} with pid ${pid} ..."
    wait $pid
done
