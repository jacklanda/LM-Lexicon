#!/bin/bash

nohup python main.py \
  --task word-interpretation \
  --base_url https://api.aigc369.com/v1/completions \
  --api_key sk-Zsigo6XBRaIThEMAF7085712D9Bc4eF3A377B580023aD9Cb \
  --model claude-3-opus-20240229 \
  --proxy http://127.0.0.1:7895 \
  --prompt_path prompts/word-interpretation.txt \
  --example_path dataset/wordnet \
  --input_path dataset/wordnet/test.jsonl \
  --output_format json \
  --retrieval_policy random \
  --evaluate \
  --shot_num 1 \
  --max_query 500 \
  --max_tokens 128 \
  --temperature 0 \
  --presence_penalty 0 \
  --frequency_penalty 0 >> "logs/word-interpretation-claude-3-opus-20240229-wordnet-random_1shot.log" 2>&1 &
  #--verbal
  
  pid=$!
  echo "Started inference claude-3-opus-20240229 on wordnet with pid ${pid} ..."
