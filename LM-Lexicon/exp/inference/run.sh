#!/bin/zsh

# run word-interpretation task with 0-shot Llama-7B model
python main.py \
  --task word-interpretation \
  --base_url  https://api.deepinfra.com/v1/openai \
  --api_key 8VI1vtpURRNpzBV2kzTZPRbFqFGAhXcE \
  --model mistralai/Mixtral-8x7B-Instruct-v0.1 \
  --proxy http://127.0.0.1:7890 \
  --prompt_path prompts/word-interpretation_fewshot.txt \
  --example_path dataset/word-interpretation/prepared/examples.json \
  --input_path dataset/word-interpretation/prepared/word-interpretation_prepared.json \
  --output_path results/word-interpretation_3-shot_Mixtral-8x7B-Instruct-v0.1.json \
  --evaluate \
  --shot_num 3 \
  --max_query 10 \
  --max_tokens 256 \
  --temperature 0 \
  --presence_penalty 0 \
  --frequency_penalty 0
