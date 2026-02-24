#!/bin/bash

# @Author: Yang Liu <yangliu.real@gmail.com>
# @Date: 2024/04/06

# This script is used to run the API server of vLLM.

# Available models:
# - lmsys/vicuna-7b-v1.5
# - NousResearch/Llama-2-7b-chat-hf
# - mistralai/Mistral-7B-Instruct-v0.2 (./Mistral-7B-Instruct-v0.2)
# - THUDM/chatglm3-6b
# - 01-ai/Yi-6B-Chat
# - deepseek-ai/deepseek-llm-7b-chat
# - deepseek-ai/deepseek-moe-16b-chat
# - google/gemma-7b-it
# - meta-llama/Llama-2-7b-chat-hf
# - meta-llama/Llama-2-13b-chat-hf
# - meta-llama/Llama-2-70b-chat-hf

# Use custom proxy and hosted mirror of HF for model downloading
#export https_proxy=http://127.0.0.1:7895 http_proxy=http://127.0.0.1:7895 all_proxy=socks5://127.0.0.1:7896
# Use custom proxy mirror of the Hugging Face
# HF_ENDPOINT=https://hf-mirror.com 

# Set CUDA devices visibility
export CUDA_VISIBLE_DEVICES=0
#export CUDA_VISIBLE_DEVICES=3

if (( $# < 1 )); then
    echo "Warning: Please provide an model name (reference: https://huggingface.co/models)!"
    exit -1
fi

# Config for inference
model=$1
host="localhost"
port=8888
gpu_memory_utilization=0.95
max_model_len=4096  # Model context length. If unspecified, will be automatically derived from the model config
max_num_seqs=4096  # Maximum number of sequences per iteration
#model_precision="half"
model_precision="bfloat16"
#model_precision="float32"
max_log_len=0
seed=42
tensor_parallel_size=1
pipeline_parallel_size=1


python -m vllm.entrypoints.openai.api_server \
    --host $host \
    --port $port \
    --model $model \
    --trust-remote-code \
    --gpu-memory-utilization $gpu_memory_utilization \
    --max-model-len $max_model_len \
    --max-num-seqs $max_num_seqs \
    --dtype $model_precision \
    --max-log-len $max_log_len \
    --seed $seed \
    --tensor-parallel-size $tensor_parallel_size \
    --disable-log-requests \
    --disable-log-stats
