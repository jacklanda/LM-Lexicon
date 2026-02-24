#!/bin/bash
# -*- coding: utf-8 -*-
#
# @author: Yang Liu <yangliu.real@gmail.com>
# @date: 2024/04/03

# Use custom proxy and hosted mirror of HF for model downloading
export https_proxy=http://127.0.0.1:7895 http_proxy=http://127.0.0.1:7895 all_proxy=socks5://127.0.0.1:7896
# Use custom proxy mirror of the Hugging Face
#export HF_ENDPOINT=https://hf-mirror.com

set -x

export CUDA_VISIBLE_DEVICES=0,1,2,3
#export NCCL_DEBUG=INFO

WORLD_SIZE=4

MODEL_TYPE=""
MODEL_NAME_OR_PATH=""
RUN_NAME=""
DO_TRAIN="False"
DO_EVAL="False"
DO_PREDICT="False"
UPLOAD_CKPT="False"
ENABLE_DEEPSPEED="False"
FSDP=""
FSDP_WRAP_LAYER=""
ENABLE_DISTRIBUTED_TRAINING="False"
ENABLE_DISTRIBUTED_INFERENCE="False"

# Hyperparameter settings
BATCH_SIZE=
PER_DEVICE_TRAIN_BATCH_SIZE=
PER_DEVICE_EVAL_BATCH_SIZE=
EVAL_SAMPLE_RATIO=
EPOCHS=
LR=
MAX_SEQ_LEN=
EVAL_TIMES=
LOGGING_STEPS=
GRAD_ACCUMULATION_STEPS=
TRAIN_ON_INPUT="True"
MASK_TERM="False"
SEQ2SEQ_TRAINING="False"
TRAINABLE_LAYERS="all"

# Path settings
OUTPUT_DIR=""
OUTPUT_FNAME_INFER=""
DATA_PATH_TRAIN=""
DATA_PATH_VALID=""
DATA_PATH_TEST=""
PRED_RESULT_PATH=""
MODEL_CONFIG_PATH="run_config/model_config.json"
#DEEPSPEED_CONFIG_PATH="run_config/deepspeed_config_zero1.json"  # ✅
#DEEPSPEED_CONFIG_PATH="run_config/deepspeed_config_zero2.json"  # ✅
DEEPSPEED_CONFIG_PATH="run_config/deepspeed_config_zero3_without_offload.json"  # ✅
#DEEPSPEED_CONFIG_PATH="run_config/deepspeed_config_zero3_with_offload.json"  # ✅
#DEEPSPEED_CONFIG_PATH="run_config/deepspeed_config_zeropp.json"  # ❌

if [ $# -lt 1 ]; then
    echo "[ERROR] Your command line contains $# arguments, please double check it!"
    exit 1
fi

# Generate configs for model & deepspeed
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_type)
            MODEL_TYPE=$2
            shift # past argument
            shift # past value
            ;;
        --model_name_or_path)
            MODEL_NAME_OR_PATH=$2
            shift # past argument
            shift # past value
            ;;
        --run_name)
            RUN_NAME=$2
            shift # past argument
            shift # past value
            ;;
        --do_train)
            DO_TRAIN="True"
            shift # past argument
            ;;
        --do_eval)
            DO_EVAL="True"
            shift # past argument
            ;;
        --do_predict)
            DO_PREDICT="True"
            shift # past argument
            ;;
        --upload_ckpt)
            UPLOAD_CKPT="True"
            shift # past argument
            ;;
        --fsdp)
            FSDP=$2
            shift
            ;;
        --fsdp_wrap_layer)
            FSDP_WRAP_LAYER=$2
            shift
            ;;
        --enable_deepspeed)
            ENABLE_DEEPSPEED="True"
            shift
            ;;
        --enable_distributed_training)
            ENABLE_DISTRIBUTED_TRAINING="True"
            shift
            ;;
        --enable_distributed_inference)
            ENABLE_DISTRIBUTED_INFERENCE="True"
            shift
            ;;
        --data_path_train)
            DATA_PATH_TRAIN=$2
            shift # past argument
            shift # past value
            ;;
        --data_path_valid)
            DATA_PATH_VALID=$2
            shift # past argument
            shift # past value
            ;;
        --data_path_test)
            DATA_PATH_TEST=$2
            shift # past argument
            shift # past value
            ;;
        --pred_result_path)
            PRED_RESULT_PATH=$2
            shift
            shift
            ;;
        -o|--output_dir)
            OUTPUT_DIR=$2
            shift # past argument
            shift # past value
            ;;
        --output_fname_infer)
            OUTPUT_FNAME_INFER=$2
            shift # past argument
            shift # past value
            ;;
        --mask_term)
            MASK_TERM=$2
            shift # past argument
            ;;
        -b|--batch_size)
            BATCH_SIZE=$2
            shift # past argument
            shift # past value
            ;;
        --per_device_train_batch_size)
            PER_DEVICE_TRAIN_BATCH_SIZE=$2
            shift # past argument
            shift # past value
            ;;
        --per_device_eval_batch_size)
            PER_DEVICE_EVAL_BATCH_SIZE=$2
            shift # past argument
            shift # past value
            ;;
        -e|--epochs)
            EPOCHS=$2
            shift # past argument
            shift # past value
            ;;
        --learning_rate)
            LR=$2
            shift # past argument
            shift # past value
            ;;
        -l|--max_seq_len)
            MAX_SEQ_LEN=$2
            shift # past argument
            shift # past value
            ;;
        --dev_set_size)
            DEV_SET_SIZE=$2
            shift # past argument
            shift # past value
            ;;
        --eval_times)
            EVAL_TIMES=$2
            shift # past argument
            shift # past value
            ;;
        --eval_sample_ratio)
            EVAL_SAMPLE_RATIO=$2
            shift # past argument
            shift # past value
            ;;
        --train_on_input)
            TRAIN_ON_INPUT=$2
            shift # past argument
            shift # past value
            ;;
        --seq2seq_training)
            SEQ2SEQ_TRAINING=$2
            shift # past argument
            shift # past value
            ;;
        --trainable_layers)
            TRAINABLE_LAYERS=$2
            shift # past argument
            shift # past value
            ;;
        --logging_steps)
            LOGGING_STEPS=$2
            shift # past argument
            shift # past value
            ;;
        --grad_accumulation_steps)
            GRAD_ACCUMULATION_STEPS=$2
            shift # past argument
            shift # past value
            ;;
        -*|--*)
            echo "[ERROR] Unknown option $1"
            exit 1
            ;;
        *)
            POSITIONAL_ARGS+=("$1") # save positional arg
            shift # past argument
            ;;
    esac
done

JSON_DATA=$(cat <<EOF
{
    "do_train": "${DO_TRAIN}",
    "do_eval": "${DO_EVAL}",
    "do_predict": "${DO_PREDICT}",
    "model_type": "${MODEL_TYPE}",
    "model_name_or_path": "${HDFS_PATH_PREFIX}${MODEL_NAME_OR_PATH}",
    "run_name": "${RUN_NAME}",
    "data_path_train": "${HDFS_PATH_PREFIX}${DATA_PATH_TRAIN}",
    "data_path_valid": "${HDFS_PATH_PREFIX}${DATA_PATH_VALID}",
    "data_path_test": "${HDFS_PATH_PREFIX}${DATA_PATH_TEST}",
    "mask_term": "${MASK_TERM}",
    "output_dir": "${OUTPUT_DIR}",
    "batch_size": ${BATCH_SIZE},
    "per_device_train_batch_size": ${PER_DEVICE_TRAIN_BATCH_SIZE},
    "per_device_eval_batch_size": ${PER_DEVICE_EVAL_BATCH_SIZE},
    "num_epochs": ${EPOCHS},
    "learning_rate": ${LR},
    "cutoff_len": ${MAX_SEQ_LEN},
    "train_on_input": "${TRAIN_ON_INPUT}",
    "seq2seq_training": "${SEQ2SEQ_TRAINING}",
    "trainable_layers": "${TRAINABLE_LAYERS}",
    "eval_times": ${EVAL_TIMES},
    "eval_sample_ratio": ${EVAL_SAMPLE_RATIO},
    "logging_steps": ${LOGGING_STEPS},
    "gradient_accumulation_steps": ${GRAD_ACCUMULATION_STEPS},
    "fsdp": "${FSDP}",
    "fsdp_wrap_layer": "${FSDP_WRAP_LAYER}"
}
EOF
)
# "output_dir": "${PWD}/${OUTPUT_DIR}",

echo $JSON_DATA
echo ${JSON_DATA} > ${MODEL_CONFIG_PATH}

# Prepare environment
echo "[INFO] Checking essential binaries path..."
echo `which pip`
echo `which conda`
echo `which python`
echo `/opt/conda/bin/python --version`
echo "[CMD] cat run_config/model_config.json"
cat ${MODEL_CONFIG_PATH}
echo "[CMD] cat run_config/deepspeed_config.json"
cat ${DEEPSPEED_CONFIG_PATH}
echo "[SUCC] Checking essential binaries path success!"

# Install dependent packages using local pip
#echo "[INFO] Install dependent packages..."
# sudo apt install -y clang  # using clang as default Cpp compiler
#echo "[CMD] sudo apt update"
#sudo apt update
#echo "[CMD] sudo apt install -y g++ & export CXX=g++"
#sudo apt install -y g++
#export CXX=g++
#echo "[CMD] sudo apt install -y libsnappy-dev"
#sudo apt install -y libsnappy-dev
#echo "[CMD] sudo apt install -y ninja-build"
#sudo apt install -y ninja-build
#echo "[CMD] sudo apt install -y git-lfs"
#sudo apt install -y git-lfs
#echo "[CMD] git lfs install"
#git lfs install

# pip
#pip install ninja
#pip install Ninja
#pip install datasets
#pip install loguru
#pip install netifaces
#pip install transformers==4.29.2
#pip install transformers==4.28.1
#pip install accelerate
#pip install --upgrade accelerate
#pip install git+https://github.com/huggingface/transformers.git
#pip install wandb
#pip install peft==0.2.0
#pip install git+https://github.com/huggingface/peft.git
#pip install python-snappy
#pip install zstandard
#pip install deepspeed
#pip install sentencepiece
#pip install tokenizers==0.13.3
#pip install bitsandbytes
#pip install fire
#pip install gradio
#pip install faiss-gpu
#pip install torchaudio
#pip install torchvision
#echo "[SUCC] Dependent packages install success!"

# do training or not

# DDP
#OMP_NUM_THREADS=12 python -m torch.distributed.run --nnodes=1 --nproc_per_node=4 finetune.py --model_config_file ${MODEL_CONFIG_PATH} || exit -1

# DeepSpeed
deepspeed --num_gpus=4 finetune.py --model_config_file ${MODEL_CONFIG_PATH} --deepspeed ${DEEPSPEED_CONFIG_PATH} || exit -1

# Run with naive pipeline parallelism implemented in transformers (w/ `device_map="auto"`)
#python -u finetune.py --model_config_file ${MODEL_CONFIG_PATH} || exit -1
#python -u finetune.py --model_config_file ${MODEL_CONFIG_PATH} --deepspeed ${DEEPSPEED_CONFIG_PATH} || exit -1

# do prediction or not
 #python -u generate.py --dev_file "<test file path>" --model_type ${MODEL_TYPE} --model_name_or_path ${MODEL_NAME_OR_PATH} --max_length ${MAX_SEQ_LEN} --dev_batch_size ${PER_DEVICE_EVAL_BATCH_SIZE} --output_file ${output_path} > /dev/stdout 2>&1 &

echo "[INFO] All tasks done!"

exit 0
